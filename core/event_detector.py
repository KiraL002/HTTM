"""
AcciVision — Module Phát Hiện & Quản Lý Sự Kiện Tai Nạn (Event Detector)

Module này quản lý vòng đời của các sự kiện tai nạn giao thông:
khởi tạo, xác nhận, gom đa phương tiện, phân loại loại tai nạn,
và đóng sự kiện chính xác khi tình huống ổn định.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class AccidentEvent:
    """
    Một sự kiện tai nạn giao thông được phát hiện và quản lý theo vòng đời.

    Vòng đời sự kiện:
    - START: Được xác nhận sau `confirmation_frames` liên tiếp vượt ngưỡng `start_threshold`.
    - ACTIVE: Đang diễn ra, liên tục cập nhật xác suất đỉnh (peak probability) và gom xe liên quan.
    - END: Tự động đóng (active=False) khi các xe đã dừng/ổn định, hoặc xác suất giảm dưới ngưỡng
           `release_threshold` trong `release_frames`. Thời điểm kết thúc (end_frame) được chốt tại
           frame cuối cùng có hoạt động va chạm, KHÔNG kéo dài đến cuối video.
    """

    event_id: int
    start_frame: int
    end_frame: int
    peak_frame: int
    probability: float
    track_ids: Set[int] = field(default_factory=set)
    vehicle_types: Dict[int, str] = field(default_factory=dict)
    accident_type: str = "unknown"
    type_title: str = "⚠️ Sự cố giao thông"
    active: bool = True
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    duration_s: float = 0.0
    description: str = ""
    snapshot_base64: Optional[str] = None
    crop_path: Optional[str] = None
    peak_bbox: Optional[Tuple[int, int, int, int]] = None
    track_bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)

    @property
    def duration_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame + 1)


class EventDetector:
    """
    Quản lý vòng đời phát hiện sự kiện tai nạn, gom sự kiện đa phương tiện (multi-vehicle aggregation),
    phân loại loại tai nạn (accident type), và đóng sự kiện chính xác (event closure).
    """

    def __init__(
        self,
        fps: float = 30.0,
        start_threshold: float = 0.75,
        confirmation_frames: int = 8,
        release_threshold: float = 0.35,
        release_frames: int = 12,
        spatial_merge_distance: float = 6.0,
        max_event_duration_frames: int = 150,
        min_track_age: int = 6,
    ) -> None:
        self.fps = float(fps if fps > 0 else 30.0)
        self.start_threshold = float(start_threshold)
        self.confirmation_frames = int(max(1, confirmation_frames))
        self.release_threshold = float(release_threshold)
        self.release_frames = int(max(1, release_frames))
        self.spatial_merge_distance = float(spatial_merge_distance)
        self.max_event_duration_frames = int(max_event_duration_frames)
        self.min_track_age = int(max(1, min_track_age))

        self._track_first_frame: Dict[int, int] = {}
        self._track_streak_start: Dict[int, int] = {}

        # Lịch sử xác suất gần nhất của từng track
        self._history: Dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        # Frame gần nhất mà track có xác suất cao
        self._track_last_accident_frame: Dict[int, int] = {}
        # Số frame liên tiếp xác suất thấp của từng track
        self._track_calm_frames: Dict[int, int] = defaultdict(int)

        # Trạng thái vị trí và hướng gần nhất của các track
        self._track_positions: Dict[int, np.ndarray] = {}
        self._track_directions: Dict[int, float] = {}
        self._track_speeds: Dict[int, float] = {}
        self._track_features: Dict[int, Dict[str, Any]] = {}
        self._track_bboxes: Dict[int, Any] = {}
        self._event_crash_positions: Dict[int, np.ndarray] = {}
        self._event_initial_directions: Dict[Tuple[int, int], float] = {}

        # Ánh xạ track_id -> AccidentEvent đang active
        self._active_by_track: Dict[int, AccidentEvent] = {}
        self._events: List[AccidentEvent] = []
        self._next_id: int = 1

    def _update_accident_classification_and_description(
        self,
        event: AccidentEvent,
    ) -> None:
        """
        Xác định chính xác loại tai nạn và tạo mô tả chi tiết, rõ ràng, tự nhiên theo chuẩn tiếng Việt.
        """
        tids = sorted(list(event.track_ids))
        num_vehicles = len(tids)
        vtypes = event.vehicle_types

        def vn_vehicle_name(cls_name: str) -> str:
            cls = str(cls_name).lower()
            if cls in ("car", "sedan", "suv"):
                return "ô tô con"
            if cls in ("truck", "lorry"):
                return "xe tải"
            if cls in ("bus",):
                return "xe buýt"
            if cls in ("motorcycle", "motorbike", "scooter"):
                return "xe máy"
            if cls in ("bicycle", "bike"):
                return "xe đạp"
            if cls in ("person", "pedestrian"):
                return "người đi bộ"
            return "phương tiện"

        labels = [f"Xe #{tid} ({vn_vehicle_name(vtypes.get(tid, vtypes.get(str(tid), 'car')))})" for tid in tids]
        classes = [str(vtypes.get(tid, vtypes.get(str(tid), "car"))).lower() for tid in tids]

        has_motorcycle = any(c in ("motorcycle", "motorbike", "scooter") for c in classes)
        has_bicycle = any(c in ("bicycle", "bike") for c in classes)
        has_car_or_large = any(c in ("car", "truck", "bus") for c in classes)
        has_pedestrian = any(c in ("person", "pedestrian", "car_person_accident", "injured") for c in classes)
        has_direct_accident_cls = any(c in ("accident", "car_car_accident") for c in classes)

        # 1. Va chạm xe máy với ô tô / xe tải / xe buýt (Ưu tiên trước vì người lái xe máy ngã ra có thể tạo thêm track person)
        if has_motorcycle and (has_car_or_large or has_direct_accident_cls):
            event.accident_type = "car_motorcycle_collision"
            event.type_title = "💥 Va chạm xe máy với ô tô"
            event.description = (
                f"Va chạm giữa {', '.join(labels)} tại giây thứ {event.start_time_s:.2f}s "
                f"(Frame #{event.start_frame} - #{event.end_frame}). Hai phương tiện xung đột quỹ đạo ở tốc độ cao, "
                f"tiếp xúc trực tiếp khiến xe máy mất thăng bằng."
            )
            return

        # 1b. Xe máy tự ngã hoặc người lái văng khỏi xe máy (chỉ có xe máy + người lái, không có ô tô/xe đạp)
        if (
            has_motorcycle
            and not has_car_or_large
            and not has_bicycle
            and len([c for c in classes if c in ("motorcycle", "motorbike", "scooter")]) == 1
        ):
            event.accident_type = "motorcycle_fall_or_crash"
            event.type_title = "🏍️ Tai nạn xe máy tự ngã / Trượt đổ"
            event.description = (
                f"Sự cố tai nạn xe máy ({', '.join(labels)}) mất thăng bằng và trượt ngã trên mặt đường "
                f"tại giây thứ {event.start_time_s:.2f}s (Frame #{event.start_frame} - #{event.end_frame})."
            )
            return

        # 1c. Va chạm giữa xe máy và xe đạp / xe 2 bánh
        if (has_motorcycle and has_bicycle) or (
            (has_motorcycle or has_bicycle) and num_vehicles >= 2 and not has_car_or_large
        ):
            event.accident_type = "car_motorcycle_collision"
            v_pair = "xe máy với xe đạp" if (has_motorcycle and has_bicycle) else "giữa các phương tiện 2 bánh"
            event.type_title = f"💥 Va chạm {v_pair}"
            event.description = (
                f"Va chạm trực tiếp giữa {', '.join(labels)} tại giây thứ {event.start_time_s:.2f}s "
                f"(Frame #{event.start_frame} - #{event.end_frame}) khiến phương tiện mất thăng bằng và ngã đổ trên mặt đường."
            )
            return

        # 2. Va chạm với người đi bộ hoặc người đi xe đạp
        if (has_pedestrian or has_bicycle) and (has_car_or_large or has_motorcycle or has_direct_accident_cls):
            event.accident_type = "vehicle_pedestrian_collision"
            target_vn = "người đi xe đạp" if has_bicycle else "người đi bộ"
            event.type_title = f"⚠️ Phương tiện va chạm {target_vn}"
            event.description = (
                f"Sự cố va chạm nghiêm trọng giữa {', '.join(labels)} tại giây thứ {event.start_time_s:.2f}s "
                f"(Frame #{event.start_frame} - #{event.end_frame}). Phát hiện va chạm trực tiếp liên quan đến {target_vn} "
                f"trên phần đường xe chạy."
            )
            return

        # 2b. Nhãn tai nạn trực tiếp từ mô hình YOLO chuyên dụng (accident / car_car_accident)
        if has_direct_accident_cls and num_vehicles == 1:
            event.accident_type = "multi_vehicle_collision"
            event.type_title = "💥 Va chạm giao thông trực tiếp"
            event.description = (
                f"Phát hiện hiện trường va chạm giao thông trực tiếp tại giây thứ {event.start_time_s:.2f}s "
                f"(Frame #{event.start_frame} - #{event.end_frame}) với độ tin cậy cao ({event.probability*100:.1f}%)."
            )
            return

        # 2c. Va chạm giữa 2 xe máy / xe 2 bánh
        if (has_motorcycle or has_bicycle) and num_vehicles >= 2 and not has_car_or_large:
            event.accident_type = "car_motorcycle_collision"
            event.type_title = "💥 Va chạm giữa các phương tiện 2 bánh"
            event.description = (
                f"Va chạm trực tiếp giữa {', '.join(labels)} tại giây thứ {event.start_time_s:.2f}s "
                f"(Frame #{event.start_frame} - #{event.end_frame}) khiến phương tiện ngã đổ trên mặt đường."
            )
            return

        # 3. Va chạm liên hoàn nhiều xe (>= 3 xe)
        if num_vehicles >= 3:
            event.accident_type = "multi_vehicle_pileup"
            event.type_title = f"⛓️ Va chạm liên hoàn ({num_vehicles} xe)"
            event.description = (
                f"Tai nạn liên hoàn liên quan đến {num_vehicles} phương tiện: {', '.join(labels)} "
                f"tại giây thứ {event.start_time_s:.2f}s (Frame #{event.start_frame} - #{event.end_frame}). "
                f"Các xe va chạm dồn toa liên tiếp do không giữ khoảng cách an toàn."
            )
            return

        # 4. Va chạm giữa 2 phương tiện (ô tô - ô tô hoặc xe tải - ô tô...)
        if num_vehicles == 2:
            t1, t2 = tids[0], tids[1]
            dir1 = self._event_initial_directions.get((event.event_id, t1), self._track_directions.get(t1))
            dir2 = self._event_initial_directions.get((event.event_id, t2), self._track_directions.get(t2))

            angle_diff = None
            if dir1 is not None and dir2 is not None:
                angle_diff = abs((dir1 - dir2 + 180.0) % 360.0 - 180.0)

            if angle_diff is not None and angle_diff > 125.0:
                event.accident_type = "head_on_collision"
                event.type_title = "⚡ Va chạm đối đầu (Trực diện)"
                event.description = (
                    f"Va chạm đối đầu trực diện giữa {labels[0]} và {labels[1]} tại giây thứ {event.start_time_s:.2f}s "
                    f"(Frame #{event.start_frame} - #{event.end_frame}). Hai xe lưu thông ngược chiều với góc lệch {angle_diff:.0f}°, "
                    f"đâm trực diện ở tốc độ cao."
                )
                return

            if angle_diff is not None and angle_diff <= 45.0:
                pos1 = self._track_positions.get(t1)
                pos2 = self._track_positions.get(t2)
                is_side_swipe = False
                if pos1 is not None and pos2 is not None and dir1 is not None:
                    rad = np.radians(float(dir1))
                    ux, uy = float(np.cos(rad)), float(np.sin(rad))
                    dx = float(pos2[0] - pos1[0])
                    dy = float(pos2[1] - pos1[1])
                    long_off = abs(dx * ux + dy * uy)
                    lat_off = abs(-dx * uy + dy * ux)
                    if lat_off > 1.75 and lat_off > long_off * 0.65:
                        is_side_swipe = True

                if is_side_swipe:
                    event.accident_type = "side_impact_collision"
                    event.type_title = "📐 Va chạm sườn xe / Chuyển làn"
                    event.description = (
                        f"Va chạm quệt sườn xe khi chuyển làn giữa {labels[0]} và {labels[1]} tại giây thứ {event.start_time_s:.2f}s "
                        f"(Frame #{event.start_frame} - #{event.end_frame}). Hai phương tiện áp sát thân xe dẫn đến xung đột quỹ đạo."
                    )
                    return

                event.accident_type = "rear_end_collision"
                event.type_title = "💥 Va chạm phía sau (Tông đuôi)"
                event.description = (
                    f"Va chạm tông đuôi phía sau giữa {labels[0]} và {labels[1]} tại giây thứ {event.start_time_s:.2f}s "
                    f"(Frame #{event.start_frame} - #{event.end_frame}). Phương tiện phía sau không làm chủ tốc độ "
                    f"khi xe phía trước giảm tốc đột ngột."
                )
                return

            # Mặc định góc cắt ngang / giao lộ / ngã tư
            event.accident_type = "side_impact_collision"
            event.type_title = "📐 Va chạm sườn xe / Giao lộ"
            ang_txt = f" (góc tiếp cận {angle_diff:.0f}°)" if angle_diff is not None else ""
            event.description = (
                f"Va chạm góc sườn xe giữa {labels[0]} và {labels[1]} tại giao lộ / ngã rẽ{ang_txt} "
                f"ở giây thứ {event.start_time_s:.2f}s (Frame #{event.start_frame} - #{event.end_frame}). "
                f"Hai xe tiếp xúc thân xe mạnh khi chuyển hướng lưu thông qua nút giao."
            )
            return

        # 5. Sự cố đơn phương tiện (Single-vehicle)
        if num_vehicles == 1:
            tid = tids[0]
            feat = self._track_features.get(tid, {})
            yaw_rate = abs(float(feat.get("yaw_rate_dps", 0.0)))
            dir_change_sum = float(feat.get("direction_change_sum_deg", 0.0))
            sudden_stop = int(feat.get("sudden_stop", 0))
            stopped_after_motion = int(feat.get("stopped_after_motion", 0))
            speed_drop = float(feat.get("speed_drop_window_mps", 0.0))
            init_spd = float(feat.get("speed_mps", 0.0)) * 3.6
            bbox_deform = int(feat.get("bbox_deformation", 0))
            ar_change = float(feat.get("bbox_aspect_change_max", 0.0))

            if has_motorcycle or has_bicycle:
                event.accident_type = "motorcycle_fall_or_crash"
                v_title = "xe đạp" if (has_bicycle and not has_motorcycle) else "xe máy"
                event.type_title = f"🏍️ Tai nạn {v_title} tự ngã / Trượt đổ"
                deform_txt = (
                    f", khung bao biến dạng đột ngột ({ar_change*100:.0f}%)"
                    if (bbox_deform == 1 or ar_change >= 0.40)
                    else ""
                )
                event.description = (
                    f"{labels[0]} mất thăng bằng và trượt ngã trên mặt đường{deform_txt} "
                    f"tại giây thứ {event.start_time_s:.2f}s (Frame #{event.start_frame} - #{event.end_frame})."
                )
                return

            if has_pedestrian:
                event.accident_type = "vehicle_pedestrian_collision"
                event.type_title = "⚠️ Phương tiện va chạm người đi bộ"
                event.description = (
                    f"Sự cố va chạm nghiêm trọng liên quan đến {labels[0]} tại giây thứ {event.start_time_s:.2f}s "
                    f"(Frame #{event.start_frame} - #{event.end_frame}) trên phần đường xe chạy."
                )
                return

            if dir_change_sum >= 90.0 or yaw_rate >= 80.0 or bbox_deform == 1:
                event.accident_type = "single_vehicle_spin_or_rollover"
                event.type_title = "🔄 Xe mất lái quay vòng / Lật xe"
                event.description = (
                    f"{labels[0]} bị mất lái nghiêm trọng, thân xe xoay góc lớn ({dir_change_sum:.0f}°) "
                    f"hoặc lật xe trên đường tại giây thứ {event.start_time_s:.2f}s (Frame #{event.start_frame})."
                )
                return

            if (sudden_stop or stopped_after_motion) and speed_drop >= 3.5:
                event.accident_type = "single_vehicle_crash"
                event.type_title = "🛑 Xe đâm vật cản / Dải phân cách"
                event.description = (
                    f"{labels[0]} va đâm vào dải phân cách hoặc chướng ngại vật ven đường tại giây thứ {event.start_time_s:.2f}s, "
                    f"giảm tốc khẩn cấp từ {init_spd:.1f} km/h về trạng thái dừng hẳn."
                )
                return

            event.accident_type = "single_vehicle_incident"
            event.type_title = "⚠️ Sự cố phương tiện trên đường"
            event.description = (
                f"Phát hiện sự cố giao thông liên quan đến {labels[0]} tại giây thứ {event.start_time_s:.2f}s "
                f"(Frame #{event.start_frame} - #{event.end_frame})."
            )
            return

        event.accident_type = "unknown"
        event.type_title = "⚠️ Sự cố giao thông"
        event.description = f"Phát hiện tình huống va chạm giao thông bất thường tại giây thứ {event.start_time_s:.2f}s."

    def _infer_accident_type(
        self,
        event: AccidentEvent,
    ) -> str:
        """
        Cập nhật loại tai nạn và mô tả chi tiết, trả về mã định danh accident_type.
        """
        self._update_accident_classification_and_description(event)
        return event.accident_type

    def _is_physically_consistent_accident(
        self,
        track_id: int,
        feat: Dict[str, Any],
        vehicle_class: str = "",
    ) -> bool:
        """
        Kiểm tra tính hợp lý vật lý trước khi xác nhận sự kiện tai nạn:
        - Va chạm đa phương tiện BẮT BUỘC phải có tiếp xúc vật lý đã xác thực (không kích hoạt trước khi chạm,
          không kích hoạt do chồng lấn phối cảnh 2D giữa 2 xe khác làn).
        - Sự cố đơn phương tiện bắt buộc track liên tục ổn định và có biến động dừng khẩn cấp / xoay / lật thực tế.
        """
        if not feat and not vehicle_class:
            return True

        # 1. Nhãn nhận diện trực tiếp từ mô hình YOLO chuyên dụng (chỉ khi độ tự tin rất cao)
        cls_name = str(vehicle_class or feat.get("class_name", "")).lower()
        if cls_name in ("accident", "car_car_accident", "car_person_accident", "injured"):
            return float(feat.get("track_confidence", 1.0)) >= 0.78

        # 2. Va chạm đa phương tiện: bắt buộc đã tiếp xúc vật lý + có xung lực động học va chạm
        has_physical_contact = bool(feat.get("has_physical_contact", 0))
        has_kinematic_impact = bool(feat.get("has_kinematic_impact", 0))
        if has_physical_contact and has_kinematic_impact:
            return True

        # 2b. Hiện trường người / xe 2 bánh ngã đổ trên mặt đường
        if bool(feat.get("is_fallen_vru", 0)):
            return True

        decel = float(feat.get("deceleration_mps2", 0.0))
        stopped_after_motion = int(feat.get("stopped_after_motion", 0))
        sudden_stop = int(feat.get("sudden_stop", 0))
        spd = float(feat.get("speed_mps", 0.0))
        speed_drop = float(feat.get("speed_drop_window_mps", 0.0))
        yaw_rate = abs(float(feat.get("yaw_rate_dps", 0.0)))
        bbox_deform = int(feat.get("bbox_deformation", 0))
        is_two_wheeler = cls_name in ("motorcycle", "motorbike", "bicycle", "bike")
        is_vru = is_two_wheeler or cls_name in ("person", "pedestrian")
        vis_acc_tag = float(feat.get("visual_accident_tag", 0.0))

        is_pedestrian = cls_name in ("person", "pedestrian")
        nearest_dist = float(feat.get("nearest_distance_m", 50.0))
        has_neighbor = int(feat.get("has_neighbor", 0))

        # Nếu 2 xe đang ở gần nhau (<= 4.5m) nhưng CHƯA tiếp xúc vật lý -> tuyệt đối không xác nhận sớm
        if has_neighbor == 1 and nearest_dist <= 4.5 and not has_physical_contact:
            return False

        # 3. Điều kiện sự cố đơn phương tiện nghiêm trọng (đâm dải phân cách / lật xe / xe máy trượt ngã)
        cont_age = int(feat.get("continuous_age", 0))
        if cont_age > 0 and cont_age < 5:
            return False

        is_severe_single_event = (
            (stopped_after_motion == 1 and decel >= 5.0 and speed_drop >= 4.0 and spd < 1.5)
            or (is_vru and decel >= 8.0 and speed_drop >= 4.5 and spd < 2.5)
            or (not is_pedestrian and sudden_stop == 1 and decel >= 6.0 and speed_drop >= 4.5 and spd < 1.2)
            or (not is_pedestrian and yaw_rate >= 95.0 and speed_drop >= 4.5 and spd < 1.8)
            or (bbox_deform == 1 and is_two_wheeler and speed_drop >= 2.8 and spd < 2.2)
            or (
                is_vru
                and vis_acc_tag >= 0.45
                and (spd < 2.2 or stopped_after_motion == 1)
                and (decel >= 2.8 or speed_drop >= 2.0 or bbox_deform == 1 or yaw_rate >= 60.0)
            )
        )

        return is_severe_single_event

    def _register_contact_partner(
        self,
        event: AccidentEvent,
        f_dict: Dict[str, Any],
        frame_id: int,
    ) -> None:
        """Gắn phương tiện đối tác đã tiếp xúc trực tiếp vào sự kiện (kể cả khi đối tác vừa bị khuất sau va chạm)."""
        partner_tid = f_dict.get("contact_partner_tid")
        if partner_tid is None:
            return
        partner_tid = int(partner_tid)
        event.track_ids.add(partner_tid)
        p_cls = f_dict.get("contact_partner_class")
        if p_cls and partner_tid not in event.vehicle_types:
            event.vehicle_types[partner_tid] = str(p_cls)
        p_bbox = f_dict.get("contact_partner_bbox")
        if p_bbox is not None and partner_tid not in event.track_bboxes:
            event.track_bboxes[partner_tid] = tuple(p_bbox)
        p_dir = f_dict.get("contact_partner_dir")
        if p_dir is not None:
            if partner_tid not in self._track_directions:
                self._track_directions[partner_tid] = float(p_dir)
            self._event_initial_directions.setdefault((event.event_id, partner_tid), float(p_dir))
        p_pos = f_dict.get("contact_partner_pos")
        if p_pos is not None and partner_tid not in self._track_positions:
            self._track_positions[partner_tid] = np.asarray(p_pos, dtype=np.float32)
        last_cf = int(f_dict.get("last_contact_frame", frame_id))
        self._track_last_accident_frame[partner_tid] = max(
            self._track_last_accident_frame.get(partner_tid, 0),
            last_cf,
        )
        if event.active:
            self._active_by_track[partner_tid] = event

    def _find_mergeable_event(
        self,
        track_id: int,
        frame_id: int,
        position: Optional[np.ndarray],
        speed: float = 0.0,
        bbox: Optional[Any] = None,
        feature_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[AccidentEvent]:
        """
        Kiểm tra xem phương tiện này có thuộc sự kiện tai nạn đang diễn ra (hoặc vừa kết thúc tại đúng hiện trường)
        để gom lại, đồng thời KHÔNG khóa vĩnh viễn phương tiện nếu tiếp tục di chuyển đến vị trí khác.
        """
        f_dict = feature_dict or {}
        partner_tid = f_dict.get("contact_partner_tid")
        pos_arr = np.asarray(position, dtype=np.float32) if position is not None else None

        for event in self._events:
            if not event.active:
                # Sự kiện đã đóng: gom các mảnh hậu quả ngã đổ tại đúng hiện trường va chạm (trong 5s)
                recent_closed = (frame_id - event.end_frame) <= int(self.fps * 5.0)
                crash_pos = self._event_crash_positions.get(event.event_id)
                near_crash_site = False
                if pos_arr is not None and crash_pos is not None:
                    near_crash_site = float(np.linalg.norm(pos_arr - crash_pos)) <= 5.5

                if track_id in event.track_ids:
                    if recent_closed or (near_crash_site and speed < 1.8):
                        return event
                elif recent_closed and near_crash_site and partner_tid in event.track_ids:
                    return event
                elif recent_closed and near_crash_site and bool(f_dict.get("is_fallen_vru", 0)):
                    return event
                continue

            # Sự kiện đang ACTIVE:
            if track_id in event.track_ids:
                return event
            if partner_tid is not None and int(partner_tid) in event.track_ids:
                return event

            # 1. Gom khi khung bao 2D thực sự chồng lấn phần thân trong (inner overlap > 0)
            if bbox is not None:
                bw = max(1.0, float(bbox[2] - bbox[0]))
                bh = max(1.0, float(bbox[3] - bbox[1]))
                ib1 = (
                    float(bbox[0]) + 0.03 * bw,
                    float(bbox[1]) + 0.03 * bh,
                    float(bbox[2]) - 0.03 * bw,
                    float(bbox[3]) - 0.03 * bh,
                )
                for existing_tid, ex_bbox in list(event.track_bboxes.items()):
                    if ex_bbox is None:
                        ex_bbox = self._track_bboxes.get(existing_tid)
                    if ex_bbox is not None:
                        ew = max(1.0, float(ex_bbox[2] - ex_bbox[0]))
                        eh = max(1.0, float(ex_bbox[3] - ex_bbox[1]))
                        ib2 = (
                            float(ex_bbox[0]) + 0.03 * ew,
                            float(ex_bbox[1]) + 0.03 * eh,
                            float(ex_bbox[2]) - 0.03 * ew,
                            float(ex_bbox[3]) - 0.03 * eh,
                        )
                        inter = max(0.0, min(ib1[2], ib2[2]) - max(ib1[0], ib2[0])) * max(
                            0.0, min(ib1[3], ib2[3]) - max(ib1[1], ib2[1])
                        )
                        if inter > 0.0 and bool(f_dict.get("has_physical_contact", 0)):
                            return event

            # 2. Gom dựa trên cự ly thực tế BEV chặt chẽ (<= 3.8m)
            if pos_arr is not None:
                for existing_tid in event.track_ids:
                    if existing_tid in self._track_positions:
                        existing_pos = self._track_positions[existing_tid]
                        dist = float(np.linalg.norm(pos_arr - existing_pos))
                        if dist <= min(3.8, self.spatial_merge_distance):
                            return event

        return None

    def update(
        self,
        track_id: int,
        frame_id: int,
        probability: float,
        position: Optional[np.ndarray] = None,
        speed: float = 0.0,
        direction: float = 0.0,
        vehicle_class: str = "vehicle",
        feature_dict: Optional[Dict[str, Any]] = None,
        bbox: Optional[Any] = None,
    ) -> Optional[AccidentEvent]:
        """
        Cập nhật kết quả phân loại cho một track tại frame hiện tại.
        """
        prob = float(probability)
        h = self._history[track_id]
        h.append(prob)
        f_dict = feature_dict or {}

        if position is not None:
            self._track_positions[track_id] = np.asarray(position, dtype=np.float32)
        if bbox is not None:
            self._track_bboxes[track_id] = bbox
        # Chỉ cập nhật hướng di chuyển khi xe còn tốc độ thực để giữ nguyên góc tiếp cận trước khi dừng hẳn
        if speed >= 1.2 or track_id not in self._track_directions:
            self._track_directions[track_id] = float(direction)
        self._track_speeds[track_id] = float(speed)
        if feature_dict is not None:
            self._track_features[track_id] = feature_dict

        # Cập nhật mức độ hoạt động của track
        if prob >= self.release_threshold:
            self._track_last_accident_frame[track_id] = frame_id
            self._track_calm_frames[track_id] = 0
        else:
            self._track_calm_frames[track_id] += 1

        # Trường hợp 1: Track đã thuộc một event đang active
        event = self._active_by_track.get(track_id)
        if event is not None and event.active:
            if prob > event.probability:
                event.probability = prob
                event.peak_frame = frame_id

            event.track_ids.add(track_id)
            event.vehicle_types[track_id] = vehicle_class
            if bbox is not None:
                event.track_bboxes[track_id] = bbox
            self._register_contact_partner(event, f_dict, frame_id)
            if prob >= self.release_threshold:
                event.end_frame = max(event.end_frame, frame_id)
                event.end_time_s = event.end_frame / self.fps
                event.duration_s = max(0.0, (event.end_frame - event.start_frame + 1) / self.fps)
            event.accident_type = self._infer_accident_type(event)
            return event

        # Ghi nhận thời điểm xuất hiện đầu tiên của track
        if track_id not in self._track_first_frame:
            self._track_first_frame[track_id] = frame_id
        track_age = frame_id - self._track_first_frame[track_id] + 1

        # Trường hợp 2: Track chưa thuộc event nào, kiểm tra chuỗi frame va chạm hợp lệ liên tiếp
        physically_valid = self._is_physically_consistent_accident(track_id, f_dict, vehicle_class=vehicle_class)
        if prob >= self.start_threshold and physically_valid:
            if track_id not in self._track_streak_start:
                self._track_streak_start[track_id] = frame_id
            streak_len = frame_id - self._track_streak_start[track_id] + 1
        else:
            self._track_streak_start.pop(track_id, None)
            streak_len = 0

        has_direct_collision = bool(f_dict.get("has_physical_contact", 0)) and bool(
            f_dict.get("has_kinematic_impact", 0)
        )
        min_age = 2 if has_direct_collision else max(5, self.min_track_age)
        track_mature = track_age >= min_age

        req_frames = max(3, self.confirmation_frames - 1) if has_direct_collision else max(4, self.confirmation_frames)
        probs_confirmed = streak_len >= req_frames

        if track_mature and probs_confirmed and physically_valid:
            recent_probs = list(h)[-streak_len:] if streak_len > 0 else [prob]
            merge_event = self._find_mergeable_event(
                track_id=track_id,
                frame_id=frame_id,
                position=position,
                speed=speed,
                bbox=bbox,
                feature_dict=f_dict,
            )
            if merge_event is not None:
                if not merge_event.active:
                    if bool(f_dict.get("is_fallen_vru", 0)):
                        merge_event.track_ids.add(track_id)
                        merge_event.vehicle_types[track_id] = vehicle_class
                        if bbox is not None:
                            merge_event.track_bboxes[track_id] = bbox
                        merge_event.end_frame = max(merge_event.end_frame, frame_id)
                        merge_event.end_time_s = merge_event.end_frame / self.fps
                        merge_event.duration_s = max(0.0, (merge_event.end_frame - merge_event.start_frame + 1) / self.fps)
                        merge_event.accident_type = self._infer_accident_type(merge_event)
                        return merge_event
                    if not has_direct_collision and (frame_id - merge_event.end_frame) <= int(self.fps * 2.5):
                        return None
                    if not has_direct_collision and speed < 1.8:
                        return None
                    merge_event = None
            if merge_event is not None:
                merge_event.track_ids.add(track_id)
                merge_event.vehicle_types[track_id] = vehicle_class
                if bbox is not None:
                    merge_event.track_bboxes[track_id] = bbox
                self._register_contact_partner(merge_event, f_dict, frame_id)
                if prob > merge_event.probability:
                    merge_event.probability = prob
                    merge_event.peak_frame = frame_id
                merge_event.end_frame = max(merge_event.end_frame, frame_id)
                merge_event.end_time_s = merge_event.end_frame / self.fps
                merge_event.duration_s = max(0.0, (merge_event.end_frame - merge_event.start_frame + 1) / self.fps)
                merge_event.accident_type = self._infer_accident_type(merge_event)
                self._active_by_track[track_id] = merge_event
                return merge_event

            # Tính start_frame chính xác: KHÔNG BAO GIỜ lùi về trước thời điểm tiếp xúc vật lý thực tế
            streak_start = self._track_streak_start.get(track_id, frame_id)
            first_contact_f = int(f_dict.get("first_contact_frame", 0))
            if first_contact_f > 0:
                if abs(streak_start - first_contact_f) <= 3:
                    start_f = first_contact_f
                else:
                    start_f = max(streak_start, first_contact_f)
                start_f = min(start_f, frame_id)
            else:
                start_f = max(1, min(streak_start, frame_id))

            peak_f = start_f if (first_contact_f > 0 and start_f == first_contact_f) else frame_id
            new_event = AccidentEvent(
                event_id=self._next_id,
                start_frame=start_f,
                end_frame=frame_id,
                peak_frame=peak_f,
                probability=float(max(recent_probs)),
                track_ids={track_id},
                vehicle_types={track_id: vehicle_class},
                accident_type="unknown",
                active=True,
                start_time_s=start_f / self.fps,
                end_time_s=frame_id / self.fps,
                duration_s=max(0.0, (frame_id - start_f + 1) / self.fps),
                track_bboxes={track_id: bbox} if bbox is not None else {},
            )
            self._next_id += 1
            init_own_dir = f_dict.get("own_contact_dir", direction)
            self._event_initial_directions[(new_event.event_id, track_id)] = float(init_own_dir)
            self._register_contact_partner(new_event, f_dict, frame_id)
            if position is not None:
                self._event_crash_positions[new_event.event_id] = np.asarray(position, dtype=np.float32)
            new_event.accident_type = self._infer_accident_type(new_event)

            self._events.append(new_event)
            self._active_by_track[track_id] = new_event
            return new_event

        return None

    def end_frame(
        self,
        frame_id: int,
        seen_track_ids: Set[int],
    ) -> None:
        """
        Gọi ở cuối mỗi frame để kiểm tra cơ chế đóng sự kiện (event closure):
        - Nếu tất cả các xe trong event đã hạ xác suất dưới release_threshold trong release_frames.
        - Hoặc các xe đã rời khỏi màn hình / đứng yên ổn định.
        - Hoặc thời lượng sự kiện vượt quá giới hạn tối đa cho một va chạm động học.
        """
        for event in self._events:
            if not event.active:
                continue

            last_active = max(
                [self._track_last_accident_frame.get(tid, event.start_frame) for tid in event.track_ids],
                default=event.start_frame,
            )

            # Kiểm tra xem toàn bộ các xe trong event đã bình ổn hoặc mất dấu đủ số frame giải phóng chưa
            inactive_frames = frame_id - last_active
            all_calm_or_lost = inactive_frames >= self.release_frames

            # Kiểm tra thời lượng tối đa
            exceeded_duration = (frame_id - event.start_frame) > self.max_event_duration_frames

            if all_calm_or_lost or exceeded_duration:
                # ĐÓNG EVENT CHÍNH XÁC:
                # end_frame là frame cuối cùng còn ghi nhận hoạt động va chạm, KHÔNG kéo dài đến hiện tại
                event.end_frame = min(last_active + 2, frame_id)
                event.active = False
                event.end_time_s = event.end_frame / self.fps
                event.duration_s = max(0.0, (event.end_frame - event.start_frame + 1) / self.fps)
                event.accident_type = self._infer_accident_type(event)

                # Giải phóng track khỏi danh sách active
                for tid in event.track_ids:
                    self._active_by_track.pop(tid, None)

    def finish(
        self,
        final_frame_id: int,
    ) -> None:
        """
        Đóng toàn bộ các sự kiện khi kết thúc video, chốt thời gian chính xác
        tại thời điểm va chạm cuối cùng thay vì frame cuối của video.
        """
        for event in self._events:
            if event.active:
                last_active = max(
                    [self._track_last_accident_frame.get(tid, event.start_frame) for tid in event.track_ids],
                    default=event.start_frame,
                )
                event.end_frame = min(last_active + 2, final_frame_id)
                event.active = False
                event.end_time_s = event.end_frame / self.fps
                event.duration_s = max(0.0, (event.end_frame - event.start_frame + 1) / self.fps)
                event.accident_type = self._infer_accident_type(event)

        self._active_by_track.clear()

    def all_events(self) -> List[AccidentEvent]:
        """Trả về toàn bộ danh sách các sự kiện tai nạn đã phát hiện."""
        return list(self._events)

    def get_active_events(self) -> List[AccidentEvent]:
        """Trả về danh sách sự kiện đang diễn ra."""
        return [e for e in self._events if e.active]
