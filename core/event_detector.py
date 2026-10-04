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
    active: bool = True
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    duration_s: float = 0.0
    description: str = ""

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

        # Ánh xạ track_id -> AccidentEvent đang active
        self._active_by_track: Dict[int, AccidentEvent] = {}
        self._events: List[AccidentEvent] = []
        self._next_id: int = 1

    def _infer_accident_type(
        self,
        event: AccidentEvent,
    ) -> str:
        """
        Phân loại tai nạn dựa trên số lượng phương tiện tham gia và quan hệ động học.
        """
        tids = list(event.track_ids)
        num_vehicles = len(tids)

        if num_vehicles >= 2:
            # Va chạm giữa nhiều xe
            t1, t2 = tids[0], tids[1]
            dir1 = self._track_directions.get(t1)
            dir2 = self._track_directions.get(t2)

            if dir1 is not None and dir2 is not None:
                angle_diff = abs((dir1 - dir2 + 180.0) % 360.0 - 180.0)
                if angle_diff > 135.0:
                    return "head_on_collision"  # Va chạm đối đầu
                if 45.0 <= angle_diff <= 135.0:
                    return "side_impact_collision"  # Va chạm góc / t-bone

            if num_vehicles > 2:
                return "multi_vehicle_pileup"  # Va chạm liên hoàn nhiều xe

            return "rear_end_collision"  # Va chạm từ phía sau

        # Tai nạn đơn xe (Single-vehicle)
        if num_vehicles == 1:
            tid = tids[0]
            feat = self._track_features.get(tid, {})
            yaw_rate = float(feat.get("yaw_rate_dps", 0.0))
            dir_change_sum = float(feat.get("direction_change_sum_deg", 0.0))
            sudden_stop = int(feat.get("sudden_stop", 0))
            stopped_after_motion = int(feat.get("stopped_after_motion", 0))

            if dir_change_sum >= 80.0 or abs(yaw_rate) >= 80.0:
                return "single_vehicle_spin_or_rollover"  # Mất lái quay vòng / lật
            if sudden_stop or stopped_after_motion:
                return "single_vehicle_sudden_stop"  # Đâm vật cản / dừng đột ngột
            return "single_vehicle_loss_of_control"  # Mất lái chệch quỹ đạo

        return "unknown"

    def _is_physically_consistent_accident(
        self,
        track_id: int,
        feat: Dict[str, Any],
    ) -> bool:
        """
        Kiểm tra tính hợp lý vật lý trước khi xác nhận sự kiện tai nạn:
        Triệt tiêu các báo động giả khi xe lưu thông bình thường hoặc chỉ vào cua thông thường.
        """
        if not feat:
            return True

        nearest_dist = float(feat.get("nearest_distance_m", 50.0))
        closing_spd = float(feat.get("closing_speed_mps", 0.0))
        has_neighbor = int(feat.get("has_neighbor", 0))

        # Điều kiện 1: Tương tác va chạm giữa các phương tiện
        # Phải có xe khác ở cự ly va chạm rất gần (<= 4.5m) hoặc tốc độ tiếp cận nhanh (>= 1.5 m/s)
        has_collision_proximity = (has_neighbor == 1 and nearest_dist <= 4.5) or closing_spd >= 1.5

        # Điều kiện 2: Sự cố đơn phương tiện nghiêm trọng (đâm vật cản, lật, phanh cháy đường)
        decel = float(feat.get("deceleration_mps2", 0.0))
        stopped_after_motion = int(feat.get("stopped_after_motion", 0))
        sudden_stop = int(feat.get("sudden_stop", 0))
        dir_sum = float(feat.get("direction_change_sum_deg", 0.0))
        yaw_rate = abs(float(feat.get("yaw_rate_dps", 0.0)))

        spd = float(feat.get("speed_mps", 0.0))

        is_severe_single_event = (
            (stopped_after_motion == 1 and decel >= 3.5)
            or (sudden_stop == 1 and decel >= 5.0 and spd < 3.0)
            or (dir_sum >= 50.0 and yaw_rate >= 50.0 and spd >= 4.0)
        )

        return has_collision_proximity or is_severe_single_event

    def _find_mergeable_event(
        self,
        track_id: int,
        position: Optional[np.ndarray],
    ) -> Optional[AccidentEvent]:
        """
        Kiểm tra xem phương tiện này có gần vị trí của một sự kiện tai nạn đang diễn ra
        để gom (aggregate) lại hay không.
        """
        if position is None:
            return None

        pos_arr = np.asarray(position, dtype=np.float32)

        for event in self._events:
            if not event.active:
                continue

            # Kiểm tra khoảng cách tới bất kỳ xe nào trong event
            for existing_tid in event.track_ids:
                if existing_tid in self._track_positions:
                    existing_pos = self._track_positions[existing_tid]
                    dist = float(np.linalg.norm(pos_arr - existing_pos))
                    if dist <= self.spatial_merge_distance:
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
    ) -> Optional[AccidentEvent]:
        """
        Cập nhật kết quả phân loại cho một track tại frame hiện tại.
        """
        prob = float(probability)
        h = self._history[track_id]
        h.append(prob)

        if position is not None:
            self._track_positions[track_id] = np.asarray(position, dtype=np.float32)
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
            event.end_frame = frame_id
            event.accident_type = self._infer_accident_type(event)
            return event

        # Ghi nhận thời điểm xuất hiện đầu tiên của track
        if track_id not in self._track_first_frame:
            self._track_first_frame[track_id] = frame_id
        track_age = frame_id - self._track_first_frame[track_id] + 1

        # Trường hợp 2: Track chưa thuộc event nào, kiểm tra xem có đạt confirmation không
        # Yêu cầu:
        # 1. Track đã ổn định qua giai đoạn khởi tạo (track_age >= min_track_age)
        # 2. Toàn bộ confirmation_frames gần nhất đều có xác suất >= start_threshold
        # 3. Phù hợp logic động học vật lý thực tế (không phải xe chạy một mình bình thường)
        track_mature = track_age >= self.min_track_age
        recent_probs = list(h)[-self.confirmation_frames:]
        probs_confirmed = (
            len(h) >= self.confirmation_frames
            and all(x >= self.start_threshold for x in recent_probs)
        )
        physically_valid = self._is_physically_consistent_accident(track_id, feature_dict or {})

        if track_mature and probs_confirmed and physically_valid:
            # Kiểm tra xem có gom được vào event tai nạn đa phương tiện lân cận không
            merge_event = self._find_mergeable_event(track_id, position)
            if merge_event is not None:
                merge_event.track_ids.add(track_id)
                merge_event.vehicle_types[track_id] = vehicle_class
                merge_event.probability = max(merge_event.probability, prob)
                merge_event.end_frame = frame_id
                merge_event.accident_type = self._infer_accident_type(merge_event)
                self._active_by_track[track_id] = merge_event
                return merge_event

            # Tạo sự kiện tai nạn mới
            start_f = max(1, frame_id - self.confirmation_frames + 1)
            new_event = AccidentEvent(
                event_id=self._next_id,
                start_frame=start_f,
                end_frame=frame_id,
                peak_frame=frame_id,
                probability=float(max(recent_probs)),
                track_ids={track_id},
                vehicle_types={track_id: vehicle_class},
                accident_type="unknown",
                active=True,
                start_time_s=start_f / self.fps,
                end_time_s=frame_id / self.fps,
                duration_s=max(0.0, (frame_id - start_f + 1) / self.fps),
            )
            self._next_id += 1
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

            # Kiểm tra xem tất cả các xe trong event đã bình ổn chưa
            all_calm = True
            for tid in event.track_ids:
                # Nếu xe vẫn còn xuất hiện và chưa đủ số frame bình ổn
                if tid in seen_track_ids:
                    if self._track_calm_frames[tid] < self.release_frames:
                        all_calm = False
                        break

            # Kiểm tra thời lượng tối đa
            exceeded_duration = (frame_id - event.start_frame) > self.max_event_duration_frames

            # Kiểm tra nếu tất cả các xe đã biến mất khỏi camera
            all_lost = all(tid not in seen_track_ids for tid in event.track_ids)

            if all_calm or exceeded_duration or all_lost:
                # ĐÓNG EVENT CHÍNH XÁC:
                # end_frame là frame cuối cùng còn ghi nhận hoạt động va chạm, KHÔNG kéo dài đến hiện tại
                last_active = max(
                    [self._track_last_accident_frame.get(tid, event.start_frame) for tid in event.track_ids],
                    default=event.start_frame,
                )
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
