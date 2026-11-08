"""
AcciVision — Module Phát Hiện & Quản Lý Sự Kiện Tai Nạn (Event Detector)

Nâng cấp:
1. Kiểm tra giao cắt Bounding Box thực tế (IoU / Proximity) để không bao giờ báo nhầm
   2 xe chạy song song trên 2 làn đường cạnh nhau (như Xe #44 và #67).
2. Lọc bỏ nhiễu rìa màn hình (Border Clipping) khi xe mới đi vào/ra khỏi khung hình.
3. Phát hiện chính xác sự cố đơn phương tiện mất lái đâm dải phân cách / quay ngang (như Xe #12).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class AccidentEvent:
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
    center_position: Optional[np.ndarray] = None
    vehicle_groups: Dict[int, Set[int]] = field(default_factory=dict)
    impact_track_ids: Set[int] = field(default_factory=set)
    track_links: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep every tracker ID while recording which IDs are estimated aliases.
        initial_tracks = set(self.track_ids)
        initial_groups = self.vehicle_groups
        if not self.impact_track_ids:
            self.impact_track_ids = set(initial_groups) if initial_groups else set(initial_tracks)
        self.vehicle_groups = {}
        for canonical, members in initial_groups.items():
            self.register_track(canonical)
            for track_id in members:
                self.register_track(track_id, alias_of=canonical)
        for track_id in sorted(initial_tracks):
            self.register_track(track_id)

    def register_track(
        self, track_id: int, vehicle_type: Optional[str] = None,
        alias_of: Optional[int] = None,
    ) -> None:
        """Register a participant, or add a tentative handoff to an estimated group.

        Only an explicit alias relation merges estimated groups; event proximity
        alone must not collapse a genuinely independent third vehicle.
        """
        if alias_of is not None and alias_of != track_id and alias_of not in self.track_ids:
            self.register_track(alias_of)
        self.track_ids.add(track_id)
        if vehicle_type is not None:
            self.vehicle_types[track_id] = vehicle_type
        existing_root = next(
            (root for root, members in self.vehicle_groups.items() if track_id in members), None
        )
        alias_root = next(
            (root for root, members in self.vehicle_groups.items() if alias_of in members), None
        ) if alias_of is not None else None
        root = alias_root if alias_root is not None else existing_root
        if root is None:
            root = track_id
            self.vehicle_groups[root] = set()
            self.impact_track_ids.add(track_id)
        if existing_root is not None and existing_root != root:
            self.vehicle_groups[root].update(self.vehicle_groups.pop(existing_root))
        self.vehicle_groups[root].add(track_id)

    @property
    def estimated_vehicle_count(self) -> int:
        return len(self.vehicle_groups)

    def participant_data(self) -> List[Dict[str, Any]]:
        """Separate observed impact anchors from unconfirmed ID associations."""
        participants = []
        for root, members in sorted(self.vehicle_groups.items()):
            anchors = members & self.impact_track_ids
            impact_id = root if root in anchors else min(anchors, default=root)
            candidates = sorted(members - self.impact_track_ids)
            participants.append({
                "canonical_track_id": root,
                "impact_track_id": impact_id,
                "track_ids": sorted(members),
                "candidate_track_ids": candidates,
                "identity_status": "unconfirmed" if candidates else "observed",
                "vehicle_type": self.vehicle_types.get(root, "vehicle"),
            })
        return participants

    @property
    def duration_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame + 1)


class EventDetector:
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
        self.spatial_merge_distance = float(max(0.1, spatial_merge_distance))
        self.temporal_merge_gap_frames = int(2.2 * self.fps)
        self.lost_grace_frames = int(0.8 * self.fps)
        self.max_event_duration_frames = int(max_event_duration_frames)
        self.min_track_age = int(max(5, min_track_age))

        self._track_first_frame: Dict[int, int] = {}
        self._history: Dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._history_frames: Dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=60))
        self._track_last_update_frame: Dict[int, int] = {}
        self._track_successors: Dict[int, int] = {}
        self._tracking_observations: Dict[int, Dict[str, Any]] = {}
        self._event_last_evidence_frame: Dict[int, int] = {}
        self._pair_history: Dict[Tuple[int, int], deque[Tuple[int, float, bool]]] = defaultdict(
            lambda: deque(maxlen=self.confirmation_frames)
        )
        self._observed_frame_id: Optional[int] = None
        self._track_last_accident_frame: Dict[int, int] = {}
        self._track_last_seen_frame: Dict[int, int] = {}
        self._track_calm_frames: Dict[int, int] = defaultdict(int)

        self._track_positions: Dict[int, np.ndarray] = {}
        self._track_bboxes: Dict[int, Tuple[float, float, float, float]] = {}
        self._track_bbox_ratios: Dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=25))
        self._track_pixel_centers: Dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=25))
        self._track_directions: Dict[int, float] = {}
        self._track_speeds: Dict[int, float] = {}
        self._track_classes: Dict[int, str] = {}
        self._track_features: Dict[int, Dict[str, Any]] = {}

        self._active_by_track: Dict[int, AccidentEvent] = {}
        self._events: List[AccidentEvent] = []
        self._next_id: int = 1

    def set_fps(self, fps: float) -> None:
        """Update all time-derived frame thresholds before processing a video."""
        self.fps = float(fps if fps > 0 else 30.0)
        self.temporal_merge_gap_frames = max(1, int(round(2.2 * self.fps)))
        self.lost_grace_frames = max(1, int(round(0.8 * self.fps)))

    def observe_frame(
        self, frame_id: int, observations: Dict[int, Dict[str, Any]]
    ) -> None:
        """Supply all current tracks before evaluating any collision partner.

        This only populates the frame context. ``update`` remains responsible
        for temporal evidence, so the result is independent of track ordering.
        Callers that update one track at a time can continue using ``update``.
        """
        self._observed_frame_id = frame_id
        for track_id, observation in observations.items():
            self._record_tracking_observation(
                track_id, frame_id, observation.get("vehicle_class", "vehicle")
            )
            position = observation.get("position")
            bbox = observation.get("bbox")
            if position is not None:
                self._track_positions[track_id] = np.asarray(position, dtype=np.float32)
            if bbox is not None:
                self._track_bboxes[track_id] = tuple(bbox)
            self._track_speeds[track_id] = float(observation.get("speed", 0.0))
            self._track_directions[track_id] = float(observation.get("direction", 0.0))
            self._track_classes[track_id] = observation.get("vehicle_class", "vehicle")
            self._track_features[track_id] = observation.get("feature_dict") or {}
            self._track_last_seen_frame[track_id] = frame_id

    def _record_tracking_observation(
        self, track_id: int, frame_id: int, vehicle_class: str,
    ) -> None:
        observation = self._tracking_observations.setdefault(
            track_id,
            {"first_frame": int(frame_id), "last_frame": int(frame_id), "class_name": vehicle_class},
        )
        observation["last_frame"] = max(observation["last_frame"], int(frame_id))
        observation["class_name"] = vehicle_class

    def tracking_data(self) -> List[Dict[str, Any]]:
        """Return actual track lifetimes; recording a late ID creates no identity link."""
        return [
            {"track_id": track_id, **observation}
            for track_id, observation in sorted(self._tracking_observations.items())
        ]

    @staticmethod
    def _bbox_iou(b1: Tuple[float, ...], b2: Tuple[float, ...]) -> float:
        intersection = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(
            0.0, min(b1[3], b2[3]) - max(b1[1], b2[1])
        )
        area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        return intersection / max(area1 + area2 - intersection, 1e-6)

    @staticmethod
    def _bboxes_touch_or_close(
        b1: Optional[Tuple[float, float, float, float]],
        b2: Optional[Tuple[float, float, float, float]],
        expand_ratio: float = 0.20,
    ) -> bool:
        """
        Kiểm tra xem 2 khung Bounding Box có thực sự tiếp xúc / đè lên nhau không.
        Hai xe chạy ở 2 làn song song cách nhau một khoảng trống (như ID:44 và ID:67) sẽ trả về False!
        """
        if b1 is None or b2 is None:
            return False

        w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
        w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
        pad_x = expand_ratio * 0.5 * (w1 + w2)
        pad_y = expand_ratio * 0.5 * (h1 + h2)

        ax1, ay1, ax2, ay2 = b1[0] - pad_x, b1[1] - pad_y, b1[2] + pad_x, b1[3] + pad_y
        bx1, by1, bx2, by2 = b2[0] - pad_x, b2[1] - pad_y, b2[2] + pad_x, b2[3] + pad_y

        inter_w = min(ax2, bx2) - max(ax1, bx1)
        inter_h = min(ay2, by2) - max(ay1, by1)
        return (inter_w > 0) and (inter_h > 0)

    @staticmethod
    def _is_near_frame_border(
        bbox: Optional[Tuple[float, float, float, float]],
        frame_shape: Optional[Tuple[int, int]],
    ) -> bool:
        """
        Kiểm tra xem phương tiện có đang bị cắt ở sát mép khung hình (mới đi vào hoặc sắp đi ra) không.
        """
        if bbox is None or frame_shape is None:
            return False
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        # Nếu cạnh dưới hoặc cạnh bên của xe chạm sát rìa 6% của màn hình (như ID:67 và ID:71)
        if x1 <= 0.04 * width or x2 >= 0.96 * width:
            return True
        if y2 >= 0.93 * height or y1 <= 0.04 * height:
            return True
        return False

    def _detect_single_vehicle_spin_or_barrier_crash(self, track_id: int) -> bool:
        """
        Phát hiện chuyên biệt cho phương tiện mất lái đâm dải phân cách / quay ngang trên đường (như Xe #12):
        1. Quỹ đạo pixel đang đi thẳng bỗng nhiên gãy góc hoặc xoay ngang (tỷ lệ Rộng/Cao của bbox biến đổi mạnh),
        2. Kèm theo xe khựng lại / giảm tốc mạnh ngay sau cú xoay.
        """
        ratios = self._track_bbox_ratios[track_id]
        px_pts = self._track_pixel_centers[track_id]
        feat = self._track_features.get(track_id, {})

        # A bend and a changing silhouette also occur during an ordinary turn.
        # Require measured loss of motion before treating them as an impact.
        spd_drop = float(feat.get("speed_drop_window_mps", 0.0))
        decel = float(feat.get("deceleration_mps2", 0.0))
        stopped_after = int(feat.get("stopped_after_motion", 0))
        if not (
            (spd_drop >= 2.5 and decel >= 3.8)
            or (stopped_after == 1 and spd_drop >= 3.0)
        ):
            return False

        if len(ratios) < 8 or len(px_pts) < 8:
            return False

        # Tỷ lệ khung hình (Width / Height) thay đổi mạnh khi xe tải/ô tô xoay ngang đâm vào dải phân cách
        min_r, max_r = min(ratios), max(ratios)
        aspect_ratio_jump = (max_r / max(min_r, 0.1)) >= 1.32

        # Tính độ cong / gãy khúc của quỹ đạo giữa nửa đầu và nửa sau cửa sổ
        mid = len(px_pts) // 2
        v_first = px_pts[mid] - px_pts[0]
        v_second = px_pts[-1] - px_pts[mid]
        norm1 = float(np.linalg.norm(v_first))
        norm2 = float(np.linalg.norm(v_second))

        # Trước đó xe phải đang chạy thực sự (norm1 >= 12 pixels)
        if norm1 < 10.0:
            return False

        # Trường hợp A: Đang lao nhanh (norm1 lớn) rồi đâm dải phân cách dừng khựng lại (norm2 sụt giảm mạnh)
        abrupt_halt = (norm2 <= 0.32 * norm1) and aspect_ratio_jump

        # Trường hợp B: Đang đi thẳng rồi văng lệch góc mạnh (> 28 độ) kèm biến dạng bbox hoặc giảm tốc
        trajectory_angle_break = False
        if norm1 >= 10.0 and norm2 >= 5.0:
            cos_sim = float(np.dot(v_first, v_second) / (norm1 * norm2))
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
            if angle_deg >= 26.0 and (aspect_ratio_jump or norm2 < 0.65 * norm1):
                trajectory_angle_break = True

        return abrupt_halt or trajectory_angle_break or (aspect_ratio_jump and (spd_drop >= 3.0 or stopped_after == 1))

    def _has_reliable_image_motion(
        self, track_id: int, *, require_change: bool = False,
    ) -> bool:
        """Require a coherent approach, rather than a wobbling detection box.

        RF motion spikes can come from occlusion, ID switches or a parked
        vehicle's changing box. Measure displacement relative to vehicle size
        so a fixed pixel threshold does not treat near and distant cars alike.
        A collision may be confirmed at contact before the trajectory bends;
        an isolated event additionally needs a measured slowdown or bend.
        """
        points = np.asarray(self._track_pixel_centers[track_id], dtype=np.float64)
        bbox = self._track_bboxes.get(track_id)
        if len(points) < 8 or bbox is None:
            return False

        middle = len(points) // 2
        before = points[middle] - points[0]
        after = points[-1] - points[middle]
        before_distance = float(np.linalg.norm(before))
        after_distance = float(np.linalg.norm(after))
        box_scale = max(1.0, float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])))
        approach_path = float(np.linalg.norm(np.diff(points[:middle + 1], axis=0), axis=1).sum())
        coherence = before_distance / max(approach_path, 1e-6)
        largest_step = float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())

        if (
            before_distance < 0.25 * box_scale
            or coherence < 0.65
            or largest_step > 0.50 * box_scale
        ):
            return False

        if require_change:
            retention = after_distance / max(before_distance, 1e-6)
            angle = 0.0
            if after_distance >= 0.10 * box_scale:
                cosine = float(np.dot(before, after)) / max(before_distance * after_distance, 1e-9)
                angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if retention > 0.75 and angle < 25.0:
                return False
            # Braking to a stop alone also describes a queue or a red light.
            # A near-zero displacement has no reliable heading; it must not
            # become an artificial 90-degree turn through arccos(0).
            if angle < 25.0 and not self._detect_single_vehicle_spin_or_barrier_crash(track_id):
                return False
        return True

    def _infer_accident_type(
        self,
        event: AccidentEvent,
    ) -> str:
        tids = sorted(event.vehicle_groups)
        num_vehicles = len(tids)

        if num_vehicles >= 2:
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    dir1 = self._track_directions.get(tids[i])
                    dir2 = self._track_directions.get(tids[j])
                    if dir1 is not None and dir2 is not None:
                        angle_diff = abs((dir1 - dir2 + 180.0) % 360.0 - 180.0)
                        if angle_diff > 135.0:
                            return "head_on_collision"
                        if 30.0 <= angle_diff <= 135.0:
                            return "side_impact_collision"
            return "rear_end_collision"

        if num_vehicles == 1:
            tid = tids[0]
            feat = self._track_features.get(tid, {})
            yaw_rate = float(feat.get("yaw_rate_dps", 0.0))
            dir_change_sum = float(feat.get("direction_change_sum_deg", 0.0))

            if self._detect_single_vehicle_spin_or_barrier_crash(tid):
                return "single_vehicle_loss_of_control"
            if dir_change_sum >= 75.0 or abs(yaw_rate) >= 75.0:
                return "single_vehicle_spin_or_rollover"
            return "single_vehicle_loss_of_control"

        return "unknown"

    def _is_in_post_accident_cooldown(
        self,
        frame_id: int,
        position: Optional[np.ndarray],
        feat: Dict[str, Any],
    ) -> bool:
        if not self._events or position is None:
            return False

        pos_arr = np.asarray(position, dtype=np.float32)
        speed_mean = float(feat.get("speed_mean_mps", 0.0))
        speed_drop = float(feat.get("speed_drop_window_mps", 0.0))

        for past_event in self._events:
            if past_event.active:
                continue
            if (
                0 < frame_id - past_event.end_frame <= self.temporal_merge_gap_frames
                and past_event.center_position is not None
            ):
                dist = float(np.linalg.norm(pos_arr - past_event.center_position))
                if dist <= 25.0 and (speed_mean < 5.5 or speed_drop < 4.5):
                    return True

        return False

    def _find_colliding_partner(
        self,
        track_id: int,
        frame_id: int,
    ) -> Optional[int]:
        """
        Chỉ xác nhận có phương tiện thứ 2 va chạm cùng nếu 2 khung Bounding Box thực sự
        chạm/giao nhau (hoặc sát cạnh nhau) VÀ có sự xung đột hướng đi / giảm tốc mạnh.
        Hai xe đi song song cùng chiều cách xa nhau (như Xe #44 và #67) sẽ trả về None!
        """
        b1 = self._track_bboxes.get(track_id)
        pos1 = self._track_positions.get(track_id)
        dir1 = self._track_directions.get(track_id, 0.0)
        if b1 is None or pos1 is None:
            return None

        for other_tid, b2 in self._track_bboxes.items():
            if other_tid == track_id:
                continue
            allowed_lag = 0 if self._observed_frame_id == frame_id else 1
            if (frame_id - self._track_last_seen_frame.get(other_tid, -999)) > allowed_lag:
                continue

            # ByteTrack can briefly assign two IDs to the same detection.
            # Near-identical boxes do not establish a two-vehicle collision.
            if self._bbox_iou(b1, b2) >= 0.75:
                continue

            # Kiểm tra 2 khung hình có thực sự chạm hoặc sát nhau không
            if not self._bboxes_touch_or_close(b1, b2, expand_ratio=0.18):
                continue

            other_spd = self._track_speeds.get(other_tid, 0.0)
            other_feat = self._track_features.get(other_tid, {})
            other_mean_spd = float(other_feat.get("speed_mean_mps", other_spd))
            if other_mean_spd < 1.5 and other_spd < 1.5:
                continue

            dir2 = self._track_directions.get(other_tid, 0.0)
            angle_diff = abs((dir1 - dir2 + 180.0) % 360.0 - 180.0)
            other_decel = float(other_feat.get("deceleration_mps2", 0.0))
            other_drop = float(other_feat.get("speed_drop_window_mps", 0.0))

            # Nếu 2 xe chạy gần như song song cùng chiều (angle_diff < 22 độ),
            # bắt buộc cả 2 phải có sụt giảm tốc độ lớn (tông đuôi) chứ không phải đi lướt qua nhau
            if angle_diff < 22.0 and other_decel < 3.5 and other_drop < 3.0:
                continue

            return other_tid

        return None

    def _is_physically_consistent_accident(
        self,
        track_id: int,
        frame_id: int,
        position: Optional[np.ndarray],
        feat: Dict[str, Any],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, Optional[int]]:
        """
        Trả về (is_valid_accident, colliding_partner_tid).
        """
        if not feat:
            return False, None

        # 1. Loại bỏ ngay các xe đang nằm sát rìa khung hình (như Xe #44, #67 mới lọt vào góc dưới camera)
        if self._is_near_frame_border(bbox, frame_shape):
            return False, None

        if self._is_in_post_accident_cooldown(frame_id, position, feat):
            return False, None

        if not self._has_reliable_image_motion(track_id):
            return False, None

        spd = float(feat.get("speed_mps", 0.0))
        speed_mean = float(feat.get("speed_mean_mps", 0.0))
        speed_drop = float(feat.get("speed_drop_window_mps", 0.0))
        decel = float(feat.get("deceleration_mps2", 0.0))
        stopped_after_motion = int(feat.get("stopped_after_motion", 0))
        sudden_stop = int(feat.get("sudden_stop", 0))
        dir_sum = float(feat.get("direction_change_sum_deg", 0.0))
        yaw_rate = abs(float(feat.get("yaw_rate_dps", 0.0)))

        # Kiểm tra tai nạn đơn xe mất lái / đâm dải phân cách (như Xe #12)
        is_barrier_or_spin_crash = self._detect_single_vehicle_spin_or_barrier_crash(track_id)

        has_kinetic_energy = (
            speed_mean >= 2.2
            or spd >= 2.8
            or speed_drop >= 2.8
            or is_barrier_or_spin_crash
        )
        if not has_kinetic_energy:
            return False, None

        # Kiểm tra va chạm thực sự giữa 2 phương tiện (Bounding box phải tiếp xúc/gần sát nhau)
        partner_tid = self._find_colliding_partner(track_id, frame_id)
        has_kinematic_shock = (
            decel >= 3.8
            or speed_drop >= 3.2
            or sudden_stop == 1
            or stopped_after_motion == 1
            or (dir_sum >= 50.0 and yaw_rate >= 45.0)
        )

        if partner_tid is not None and has_kinematic_shock:
            # Broad padding is useful for finding a possible partner, but is
            # insufficient evidence of contact between two separate lanes.
            if not self._bboxes_touch_or_close(
                bbox, self._track_bboxes.get(partner_tid), expand_ratio=0.03,
            ):
                return False, None
            return True, partner_tid

        # Kiểm tra sự cố đơn phương tiện (Xe #12 mất lái đâm dải phân cách hoặc xe lật)
        is_severe_single_event = (
            is_barrier_or_spin_crash
            or (stopped_after_motion == 1 and decel >= 4.2 and speed_drop >= 4.0)
            or (sudden_stop == 1 and decel >= 5.2 and speed_drop >= 4.0 and dir_sum >= 35.0)
            or (
                dir_sum >= 75.0 and yaw_rate >= 65.0 and speed_mean >= 3.2
                and decel >= 3.8 and speed_drop >= 2.8
            )
        )

        if is_severe_single_event and self._has_reliable_image_motion(
            track_id, require_change=True,
        ):
            return True, None

        return False, None

    def _find_mergeable_event(
        self,
        frame_id: int,
        position: Optional[np.ndarray],
    ) -> Optional[AccidentEvent]:
        if position is None:
            return None
        current_position = np.asarray(position, dtype=np.float32)
        for event in reversed(self._events):
            # Both TIME and SPACE constraints must match to merge incidents.
            if not (event.active or 0 <= frame_id - event.end_frame <= self.temporal_merge_gap_frames):
                continue
            if event.center_position is None:
                continue
            distance = float(np.linalg.norm(current_position - event.center_position))
            if distance <= self.spatial_merge_distance:
                return event
        return None

    def _short_contact_evidence(
        self, track_id: int, partner_tid: Optional[int], frame_id: int,
        probability: float, feat: Dict[str, Any],
    ) -> List[Tuple[int, float, bool]]:
        """Confirm brief impacts using repeated contact and a real motion shock.

        A physical impact may last fewer frames than the ordinary RF window,
        especially when tracking fragments. This path still requires the RF
        threshold on two adjacent frames and at least one severe shock.
        """
        if partner_tid is None or probability < self.start_threshold:
            return []
        b1 = self._track_bboxes.get(track_id)
        b2 = self._track_bboxes.get(partner_tid)
        if not self._bboxes_touch_or_close(b1, b2, expand_ratio=0.0):
            return []
        decel = float(feat.get("deceleration_mps2", 0.0))
        drop = float(feat.get("speed_drop_window_mps", 0.0))
        stopped = int(feat.get("stopped_after_motion", 0))
        if decel < 3.8 and drop < 3.2 and stopped != 1:
            return []
        severe = decel >= 8.0 or drop >= 3.2 or stopped == 1
        pair = tuple(sorted((track_id, partner_tid)))
        history = self._pair_history[pair]
        if history and frame_id - history[-1][0] > 1:
            history.clear()
        if history and history[-1][0] == frame_id:
            prior = history.pop()
            history.append((frame_id, max(probability, prior[1]), severe or prior[2]))
        else:
            history.append((frame_id, probability, severe))
        needed = min(2, self.confirmation_frames)
        recent = list(history)[-needed:]
        if len(recent) == needed and any(sample[2] for sample in recent):
            return recent
        return []

    def _attach_fragmented_track(
        self, track_id: int, frame_id: int, position: Optional[np.ndarray],
        bbox: Optional[Tuple[float, float, float, float]], vehicle_class: str,
        speed: float, direction: float,
    ) -> Optional[AccidentEvent]:
        """Reconnect a new ID to a recently lost participant of a known impact."""
        if track_id in self._track_first_frame or position is None or bbox is None:
            return None
        best: Optional[AccidentEvent] = None
        best_lost_tid: Optional[int] = None
        best_iou = 0.05
        for event in self._events:
            if not event.active or not event.accident_type.endswith("collision"):
                continue
            for lost_tid in event.track_ids:
                if lost_tid in self._track_successors:
                    continue
                gap = frame_id - self._track_last_seen_frame.get(lost_tid, frame_id)
                if not 0 < gap <= self.lost_grace_frames:
                    continue
                old_bbox = self._track_bboxes.get(lost_tid)
                old_pos = self._track_positions.get(lost_tid)
                old_class = self._track_classes.get(lost_tid, "vehicle")
                if old_bbox is None or old_pos is None:
                    continue
                if old_class != vehicle_class and "vehicle" not in (old_class, vehicle_class):
                    continue
                if np.linalg.norm(np.asarray(position) - old_pos) > self.spatial_merge_distance:
                    continue
                overlap = self._bbox_iou(bbox, old_bbox)
                if overlap > best_iou:
                    best_iou, best, best_lost_tid = overlap, event, lost_tid
        if best_lost_tid is not None and speed >= 1.5:
            old_speed = self._track_speeds.get(best_lost_tid, 0.0)
            old_direction = self._track_directions.get(best_lost_tid, 0.0)
            heading_difference = abs((direction - old_direction + 180.0) % 360.0 - 180.0)
            if old_speed >= 1.5 and heading_difference > 135.0:
                return None
        if best is not None:
            best.register_track(track_id, vehicle_class, alias_of=best_lost_tid)
            best.track_links.append({
                "source_track_id": best_lost_tid,
                "target_track_id": track_id,
                "frame": frame_id,
                "reason": "geometry",
                "identity_status": "unconfirmed",
            })
            self._active_by_track[track_id] = best
            self._track_last_accident_frame[track_id] = frame_id
            if best_lost_tid is not None:
                self._track_successors[best_lost_tid] = track_id
        return best

    def _register_event_track(
        self, event: AccidentEvent, track_id: int, vehicle_class: str,
    ) -> None:
        """Preserve an estimated handoff when an existing incident is merged again."""
        predecessors = {successor: predecessor for predecessor, successor in self._track_successors.items()}
        pending = [track_id]
        visited: Set[int] = set()
        while pending:
            related_id = pending.pop()
            if related_id in visited:
                continue
            visited.add(related_id)
            if related_id != track_id and related_id in event.track_ids:
                event.register_track(track_id, vehicle_class, alias_of=related_id)
                return
            pending.extend(
                related for related in (
                    predecessors.get(related_id), self._track_successors.get(related_id)
                ) if related is not None
            )
        event.register_track(track_id, vehicle_class)

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
        bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[AccidentEvent]:
        raw_prob = float(probability)
        feat = feature_dict or {}
        self._record_tracking_observation(track_id, frame_id, vehicle_class)

        previous_update = self._track_last_update_frame.get(track_id)
        if previous_update is not None and frame_id - previous_update > 1:
            # Evidence before an occlusion cannot confirm a later observation.
            self._history[track_id].clear()
            self._history_frames[track_id].clear()
            self._track_bbox_ratios[track_id].clear()
            self._track_pixel_centers[track_id].clear()
            self._track_first_frame[track_id] = frame_id
        self._track_last_update_frame[track_id] = frame_id
        self._attach_fragmented_track(
            track_id, frame_id, position, bbox, vehicle_class, speed, direction
        )

        if position is not None:
            self._track_positions[track_id] = np.asarray(position, dtype=np.float32)
        if bbox is not None:
            self._track_bboxes[track_id] = bbox
            w = max(1.0, float(bbox[2] - bbox[0]))
            h = max(1.0, float(bbox[3] - bbox[1]))
            self._track_bbox_ratios[track_id].append(w / h)
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            self._track_pixel_centers[track_id].append(np.array([cx, cy], dtype=np.float32))

        self._track_directions[track_id] = float(direction)
        self._track_speeds[track_id] = float(speed)
        self._track_classes[track_id] = vehicle_class
        self._track_features[track_id] = feat
        self._track_last_seen_frame[track_id] = frame_id

        physically_valid, partner_tid = self._is_physically_consistent_accident(
            track_id=track_id,
            frame_id=frame_id,
            position=position,
            feat=feat,
            bbox=bbox,
            frame_shape=frame_shape,
        )

        # Geometry may reject a prediction, but it must not invent RF confidence.
        effective_prob = raw_prob if physically_valid else min(raw_prob, 0.20)

        h = self._history[track_id]
        h.append(effective_prob)

        frame_history = self._history_frames[track_id]
        frame_history.append(frame_id)

        event = self._active_by_track.get(track_id)
        collision_event = (
            event is not None and event.active
            and event.accident_type.endswith("collision")
        )
        residual_shock = (
            float(feat.get("deceleration_mps2", 0.0)) >= 3.8
            and float(feat.get("speed_drop_window_mps", 0.0)) >= 0.5
        )
        if event is not None and event.active and raw_prob >= self.start_threshold:
            if physically_valid or (collision_event and residual_shock):
                self._event_last_evidence_frame[event.event_id] = frame_id
        continuing_motion = (
            collision_event
            and frame_id - self._event_last_evidence_frame.get(event.event_id, event.start_frame)
            <= self.temporal_merge_gap_frames
            and float(feat.get("speed_mps", speed)) >= 1.5
            and not self._is_near_frame_border(bbox, frame_shape)
        )

        if (effective_prob >= self.release_threshold and physically_valid) or continuing_motion:
            self._track_last_accident_frame[track_id] = frame_id
            self._track_calm_frames[track_id] = 0
        else:
            self._track_calm_frames[track_id] += 1

        if event is not None and event.active:
            if effective_prob > event.probability and physically_valid:
                event.probability = min(effective_prob, 0.965)
                event.peak_frame = frame_id

            if physically_valid or continuing_motion:
                event.end_frame = frame_id
                event.end_time_s = event.end_frame / self.fps
                event.duration_s = max(
                    0.1, (event.end_frame - event.start_frame + 1) / self.fps
                )
            return event

        if track_id not in self._track_first_frame:
            self._track_first_frame[track_id] = frame_id
        track_age = frame_id - self._track_first_frame[track_id] + 1

        track_mature = track_age >= self.min_track_age
        window_size = self.confirmation_frames
        recent_probs = list(h)[-window_size:]
        high_count = sum(1 for x in recent_probs if x >= self.start_threshold)
        required_high = min(window_size, max(3, window_size - 3))

        probs_confirmed = (
            len(recent_probs) >= required_high
            and high_count >= required_high
            and effective_prob >= self.start_threshold
        )

        contact_evidence = self._short_contact_evidence(
            track_id, partner_tid, frame_id, raw_prob, feat
        ) if physically_valid else []

        if track_mature and (probs_confirmed or contact_evidence) and physically_valid:
            if contact_evidence:
                start_f = contact_evidence[0][0]
                evidence_probability = float(np.mean([sample[1] for sample in contact_evidence]))
            else:
                recent_frames = list(frame_history)[-window_size:]
                high_samples = [
                    (sample_frame, sample_prob)
                    for sample_frame, sample_prob in zip(recent_frames, recent_probs)
                    if sample_prob >= self.start_threshold
                ]
                start_f = high_samples[0][0]
                evidence_probability = float(np.mean([sample[1] for sample in high_samples]))
            merge_event = self._find_mergeable_event(frame_id, position)
            if merge_event is not None:
                merge_event.active = True
                self._event_last_evidence_frame[merge_event.event_id] = frame_id
                self._register_event_track(merge_event, track_id, vehicle_class)
                if partner_tid is not None:
                    self._register_event_track(
                        merge_event, partner_tid, self._track_classes.get(partner_tid, "car")
                    )
                merge_event.probability = min(
                    max(merge_event.probability, evidence_probability),
                    0.965,
                )
                merge_event.end_frame = max(merge_event.end_frame, frame_id)
                merge_event.end_time_s = merge_event.end_frame / self.fps
                merge_event.duration_s = max(
                    0.1, (merge_event.end_frame - merge_event.start_frame + 1) / self.fps
                )
                merge_event.accident_type = self._infer_accident_type(merge_event)
                for tid in merge_event.track_ids:
                    self._active_by_track[tid] = merge_event
                return merge_event

            pos_copy = (
                np.asarray(position, dtype=np.float32).copy()
                if position is not None
                else None
            )
            new_event = AccidentEvent(
                event_id=self._next_id,
                start_frame=start_f,
                end_frame=frame_id,
                peak_frame=frame_id,
                probability=min(evidence_probability, 0.965),
                track_ids={track_id},
                impact_track_ids={track_id} if partner_tid is None else {track_id, partner_tid},
                vehicle_types={track_id: vehicle_class},
                accident_type="unknown",
                active=True,
                start_time_s=start_f / self.fps,
                end_time_s=frame_id / self.fps,
                duration_s=max(0.1, (frame_id - start_f + 1) / self.fps),
                center_position=pos_copy,
            )
            self._next_id += 1
            if partner_tid is not None:
                self._register_event_track(
                    new_event, partner_tid, self._track_classes.get(partner_tid, "car")
                )
            new_event.accident_type = self._infer_accident_type(new_event)

            self._events.append(new_event)
            self._event_last_evidence_frame[new_event.event_id] = frame_id
            for tid in new_event.track_ids:
                self._active_by_track[tid] = new_event
            return new_event

        return None

    def end_frame(self, frame_id: int, seen_track_ids: Set[int]) -> None:
        for event in self._events:
            if not event.active:
                continue

            # Each participant may independently become calm OR be lost.
            # A mixture of those states must also release the event.
            all_settled = bool(event.track_ids) and all(
                (
                    tid in seen_track_ids
                    and self._track_calm_frames[tid] >= self.release_frames
                )
                or (
                    frame_id - self._track_last_seen_frame.get(tid, frame_id)
                    > self.lost_grace_frames
                )
                for tid in event.track_ids
            )

            exceeded_duration = (
                frame_id - event.start_frame
            ) > self.max_event_duration_frames

            if all_settled or exceeded_duration:
                last_active = max(
                    [
                        self._track_last_accident_frame.get(tid, event.start_frame)
                        for tid in event.track_ids
                    ],
                    default=event.start_frame,
                )
                event.end_frame = min(max(event.end_frame, last_active + 2), frame_id)
                event.active = False
                event.end_time_s = event.end_frame / self.fps
                event.duration_s = max(
                    0.1, (event.end_frame - event.start_frame + 1) / self.fps
                )

                for tid in list(event.track_ids):
                    self._active_by_track.pop(tid, None)

    def finish(self, final_frame_id: int) -> None:
        for event in self._events:
            if event.active:
                last_active = max(
                    [
                        self._track_last_accident_frame.get(tid, event.start_frame)
                        for tid in event.track_ids
                    ],
                    default=event.start_frame,
                )
                event.end_frame = min(
                    max(event.end_frame, last_active + 2), final_frame_id
                )
                event.active = False
                event.end_time_s = event.end_frame / self.fps
                event.duration_s = max(
                    0.1, (event.end_frame - event.start_frame + 1) / self.fps
                )

        self._active_by_track.clear()

    def all_events(self) -> List[AccidentEvent]:
        return list(self._events)

    def get_active_events(self) -> List[AccidentEvent]:
        return [e for e in self._events if e.active]
