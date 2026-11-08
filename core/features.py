"""
AcciVision — Module Trích Xuất Đặc Trưng Động Học (Kinematic Feature Extractor)

Đặc trưng được tính từ vị trí đã quy đổi ở bước trước.
Độ chính xác vật lý phụ thuộc vào việc hiệu chỉnh camera/phép biến đổi phối cảnh.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureVector:
    track_id: int
    object_type: str
    x_m: float
    y_m: float
    speed_mps: float
    acceleration_mps2: float
    deceleration_mps2: float
    speed_delta_mps: float
    direction_deg: float
    direction_change_deg: float
    yaw_rate_dps: float
    lateral_acceleration_mps2: float
    jerk_mps3: float
    nearest_distance_m: float
    neighbor_count: int
    has_neighbor: int
    nearest_distance_valid: int
    closing_speed_mps: float
    interaction_count: int
    sudden_stop: int
    trajectory_change: int
    speed_drop_window_mps: float
    direction_change_max_deg: float
    direction_change_sum_deg: float
    speed_mean_mps: float
    speed_std_mps: float
    acceleration_std_mps2: float
    stopped_after_motion: int
    temporal_window: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {k: v for k, v in self.__dict__.items()}


class FeatureExtractor:
    """
    Trích xuất đặc trưng động học từ vị trí track đã được quy đổi.
    """

    def __init__(
        self,
        fps: float,
        sudden_stop_deceleration: float = 3.5,
        direction_change_threshold: float = 35.0,
        temporal_window: int = 15,
        max_interaction_distance: float = 50.0,
        velocity_smoothing_alpha: float = 0.35,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")

        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.sudden_stop_deceleration = float(sudden_stop_deceleration)
        self.direction_change_threshold = float(direction_change_threshold)
        self.temporal_window = int(max(10, temporal_window))
        self.max_interaction_distance = float(max_interaction_distance)
        self.velocity_smoothing_alpha = float(velocity_smoothing_alpha)
        self.position_smoothing_alpha = 0.50

        self._pos_history = defaultdict(lambda: deque(maxlen=12))
        self._pixel_history = defaultdict(lambda: deque(maxlen=12))
        self._bbox_history = defaultdict(lambda: deque(maxlen=15))
        self._smoothed_positions: Dict[int, np.ndarray] = {}
        self._speed_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._accel_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._dir_change_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._time_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._smoothed_velocities = defaultdict(lambda: np.zeros(2, dtype=np.float32))
        self._previous_accel = defaultdict(float)
        self._track_age = defaultdict(int)
        self._last_frame_id: Dict[int, int] = {}

    def get_smoothed_velocity(self, track_id: int) -> np.ndarray:
        if track_id in self._smoothed_velocities:
            return self._smoothed_velocities[track_id].copy()
        return np.zeros(2, dtype=np.float32)

    def _smooth_position(self, track_id: int, raw_position: np.ndarray) -> np.ndarray:
        raw_pos = np.asarray(raw_position, dtype=np.float32)
        if track_id not in self._smoothed_positions:
            self._smoothed_positions[track_id] = raw_pos.copy()
            return raw_pos

        prev_pos = self._smoothed_positions[track_id]
        smoothed = (
            self.position_smoothing_alpha * raw_pos
            + (1.0 - self.position_smoothing_alpha) * prev_pos
        )
        self._smoothed_positions[track_id] = smoothed
        return smoothed

    def _is_jittering_stationary(
        self,
        track_id: int,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """Suppress small oscillations while preserving consistent slow motion."""
        px_hist = self._pixel_history[track_id]
        if len(px_hist) >= 5 and bbox is not None:
            w = max(12.0, float(bbox[2] - bbox[0]))
            h = max(12.0, float(bbox[3] - bbox[1]))
            box_scale = 0.5 * (w + h)
            positions = np.asarray(px_hist)
            jitter_radius = 0.16 * box_scale
        else:
            pos_hist = self._pos_history[track_id]
            if len(pos_hist) < 5:
                return False
            positions = np.asarray(pos_hist)
            jitter_radius = 0.30

        steps = np.diff(positions, axis=0)
        path_length = float(np.linalg.norm(steps, axis=1).sum())
        if path_length <= 1e-6:
            return True
        net_disp = float(np.linalg.norm(positions[-1] - positions[0]))
        extent = float(np.linalg.norm(positions - positions[0], axis=1).max())
        # Net displacement alone also labels a slowly approaching vehicle as
        # parked. Consistent motion has net/path near one, regardless of scale.
        coherence = net_disp / path_length
        return extent < jitter_radius and coherence < 0.5

    def calculate_velocity(
        self,
        previous_position: np.ndarray,
        current_position: np.ndarray,
        track_id: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> np.ndarray:
        effective_dt = self.dt if elapsed_seconds is None else max(1e-6, float(elapsed_seconds))
        raw_velocity = (
            np.asarray(current_position, dtype=np.float32)
            - np.asarray(previous_position, dtype=np.float32)
        ) / effective_dt

        if track_id is not None and track_id in self._smoothed_velocities:
            prev_v = self._smoothed_velocities[track_id]
            smoothed_v = (
                self.velocity_smoothing_alpha * raw_velocity
                + (1.0 - self.velocity_smoothing_alpha) * prev_v
            )
            self._smoothed_velocities[track_id] = smoothed_v
            return smoothed_v

        if track_id is not None:
            self._smoothed_velocities[track_id] = raw_velocity

        return raw_velocity

    def calculate_speed(self, velocity: np.ndarray) -> float:
        return float(np.linalg.norm(velocity))

    def calculate_acceleration(
        self,
        previous_velocity: np.ndarray,
        current_velocity: np.ndarray,
        elapsed_seconds: Optional[float] = None,
    ) -> np.ndarray:
        effective_dt = self.dt if elapsed_seconds is None else max(1e-6, float(elapsed_seconds))
        return (
            np.asarray(current_velocity, dtype=np.float32)
            - np.asarray(previous_velocity, dtype=np.float32)
        ) / effective_dt

    def calculate_direction(self, velocity: np.ndarray) -> float:
        if np.linalg.norm(velocity) < 1e-4:
            return 0.0
        return float(np.degrees(np.arctan2(velocity[1], velocity[0])))

    def calculate_direction_change(
        self,
        previous_direction: float,
        current_direction: float,
    ) -> float:
        return abs(float((current_direction - previous_direction + 180.0) % 360.0 - 180.0))

    def calculate_distance(
        self,
        position_a: np.ndarray,
        position_b: np.ndarray,
    ) -> float:
        return float(
            np.linalg.norm(
                np.asarray(position_a, dtype=np.float32)
                - np.asarray(position_b, dtype=np.float32)
            )
        )

    def calculate_vehicle_interaction(
        self,
        current_track_id: int,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        all_vehicle_positions: Dict[int, np.ndarray],
        all_vehicle_velocities: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[float, int, int, int, float]:
        distances: List[Tuple[float, int]] = []
        curr_pos = np.asarray(current_position, dtype=np.float32)

        for track_id, pos in all_vehicle_positions.items():
            if track_id == current_track_id or pos is None:
                continue
            pos_arr = np.asarray(pos, dtype=np.float32)
            if np.all(np.isfinite(pos_arr)):
                dist = self.calculate_distance(curr_pos, pos_arr)
                distances.append((dist, track_id))

        if not distances:
            return (self.max_interaction_distance, 0, 0, 0, 0.0)

        distances.sort(key=lambda x: x[0])
        min_dist, nearest_tid = distances[0]
        capped_dist = float(min(min_dist, self.max_interaction_distance))
        has_neighbor = 1 if min_dist <= 25.0 else 0
        valid = 1

        closing_speed = 0.0
        if all_vehicle_velocities is not None and nearest_tid in all_vehicle_velocities:
            nearest_pos = np.asarray(all_vehicle_positions[nearest_tid], dtype=np.float32)
            nearest_vel = np.asarray(all_vehicle_velocities[nearest_tid], dtype=np.float32)

            rel_pos = nearest_pos - curr_pos
            pos_norm = np.linalg.norm(rel_pos)
            if pos_norm > 1e-3:
                rel_vel = current_velocity - nearest_vel
                closing_speed = float(np.dot(rel_vel, rel_pos) / pos_norm)

        return (capped_dist, len(distances), has_neighbor, valid, closing_speed)

    def extract_frame(
        self,
        track_inputs: List[dict],
        frame_id: Optional[int] = None,
    ) -> Dict[int, FeatureVector]:
        """Extract kinematics, then interactions using one current-frame snapshot.

        Each input contains the per-track arguments accepted by ``extract``;
        omit ``all_vehicle_positions`` and ``all_vehicle_velocities``. Previous
        states must all come from the preceding observation of each track.
        """
        features: Dict[int, FeatureVector] = {}
        for track_input in track_inputs:
            track_id = int(track_input["track_id"])
            if track_id in features:
                raise ValueError(f"Duplicate track_id in frame: {track_id}")
            features[track_id] = self.extract(
                **track_input,
                all_vehicle_positions={},
                frame_id=frame_id,
            )

        positions = {
            track_id: np.asarray([feature.x_m, feature.y_m], dtype=np.float32)
            for track_id, feature in features.items()
        }
        velocities = {
            track_id: self.get_smoothed_velocity(track_id)
            for track_id in features
        }
        for track_id, feature in features.items():
            (
                feature.nearest_distance_m,
                feature.neighbor_count,
                feature.has_neighbor,
                feature.nearest_distance_valid,
                feature.closing_speed_mps,
            ) = self.calculate_vehicle_interaction(
                current_track_id=track_id,
                current_position=positions[track_id],
                current_velocity=velocities[track_id],
                all_vehicle_positions=positions,
                all_vehicle_velocities=velocities,
            )
            feature.interaction_count = feature.neighbor_count
        return features

    def extract(
        self,
        track_id: int,
        current_position: np.ndarray,
        previous_position: Optional[np.ndarray],
        previous_velocity: Optional[np.ndarray],
        previous_direction: Optional[float],
        previous_speed: float,
        all_vehicle_positions: Dict[int, np.ndarray],
        all_vehicle_velocities: Optional[Dict[int, np.ndarray]] = None,
        object_type: str = "vehicle",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_id: Optional[int] = None,
    ) -> FeatureVector:
        """
        Trích xuất vector đặc trưng cho phương tiện theo từng track và thời gian thực.
        """
        # The same track may disappear for several frames during occlusion.
        # Account for elapsed time instead of treating the gap as one frame.
        frames_elapsed = 1
        if frame_id is not None:
            last_id = self._last_frame_id.get(track_id)
            if last_id is not None:
                frames_elapsed = max(1, int(frame_id) - last_id)
            if frames_elapsed > 15:
                self.reset_track(track_id)
                previous_position = None
                previous_velocity = None
                previous_direction = None
                previous_speed = 0.0
                frames_elapsed = 1
            self._last_frame_id[track_id] = int(frame_id)
        elapsed_seconds = frames_elapsed * self.dt

        if bbox is not None:
            bx = 0.5 * (bbox[0] + bbox[2])
            by = bbox[3]
            self._pixel_history[track_id].append(np.array([bx, by], dtype=np.float32))
            self._bbox_history[track_id].append(bbox)

        prev_smoothed_pos = self._smoothed_positions.get(track_id)
        smoothed_pos = self._smooth_position(track_id, current_position)
        self._pos_history[track_id].append(smoothed_pos.copy())

        self._track_age[track_id] += 1
        age = self._track_age[track_id]

        if prev_smoothed_pos is None or previous_position is None:
            velocity = np.zeros(2, dtype=np.float32)
        else:
            velocity = self.calculate_velocity(
                prev_smoothed_pos, smoothed_pos,
                track_id=track_id, elapsed_seconds=elapsed_seconds
            )

        # Do not boost speed using bbox size: it makes train/inference
        # feature distributions inconsistent and amplifies tracking jitter.
        # Position calibration belongs in the camera homography upstream.

        is_stationary_jitter = self._is_jittering_stationary(track_id, bbox=bbox)
        if is_stationary_jitter:
            velocity = np.zeros(2, dtype=np.float32)
            self._smoothed_velocities[track_id] = velocity.copy()

        speed = self.calculate_speed(velocity)

        if speed >= 1.5 and not is_stationary_jitter:
            direction = self.calculate_direction(velocity)
        else:
            direction = float(previous_direction) if previous_direction is not None else 0.0

        # 2. Gia tốc dọc & giảm tốc
        if age < 3 or previous_velocity is None or np.all(previous_velocity == 0):
            acceleration = 0.0
            deceleration = 0.0
            speed_delta = 0.0
        elif is_stationary_jitter and previous_speed < 1.8:
            acceleration = 0.0
            deceleration = 0.0
            speed_delta = 0.0
        else:
            acceleration_vector = self.calculate_acceleration(
                previous_velocity, velocity, elapsed_seconds=elapsed_seconds
            )
            if speed > 1e-4:
                direction_unit = velocity / speed
                acceleration = float(np.dot(acceleration_vector, direction_unit))
            else:
                acceleration = float(np.linalg.norm(acceleration_vector))
                if speed < previous_speed:
                    acceleration = -acceleration

            acceleration = float(np.clip(acceleration, -18.0, 15.0))
            deceleration = float(np.clip(max(0.0, -acceleration), 0.0, 18.0))
            speed_delta = float(np.clip(speed - previous_speed, -25.0, 25.0))

        # 3. Hướng & Tốc độ quay
        prev_a = self._previous_accel[track_id]
        if age < 3 or previous_direction is None or speed < 1.5 or is_stationary_jitter:
            direction_change = 0.0
            yaw_rate = 0.0
            lateral_accel = 0.0
            jerk = 0.0
        else:
            direction_change = self.calculate_direction_change(previous_direction, direction)
            direction_change = float(np.clip(direction_change, 0.0, 180.0))
            yaw_rate = float(np.clip(direction_change / elapsed_seconds, 0.0, 180.0))

            yaw_rate_rad = np.radians(yaw_rate)
            lateral_accel = float(np.clip(speed * yaw_rate_rad, 0.0, 30.0))
            jerk = float(np.clip((acceleration - prev_a) / elapsed_seconds, -60.0, 60.0))

        self._previous_accel[track_id] = acceleration

        # 4. Tương tác giữa các phương tiện
        (
            nearest_dist,
            neighbor_cnt,
            has_neighbor,
            dist_valid,
            closing_spd,
        ) = self.calculate_vehicle_interaction(
            current_track_id=track_id,
            current_position=smoothed_pos,
            current_velocity=velocity,
            all_vehicle_positions=all_vehicle_positions,
            all_vehicle_velocities=all_vehicle_velocities,
        )

        if is_stationary_jitter and speed < 0.5:
            closing_spd = 0.0

        # 5. Cập nhật chuỗi thời gian (Temporal sliding window)
        speeds = self._speed_history[track_id]
        accels = self._accel_history[track_id]
        dir_changes = self._dir_change_history[track_id]
        times = self._time_history[track_id]

        times.append((times[-1] if times else 0.0) + elapsed_seconds)
        speeds.append(speed)
        accels.append(acceleration)
        dir_changes.append(direction_change)

        max_speed_window = float(max(speeds)) if speeds else speed
        speed_drop_window = float(max_speed_window - speed)
        dir_change_max = float(max(dir_changes)) if dir_changes else direction_change
        dir_change_sum = float(sum(dir_changes)) if dir_changes else direction_change

        speed_mean = float(np.mean(speeds)) if speeds else speed
        speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        accel_std = float(np.std(accels)) if len(accels) > 1 else 0.0

        # Nếu có giảm tốc mạnh trong cửa sổ thời gian (ví dụ Xe #12 đâm vào dải phân cách rồi khựng lại)
        if len(speeds) >= 4 and speed_drop_window >= 3.0:
            window_seconds = times[-1] - times[0] + self.dt
            effective_window_decel = speed_drop_window / window_seconds
            deceleration = max(deceleration, float(np.clip(effective_window_decel * 1.5, 0.0, 18.0)))

        stopped_after_motion = (
            1
            if (max_speed_window >= 3.2 and (speed < 1.5 or speed <= 0.35 * max_speed_window) and speed_drop_window >= 2.5)
            else 0
        )

        sudden_stop_flag = (
            1
            if (
                (deceleration >= self.sudden_stop_deceleration or speed_drop_window >= 3.5)
                and max_speed_window >= 3.5
            )
            else 0
        )
        trajectory_change_flag = (
            1
            if (
                (direction_change >= 28.0 or dir_change_sum >= 55.0)
                and max_speed_window >= 2.8
                and not is_stationary_jitter
            )
            else 0
        )

        return FeatureVector(
            track_id=track_id,
            object_type=object_type,
            x_m=float(smoothed_pos[0]),
            y_m=float(smoothed_pos[1]),
            speed_mps=float(speed),
            acceleration_mps2=float(acceleration),
            deceleration_mps2=float(deceleration),
            speed_delta_mps=float(speed_delta),
            direction_deg=float(direction),
            direction_change_deg=float(direction_change),
            yaw_rate_dps=float(yaw_rate),
            lateral_acceleration_mps2=float(lateral_accel),
            jerk_mps3=float(jerk),
            nearest_distance_m=float(nearest_dist),
            neighbor_count=int(neighbor_cnt),
            has_neighbor=int(has_neighbor),
            nearest_distance_valid=int(dist_valid),
            closing_speed_mps=float(closing_spd),
            interaction_count=int(neighbor_cnt),
            sudden_stop=int(sudden_stop_flag),
            trajectory_change=int(trajectory_change_flag),
            speed_drop_window_mps=float(speed_drop_window),
            direction_change_max_deg=float(dir_change_max),
            direction_change_sum_deg=float(dir_change_sum),
            speed_mean_mps=float(speed_mean),
            speed_std_mps=float(speed_std),
            acceleration_std_mps2=float(accel_std),
            stopped_after_motion=int(stopped_after_motion),
            temporal_window=int(len(speeds)),
        )

    def reset_track(self, track_id: int) -> None:
        self._pos_history.pop(track_id, None)
        self._pixel_history.pop(track_id, None)
        self._bbox_history.pop(track_id, None)
        self._smoothed_positions.pop(track_id, None)
        self._speed_history.pop(track_id, None)
        self._accel_history.pop(track_id, None)
        self._dir_change_history.pop(track_id, None)
        self._time_history.pop(track_id, None)
        self._smoothed_velocities.pop(track_id, None)
        self._previous_accel.pop(track_id, None)
        self._track_age.pop(track_id, None)
        self._last_frame_id.pop(track_id, None)
    def reset(self) -> None:
        """Reset all temporal state at the start of a new input video."""
        for tid in list(self._track_age):
            self.reset_track(tid)
