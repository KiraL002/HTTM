"""
AcciVision — Module Trích Xuất Đặc Trưng Động Học (Kinematic Feature Extractor)

Module này trích xuất các đặc trưng vật lý từ quỹ đạo chuyển động
của phương tiện: tốc độ, gia tốc, hướng di chuyển, tương tác giữa
các phương tiện, và các chỉ số cảnh báo tai nạn đơn xe/đa xe.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureVector:
    """
    Tập đặc trưng động học và tương tác của phương tiện tại một frame.

    Semantics rõ ràng:
    - ``acceleration_mps2``: Gia tốc dọc có dấu (m/s²). Dương là tăng tốc, âm là giảm tốc.
    - ``deceleration_mps2``: Độ lớn giảm tốc / phanh (m/s²), luôn >= 0. deceleration = max(0, -acceleration).
    - ``speed_delta_mps``: Độ thay đổi tốc độ (v_t - v_{t-1}).
    - ``nearest_distance_m``: Khoảng cách tới xe gần nhất, giới hạn trần 50.0m để tránh bias dữ liệu.
    - ``has_neighbor``: 1 nếu có xe lân cận trong phạm vi tương tác (<= 25m), 0 nếu là xe đơn độc.
    - ``closing_speed_mps``: Tốc độ tiếp cận xe phía trước/gần nhất (m/s).
    - Single-vehicle features: ``yaw_rate_dps``, ``lateral_acceleration_mps2``, ``jerk_mps3``,
      ``sudden_stop``, ``trajectory_change``, ``speed_drop_window_mps``, ``stopped_after_motion``.
    - Temporal features: Thống kê thực tế trong cửa sổ trượt (window).
    """

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
    Trích xuất đặc trưng động học cho phương tiện, hỗ trợ cả tai nạn đơn xe (single-vehicle)
    và tai nạn va chạm nhiều xe (multi-vehicle interaction).
    """

    def __init__(
        self,
        fps: float,
        sudden_stop_deceleration: float = 3.5,
        direction_change_threshold: float = 35.0,
        temporal_window: int = 15,
        max_interaction_distance: float = 50.0,
        velocity_smoothing_alpha: float = 0.30,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")

        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.sudden_stop_deceleration = float(sudden_stop_deceleration)
        self.direction_change_threshold = float(direction_change_threshold)
        self.temporal_window = int(max(3, temporal_window))
        self.max_interaction_distance = float(max_interaction_distance)
        self.velocity_smoothing_alpha = float(velocity_smoothing_alpha)

        # Lịch sử chuỗi thời gian cho từng track
        self._speed_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._accel_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._dir_change_history = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._smoothed_velocities = defaultdict(lambda: np.zeros(2, dtype=np.float32))
        self._previous_accel = defaultdict(float)
        self._track_age = defaultdict(int)

    def get_smoothed_velocity(self, track_id: int) -> np.ndarray:
        """Lấy vận tốc đã làm mịn của track_id."""
        if track_id in self._smoothed_velocities:
            return self._smoothed_velocities[track_id].copy()
        return np.zeros(2, dtype=np.float32)

    def calculate_velocity(
        self,
        previous_position: np.ndarray,
        current_position: np.ndarray,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Tính velocity (vx, vy) từ 2 vị trí và áp dụng EMA filter nhẹ
        để triệt tiêu nhiễu rung bounding box giữa 2 frame liên tiếp.
        """
        raw_velocity = (np.asarray(current_position, dtype=np.float32) - np.asarray(previous_position, dtype=np.float32)) / self.dt

        if track_id is not None and track_id in self._smoothed_velocities:
            prev_v = self._smoothed_velocities[track_id]
            # EMA smoothing
            smoothed_v = self.velocity_smoothing_alpha * raw_velocity + (1.0 - self.velocity_smoothing_alpha) * prev_v
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
    ) -> np.ndarray:
        return (np.asarray(current_velocity, dtype=np.float32) - np.asarray(previous_velocity, dtype=np.float32)) / self.dt

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
        return float(np.linalg.norm(np.asarray(position_a, dtype=np.float32) - np.asarray(position_b, dtype=np.float32)))

    def calculate_vehicle_interaction(
        self,
        current_track_id: int,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        all_vehicle_positions: Dict[int, np.ndarray],
        all_vehicle_velocities: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[float, int, int, int, float]:
        """
        Chỉ tính tương tác giữa CÁC PHƯƠNG TIỆN (vehicles only).
        Trả về:
            - nearest_distance_m: khoảng cách tới xe gần nhất (giới hạn trần max_interaction_distance)
            - neighbor_count: tổng số xe khác trong frame
            - has_neighbor: 1 nếu có xe lân cận <= 25m, 0 nếu không có
            - nearest_distance_valid: 1 nếu có ít nhất 1 xe khác
            - closing_speed_mps: tốc độ tiếp cận xe gần nhất
        """
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
            # Single vehicle scenario: không có xe nào khác trong frame
            return (
                self.max_interaction_distance,  # 50m neutral sentinel, tránh cực đoan 1000m
                0,                              # neighbor_count
                0,                              # has_neighbor
                0,                              # nearest_distance_valid
                0.0,                            # closing_speed_mps
            )

        distances.sort(key=lambda x: x[0])
        min_dist, nearest_tid = distances[0]
        capped_dist = float(min(min_dist, self.max_interaction_distance))
        has_neighbor = 1 if min_dist <= 25.0 else 0
        valid = 1

        # Tính closing speed (tốc độ tiếp cận) với xe gần nhất
        closing_speed = 0.0
        if all_vehicle_velocities is not None and nearest_tid in all_vehicle_velocities:
            nearest_pos = np.asarray(all_vehicle_positions[nearest_tid], dtype=np.float32)
            nearest_vel = np.asarray(all_vehicle_velocities[nearest_tid], dtype=np.float32)

            rel_pos = nearest_pos - curr_pos
            pos_norm = np.linalg.norm(rel_pos)
            if pos_norm > 1e-3:
                rel_vel = current_velocity - nearest_vel
                # Chiếu vận tốc tương đối lên hướng vector vị trí
                closing_speed = float(np.dot(rel_vel, rel_pos) / pos_norm)

        return (capped_dist, len(distances), has_neighbor, valid, closing_speed)

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
    ) -> FeatureVector:
        """
        Trích xuất đầy đủ vector đặc trưng cho một phương tiện.
        """
        current_position = np.asarray(current_position, dtype=np.float32)
        self._track_age[track_id] += 1
        age = self._track_age[track_id]

        # 1. Tính vận tốc & tốc độ
        if previous_position is None:
            velocity = np.zeros(2, dtype=np.float32)
        else:
            velocity = self.calculate_velocity(previous_position, current_position, track_id=track_id)

        speed = self.calculate_speed(velocity)
        # Khóa hướng khi xe đi quá chậm (< 1.8 m/s ~ 6.5 km/h) hoặc dừng
        # để triệt tiêu nhiễu pixel làm lật góc 180 độ gây hiểu nhầm xe bị xoay/lật
        if speed >= 1.8:
            direction = self.calculate_direction(velocity)
        else:
            direction = float(previous_direction) if previous_direction is not None else 0.0

        # 2. Gia tốc dọc & giảm tốc (semantics thống nhất)
        # Bắt buộc qua giai đoạn warm-up (age >= 3) để tránh cú nhảy gia tốc ảo
        # khi xe mới lọt vào camera từ vận tốc [0, 0] lên vận tốc chạy thực tế
        if age < 3 or previous_velocity is None or np.all(previous_velocity == 0):
            acceleration_vector = np.zeros(2, dtype=np.float32)
            acceleration = 0.0
            deceleration = 0.0
            speed_delta = 0.0
        else:
            acceleration_vector = self.calculate_acceleration(previous_velocity, velocity)
            if speed > 1e-4:
                # Chiếu vector gia tốc lên vector hướng chuyển động
                direction_unit = velocity / speed
                acceleration = float(np.dot(acceleration_vector, direction_unit))
            else:
                acceleration = float(np.linalg.norm(acceleration_vector))
                if speed < previous_speed:
                    acceleration = -acceleration

            # Giới hạn vật lý thực tế: xe giao thông không thể gia tốc/giảm tốc vượt quá 15 m/s² (~1.5g)
            acceleration = float(np.clip(acceleration, -18.0, 15.0))
            deceleration = float(np.clip(max(0.0, -acceleration), 0.0, 18.0))
            speed_delta = float(np.clip(speed - previous_speed, -25.0, 25.0))

        # 3. Hướng & Tốc độ quay (Single-vehicle dynamics)
        prev_a = self._previous_accel[track_id]
        if age < 3 or previous_direction is None or speed < 1.8:
            direction_change = 0.0
            yaw_rate = 0.0
            lateral_accel = 0.0
            jerk = 0.0 if age < 3 else float(np.clip((acceleration - prev_a) / self.dt, -60.0, 60.0))
        else:
            direction_change = self.calculate_direction_change(previous_direction, direction)
            direction_change = float(np.clip(direction_change, 0.0, 180.0))
            yaw_rate = float(np.clip(direction_change / self.dt, 0.0, 180.0))  # deg/s

            yaw_rate_rad = np.radians(yaw_rate)
            lateral_accel = float(np.clip(speed * yaw_rate_rad, 0.0, 30.0))
            jerk = float(np.clip((acceleration - prev_a) / self.dt, -60.0, 60.0))

        self._previous_accel[track_id] = acceleration

        # 4. Chỉ tính tương tác xe với xe (Vehicle interaction only)
        (
            nearest_dist,
            neighbor_cnt,
            has_neighbor,
            dist_valid,
            closing_spd,
        ) = self.calculate_vehicle_interaction(
            current_track_id=track_id,
            current_position=current_position,
            current_velocity=velocity,
            all_vehicle_positions=all_vehicle_positions,
            all_vehicle_velocities=all_vehicle_velocities,
        )

        # 5. Cập nhật chuỗi thời gian thực sự (Temporal sliding window)
        speeds = self._speed_history[track_id]
        accels = self._accel_history[track_id]
        dir_changes = self._dir_change_history[track_id]

        speeds.append(speed)
        accels.append(acceleration)
        dir_changes.append(direction_change)

        # Temporal features
        max_speed_window = float(max(speeds)) if speeds else speed
        speed_drop_window = float(max_speed_window - speed)
        dir_change_max = float(max(dir_changes)) if dir_changes else direction_change
        dir_change_sum = float(sum(dir_changes)) if dir_changes else direction_change

        speed_mean = float(np.mean(speeds)) if speeds else speed
        speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        accel_std = float(np.std(accels)) if len(accels) > 1 else 0.0

        # stopped_after_motion: xe từng chuyển động (> 3.5 m/s) và giờ phanh dừng hẳn (< 1.0 m/s)
        stopped_after_motion = 1 if (max_speed_window >= 3.5 and speed < 1.0 and deceleration >= 2.0) else 0

        # Cờ tai nạn đơn xe sơ bộ (chỉ kích hoạt khi xe thực sự đang chuyển động, không do nhiễu pixel khi đi chậm)
        sudden_stop_flag = 1 if (deceleration >= self.sudden_stop_deceleration and max_speed_window >= 4.5 and speed_drop_window >= 3.0) else 0
        trajectory_change_flag = 1 if (direction_change >= self.direction_change_threshold and speed >= 2.5) else 0

        return FeatureVector(
            track_id=track_id,
            object_type=object_type,
            x_m=float(current_position[0]),
            y_m=float(current_position[1]),
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
        """Xóa cache khi một track kết thúc."""
        self._speed_history.pop(track_id, None)
        self._accel_history.pop(track_id, None)
        self._dir_change_history.pop(track_id, None)
        self._smoothed_velocities.pop(track_id, None)
        self._previous_accel.pop(track_id, None)
        self._track_age.pop(track_id, None)
