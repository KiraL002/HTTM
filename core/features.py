from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class FeatureVector:
    """
    Feature vector của một vehicle tại một frame.
    """

    track_id: int

    x_m: float
    y_m: float

    speed_mps: float
    acceleration_mps2: float

    direction_deg: float
    direction_change_deg: float

    nearest_distance_m: float

    sudden_stop: int

    trajectory_change: int

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "track_id": self.track_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "speed_mps": self.speed_mps,
            "acceleration_mps2": (
                self.acceleration_mps2
            ),
            "direction_deg": self.direction_deg,
            "direction_change_deg": (
                self.direction_change_deg
            ),
            "nearest_distance_m": (
                self.nearest_distance_m
            ),
            "sudden_stop": self.sudden_stop,
            "trajectory_change": (
                self.trajectory_change
            ),
        }


class FeatureExtractor:
    """
    Tính đặc trưng động học.

    Tất cả tọa độ đầu vào nên đã là tọa độ thực tế
    trong hệ BEV, đơn vị meter.
    """

    def __init__(
        self,
        fps: float,
        sudden_stop_deceleration: float = 3.0,
        direction_change_threshold: float = 45.0,
    ) -> None:

        if fps <= 0:
            raise ValueError(
                "fps must be > 0"
            )

        self.fps = float(fps)
        self.dt = 1.0 / self.fps

        self.sudden_stop_deceleration = float(
            sudden_stop_deceleration
        )

        self.direction_change_threshold = float(
            direction_change_threshold
        )

    def calculate_velocity(
        self,
        previous_position: np.ndarray,
        current_position: np.ndarray,
    ) -> np.ndarray:

        return (
            current_position
            - previous_position
        ) / self.dt

    def calculate_speed(
        self,
        velocity: np.ndarray,
    ) -> float:

        return float(
            np.linalg.norm(velocity)
        )

    def calculate_acceleration(
        self,
        previous_velocity: np.ndarray,
        current_velocity: np.ndarray,
    ) -> np.ndarray:

        return (
            current_velocity
            - previous_velocity
        ) / self.dt

    def calculate_direction(
        self,
        velocity: np.ndarray,
    ) -> float:

        vx, vy = velocity

        if np.linalg.norm(velocity) < 1e-8:
            return 0.0

        angle = np.degrees(
            np.arctan2(vy, vx)
        )

        return float(angle)

    def calculate_direction_change(
        self,
        previous_direction: float,
        current_direction: float,
    ) -> float:

        delta = (
            current_direction
            - previous_direction
        )

        # đưa về [-180, 180]
        delta = (
            delta + 180.0
        ) % 360.0 - 180.0

        return abs(float(delta))

    def calculate_distance(
        self,
        position_a: np.ndarray,
        position_b: np.ndarray,
    ) -> float:

        return float(
            np.linalg.norm(
                position_a - position_b
            )
        )

    def nearest_distance(
        self,
        current_position: np.ndarray,
        all_positions: Dict[int, np.ndarray],
        current_track_id: int,
    ) -> float:

        distances = []

        for track_id, position in all_positions.items():

            if track_id == current_track_id:
                continue

            if position is None:
                continue

            if not np.all(
                np.isfinite(position)
            ):
                continue

            distance = (
                self.calculate_distance(
                    current_position,
                    position,
                )
            )

            distances.append(distance)

        if not distances:
            return float("inf")

        return float(min(distances))

    def detect_sudden_stop(
        self,
        previous_speed: float,
        current_speed: float,
    ) -> int:

        deceleration = (
            previous_speed
            - current_speed
        ) / self.dt

        return int(
            deceleration
            >= self.sudden_stop_deceleration
        )

    def detect_trajectory_change(
        self,
        direction_change_deg: float,
    ) -> int:

        return int(
            direction_change_deg
            >= self.direction_change_threshold
        )

    def extract(
        self,
        track_id: int,
        current_position: np.ndarray,
        previous_position: Optional[np.ndarray],
        previous_velocity: Optional[np.ndarray],
        previous_direction: Optional[float],
        previous_speed: float,
        all_current_positions: Dict[
            int,
            np.ndarray
        ],
    ) -> FeatureVector:

        current_position = np.asarray(
            current_position,
            dtype=np.float32,
        )

        # Chưa đủ 2 frame để tính velocity.
        if previous_position is None:

            velocity = np.zeros(
                2,
                dtype=np.float32,
            )

            speed = 0.0

        else:

            velocity = self.calculate_velocity(
                previous_position,
                current_position,
            )

            speed = self.calculate_speed(
                velocity
            )

        direction = self.calculate_direction(
            velocity
        )

        if previous_velocity is None:

            acceleration_vector = (
                np.zeros(
                    2,
                    dtype=np.float32,
                )
            )

        else:

            acceleration_vector = (
                self.calculate_acceleration(
                    previous_velocity,
                    velocity,
                )
            )

        acceleration = float(
            np.linalg.norm(
                acceleration_vector
            )
        )

        if previous_direction is None:

            direction_change = 0.0

        else:

            direction_change = (
                self.calculate_direction_change(
                    previous_direction,
                    direction,
                )
            )

        nearest = self.nearest_distance(
            current_position=current_position,
            all_positions=all_current_positions,
            current_track_id=track_id,
        )

        sudden_stop = self.detect_sudden_stop(
            previous_speed=previous_speed,
            current_speed=speed,
        )

        trajectory_change = (
            self.detect_trajectory_change(
                direction_change
            )
        )

        return FeatureVector(
            track_id=track_id,
            x_m=float(current_position[0]),
            y_m=float(current_position[1]),
            speed_mps=speed,
            acceleration_mps2=acceleration,
            direction_deg=direction,
            direction_change_deg=direction_change,
            nearest_distance_m=nearest,
            sudden_stop=sudden_stop,
            trajectory_change=trajectory_change,
        )