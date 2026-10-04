"""
AcciVision — Module Quản Lý Quỹ Đạo Chuyển Động (Trajectory Manager)

Module này lưu trữ và quản lý lịch sử chuyển động (trajectory)
của từng phương tiện, hỗ trợ nội suy khi mất phát hiện tạm thời.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrajectoryPoint:
    """
    Một điểm trong trajectory.
    """

    frame_id: int
    timestamp: float

    x: Optional[float]
    y: Optional[float]

    confidence: Optional[float] = None
    interpolated: bool = False


@dataclass
class TrackTrajectory:
    """
    Lịch sử chuyển động của một track.
    """

    track_id: int

    points: List[TrajectoryPoint] = field(
        default_factory=list
    )

    missed_frames: int = 0

    def add_point(
        self,
        frame_id: int,
        timestamp: float,
        x: Optional[float],
        y: Optional[float],
        confidence: Optional[float] = None,
        interpolated: bool = False,
    ) -> None:

        self.points.append(
            TrajectoryPoint(
                frame_id=frame_id,
                timestamp=timestamp,
                x=x,
                y=y,
                confidence=confidence,
                interpolated=interpolated,
            )
        )

    def last_point(
        self,
    ) -> Optional[TrajectoryPoint]:

        if not self.points:
            return None

        return self.points[-1]

    def valid_points(self) -> List[TrajectoryPoint]:

        return [
            p
            for p in self.points
            if p.x is not None
            and p.y is not None
            and np.isfinite(p.x)
            and np.isfinite(p.y)
        ]


class TrajectoryManager:
    """
    Quản lý trajectory của tất cả vehicles.
    """

    def __init__(
        self,
        max_history: int = 300,
        max_missed_frames: int = 15,
    ) -> None:

        self.max_history = max_history
        self.max_missed_frames = max_missed_frames

        self.trajectories: Dict[
            int,
            TrackTrajectory
        ] = {}

    def update_track(
        self,
        track_id: int,
        frame_id: int,
        timestamp: float,
        x: float,
        y: float,
        confidence: Optional[float] = None,
    ) -> None:

        if track_id not in self.trajectories:

            self.trajectories[track_id] = (
                TrackTrajectory(
                    track_id=track_id
                )
            )

        trajectory = self.trajectories[track_id]

        trajectory.add_point(
            frame_id=frame_id,
            timestamp=timestamp,
            x=float(x),
            y=float(y),
            confidence=confidence,
            interpolated=False,
        )

        trajectory.missed_frames = 0

        self._limit_history(trajectory)

    def mark_missed(
        self,
        active_track_ids: set[int],
        frame_id: int,
        timestamp: float,
    ) -> None:
        """
        Với những track không xuất hiện ở frame hiện tại,
        thêm một điểm None.

        Các điểm None sau đó có thể được nội suy.
        """

        for track_id, trajectory in self.trajectories.items():

            if track_id in active_track_ids:
                continue

            trajectory.missed_frames += 1

            # Nếu track đã mất quá lâu, không tiếp tục
            # tạo trajectory vô hạn.
            if (
                trajectory.missed_frames
                > self.max_missed_frames
            ):
                continue

            trajectory.add_point(
                frame_id=frame_id,
                timestamp=timestamp,
                x=None,
                y=None,
                confidence=None,
                interpolated=False,
            )

            self._limit_history(trajectory)

    def get(
        self,
        track_id: int,
    ) -> Optional[TrackTrajectory]:

        return self.trajectories.get(track_id)

    def get_points(
        self,
        track_id: int,
    ) -> np.ndarray:

        trajectory = self.get(track_id)

        if trajectory is None:
            return np.empty(
                (0, 2),
                dtype=np.float32
            )

        points = []

        for p in trajectory.points:

            if p.x is None or p.y is None:
                points.append(
                    [np.nan, np.nan]
                )
            else:
                points.append(
                    [p.x, p.y]
                )

        return np.asarray(
            points,
            dtype=np.float32
        )

    def active_ids(self) -> List[int]:

        return list(
            self.trajectories.keys()
        )

    def _limit_history(
        self,
        trajectory: TrackTrajectory,
    ) -> None:

        if (
            len(trajectory.points)
            > self.max_history
        ):
            trajectory.points = (
                trajectory.points[
                    -self.max_history:
                ]
            )

    def remove_dead_tracks(self) -> None:
        """
        Có thể gọi định kỳ để giải phóng memory.
        """

        dead_ids = [
            track_id
            for track_id, trajectory
            in self.trajectories.items()
            if trajectory.missed_frames
            > self.max_missed_frames
        ]

        for track_id in dead_ids:
            del self.trajectories[track_id]