from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class AccidentEvent:
    track_id: int

    start_frame: int
    last_frame: int

    probability: float

    active: bool = True


class EventDetector:
    """
    Phát hiện accident event theo time window.

    Không chỉ nhìn một frame.

    Logic:

        probability >= start_threshold
            trong nhiều frame liên tiếp
            -> bắt đầu accident event

        Sau khi event bắt đầu:
            không tiếp tục cập nhật bằng
            các vehicle bình thường khác.

    """

    def __init__(
        self,
        sequence_length: int = 30,
        start_threshold: float = 0.70,
        confirmation_frames: int = 5,
        release_threshold: float = 0.30,
    ) -> None:

        self.sequence_length = sequence_length
        self.start_threshold = start_threshold
        self.confirmation_frames = confirmation_frames
        self.release_threshold = release_threshold

        self.history: Dict[
            int,
            Deque[float]
        ] = {}

        self.events: Dict[
            int,
            AccidentEvent
        ] = {}

    def update(
        self,
        track_id: int,
        frame_id: int,
        probability: float,
    ) -> Optional[AccidentEvent]:

        if track_id not in self.history:

            self.history[track_id] = deque(
                maxlen=self.sequence_length
            )

        self.history[track_id].append(
            float(probability)
        )

        history = self.history[track_id]

        # --------------------------------------------------
        # Nếu event đã xảy ra
        # --------------------------------------------------

        if track_id in self.events:

            event = self.events[track_id]

            event.last_frame = frame_id
            event.probability = max(
                event.probability,
                probability
            )

            return event

        # --------------------------------------------------
        # Chưa đủ dữ liệu
        # --------------------------------------------------

        if len(history) < self.confirmation_frames:
            return None

        recent = list(history)[
            -self.confirmation_frames:
        ]

        # Cần nhiều frame liên tiếp vượt threshold.
        if all(
            p >= self.start_threshold
            for p in recent
        ):

            event = AccidentEvent(
                track_id=track_id,
                start_frame=(
                    frame_id
                    - self.confirmation_frames
                    + 1
                ),
                last_frame=frame_id,
                probability=max(recent),
            )

            self.events[track_id] = event

            return event

        return None

    def has_event(
        self,
        track_id: int,
    ) -> bool:

        return track_id in self.events

    def get_event(
        self,
        track_id: int,
    ) -> Optional[AccidentEvent]:

        return self.events.get(track_id)

    def all_events(self):

        return list(
            self.events.values()
        )