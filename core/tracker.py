from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Track:
    """
    Một object đang được tracking.
    """

    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def bottom_center(self) -> tuple[float, float]:
        """
        Điểm giữa cạnh dưới bounding box.
        """
        x = (self.x1 + self.x2) / 2.0
        y = self.y2

        return x, y


class ByteTrackTracker:
    """
    YOLO + ByteTrack.

    Ultralytics quản lý:
        - Detection
        - Data association
        - Track ID
        - Track state

    persist=True:
        giữ state giữa các frame liên tiếp.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.15,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> None:

        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config
        self.device = device
        self.classes = classes

        self.model = YOLO(model_path)

    def update(
        self,
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Nhận một frame và trả về danh sách tracks.
        """

        kwargs = {
            "source": frame,
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        if self.classes is not None:
            kwargs["classes"] = self.classes

        results = self.model.track(**kwargs)

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        # Chưa tracking được object nào
        if not result.boxes.is_track:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        tracks: List[Track] = []

        for bbox, confidence, class_id, track_id in zip(
            boxes,
            confidences,
            class_ids,
            track_ids,
        ):

            class_name = result.names[int(class_id)]

            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=np.asarray(
                        bbox,
                        dtype=np.float32,
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )

        return tracks

    def reset(self) -> None:
        """
        Reset tracker state bằng cách tạo lại model.

        Hữu ích khi chuyển sang video mới.
        """
        self.model = YOLO(self.model_path)

    def __call__(
        self,
        frame: np.ndarray,
    ) -> List[Track]:

        return self.update(frame)