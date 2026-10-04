"""
AcciVision — Module Phát Hiện Phương Tiện (Vehicle Detector)

Module này chứa wrapper cho YOLOv8 để phát hiện các phương tiện
giao thông (ô tô, xe máy, xe buýt, xe tải) trong từng frame video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """
    Một object được YOLO phát hiện.
    """

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
        Lấy điểm giữa cạnh dưới bounding box.

        Đây là điểm đại diện cho vị trí tiếp xúc
        của phương tiện với mặt đường.
        """
        x = (self.x1 + self.x2) / 2.0
        y = self.y2

        return x, y


class YOLODetector:
    """
    Wrapper cho YOLO.

    Dùng cho detection độc lập.
    Tracking sẽ được xử lý trong tracker.py.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:

        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = YOLO(model_path)

    def predict(
        self,
        frame: np.ndarray,
    ) -> List[Detection]:
        """
        Chạy YOLO detection trên một frame.
        """

        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []

        for bbox, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids,
        ):
            class_name = result.names[int(class_id)]

            detections.append(
                Detection(
                    bbox=np.asarray(
                        bbox,
                        dtype=np.float32,
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )

        return detections

    def __call__(
        self,
        frame: np.ndarray,
    ) -> List[Detection]:

        return self.predict(frame)