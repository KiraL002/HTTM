"""
AcciVision — Module Theo Dõi Đa Đối Tượng (Multi-Object Tracker)

Module này kết hợp YOLOv8 và ByteTrack để theo dõi liên tục
các phương tiện giao thông qua nhiều frame video, gán mã định danh
phiên theo dõi (Track ID) cho từng phương tiện.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
from ultralytics import YOLO


# Bảng chuẩn hóa loại đối tượng từ COCO classes
OBJECT_TYPE_MAP: Dict[str, str] = {
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",  # Ho tro cac xe bi loa den ban dem nhan nham thanh train
    "bicycle": "two_wheeler",
    "person": "pedestrian",
}

DEFAULT_VEHICLE_CLASSES: Set[str] = {"car", "motorcycle", "bus", "truck", "train"}


@dataclass
class Track:
    """
    Một đối tượng (object) đang được tracking.

    LƯU Ý QUAN TRỌNG:
    - ``track_id`` là ID tracking tạm thời do ByteTrack quản lý theo từng video/session,
      KHÔNG PHẢI là định danh vĩnh viễn (persistent vehicle ID / biển số xe) trong thế giới thực.
    - ``object_type`` là phân loại chuẩn hóa ngữ nghĩa ('vehicle', 'two_wheeler', 'pedestrian', 'other').
    """

    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    object_type: str = "vehicle"

    @property
    def is_vehicle(self) -> bool:
        """Kiểm tra đối tượng có phải là phương tiện giao thông (ô tô, xe máy, xe buýt, xe tải)."""
        return self.object_type == "vehicle"

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
        Điểm giữa cạnh dưới bounding box (thường tiếp xúc mặt đường).
        """
        x = (self.x1 + self.x2) / 2.0
        y = self.y2
        return x, y

    @property
    def center(self) -> tuple[float, float]:
        """Tâm bounding box."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class ByteTrackTracker:
    """
    YOLO + ByteTrack.

    Ultralytics quản lý:
        - Detection
        - Data association
        - Track ID
        - Track state

    Chuẩn hóa:
        - object_type rõ ràng
        - Lọc phương tiện giao thông (vehicle filtering)
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        vehicle_class_names: Optional[List[str]] = None,
        vehicle_only: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.tracker_config = tracker_config
        self.device = device
        self.classes = classes
        self.vehicle_only = bool(vehicle_only)

        if vehicle_class_names:
            self.vehicle_class_names = {x.lower().strip() for x in vehicle_class_names}
        else:
            self.vehicle_class_names = set(DEFAULT_VEHICLE_CLASSES)

        self.model = YOLO(model_path)

    def update(
        self,
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Nhận một frame và trả về danh sách tracks đã được chuẩn hóa object_type
        và lọc theo vehicle nếu vehicle_only=True.
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

        if result.boxes is None or not result.boxes.is_track:
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
            raw_class_name = str(result.names[int(class_id)])
            lower_name = raw_class_name.lower().strip()

            # Xác định object_type chuẩn hóa
            object_type = OBJECT_TYPE_MAP.get(lower_name, "other")
            if lower_name in self.vehicle_class_names:
                object_type = "vehicle"

            # Vehicle filtering: nếu chỉ theo dõi phương tiện giao thông
            if self.vehicle_only and object_type != "vehicle":
                continue

            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=np.asarray(bbox, dtype=np.float32),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=raw_class_name,
                    object_type=object_type,
                )
            )

        return tracks

    def reset(self) -> None:
        """
        Reset tracker state khi chuyển video mới.
        """
        self.model = YOLO(self.model_path)

    def __call__(
        self,
        frame: np.ndarray,
    ) -> List[Track]:
        return self.update(frame)
