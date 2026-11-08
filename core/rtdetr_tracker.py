"""RT-DETR native full-frame detections with explicit ByteTrack association.

No event labels, source identities, event regions or timestamps enter this adapter.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from ultralytics import RTDETR
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from .tracker import Track, OBJECT_TYPE_MAP


def filter_detections(rows, shape, confidence, classes, iou=.5, max_det=300):
    a = np.asarray(rows, dtype=np.float32).reshape(-1, 6).copy()
    if not np.isfinite(a).all():
        raise ValueError("Non-finite detector output")
    height, width = shape[:2]
    a[:, [0, 2]] = np.clip(a[:, [0, 2]], 0, width)
    a[:, [1, 3]] = np.clip(a[:, [1, 3]], 0, height)
    a = a[(a[:, 2] > a[:, 0]) & (a[:, 3] > a[:, 1]) &
          (a[:, 4] >= confidence) & np.isin(a[:, 5], classes)]
    order = np.argsort(-a[:, 4], kind="stable")
    kept = []
    areas = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    while len(order) and len(kept) < max_det:
        index, rest = order[0], order[1:]
        kept.append(index)
        overlap = np.maximum(0, np.minimum(a[index, 2:4], a[rest, 2:4]) -
                             np.maximum(a[index, :2], a[rest, :2])).prod(axis=1)
        ratio = overlap / np.maximum(1e-6, areas[index] + areas[rest] - overlap)
        order = rest[ratio <= iou]
    return a[kept]


class RTDETRTracker:
    def __init__(self, model_path, confidence=.4, iou_threshold=.5,
                 tracker_config="configs/bytetrack_detail.yaml", device="cpu",
                 classes=None, imgsz=640, max_det=300, **unused):
        self.model = RTDETR(str(model_path))
        self.classes = classes if classes is not None else [2, 5, 7]
        if any(self.model.names[i] != name for i, name in ((2, "car"), (5, "bus"), (7, "truck"))):
            raise ValueError("Expected pretrained COCO vehicle classes")
        self.confidence, self.iou = confidence, iou_threshold
        self.imgsz, self.max_det, self.device = imgsz, max_det, device
        self.tracker_settings = yaml.safe_load(Path(tracker_config).read_text(encoding="utf-8"))
        self.reset()

    def reset(self):
        self.tracker = BYTETracker(SimpleNamespace(**self.tracker_settings))

    def update(self, frame):
        result = self.model.predict(frame, conf=self.confidence, imgsz=self.imgsz,
                                    classes=self.classes, device=self.device,
                                    max_det=self.max_det, verbose=False)[0]
        rows = filter_detections(result.boxes.data.cpu().numpy(), frame.shape,
                                 self.confidence, self.classes, self.iou, self.max_det)
        # Empty detections still advance the tracker by one native source frame.
        observed = self.tracker.update(Boxes(rows, frame.shape[:2]))
        return [Track(track_id=int(r[4]), bbox=np.asarray(r[:4], np.float32),
                      confidence=float(r[5]), class_id=int(r[6]),
                      class_name=self.model.names[int(r[6])],
                      object_type=OBJECT_TYPE_MAP.get(self.model.names[int(r[6])], "other"))
                for r in observed]

    __call__ = update
