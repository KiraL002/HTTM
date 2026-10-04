"""
AcciVision — Các module xử lý lõi (Core Processing Modules)

Bao gồm:
- AccidentClassifier: Phân loại tai nạn (Random Forest)
- YOLODetector: Phát hiện phương tiện (YOLOv8)
- EventDetector: Phát hiện & quản lý sự kiện tai nạn
- FeatureExtractor: Trích xuất đặc trưng động học
- ByteTrackTracker: Theo dõi đa đối tượng
- TrajectoryManager: Quản lý quỹ đạo chuyển động
- PerspectiveTransformer: Chuyển đổi phối cảnh
"""

from .classifier import AccidentClassifier

from .detector import (
    Detection,
    YOLODetector,
)

from .event_detector import (
    AccidentEvent,
    EventDetector,
)

from .features import (
    FeatureExtractor,
    FeatureVector,
)

from .tracker import (
    ByteTrackTracker,
    Track,
)

from .trajectory import (
    TrackTrajectory,
    TrajectoryManager,
    TrajectoryPoint,
)

from .transformer import (
    PerspectiveTransformer,
)


__all__ = [
    "AccidentClassifier",

    "Detection",
    "YOLODetector",

    "AccidentEvent",
    "EventDetector",

    "FeatureExtractor",
    "FeatureVector",

    "ByteTrackTracker",
    "Track",

    "TrackTrajectory",
    "TrajectoryManager",
    "TrajectoryPoint",

    "PerspectiveTransformer",
]