"""
AcciVision — Các module xử lý lõi (Core Processing Modules)

Bao gồm:
- AccidentClassifier: Phân loại tai nạn (Random Forest tối ưu Cython)
- VisualAccidentVerifier: Xác thực thị giác kép (MobileNetV3)
- EventDetector: Phát hiện & quản lý vòng đời sự kiện tai nạn
- FeatureExtractor: Trích xuất đặc trưng động học & biến dạng hình học khung bao
- ByteTrackTracker: Phát hiện (YOLO11) & Theo dõi đa đối tượng (ByteTrack)
- TrajectoryManager: Quản lý quỹ đạo chuyển động
- PerspectiveTransformer: Chuyển đổi phối cảnh BEV & Tự động hiệu chỉnh tỷ lệ (Auto-Scale)
"""

from .classifier import AccidentClassifier

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

from .visual_classifier import (
    VisualAccidentVerifier,
)


__all__ = [
    "AccidentClassifier",
    "VisualAccidentVerifier",
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
