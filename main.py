"""
AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh
Module chính: Pipeline xử lý đa tầng & Giao diện dòng lệnh (CLI)

Pipeline xử lý:
    OpenCV (đọc video)
      ↓
    YOLOv8 (phát hiện phương tiện) + ByteTrack (theo dõi đa đối tượng)
      ↓
    Quản lý quỹ đạo (Trajectory Management)
      ↓
    Chuyển đổi phối cảnh (Perspective Transform → Bird's Eye View)
      ↓
    Trích xuất đặc trưng động học (Kinematic Feature Extraction)
      ↓
    Phân loại tai nạn (Random Forest Classifier)
      ↓
    Phát hiện & quản lý sự kiện (Event Detection & Lifecycle)
      ↓
    Video kết quả + CSV đặc trưng
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import yaml

from core import (
    AccidentClassifier,
    ByteTrackTracker,
    FeatureExtractor,
    PerspectiveTransformer,
    TrajectoryManager,
)
from core.event_detector import EventDetector
from utils import (
    draw_detection,
    draw_fps,
    draw_speed,
    draw_status,
    draw_track_point,
    draw_trajectory,
    draw_world_position,
)


# ============================================================
# NẠP CẤU HÌNH (Configuration Loader)
# ============================================================

def load_config(config_path: str) -> dict:
    """
    Nạp tệp cấu hình YAML.

    Parameters
    ----------
    config_path : str
        Đường dẫn đến tệp config.yaml

    Returns
    -------
    dict
        Dictionary chứa cấu hình hệ thống
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            "Config file is empty."
        )

    return config


# ============================================================
# XỬ LÝ ĐƯỜNG DẪN (Path Resolution)
# ============================================================

def resolve_path(
    project_root: Path,
    configured_path: str,
) -> Path:
    """
    Giải quyết đường dẫn tương đối từ thư mục gốc dự án.
    """

    path = Path(configured_path)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_parent_directory(
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PIPELINE XỬ LÝ CHÍNH (Main Processing Pipeline)
# ============================================================

class TrafficAccidentPipeline:
    """
    Pipeline phát hiện tai nạn giao thông đầu-cuối (end-to-end).

    Quy trình xử lý:
        [1] Đọc video (OpenCV)
        [2] Phát hiện phương tiện (YOLOv8) + Theo dõi (ByteTrack)
        [3] Quản lý quỹ đạo chuyển động (Trajectory)
        [4] Chuyển đổi phối cảnh (Perspective Transform → BEV)
        [5] Trích xuất đặc trưng động học (Feature Extraction)
        [6] Phân loại tai nạn (Random Forest Classifier)
        [7] Phát hiện & quản lý sự kiện (Event Detection)
        [8] Xuất video kết quả + CSV đặc trưng
    """

    def __init__(
        self,
        config: dict,
        project_root: Optional[Path] = None,
    ) -> None:

        self.config = config

        self.project_root = (
            project_root
            if project_root is not None
            else Path(__file__).resolve().parent
        )
        # ----------------------------------------------------
        # Bộ phát hiện sự kiện tai nạn (Accident Event Detector)
        # ----------------------------------------------------

        classifier_thresh = float(
            config.get("classifier", {}).get("threshold", 0.75)
        )
        confirmation_frames = int(
            config.get("classifier", {}).get("confirmation_frames", 8)
        )

        self.event_detector = EventDetector(
            fps=30.0,
            start_threshold=classifier_thresh,
            confirmation_frames=confirmation_frames,
            release_threshold=0.35,
            release_frames=12,
            spatial_merge_distance=6.0,
            min_track_age=int(config.get("classifier", {}).get("min_track_age", 2)),
        )

        self.accident_detected = False
        self.accident_start_frame = None
        self.accident_probability = 0.0
        self.accident_track_ids = set()
        # ----------------------------------------------------
        # Cấu hình Video đầu vào/đầu ra
        # ----------------------------------------------------

        video_cfg = config["video"]

        self.video_input = resolve_path(
            self.project_root,
            video_cfg["input"],
        )

        self.video_output = resolve_path(
            self.project_root,
            video_cfg["output"],
        )

        self.features_csv = resolve_path(
            self.project_root,
            video_cfg["features_csv"],
        )

        # ----------------------------------------------------
        # Mô hình phát hiện (YOLO) & Theo dõi (ByteTrack)
        # ----------------------------------------------------

        yolo_cfg = config["yolo"]

        self.yolo_model_path = resolve_path(
            self.project_root,
            yolo_cfg["model"],
        )

        self.yolo_confidence = float(
            yolo_cfg.get(
                "confidence",
                0.25,
            )
        )

        self.yolo_iou = float(
            yolo_cfg.get(
                "iou",
                0.50,
            )
        )

        self.yolo_device = yolo_cfg.get(
            "device"
        )

        self.vehicle_classes = yolo_cfg.get(
            "classes"
        )

        tracker_cfg = config["tracker"]

        self.tracker_config = tracker_cfg.get(
            "config",
            "bytetrack.yaml",
        )

        # ----------------------------------------------------
        # Quản lý quỹ đạo (Trajectory Manager)
        # ----------------------------------------------------

        self.trajectory_manager = (
            TrajectoryManager(
                max_history=int(
                    tracker_cfg.get(
                        "max_history",
                        300,
                    )
                ),
                max_missed_frames=int(
                    tracker_cfg.get(
                        "max_missed_frames",
                        15,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # Chuyển đổi phối cảnh (Perspective Transform)
        # ----------------------------------------------------

        perspective_cfg = config["perspective"]

        self.perspective_enabled = bool(
            perspective_cfg.get(
                "enabled",
                True,
            )
        )

        self.transformer = None

        if self.perspective_enabled:

            source_points = np.asarray(
                perspective_cfg[
                    "source_points"
                ],
                dtype=np.float32,
            )

            destination_points = np.asarray(
                perspective_cfg[
                    "destination_points"
                ],
                dtype=np.float32,
            )

            self.transformer = (
                PerspectiveTransformer(
                    source_points=source_points,
                    destination_points=destination_points,
                    meters_per_pixel_x=float(
                        perspective_cfg[
                            "meters_per_pixel_x"
                        ]
                    ),
                    meters_per_pixel_y=float(
                        perspective_cfg[
                            "meters_per_pixel_y"
                        ]
                    ),
                )
            )

        # ----------------------------------------------------
        # Trích xuất đặc trưng động học (Feature Extractor)
        # ----------------------------------------------------

        feature_cfg = config["features"]

        self.feature_extractor = (
            FeatureExtractor(
                fps=30.0,
                sudden_stop_deceleration=float(
                    feature_cfg.get(
                        "sudden_stop_deceleration",
                        4.5,
                    )
                ),
                direction_change_threshold=float(
                    feature_cfg.get(
                        "direction_change_threshold",
                        45.0,
                    )
                ),
                temporal_window=int(feature_cfg.get("temporal_window", 8)),
                velocity_smoothing_alpha=float(
                    feature_cfg.get("velocity_smoothing_alpha", 0.30)
                ),
            )
        )

        # ----------------------------------------------------
        # Mô hình phân loại tai nạn (Accident Classifier)
        # ----------------------------------------------------

        classifier_cfg = config[
            "classifier"
        ]

        self.classifier = None

        classifier_enabled = bool(
            classifier_cfg.get(
                "enabled",
                True,
            )
        )

        if classifier_enabled:

            classifier_path = resolve_path(
                self.project_root,
                classifier_cfg["model"],
            )

            if classifier_path.exists():

                try:

                    self.classifier = (
                        AccidentClassifier(
                            model_path=str(
                                classifier_path
                            ),
                            threshold=float(
                                classifier_cfg.get(
                                    "threshold",
                                    0.70,
                                )
                            ),
                        )
                    )

                except Exception as exc:

                    print(
                        "[WARNING] "
                        "Cannot load classifier: "
                        f"{exc}"
                    )

                    self.classifier = None

            else:

                print(
                    "[WARNING] "
                    f"Classifier not found: "
                    f"{classifier_path}"
                )

        # ----------------------------------------------------
        # Trạng thái trước đó (Previous States)
        # Dùng để tính toán: vận tốc, gia tốc, hướng di chuyển
        # ----------------------------------------------------

        self.previous_positions: Dict[
            int,
            np.ndarray
        ] = {}

        self.previous_velocities: Dict[
            int,
            np.ndarray
        ] = {}

        self.previous_directions: Dict[
            int,
            float
        ] = {}

        self.previous_speeds: Dict[
            int,
            float
        ] = {}

        # Đếm frame hiện tại
        self.frame_id = 0
        self.fps = 30.0

        # Trạng thái phát hiện tai nạn tổng thể
        self.global_status = "BÌNH THƯỜNG"
        self.global_probability = 0.0

        # Các trường dữ liệu xuất CSV
        self.csv_fields = [
            "video_id",
            "frame_id",
            "timestamp",
            "track_id",
            "class_name",
            "object_type",
            "x_m",
            "y_m",
            "speed_mps",
            "acceleration_mps2",
            "deceleration_mps2",
            "speed_delta_mps",
            "direction_deg",
            "direction_change_deg",
            "yaw_rate_dps",
            "lateral_acceleration_mps2",
            "jerk_mps3",
            "nearest_distance_m",
            "neighbor_count",
            "has_neighbor",
            "nearest_distance_valid",
            "closing_speed_mps",
            "interaction_count",
            "sudden_stop",
            "trajectory_change",
            "speed_drop_window_mps",
            "direction_change_max_deg",
            "direction_change_sum_deg",
            "speed_mean_mps",
            "speed_std_mps",
            "acceleration_std_mps2",
            "stopped_after_motion",
            "temporal_window",
            "interpolated",
            "accident_probability",
            "status",
        ]

    # ========================================================
    # KHỞI TẠO MÔ HÌNH (Model Initialization)
    # ========================================================

    def _initialize_models(self) -> None:

        self.tracker = ByteTrackTracker(
            model_path=str(
                self.yolo_model_path
            ),
            confidence=self.yolo_confidence,
            iou_threshold=self.yolo_iou,
            tracker_config=self.tracker_config,
            device=self.yolo_device,
            classes=self.vehicle_classes,
            vehicle_only=True,
        )

    # ========================================================
    # ĐỌC VIDEO ĐẦU VÀO (Video Input)
    # ========================================================

    def _open_video(self):

        if not self.video_input.exists():
            raise FileNotFoundError(
                f"Input video not found: "
                f"{self.video_input}"
            )

        cap = cv2.VideoCapture(
            str(self.video_input)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: "
                f"{self.video_input}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0 or not np.isfinite(fps):
            fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        return (
            cap,
            float(fps),
            width,
            height,
        )

    # ========================================================
    # XUẤT CSV ĐẶC TRƯNG (Feature CSV Output)
    # ========================================================

    def _open_csv(self):

        ensure_parent_directory(
            self.features_csv
        )

        file = self.features_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            file,
            fieldnames=self.csv_fields,
        )

        writer.writeheader()

        return file, writer

    # ========================================================
    # TỌA ĐỘ THỰC TẾ (World Positions)
    # ========================================================

    def _get_world_positions(
        self,
        tracks,
    ) -> Dict[int, np.ndarray]:

        positions = {}

        for track in tracks:

            pixel_x, pixel_y = (
                track.bottom_center
            )

            if self.transformer is not None:

                world_x, world_y = (
                    self.transformer.image_to_world(
                        pixel_x,
                        pixel_y,
                    )
                )

            else:

                # Fallback chỉ để debug.
                # Đây là PIXEL chứ không phải meter.
                world_x = pixel_x
                world_y = pixel_y

            positions[
                track.track_id
            ] = np.asarray(
                [world_x, world_y],
                dtype=np.float64,
            )

        return positions

    # ========================================================
    # TRÍCH XUẤT ĐẶC TRƯNG & PHÂN LOẠI (Feature + Classification)
    # ========================================================

    @staticmethod
    def _infer_accident_type(feature) -> str:
        """Phân loại sơ bộ loại tai nạn dựa trên đặc trưng động học."""
        if feature.neighbor_count == 0:
            return "single_vehicle_loss_of_control" if feature.trajectory_change else "single_vehicle_sudden_stop"
        if feature.trajectory_change:
            return "side_collision"
        if feature.deceleration_mps2 >= 3.0:
            return "rear_end_collision"
        return "vehicle_collision"

    def _process_track_features(
        self,
        tracks,
        world_positions,
    ):

        frame_features = []

        max_probability = 0.0
        frame_status = "NORMAL"

        for track in tracks:

            track_id = track.track_id

            current_position = (
                world_positions[track_id]
            )

            previous_position = (
                self.previous_positions.get(
                    track_id
                )
            )

            previous_velocity = (
                self.previous_velocities.get(
                    track_id
                )
            )

            previous_direction = (
                self.previous_directions.get(
                    track_id
                )
            )

            previous_speed = (
                self.previous_speeds.get(
                    track_id,
                    0.0
                )
            )

            feature = (
                self.feature_extractor.extract(
                    track_id=track_id,
                    current_position=current_position,
                    previous_position=previous_position,
                    previous_velocity=previous_velocity,
                    previous_direction=previous_direction,
                    previous_speed=previous_speed,
                    all_vehicle_positions=world_positions,
                    all_vehicle_velocities=self.previous_velocities,
                    object_type=track.object_type,
                )
            )

            features_dict = feature.to_dict()

            probability = 0.0
            status = "NORMAL"

            # ------------------------------------------------
            # Phân loại tai nạn (Classifier)
            # ------------------------------------------------

            if self.classifier is not None:

                try:

                    status, probability = (
                        self.classifier
                        .predict_with_probability(
                            features_dict
                        )
                    )

                except Exception as exc:

                    # Rất quan trọng:
                    # Không làm chết cả pipeline nếu
                    # RF model cũ không tương thích.

                    print(
                        "\n[WARNING] "
                        "Classifier error for "
                        f"track {track_id}: "
                        f"{exc}"
                    )

                    status = (
                        "CLASSIFIER_ERROR"
                    )
                # ------------------------------------------------
                # Phát hiện sự kiện tai nạn (Event Detection)
                # ------------------------------------------------

                event = self.event_detector.update(
                    track_id=track_id,
                    frame_id=self.frame_id,
                    probability=probability,
                    position=current_position,
                    speed=feature.speed_mps,
                    direction=feature.direction_deg,
                    vehicle_class=track.class_name,
                    feature_dict=features_dict,
                )

                if event is not None:

                    self.accident_detected = True

                    self.accident_start_frame = (
                        event.start_frame
                    )

                    self.accident_probability = (
                        event.probability
                    )

                    self.accident_track_ids.update(event.track_ids)
            # ------------------------------------------------
            # Trạng thái tổng thể (Global Status)
            # ------------------------------------------------

            if probability > max_probability:

                max_probability = (
                    probability
                )

            if status in ("ACCIDENT", "POSSIBLE ACCIDENT"):
                frame_status = "ACCIDENT"

            # ------------------------------------------------
            # Cập nhật trạng thái trước đó
            # ------------------------------------------------

            if previous_position is not None:
                velocity = self.feature_extractor.get_smoothed_velocity(track_id)
            else:
                velocity = np.zeros(
                    2,
                    dtype=np.float64,
                )

            self.previous_positions[
                track_id
            ] = current_position.copy()

            self.previous_velocities[
                track_id
            ] = velocity.copy()

            self.previous_directions[
                track_id
            ] = feature.direction_deg

            self.previous_speeds[
                track_id
            ] = feature.speed_mps

            # ------------------------------------------------
            # Dòng dữ liệu CSV
            # ------------------------------------------------

            row = {
                "video_id": self.video_input.stem,
                "frame_id": self.frame_id,
                "timestamp": (
                    self.frame_id
                    / self.fps
                ),
                "track_id": track_id,
                "class_name": track.class_name,
                "object_type": track.object_type,
                "x_m": feature.x_m,
                "y_m": feature.y_m,
                "speed_mps": (
                    feature.speed_mps
                ),
                "acceleration_mps2": (
                    feature.acceleration_mps2
                ),
                "deceleration_mps2": feature.deceleration_mps2,
                "speed_delta_mps": feature.speed_delta_mps,
                "direction_deg": (
                    feature.direction_deg
                ),
                "direction_change_deg": (
                    feature.direction_change_deg
                ),
                "yaw_rate_dps": feature.yaw_rate_dps,
                "lateral_acceleration_mps2": feature.lateral_acceleration_mps2,
                "jerk_mps3": feature.jerk_mps3,
                "nearest_distance_m": (
                    feature.nearest_distance_m
                ),
                "neighbor_count": feature.neighbor_count,
                "has_neighbor": feature.has_neighbor,
                "nearest_distance_valid": feature.nearest_distance_valid,
                "closing_speed_mps": feature.closing_speed_mps,
                "interaction_count": feature.interaction_count,
                "sudden_stop": (
                    feature.sudden_stop
                ),
                "trajectory_change": (
                    feature.trajectory_change
                ),
                "speed_drop_window_mps": feature.speed_drop_window_mps,
                "direction_change_max_deg": feature.direction_change_max_deg,
                "direction_change_sum_deg": feature.direction_change_sum_deg,
                "speed_mean_mps": feature.speed_mean_mps,
                "speed_std_mps": feature.speed_std_mps,
                "acceleration_std_mps2": feature.acceleration_std_mps2,
                "stopped_after_motion": feature.stopped_after_motion,
                "temporal_window": feature.temporal_window,
                "interpolated": 0,
                "accident_probability": (
                    probability
                ),
                "status": status,
            }

            frame_features.append(
                (
                    track,
                    feature,
                    status,
                    probability,
                    row,
                )
            )

        active_events = self.event_detector.get_active_events()
        all_events = self.event_detector.all_events()
        if active_events or all_events:
            self.global_status = "ACCIDENT DETECTED"
            self.global_probability = max((e.probability for e in (active_events or all_events)), default=0.0)
        else:
            self.global_status = "NORMAL"
            self.global_probability = max_probability

        return frame_features

    # ========================================================
    # VẼ KẾT QUẢ LÊN FRAME (Drawing / Visualization)
    # ========================================================

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks,
        frame_features,
        processing_fps: float,
    ) -> np.ndarray:

        feature_by_id = {
            item[0].track_id: item
            for item in frame_features
        }

        output_cfg = self.config[
            "output"
        ]

        for track in tracks:

            item = feature_by_id.get(
                track.track_id
            )

            # Safety
            if item is None:
                continue

            (
                _track,
                feature,
                status,
                probability,
                _row,
            ) = item

            # ------------------------------------------------
            # Vẽ bounding box
            # ------------------------------------------------

            draw_detection(
                frame,
                bbox=track.bbox,
                label=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
            )

            # ------------------------------------------------
            # Vẽ điểm đáy (bottom center)
            # ------------------------------------------------

            draw_track_point(
                frame,
                track.bottom_center,
            )

            # ------------------------------------------------
            # Hiển thị tốc độ
            # ------------------------------------------------

            if output_cfg.get(
                "draw_speed",
                True,
            ):

                draw_speed(
                    frame,
                    position=(
                        int(track.x1),
                        int(track.y2 + 20),
                    ),
                    speed_mps=feature.speed_mps,
                )

            # ------------------------------------------------
            # Hiển thị tọa độ thực tế (mét)
            # ------------------------------------------------

            if (
                output_cfg.get(
                    "draw_world_position",
                    True,
                )
                and self.transformer is not None
            ):

                draw_world_position(
                    frame,
                    position=(
                        int(track.x1),
                        int(track.y2 + 40),
                    ),
                    world_x=feature.x_m,
                    world_y=feature.y_m,
                )

            # ------------------------------------------------
            # Vẽ quỹ đạo chuyển động
            # ------------------------------------------------

            if output_cfg.get(
                "draw_trajectory",
                True,
            ):

                trajectory = (
                    self.trajectory_manager
                    .get_points(
                        track.track_id
                    )
                )

                trajectory_length = (
                    int(
                        output_cfg.get(
                            "trajectory_length",
                            50,
                        )
                    )
                )

                if (
                    len(trajectory)
                    > trajectory_length
                ):

                    trajectory = (
                        trajectory[
                            -trajectory_length:
                        ]
                    )

                # Chỉ vẽ nếu đang dùng pixel.
                #
                # World coordinates không thể vẽ trực tiếp
                # lên ảnh camera.
                #
                # Do đó trajectory visualization ở đây
                # lấy lại pixel bottom-center từ bbox.
                #
                # Ta xây trajectory từ bbox points ở video
                # trong bản MVP này.

        # ----------------------------------------------------
        # Hiển thị trạng thái phát hiện
        # ----------------------------------------------------

        draw_status(
            frame,
            self.global_status,
            self.global_probability,
        )

        if output_cfg.get(
            "draw_fps",
            True,
        ):

            draw_fps(
                frame,
                processing_fps,
            )

        return frame

    # ========================================================
    # CHẠY PIPELINE (Run Pipeline)
    # ========================================================

    def run(
        self,
        progress_callback: Optional[
            Callable[[float], None]
        ] = None,
    ) -> dict:

        self._initialize_models()

        (
            cap,
            self.fps,
            width,
            height,
        ) = self._open_video()

        # Cập nhật FPS thực tế cho FeatureExtractor & EventDetector
        self.feature_extractor.fps = (
            self.fps
        )

        self.feature_extractor.dt = (
            1.0 / self.fps
        )

        self.event_detector.fps = (
            self.fps
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        output_cfg = self.config[
            "output"
        ]

        video_cfg = self.config[
            "video"
        ]

        writer = None

        if (
            output_cfg.get(
                "save_video",
                True,
            )
            and video_cfg.get(
                "save_output",
                True,
            )
        ):

            ensure_parent_directory(
                self.video_output
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                str(self.video_output),
                fourcc,
                self.fps,
                (width, height),
            )

            if not writer.isOpened():
                writer.release()

                raise RuntimeError(
                    "Cannot create output video."
                )

        csv_file = None
        csv_writer = None

        if output_cfg.get(
            "save_csv",
            True,
        ):

            csv_file, csv_writer = (
                self._open_csv()
            )

        processing_times = []

        try:

            while True:

                start_time = (
                    time.perf_counter()
                )

                success, frame = cap.read()

                if not success:
                    break

                self.frame_id += 1

                # ------------------------------------------------
                # Bước 1: Phát hiện & Theo dõi (YOLO + ByteTrack)
                # ------------------------------------------------

                tracks = self.tracker.update(
                    frame
                )

                # Lọc chỉ giữ phương tiện giao thông (vehicle filtering)
                tracks = [t for t in tracks if t.is_vehicle]

                active_ids = {
                    track.track_id
                    for track in tracks
                }

                # ------------------------------------------------
                # Bước 2: Tính tọa độ thực tế (World Coordinates)
                # ------------------------------------------------

                world_positions = (
                    self._get_world_positions(
                        tracks
                    )
                )

                # ------------------------------------------------
                # Bước 3: Cập nhật quỹ đạo (Trajectory Update)
                # ------------------------------------------------

                timestamp = (
                    self.frame_id
                    / self.fps
                )

                for track in tracks:

                    x, y = (
                        world_positions[
                            track.track_id
                        ]
                    )

                    self.trajectory_manager.update_track(
                        track_id=track.track_id,
                        frame_id=self.frame_id,
                        timestamp=timestamp,
                        x=float(x),
                        y=float(y),
                        confidence=track.confidence,
                    )

                # ------------------------------------------------
                # Bước 4: Đánh dấu track mất tích
                # ------------------------------------------------

                self.trajectory_manager.mark_missed(
                    active_track_ids=active_ids,
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                )

                # ------------------------------------------------
                # Bước 5: Trích xuất đặc trưng động học
                # ------------------------------------------------

                frame_features = (
                    self._process_track_features(
                        tracks,
                        world_positions,
                    )
                )
                self.event_detector.end_frame(self.frame_id, set(active_ids))

                # ------------------------------------------------
                # Bước 6: Ghi dữ liệu CSV
                # ------------------------------------------------

                if csv_writer is not None:

                    for item in frame_features:

                        row = item[4]

                        csv_writer.writerow(row)

                # ------------------------------------------------
                # Bước 7: Tính FPS xử lý
                # ------------------------------------------------

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                processing_fps = (
                    1.0 / elapsed
                    if elapsed > 0
                    else 0.0
                )

                processing_times.append(
                    elapsed
                )

                # ------------------------------------------------
                # Bước 8: Vẽ kết quả lên frame
                # ------------------------------------------------

                frame = self._draw_frame(
                    frame=frame,
                    tracks=tracks,
                    frame_features=frame_features,
                    processing_fps=processing_fps,
                )

                # ------------------------------------------------
                # Bước 9: Ghi video kết quả
                # ------------------------------------------------

                if writer is not None:
                    writer.write(frame)

                # ------------------------------------------------
                # Bước 10: Hiển thị cửa sổ OpenCV
                # ------------------------------------------------

                if video_cfg.get(
                    "display",
                    True,
                ):

                    cv2.imshow(
                        "AcciVision - Phat Hien Tai Nan Giao Thong",
                        frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                # ------------------------------------------------
                # Bước 11: Cập nhật tiến trình (Progress Callback)
                # ------------------------------------------------

                if (
                    progress_callback
                    is not None
                    and total_frames > 0
                    and (self.frame_id % 5 == 0 or self.frame_id == total_frames)
                ):

                    progress = min(
                        self.frame_id
                        / total_frames,
                        1.0,
                    )

                    progress_callback(
                        progress
                    )

        finally:

            cap.release()

            if writer is not None:
                writer.release()

            if csv_file is not None:
                csv_file.close()

            cv2.destroyAllWindows()

        self.event_detector.finish(self.frame_id)

        average_fps = 0.0

        if processing_times:

            average_time = (
                sum(processing_times)
                / len(processing_times)
            )

            if average_time > 0:
                average_fps = (
                    1.0 / average_time
                )

        confirmed_events = self.event_detector.all_events()
        has_accident = len(confirmed_events) > 0
        overall_status = "ACCIDENT DETECTED" if has_accident else "NORMAL"
        overall_verdict = "CÓ TAI NẠN" if has_accident else "KHÔNG CÓ TAI NẠN"
        overall_probability = max((e.probability for e in confirmed_events), default=0.0)

        return {
            "video_input": str(
                self.video_input
            ),
            "video_output": str(
                self.video_output
            ),
            "features_csv": str(
                self.features_csv
            ),
            "frames_processed": (
                self.frame_id
            ),
            "average_fps": average_fps,
            # Kết luận video có tai nạn hay không (Nhị phân)
            "has_accident": has_accident,
            "verdict": overall_verdict,
            "status": overall_status,
            "accident_probability": overall_probability,
            "events_count": len(confirmed_events),
            "events": [
                {
                    "event_id": e.event_id,
                    "accident_type": e.accident_type,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "start_time_s": round(e.start_time_s, 2),
                    "end_time_s": round(e.end_time_s, 2),
                    "duration_s": round(e.duration_s, 2),
                    # Mã theo dõi ByteTrack (không phải biển số xe)
                    "track_ids": sorted(list(e.track_ids)),
                    "vehicle_types": e.vehicle_types,
                    "probability": round(e.probability, 4),
                    "active": e.active,
                    "description": e.description,
                }
                for e in confirmed_events
            ],
        }


# ============================================================
# GIAO DIỆN DÒNG LỆNH (Command Line Interface)
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AcciVision — Hệ Thống Phát Hiện "
            "Tai Nạn Giao Thông Thông Minh"
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Đường dẫn đến tệp cấu hình config.yaml",
    )

    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Đường dẫn video đầu vào (ghi đè video.input trong config.yaml)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Đường dẫn video đầu ra (ghi đè video.output trong config.yaml)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Tắt cửa sổ hiển thị OpenCV (chế độ server)",
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        args.config
    )

    if args.input:
        config["video"]["input"] = args.input

    if args.output:
        config["video"]["output"] = args.output

    if args.no_display:
        config["video"]["display"] = False

    pipeline = (
        TrafficAccidentPipeline(
            config=config,
            project_root=project_root,
        )
    )

    print("=" * 60)
    print(
        "ACCIVISION — PHÁT HIỆN TAI NẠN GIAO THÔNG THÔNG MINH"
    )
    print("=" * 60)

    print(
        f"Video đầu vào : {pipeline.video_input}"
    )

    print(
        f"Mô hình YOLO  : {pipeline.yolo_model_path}"
    )

    print("=" * 60)

    result = pipeline.run()

    print()
    print("=" * 60)
    print("KẾT QUẢ PHÂN TÍCH VIDEO")
    print("=" * 60)
    verdict_str = "CÓ TAI NẠN (ACCIDENT DETECTED)" if result["has_accident"] else "KHÔNG CÓ TAI NẠN (NORMAL)"
    print(f"📌 KẾT LUẬN VIDEO   : {verdict_str}")
    print(f"🚨 Số vụ tai nạn    : {result['events_count']}")
    print(f"🎞️ Tổng số frame    : {result['frames_processed']}")
    print(f"⚡ Tốc độ xử lý     : {result['average_fps']:.1f} FPS")
    print(f"📁 Video đầu ra     : {result['video_output']}")
    print(f"📊 CSV đặc trưng    : {result['features_csv']}")
    if result["has_accident"]:
        print("-" * 60)
        print("CHI TIẾT CÁC VỤ TAI NẠN:")
        for idx, ev in enumerate(result.get("events", []), 1):
            print(
                f"  [{idx}] Loại: {ev['accident_type']} | "
                f"Thời gian: {ev['start_time_s']}s -> {ev['end_time_s']}s "
                f"(Thời lượng: {ev['duration_s']}s) | Xe ID: {list(ev['vehicle_types'].keys())}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
