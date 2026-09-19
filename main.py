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
# CONFIG
# ============================================================

def load_config(config_path: str) -> dict:
    """
    Load YAML configuration.
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
# PATH
# ============================================================

def resolve_path(
    project_root: Path,
    configured_path: str,
) -> Path:
    """
    Resolve relative path from project root.
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
# MAIN PIPELINE
# ============================================================

class TrafficAccidentPipeline:
    """
    End-to-end traffic accident detection pipeline.

    Pipeline:

        OpenCV
          ↓
        YOLO + ByteTrack
          ↓
        Trajectory
          ↓
        Perspective Transform
          ↓
        Feature Extraction
          ↓
        Random Forest
          ↓
        Output Video + CSV
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
        # Accident Event Detector
        # ----------------------------------------------------

        self.event_detector = EventDetector(
            sequence_length=30,
            start_threshold=0.70,
            confirmation_frames=5,
            release_threshold=0.30,
        )

        self.accident_detected = False
        self.accident_start_frame = None
        self.accident_probability = 0.0
        self.accident_track_ids = set()
        # ----------------------------------------------------
        # Video
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
        # YOLO / ByteTrack
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
        # Tracker / trajectory
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
        # Perspective
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
        # Feature extractor
        # ----------------------------------------------------

        feature_cfg = config["features"]

        self.feature_extractor = (
            FeatureExtractor(
                fps=30.0,
                sudden_stop_deceleration=float(
                    feature_cfg.get(
                        "sudden_stop_deceleration",
                        3.0,
                    )
                ),
                direction_change_threshold=float(
                    feature_cfg.get(
                        "direction_change_threshold",
                        45.0,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # Classifier
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
        # Previous states
        #
        # Used for calculating:
        # velocity
        # acceleration
        # direction
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

        # Current timestamp
        self.frame_id = 0
        self.fps = 30.0

        # Global accident status
        self.global_status = "NORMAL"
        self.global_probability = 0.0

        # Output CSV fields
        self.csv_fields = [
            "video_id",
            "frame_id",
            "timestamp",
            "track_id",
            "class_name",
            "x_m",
            "y_m",
            "speed_mps",
            "acceleration_mps2",
            "direction_deg",
            "direction_change_deg",
            "nearest_distance_m",
            "sudden_stop",
            "trajectory_change",
            "interpolated",
            "accident_probability",
            "status",
        ]

    # ========================================================
    # INITIALIZATION
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
        )

    # ========================================================
    # VIDEO
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
    # OUTPUT CSV
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
    # CURRENT WORLD POSITIONS
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
    # FEATURE + CLASSIFICATION
    # ========================================================

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
                    all_current_positions=world_positions,
                )
            )

            features_dict = feature.to_dict()

            probability = 0.0
            status = "NORMAL"

            # ------------------------------------------------
            # Classifier
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
                # Accident Event Detection
                # ------------------------------------------------

                event = self.event_detector.update(
                    track_id=track_id,
                    frame_id=self.frame_id,
                    probability=probability,
                )

                if event is not None:

                    self.accident_detected = True

                    self.accident_start_frame = (
                        event.start_frame
                    )

                    self.accident_probability = (
                        event.probability
                    )

                    self.accident_track_ids.add(
                        event.track_id
                    )
            # ------------------------------------------------
            # Global status
            # ------------------------------------------------

            if probability > max_probability:

                max_probability = (
                    probability
                )

            if (
                status == "POSSIBLE ACCIDENT"
            ):
                frame_status = (
                    "POSSIBLE ACCIDENT"
                )

            # ------------------------------------------------
            # Update previous state
            # ------------------------------------------------

            if previous_position is not None:

                velocity = (
                    self.feature_extractor
                    .calculate_velocity(
                        previous_position,
                        current_position,
                    )
                )

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
            # CSV row
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
                "x_m": feature.x_m,
                "y_m": feature.y_m,
                "speed_mps": (
                    feature.speed_mps
                ),
                "acceleration_mps2": (
                    feature.acceleration_mps2
                ),
                "direction_deg": (
                    feature.direction_deg
                ),
                "direction_change_deg": (
                    feature.direction_change_deg
                ),
                "nearest_distance_m": (
                    feature.nearest_distance_m
                ),
                "sudden_stop": (
                    feature.sudden_stop
                ),
                "trajectory_change": (
                    feature.trajectory_change
                ),
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

        self.global_status = frame_status
        self.global_probability = (
            max_probability
        )

        return frame_features

    # ========================================================
    # DRAW
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
            # BBOX
            # ------------------------------------------------

            draw_detection(
                frame,
                bbox=track.bbox,
                label=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
            )

            # ------------------------------------------------
            # Bottom center
            # ------------------------------------------------

            draw_track_point(
                frame,
                track.bottom_center,
            )

            # ------------------------------------------------
            # Speed
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
            # World position
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
            # Trajectory
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
        # Status
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
    # RUN
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

        # FeatureExtractor cần FPS thực tế.
        self.feature_extractor.fps = (
            self.fps
        )

        self.feature_extractor.dt = (
            1.0 / self.fps
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
                # 1. YOLO + ByteTrack
                # ------------------------------------------------

                tracks = self.tracker.update(
                    frame
                )

                active_ids = {
                    track.track_id
                    for track in tracks
                }

                # ------------------------------------------------
                # 2. World coordinates
                # ------------------------------------------------

                world_positions = (
                    self._get_world_positions(
                        tracks
                    )
                )

                # ------------------------------------------------
                # 3. Trajectory
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
                # 4. Mark missing tracks
                # ------------------------------------------------

                self.trajectory_manager.mark_missed(
                    active_track_ids=active_ids,
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                )

                # ------------------------------------------------
                # 5. Feature extraction
                # ------------------------------------------------

                frame_features = (
                    self._process_track_features(
                        tracks,
                        world_positions,
                    )
                )

                # ------------------------------------------------
                # 6. CSV
                # ------------------------------------------------

                if csv_writer is not None:

                    for item in frame_features:

                        row = item[4]

                        csv_writer.writerow(row)

                # ------------------------------------------------
                # 7. Processing FPS
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
                # 8. Draw
                # ------------------------------------------------

                frame = self._draw_frame(
                    frame=frame,
                    tracks=tracks,
                    frame_features=frame_features,
                    processing_fps=processing_fps,
                )

                # ------------------------------------------------
                # 9. Write video
                # ------------------------------------------------

                if writer is not None:
                    writer.write(frame)

                # ------------------------------------------------
                # 10. Display
                # ------------------------------------------------

                if video_cfg.get(
                    "display",
                    True,
                ):

                    cv2.imshow(
                        "HTTM - Traffic Accident Detection",
                        frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                # ------------------------------------------------
                # 11. Progress callback
                # ------------------------------------------------

                if (
                    progress_callback
                    is not None
                    and total_frames > 0
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
            "status": self.global_status,
            "accident_probability": (
                self.global_probability
            ),
        }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "HTTM Traffic Accident "
            "Detection Pipeline"
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        args.config
    )

    pipeline = (
        TrafficAccidentPipeline(
            config=config,
            project_root=project_root,
        )
    )

    print("=" * 60)
    print(
        "HTTM TRAFFIC ACCIDENT DETECTION"
    )
    print("=" * 60)

    print(
        f"Input : {pipeline.video_input}"
    )

    print(
        f"Model : {pipeline.yolo_model_path}"
    )

    print("=" * 60)

    result = pipeline.run()

    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()