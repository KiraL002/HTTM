"""
AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh
Module chính: Pipeline xử lý đa tầng & Giao diện dòng lệnh (CLI)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from core import (
    AccidentClassifier,
    ByteTrackTracker,
    TrajectoryManager,
)
from core.event_detector import EventDetector
from core.feature_pipeline import MotionFeaturePipeline, tracker_parameters
from utils import (
    draw_detection,
    draw_fps,
    draw_speed,
    draw_status,
    draw_track_point,
    draw_trajectory,
    draw_world_position,
)
from utils.video_writer import BrowserVideoWriter


# ============================================================
# NẠP CẤU HÌNH (Configuration Loader)
# ============================================================

def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError("Config file is empty.")

    return config


# ============================================================
# XỬ LÝ ĐƯỜNG DẪN (Path Resolution)
# ============================================================

def resolve_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# PIPELINE XỬ LÝ CHÍNH (Main Processing Pipeline)
# ============================================================

class TrafficAccidentPipeline:
    """
    Pipeline phát hiện tai nạn giao thông đầu-cuối (end-to-end).
    """

    ANALYSIS_VERSION = "2026-10-22.2"

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

        # Bộ phát hiện sự kiện tai nạn (Accident Event Detector)
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

        # Cấu hình Video đầu vào/đầu ra
        video_cfg = config["video"]
        self.video_input = resolve_path(self.project_root, video_cfg["input"])
        self.video_output = resolve_path(self.project_root, video_cfg["output"])
        self.features_csv = resolve_path(self.project_root, video_cfg["features_csv"])

        # Mô hình phát hiện (YOLO) & Theo dõi (ByteTrack)
        yolo_cfg = config["yolo"]
        self.yolo_model_path = resolve_path(self.project_root, yolo_cfg["model"])
        self.yolo_confidence = float(yolo_cfg.get("confidence", 0.25))
        self.yolo_iou = float(yolo_cfg.get("iou", 0.50))
        self.yolo_imgsz = yolo_cfg.get("imgsz", 640)
        self.yolo_max_det = yolo_cfg.get("max_det", 300)
        self.yolo_agnostic_nms = yolo_cfg.get("agnostic_nms", False)
        self.yolo_device = yolo_cfg.get("device")
        self.vehicle_classes = yolo_cfg.get("classes")

        tracker_cfg = config["tracker"]
        configured_tracker = tracker_cfg.get("config", "bytetrack.yaml")
        local_tracker = resolve_path(self.project_root, configured_tracker)
        self.tracker_config = str(local_tracker) if local_tracker.is_file() else configured_tracker

        # Quản lý quỹ đạo (Trajectory Manager)
        self.trajectory_manager = TrajectoryManager(
            max_history=int(tracker_cfg.get("max_history", 300)),
            max_missed_frames=int(tracker_cfg.get("max_missed_frames", 15)),
        )
        # Lưu quỹ đạo pixel để vẽ trực tiếp lên video
        self.pixel_trajectories: Dict[int, List[Tuple[float, float]]] = {}

        # The same motion implementation is used by the batch extractor.
        self.motion = MotionFeaturePipeline(config)
        self.feature_extractor = self.motion.extractor
        self.transformer = self.motion.transformer
        self.perspective_enabled = self.transformer is not None
        self.fallback_meters_per_pixel_x = self.motion.scale_x
        self.fallback_meters_per_pixel_y = self.motion.scale_y

        # Mô hình phân loại tai nạn (Accident Classifier)
        classifier_cfg = config["classifier"]
        self.classifier = None
        classifier_enabled = bool(classifier_cfg.get("enabled", True))

        if classifier_enabled:
            classifier_path = resolve_path(
                self.project_root,
                classifier_cfg["model"],
            )
            if classifier_path.exists():
                try:
                    self.classifier = AccidentClassifier(
                        model_path=str(classifier_path),
                        threshold=float(classifier_cfg.get("threshold", 0.70)),
                    )
                except Exception as exc:
                    raise RuntimeError(f"Không thể nạp mô hình phân loại tai nạn: {classifier_path}") from exc
            else:
                raise FileNotFoundError(f"Không tìm thấy mô hình phân loại tai nạn: {classifier_path}")

        # Mô hình xác thực thị giác kép (Visual Verifier - MobileNetV3)
        self.visual_verifier = None
        visual_cfg = config.get("visual_verification", {})
        if visual_cfg.get("enabled", True):
            try:
                from core.visual_classifier import VisualAccidentVerifier
                v_model_path = resolve_path(
                    self.project_root,
                    visual_cfg.get("model", "models/visual_accident_classifier.pth"),
                )
                self.visual_verifier = VisualAccidentVerifier(
                    model_path=str(v_model_path),
                    threshold=float(visual_cfg.get("threshold", 0.40)),
                    device=self.yolo_device,
                )
            except Exception as e:
                print(f"[WARNING] Cannot init visual verifier: {e}")

        self.frame_id = 0
        self.fps = 30.0
        self.global_status = "NORMAL"
        self.global_probability = 0.0

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

    def _initialize_models(self) -> None:
        self.tracker = ByteTrackTracker(**tracker_parameters(self.config, self.project_root))

    def _open_video(self):
        if not self.video_input.exists():
            raise FileNotFoundError(f"Input video not found: {self.video_input}")

        cap = cv2.VideoCapture(str(self.video_input))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_input}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not np.isfinite(fps):
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return cap, float(fps), width, height

    def _create_browser_compatible_writer(
        self,
        width: int,
        height: int,
    ) -> BrowserVideoWriter:
        """Encode H.264/yuv420p MP4 with the source video's actual FPS."""
        ensure_parent_directory(self.video_output)
        self.video_output = self.video_output.with_suffix(".mp4")
        return BrowserVideoWriter(self.video_output, self.fps, (width, height))

    def _open_csv(self):
        ensure_parent_directory(self.features_csv)
        file = self.features_csv.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(file, fieldnames=self.csv_fields)
        writer.writeheader()
        return file, writer

    def _get_world_positions(self, tracks) -> Dict[int, np.ndarray]:
        """Convert ground-contact image points with ONE stable mapping per video.

        Never derive meters-per-pixel from the median size of detections:
        traffic density changes this median even if the camera is stationary.
        Without calibrated perspective, the fallback is only an approximate
        pixel scale and must not be reported as actual physical distance.
        """
        return self.motion.world_positions(tracks)

    @staticmethod
    def _infer_accident_type(feature) -> str:
        if feature.neighbor_count == 0:
            return (
                "single_vehicle_loss_of_control"
                if feature.trajectory_change
                else "single_vehicle_sudden_stop"
            )
        if feature.trajectory_change:
            return "side_collision"
        if feature.deceleration_mps2 >= 3.0:
            return "rear_end_collision"
        return "vehicle_collision"

    def _process_track_features(
        self,
        frame: np.ndarray,
        tracks,
        world_positions,
    ):
        frame_features = []
        max_probability = 0.0
        frame_status = "NORMAL"

        features_by_track = self.motion.extract(tracks, world_positions, self.frame_id)
        self.event_detector.observe_frame(
            self.frame_id,
            {
                track.track_id: {
                    "position": world_positions[track.track_id],
                    "bbox": track.bbox,
                    "speed": features_by_track[track.track_id].speed_mps,
                    "direction": features_by_track[track.track_id].direction_deg,
                    "vehicle_class": track.class_name,
                    "feature_dict": features_by_track[track.track_id].to_dict(),
                }
                for track in tracks
            },
        )

        batch_predictions = {}
        if self.classifier is not None:
            predictions = self.classifier.predict_many_with_probability([
                features_by_track[track.track_id].to_dict() for track in tracks
            ])
            batch_predictions = dict(zip([track.track_id for track in tracks], predictions))

        for track in tracks:
            track_id = track.track_id
            current_position = world_positions[track_id]
            feature = features_by_track[track_id]

            features_dict = feature.to_dict()
            probability = 0.0
            status = "NORMAL"

            if self.classifier is not None:
                try:
                    status, probability = batch_predictions[track_id]
                except Exception as exc:
                    raise RuntimeError(
                        f"Phân loại thất bại tại frame {self.frame_id}, xe #{track_id}."
                    ) from exc

                if (
                    status != "CLASSIFIER_ERROR"
                    and self.visual_verifier is not None
                    and self.visual_verifier.enabled
                    and probability >= 0.50
                ):
                    try:
                        visual_prob = self.visual_verifier.verify_crop(frame, track.bbox)
                    except Exception as exc:
                        # Do not discard valid RF prediction if visual CNN fails.
                        print(f"[WARNING] Visual verifier failed: {exc}")
                        self.visual_verifier.enabled = False
                    else:
                        if probability < 0.85 and visual_prob < 0.25:
                            probability *= 0.45
                        # The older CNN has not been calibrated jointly with
                        # this RF. It may veto a marginal prediction, but must
                        # not promote a sub-threshold RF score into an event.
                        # Always recompute status AFTER fusion; no stale labels.
                        threshold = float(
                            self.config.get("classifier", {}).get("threshold", 0.75)
                        )
                        status = "ACCIDENT" if probability >= threshold else "NORMAL"

                event = self.event_detector.update(
                    track_id=track_id,
                    frame_id=self.frame_id,
                    probability=probability,
                    position=current_position,
                    speed=feature.speed_mps,
                    direction=feature.direction_deg,
                    vehicle_class=track.class_name,
                    feature_dict=features_dict,
                    bbox=track.bbox,
                    frame_shape=frame.shape[:2],
                )

                status = "ACCIDENT" if event is not None else "NORMAL"

                if event is not None:
                    self.accident_detected = True
                    self.accident_start_frame = event.start_frame
                    self.accident_probability = event.probability
                    self.accident_track_ids.update(event.track_ids)

            if probability > max_probability:
                max_probability = probability

            if status in ("ACCIDENT", "POSSIBLE ACCIDENT"):
                frame_status = "ACCIDENT"

            row = {
                "video_id": self.video_input.stem,
                "frame_id": self.frame_id,
                "timestamp": self.frame_id / self.fps,
                "track_id": track_id,
                "class_name": track.class_name,
                "object_type": track.object_type,
                "x_m": feature.x_m,
                "y_m": feature.y_m,
                "speed_mps": feature.speed_mps,
                "acceleration_mps2": feature.acceleration_mps2,
                "deceleration_mps2": feature.deceleration_mps2,
                "speed_delta_mps": feature.speed_delta_mps,
                "direction_deg": feature.direction_deg,
                "direction_change_deg": feature.direction_change_deg,
                "yaw_rate_dps": feature.yaw_rate_dps,
                "lateral_acceleration_mps2": feature.lateral_acceleration_mps2,
                "jerk_mps3": feature.jerk_mps3,
                "nearest_distance_m": feature.nearest_distance_m,
                "neighbor_count": feature.neighbor_count,
                "has_neighbor": feature.has_neighbor,
                "nearest_distance_valid": feature.nearest_distance_valid,
                "closing_speed_mps": feature.closing_speed_mps,
                "interaction_count": feature.interaction_count,
                "sudden_stop": feature.sudden_stop,
                "trajectory_change": feature.trajectory_change,
                "speed_drop_window_mps": feature.speed_drop_window_mps,
                "direction_change_max_deg": feature.direction_change_max_deg,
                "direction_change_sum_deg": feature.direction_change_sum_deg,
                "speed_mean_mps": feature.speed_mean_mps,
                "speed_std_mps": feature.speed_std_mps,
                "acceleration_std_mps2": feature.acceleration_std_mps2,
                "stopped_after_motion": feature.stopped_after_motion,
                "temporal_window": feature.temporal_window,
                "interpolated": 0,
                "accident_probability": probability,
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
        if active_events:
            self.global_status = "ACCIDENT DETECTED"
            self.global_probability = max(
                (e.probability for e in active_events), default=0.0
            )
        else:
            self.global_status = "NORMAL"
            self.global_probability = max_probability

        return frame_features

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks,
        frame_features,
        processing_fps: float,
    ) -> np.ndarray:

        feature_by_id = {item[0].track_id: item for item in frame_features}
        output_cfg = self.config["output"]

        active_events = self.event_detector.get_active_events()
        active_accident_ids = set()
        for ev in active_events:
            active_accident_ids.update(ev.track_ids)

        for track in tracks:
            item = feature_by_id.get(track.track_id)
            if item is None:
                continue

            _track, feature, status, probability, _row = item

            is_vehicle_in_accident = (
                track.track_id in active_accident_ids
                or status in ("ACCIDENT", "POSSIBLE ACCIDENT")
            )

            # Vẽ quỹ đạo chuyển động (Pixel Trajectory)
            if output_cfg.get("draw_trajectory", True):
                pts = self.pixel_trajectories.get(track.track_id, [])
                trajectory_length = int(output_cfg.get("trajectory_length", 40))
                if len(pts) > 1:
                    draw_trajectory(
                        frame,
                        np.asarray(pts[-trajectory_length:], dtype=np.float64),
                    )

            # Vẽ bounding box (Đỏ nếu gặp tai nạn, Xanh nếu bình thường)
            draw_detection(
                frame,
                bbox=track.bbox,
                label=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
                is_accident=is_vehicle_in_accident,
            )

            # Vẽ điểm đáy (bottom center)
            draw_track_point(
                frame,
                track.bottom_center,
            )

            # Hiển thị tốc độ
            if output_cfg.get("draw_speed", True):
                draw_speed(
                    frame,
                    position=(int(track.x1), int(track.y2 + 20)),
                    speed_mps=feature.speed_mps,
                )

            # Hiển thị tọa độ thực tế (mét)
            if (
                output_cfg.get("draw_world_position", True)
                and self.transformer is not None
            ):
                draw_world_position(
                    frame,
                    position=(int(track.x1), int(track.y2 + 40)),
                    world_x=feature.x_m,
                    world_y=feature.y_m,
                )

        # Hiển thị trạng thái phát hiện + Thời gian video
        draw_status(
            frame,
            self.global_status,
            self.global_probability,
            timestamp_s=self.frame_id / self.fps,
            frame_id=self.frame_id,
        )

        if output_cfg.get("draw_fps", True):
            draw_fps(frame, processing_fps)

        return frame

    def run(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:

        if self.classifier is None:
            raise RuntimeError("Phân tích tai nạn cần mô hình phân loại đang hoạt động.")
        self._initialize_models()
        cap, self.fps, width, height = self._open_video()

        self.feature_extractor.fps = self.fps
        self.feature_extractor.dt = 1.0 / self.fps
        self.event_detector.set_fps(self.fps)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_cfg = self.config["output"]
        video_cfg = self.config["video"]

        writer = None
        csv_file = None
        csv_writer = None

        processing_times = []
        csv_interval = max(1, int(output_cfg.get("csv_interval_frames", 1)))
        display_interval = max(1, int(video_cfg.get("display_every_n_frames", 1)))

        try:
            if output_cfg.get("save_video", True) and video_cfg.get("save_output", True):
                writer = self._create_browser_compatible_writer(width, height)
            if output_cfg.get("save_csv", False):
                csv_file, csv_writer = self._open_csv()
            while True:
                start_time = time.perf_counter()
                success, frame = cap.read()
                if not success:
                    break

                self.frame_id += 1

                # Bước 1: Phát hiện & Theo dõi (YOLO + ByteTrack)
                tracks = self.tracker.update(frame)
                tracks = [t for t in tracks if t.is_vehicle]
                active_ids = {track.track_id for track in tracks}

                # Bước 2: Tính tọa độ thực tế (World Coordinates)
                world_positions = self._get_world_positions(tracks)

                # Bước 3: Cập nhật quỹ đạo (Trajectory Update)
                timestamp = self.frame_id / self.fps
                for track in tracks:
                    x, y = world_positions[track.track_id]
                    self.trajectory_manager.update_track(
                        track_id=track.track_id,
                        frame_id=self.frame_id,
                        timestamp=timestamp,
                        x=float(x),
                        y=float(y),
                        confidence=track.confidence,
                    )
                    self.pixel_trajectories.setdefault(track.track_id, []).append(
                        track.bottom_center
                    )

                # Bước 4: Đánh dấu track mất tích
                self.trajectory_manager.mark_missed(
                    active_track_ids=active_ids,
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                )

                # Bước 5: Trích xuất đặc trưng động học & Phân loại
                frame_features = self._process_track_features(
                    frame,
                    tracks,
                    world_positions,
                )
                self.event_detector.end_frame(self.frame_id, set(active_ids))
                active_events = self.event_detector.get_active_events()
                self.global_status = "ACCIDENT DETECTED" if active_events else "NORMAL"
                self.global_probability = max(
                    (event.probability for event in active_events), default=0.0
                )

                # Bước 6: Ghi dữ liệu CSV
                if csv_writer is not None and self.frame_id % csv_interval == 0:
                    for item in frame_features:
                        csv_writer.writerow(item[4])

                # Bước 7: Tính FPS xử lý
                elapsed = time.perf_counter() - start_time
                processing_fps = 1.0 / elapsed if elapsed > 0 else 0.0

                # Bước 8: Vẽ kết quả lên frame
                frame = self._draw_frame(
                    frame=frame,
                    tracks=tracks,
                    frame_features=frame_features,
                    processing_fps=processing_fps,
                )

                # Bước 9: Ghi video kết quả
                if writer is not None:
                    writer.write(frame)

                # Bước 10: Hiển thị cửa sổ OpenCV
                if video_cfg.get("display", True) and self.frame_id % display_interval == 0:
                    cv2.imshow(
                        "AcciVision - Phat Hien Tai Nan Giao Thong",
                        frame,
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                # Đo FPS xử lý thực tế (gồm vẽ và ghi video), tránh báo cao giả.
                processing_times.append(time.perf_counter() - start_time)

                # Bước 11: Cập nhật tiến trình (Progress Callback)
                if (
                    progress_callback is not None
                    and total_frames > 0
                    and (self.frame_id % 5 == 0 or self.frame_id == total_frames)
                ):
                    progress = min(self.frame_id / total_frames, 1.0)
                    progress_callback(progress)

            if self.frame_id == 0:
                raise RuntimeError("Không đọc được khung hình nào từ video đầu vào.")

        finally:
            cap.release()
            try:
                if writer is not None:
                    writer.release()
            finally:
                if csv_file is not None:
                    csv_file.close()
                cv2.destroyAllWindows()

        self.event_detector.finish(self.frame_id)

        average_fps = 0.0
        if processing_times:
            average_time = sum(processing_times) / len(processing_times)
            if average_time > 0:
                average_fps = 1.0 / average_time

        confirmed_events = self.event_detector.all_events()
        has_accident = len(confirmed_events) > 0
        overall_status = "ACCIDENT DETECTED" if has_accident else "NORMAL"
        overall_verdict = "CÓ TAI NẠN" if has_accident else "KHÔNG CÓ TAI NẠN"
        overall_probability = max(
            (e.probability for e in confirmed_events), default=0.0
        )

        return {
            "analysis_version": self.ANALYSIS_VERSION,
            "detector_settings": {
                "model": str(self.yolo_model_path), "confidence": self.yolo_confidence,
                "iou": self.yolo_iou, "imgsz": self.yolo_imgsz,
                "max_det": self.yolo_max_det, "agnostic_nms": self.yolo_agnostic_nms,
                "tracker_config": self.tracker_config,
            },
            "video_input": str(self.video_input),
            "video_output": str(self.video_output),
            "features_csv": str(self.features_csv),
            "frames_processed": self.frame_id,
            "average_fps": average_fps,
            "has_accident": has_accident,
            "verdict": overall_verdict,
            "status": overall_status,
            "accident_probability": overall_probability,
            "events_count": len(confirmed_events),
            "tracking_observations": [
                {
                    **observation,
                    "first_time_s": round(observation["first_frame"] / self.fps, 2),
                    "last_time_s": round(observation["last_frame"] / self.fps, 2),
                }
                for observation in self.event_detector.tracking_data()
            ],
            "events": [
                {
                    "event_id": e.event_id,
                    "accident_type": e.accident_type,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "start_time_s": round(e.start_time_s, 2),
                    "end_time_s": round(e.end_time_s, 2),
                    "duration_s": round(e.duration_s, 2),
                    "track_ids": sorted(list(e.track_ids)),
                    "vehicle_types": e.vehicle_types,
                    "vehicle_count": e.estimated_vehicle_count,
                    "vehicle_count_is_estimate": True,
                    "impact_track_ids": sorted(e.impact_track_ids),
                    "track_links": list(e.track_links),
                    "vehicles": [
                        {
                            "vehicle_id": participant["canonical_track_id"],
                            "impact_track_id": participant["impact_track_id"],
                            "track_ids": participant["track_ids"],
                            "candidate_track_ids": participant["candidate_track_ids"],
                            "identity_status": participant["identity_status"],
                            "class_name": participant["vehicle_type"],
                        }
                        for participant in e.participant_data()
                    ],
                    "probability": round(e.probability, 4),
                    "active": e.active,
                    "description": e.description,
                }
                for e in confirmed_events
            ],
        }


def main():
    parser = argparse.ArgumentParser(
        description="AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh"
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
        help="Đường dẫn video đầu vào",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Đường dẫn video đầu ra",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Tắt cửa sổ hiển thị OpenCV",
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(args.config)

    if args.input:
        config["video"]["input"] = args.input
    if args.output:
        config["video"]["output"] = args.output
    if args.no_display:
        config["video"]["display"] = False

    if config.get("demo", {}).get("model"):
        from utils.pair_demo import run_analysis
        source = resolve_path(project_root, config["video"]["input"])
        destination = resolve_path(project_root, config["video"]["output"])
        ensure_parent_directory(destination)
        result = run_analysis(source, destination)
        print(f"Kết quả phân tích: {result['label']} | sự kiện: {len(result['events'])}")
        for event in result["events"]:
            print(f"Mốc nghi sự cố: ~{event['time_s']:.2f}s | ID xe: {event['track_ids']}")
        print(f"Video đầu ra: {destination}")
        return

    pipeline = TrafficAccidentPipeline(
        config=config,
        project_root=project_root,
    )

    result = pipeline.run()
    print(f"Video đầu ra: {result['video_output']}")


if __name__ == "__main__":
    main()
