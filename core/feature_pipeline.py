"""Shared detection settings and motion state for inference and data extraction."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .features import FeatureExtractor
from .transformer import PerspectiveTransformer


FEATURE_PIPELINE_VERSION = "1"
RF_FEATURE_COLUMNS = (
    "speed_mps", "acceleration_mps2", "deceleration_mps2", "speed_delta_mps",
    "direction_deg", "direction_change_deg", "yaw_rate_dps", "lateral_acceleration_mps2",
    "jerk_mps3", "nearest_distance_m", "neighbor_count", "has_neighbor",
    "nearest_distance_valid", "closing_speed_mps", "interaction_count", "sudden_stop",
    "trajectory_change", "speed_drop_window_mps", "direction_change_max_deg",
    "direction_change_sum_deg", "speed_mean_mps", "speed_std_mps",
    "acceleration_std_mps2", "stopped_after_motion", "temporal_window",
)


def tracker_parameters(config: dict, project_root: Path) -> dict:
    yolo = config["yolo"]
    model = Path(yolo["model"])
    if not model.is_absolute():
        model = project_root / model
    tracker = config["tracker"].get("config", "bytetrack.yaml")
    local = project_root / tracker
    return {
        "model_path": str(model), "confidence": float(yolo.get("confidence", .25)),
        "iou_threshold": float(yolo.get("iou", .50)),
        "tracker_config": str(local) if local.is_file() else tracker,
        "device": yolo.get("device"), "classes": yolo.get("classes"),
        "vehicle_only": True, "imgsz": yolo.get("imgsz", 640),
        "max_det": yolo.get("max_det", 300), "agnostic_nms": yolo.get("agnostic_nms", False),
    }


class MotionFeaturePipeline:
    """One state per video. No labels, predictions, video writes or RF/CNN loads."""

    def __init__(self, config: dict, fps: float = 30.) -> None:
        perspective, features = config["perspective"], config["features"]
        self.scale_x = float(perspective.get("meters_per_pixel_x", .05))
        self.scale_y = float(perspective.get("meters_per_pixel_y", .05))
        self.transformer = None
        if perspective.get("enabled", True):
            self.transformer = PerspectiveTransformer(
                source_points=np.asarray(perspective["source_points"], dtype=np.float32),
                destination_points=np.asarray(perspective["destination_points"], dtype=np.float32),
                meters_per_pixel_x=self.scale_x, meters_per_pixel_y=self.scale_y,
            )
        self.extractor = FeatureExtractor(
            fps=fps,
            sudden_stop_deceleration=float(features.get("sudden_stop_deceleration", 4.5)),
            direction_change_threshold=float(features.get("direction_change_threshold", 45.)),
            temporal_window=int(features.get("temporal_window", 8)),
            velocity_smoothing_alpha=float(features.get("velocity_smoothing_alpha", .30)),
        )
        self.previous_positions = {}
        self.previous_velocities = {}
        self.previous_directions = {}
        self.previous_speeds = {}

    def world_positions(self, tracks) -> dict:
        positions = {}
        for track in tracks:
            px, py = track.bottom_center
            x, y = (self.transformer.image_to_world(px, py) if self.transformer is not None
                    else (px * self.scale_x, py * self.scale_y))
            positions[track.track_id] = np.asarray([x, y], dtype=np.float64)
        return positions

    def extract(self, tracks, positions: dict, frame_id: int) -> dict:
        inputs = [
            {"track_id": t.track_id, "current_position": positions[t.track_id],
             "previous_position": self.previous_positions.get(t.track_id),
             "previous_velocity": self.previous_velocities.get(t.track_id),
             "previous_direction": self.previous_directions.get(t.track_id),
             "previous_speed": self.previous_speeds.get(t.track_id, 0.),
             "object_type": t.object_type, "bbox": t.bbox}
            for t in tracks
        ]
        result = self.extractor.extract_frame(inputs, frame_id=frame_id)
        # Update only after every vehicle's interactions use the same snapshot.
        for t in tracks:
            tid, feature = t.track_id, result[t.track_id]
            velocity = (self.extractor.get_smoothed_velocity(tid)
                        if tid in self.previous_positions else np.zeros(2, dtype=np.float64))
            self.previous_positions[tid] = positions[tid].copy()
            self.previous_velocities[tid] = velocity.copy()
            self.previous_directions[tid] = feature.direction_deg
            self.previous_speeds[tid] = feature.speed_mps
        return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_signature(config: dict, project_root: Path) -> str:
    """Portable identity for features/track IDs: config, weights, code and runtime."""
    settings = tracker_parameters(config, project_root)
    settings["model_path"] = file_sha256(Path(settings["model_path"]))
    tracker = Path(settings["tracker_config"])
    if not tracker.is_file():
        import ultralytics
        tracker = Path(ultralytics.__file__).parent / "cfg" / "trackers" / tracker
    settings["tracker_config"] = file_sha256(tracker)
    core = Path(__file__).resolve().parent
    import cv2
    payload = {
        "opencv": cv2.__version__,
        "version": FEATURE_PIPELINE_VERSION, "columns": RF_FEATURE_COLUMNS,
        "tracker": settings, "features": config["features"], "perspective": config["perspective"],
        "code": {name: file_sha256(core / name) for name in
                 ("feature_pipeline.py", "features.py", "tracker.py", "transformer.py")},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("ultralytics", "numpy", "torch")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
