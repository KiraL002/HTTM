"""Extract unlabeled vehicle features with the runtime pipeline (explicit CLI only)."""
from __future__ import annotations

import argparse
import csv
import math
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2

from core.feature_pipeline import (
    FEATURE_PIPELINE_VERSION, RF_FEATURE_COLUMNS, MotionFeaturePipeline,
    feature_signature, file_sha256, tracker_parameters,
)
from core.tracker import ByteTrackTracker
from main import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FIELDS = [
    "extraction_id", "feature_signature", "feature_schema_version", "video_id",
    "source_path", "source_sha256", "source_group", "camera_group", "split_group",
    "detection_profile", "coordinate_mode", "source_fps", "width", "height",
    "frame_id", "source_time_s", "runtime_time_s", "track_id", "class_name",
    "detection_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "x_m", "y_m",
    *RF_FEATURE_COLUMNS,
]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def process_video(video_path: Path, tracker, config: dict, writer, identity: dict,
                  sample_every: int = 1, max_frames: int | None = None) -> dict:
    if type(sample_every) is not int or sample_every <= 0:
        raise ValueError("sample_every must be a positive integer")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.  # Same fallback as TrafficAccidentPipeline._open_video.
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        motion = MotionFeaturePipeline(config, fps=fps)
        rows, frame_id = 0, 0
        while max_frames is None or frame_id < max_frames:
            success, frame = capture.read()
            if not success:
                break
            frame_id += 1
            tracks = [t for t in tracker.update(frame) if t.is_vehicle]
            features = motion.extract(tracks, motion.world_positions(tracks), frame_id)
            # Process every frame; sampling only reduces exported rows.
            if frame_id % sample_every:
                continue
            for track in tracks:
                feature = features[track.track_id]
                values = {name: getattr(feature, name) for name in RF_FEATURE_COLUMNS}
                if not all(math.isfinite(float(value)) for value in values.values()):
                    raise ValueError(f"Non-finite features: {video_path}, frame {frame_id}, track {track.track_id}")
                writer.writerow({
                    **identity, "feature_schema_version": FEATURE_PIPELINE_VERSION,
                    "coordinate_mode": "configured_homography" if motion.transformer is not None else "approximate_scale",
                    "source_fps": fps, "width": width, "height": height,
                    "frame_id": frame_id, "source_time_s": (frame_id - 1) / fps,
                    "runtime_time_s": frame_id / fps, "track_id": track.track_id,
                    "class_name": track.class_name, "detection_confidence": track.confidence,
                    **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), map(float, track.bbox))),
                    "x_m": feature.x_m, "y_m": feature.y_m, **values,
                })
                rows += 1
        if frame_id == 0:
            raise RuntimeError(f"No decodable frames: {video_path}")
        return {"frames": frame_id, "rows": rows, "fps": fps}
    finally:
        capture.release()


def extract_dataset(video_dir: Path, output: Path, config: dict, *, project_root=PROJECT_ROOT,
                    limit_videos=None, max_frames=None, sample_every=None, videos=None,
                    camera_group="", overwrite=False) -> dict:
    video_dir, output = Path(video_dir).resolve(), Path(output).resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if output.suffix.lower() != ".csv":
        raise ValueError("Output must be an explicitly selected .csv file")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
    if videos:
        files = [(video_dir / name).resolve() for name in videos]
        if any(not p.is_relative_to(video_dir) for p in files):
            raise ValueError("Selected videos must be inside video_dir")
    else:
        files = sorted(p for p in video_dir.rglob("*") if p.is_file()
                       and p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"))
    if limit_videos is not None:
        if type(limit_videos) is not int or limit_videos <= 0:
            raise ValueError("limit_videos must be a positive integer")
        files = files[:limit_videos]
    if not files or any(not p.is_file() for p in files):
        raise ValueError("No selected source videos are available")
    interval = sample_every if sample_every is not None else int(config.get("output", {}).get("csv_interval_frames", 1))
    if type(interval) is not int or interval <= 0:
        raise ValueError("sample_every must be a positive integer")
    signature = feature_signature(config, Path(project_root))
    extraction_id = uuid4().hex
    if config["perspective"].get("enabled", True):
        print("[NOTE] Config homography must match these cameras; one intersection calibration does not calibrate unrelated Kaggle cameras.")
    else:
        print("[NOTE] Approximate scaling: m/mps fields are not calibrated measurements. Use the same config at inference.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    total = {"videos": 0, "frames": 0, "rows": 0, "duplicates_skipped": 0,
             "extraction_id": extraction_id, "feature_signature": signature}
    hashes = set()
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=output.parent,
                                         prefix=".features-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=FEATURE_FIELDS)
            writer.writeheader()
            for path in files:
                sha = file_sha256(path)
                if sha in hashes:
                    total["duplicates_skipped"] += 1
                    continue
                hashes.add(sha)
                source_group = "sha256:" + sha
                relative = path.relative_to(video_dir)
                identity = {"extraction_id": extraction_id, "feature_signature": signature,
                            "video_id": relative.with_suffix("").as_posix(), "source_path": relative.as_posix(),
                            "source_sha256": sha, "source_group": source_group, "camera_group": camera_group,
                            "split_group": camera_group or source_group,
                            "detection_profile": "fixed"}
                # A fresh tracker and state prevent cross-video history/identities.
                tracker = ByteTrackTracker(**tracker_parameters(config, Path(project_root)))
                result = process_video(path, tracker, config, writer, identity, interval, max_frames)
                total["videos"] += 1
                total["frames"] += result["frames"]
                total["rows"] += result["rows"]
                print(f"{relative}: {result['frames']} frames, {result['rows']} rows")
        if not total["rows"]:
            raise ValueError("No vehicle feature rows were produced; no CSV was saved")
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        temporary.replace(output)
        return total
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Only this CSV is retained; no implicit output path")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--videos", nargs="+", help="Specific paths relative to video-dir; otherwise find videos recursively")
    parser.add_argument("--limit-videos", type=positive_int)
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--sample-every", type=positive_int)
    parser.add_argument("--camera-group", default="", help="Only group videos verified to share a camera")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    print(extract_dataset(args.video_dir, args.output, config, videos=args.videos,
                          limit_videos=args.limit_videos, max_frames=args.max_frames,
                          sample_every=args.sample_every, camera_group=args.camera_group,
                          overwrite=args.overwrite))


if __name__ == "__main__":
    main()
