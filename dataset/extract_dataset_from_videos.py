"""
Công cụ trích xuất đặc trưng hàng loạt từ thư mục video (Batch Video Feature Extractor).

Dùng để tự động xử lý nhiều video giao thông / tai nạn thực tế
và tạo ra file CSV đặc trưng phục vụ huấn luyện mô hình.

Cách dùng:
    python dataset/extract_dataset_from_videos.py --video-dir dataset/videos --label-accident --output dataset/extracted_features.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.tracker import ByteTrackTracker
from core.features import FeatureExtractor
from core.transformer import PerspectiveTransformer


def process_video(
    video_path: Path,
    tracker: ByteTrackTracker,
    feature_extractor: FeatureExtractor,
    transformer: Optional[PerspectiveTransformer],
    is_accident_video: bool,
    csv_writer: csv.DictWriter,
    video_id: str,
    max_frames: Optional[int] = None,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0

    feature_extractor.fps = fps
    feature_extractor.dt = 1.0 / fps

    frame_id = 0
    previous_positions: Dict[int, np.ndarray] = {}
    previous_velocities: Dict[int, np.ndarray] = {}
    previous_directions: Dict[int, float] = {}
    previous_speeds: Dict[int, float] = {}

    written_rows = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if max_frames and frame_id > max_frames:
                break

            tracks = tracker.update(frame)
            # Lọc chỉ các phương tiện giao thông
            tracks = [t for t in tracks if t.is_vehicle]

            # Tọa độ thế giới
            world_positions: Dict[int, np.ndarray] = {}
            for t in tracks:
                px, py = t.bottom_center
                if transformer is not None:
                    wx, wy = transformer.image_to_world(px, py)
                else:
                    wx, wy = px * 0.05, py * 0.05
                world_positions[t.track_id] = np.array([wx, wy], dtype=np.float32)

            for t in tracks:
                tid = t.track_id
                curr_pos = world_positions[tid]
                prev_pos = previous_positions.get(tid)
                prev_vel = previous_velocities.get(tid)
                prev_dir = previous_directions.get(tid)
                prev_spd = previous_speeds.get(tid, 0.0)

                feat = feature_extractor.extract(
                    track_id=tid,
                    current_position=curr_pos,
                    previous_position=prev_pos,
                    previous_velocity=prev_vel,
                    previous_direction=prev_dir,
                    previous_speed=prev_spd,
                    all_vehicle_positions=world_positions,
                    all_vehicle_velocities=previous_velocities,
                    object_type=t.object_type,
                )

                if prev_pos is not None:
                    vel = feature_extractor.calculate_velocity(prev_pos, curr_pos, track_id=tid)
                else:
                    vel = np.zeros(2, dtype=np.float32)

                previous_positions[tid] = curr_pos.copy()
                previous_velocities[tid] = vel.copy()
                previous_directions[tid] = feat.direction_deg
                previous_speeds[tid] = feat.speed_mps

                # Xác định nhãn
                # Với video tai nạn, các frame có phanh gấp hoặc đổi hướng mạnh sẽ là ACCIDENT
                if is_accident_video:
                    is_crash = (feat.deceleration_mps2 >= 3.5 or feat.direction_change_deg >= 35.0 or feat.speed_drop_window_mps >= 5.0)
                    status = "POSSIBLE ACCIDENT" if is_crash else "NORMAL"
                    prob = 0.85 if is_crash else 0.05
                else:
                    status = "NORMAL"
                    prob = 0.02

                row = {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "timestamp": frame_id / fps,
                    "track_id": tid,
                    "class_name": t.class_name,
                    "object_type": t.object_type,
                    "x_m": feat.x_m,
                    "y_m": feat.y_m,
                    "speed_mps": feat.speed_mps,
                    "acceleration_mps2": feat.acceleration_mps2,
                    "deceleration_mps2": feat.deceleration_mps2,
                    "speed_delta_mps": feat.speed_delta_mps,
                    "direction_deg": feat.direction_deg,
                    "direction_change_deg": feat.direction_change_deg,
                    "yaw_rate_dps": feat.yaw_rate_dps,
                    "lateral_acceleration_mps2": feat.lateral_acceleration_mps2,
                    "jerk_mps3": feat.jerk_mps3,
                    "nearest_distance_m": feat.nearest_distance_m,
                    "neighbor_count": feat.neighbor_count,
                    "has_neighbor": feat.has_neighbor,
                    "nearest_distance_valid": feat.nearest_distance_valid,
                    "closing_speed_mps": feat.closing_speed_mps,
                    "interaction_count": feat.interaction_count,
                    "sudden_stop": feat.sudden_stop,
                    "trajectory_change": feat.trajectory_change,
                    "speed_drop_window_mps": feat.speed_drop_window_mps,
                    "direction_change_max_deg": feat.direction_change_max_deg,
                    "direction_change_sum_deg": feat.direction_change_sum_deg,
                    "speed_mean_mps": feat.speed_mean_mps,
                    "speed_std_mps": feat.speed_std_mps,
                    "acceleration_std_mps2": feat.acceleration_std_mps2,
                    "stopped_after_motion": feat.stopped_after_motion,
                    "temporal_window": feat.temporal_window,
                    "interpolated": 0,
                    "accident_probability": prob,
                    "status": status,
                }
                csv_writer.writerow(row)
                written_rows += 1

    finally:
        cap.release()

    return written_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from multiple videos.")
    parser.add_argument("--video-dir", required=True, help="Directory containing .mp4 or .avi files")
    parser.add_argument("--output", default="dataset/extracted_features.csv", help="Output CSV path")
    parser.add_argument("--yolo-model", default="models/yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--label-accident", action="store_true", help="Mark anomalies in these videos as accident")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video to process")
    args = parser.parse_args()

    v_dir = Path(args.video_dir)
    if not v_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {v_dir}")

    video_files = list(v_dir.glob("*.mp4")) + list(v_dir.glob("*.avi")) + list(v_dir.glob("*.mkv"))
    if not video_files:
        raise ValueError(f"No video files found in {v_dir}")

    print(f"Found {len(video_files)} videos in {v_dir}")

    tracker = ByteTrackTracker(model_path=args.yolo_model, confidence=0.25, vehicle_only=True)
    feature_extractor = FeatureExtractor(fps=30.0)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    from main import load_config
    cfg_p = Path(__file__).resolve().parent.parent / "config.yaml"
    transformer = None
    if cfg_p.exists():
        cfg = load_config(str(cfg_p))
        if cfg.get("perspective", {}).get("enabled", False):
            p_cfg = cfg["perspective"]
            transformer = PerspectiveTransformer(
                source_points=np.array(p_cfg["source_points"], dtype=np.float32),
                destination_points=np.array(p_cfg["destination_points"], dtype=np.float32),
                meters_per_pixel_x=p_cfg.get("meters_per_pixel_x", 0.05),
                meters_per_pixel_y=p_cfg.get("meters_per_pixel_y", 0.05),
            )

    fieldnames = [
        "video_id", "frame_id", "timestamp", "track_id", "class_name", "object_type",
        "x_m", "y_m", "speed_mps", "acceleration_mps2", "deceleration_mps2", "speed_delta_mps",
        "direction_deg", "direction_change_deg", "yaw_rate_dps", "lateral_acceleration_mps2",
        "jerk_mps3", "nearest_distance_m", "neighbor_count", "has_neighbor", "nearest_distance_valid",
        "closing_speed_mps", "interaction_count", "sudden_stop", "trajectory_change",
        "speed_drop_window_mps", "direction_change_max_deg", "direction_change_sum_deg",
        "speed_mean_mps", "speed_std_mps", "acceleration_std_mps2", "stopped_after_motion",
        "temporal_window", "interpolated", "accident_probability", "status",
    ]

    total_rows = 0
    with out_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, vf in enumerate(video_files):
            print(f"[{idx+1}/{len(video_files)}] Processing: {vf.name} ...")
            cnt = process_video(
                video_path=vf,
                tracker=tracker,
                feature_extractor=feature_extractor,
                transformer=transformer,
                is_accident_video=args.label_accident,
                csv_writer=writer,
                video_id=vf.stem,
                max_frames=args.max_frames,
            )
            total_rows += cnt

    print(f"\nExtraction complete! Total {total_rows} rows written to: {out_p}")


if __name__ == "__main__":
    main()
