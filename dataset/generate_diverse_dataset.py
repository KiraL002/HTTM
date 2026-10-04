"""
Tạo bộ dữ liệu đặc trưng đa dạng kịch bản (Diverse Scenarios Dataset Generator)
cho mô hình phát hiện tai nạn giao thông (HTTM).

Bao gồm:
- Giao thông bình thường: Cao tốc, đô thị dừng-chạy, chuyển làn, quay đầu, ùn tắc.
- Tai nạn đa xe: Va chạm phía sau (rear-end), đối đầu (head-on), sườn góc (t-bone), dồn toa (pileup).
- Tai nạn đơn xe: Mất lái quay vòng (spin/rollover), đâm vật cản phanh gấp (barrier crash/sudden stop).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CSV_COLUMNS = [
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


def generate_scenario_samples(
    scenario_type: str,
    num_sequences: int = 20,
    seq_length: int = 30,
    fps: float = 30.0,
    start_video_id: int = 1,
) -> List[Dict]:
    """
    Sinh chuỗi dữ liệu động học mô phỏng theo vật lý thực tế cho từng kịch bản.
    """
    dt = 1.0 / fps
    rows: List[Dict] = []
    vehicle_types = ["car", "truck", "bus", "motorcycle"]

    for seq_idx in range(num_sequences):
        video_name = f"{scenario_type}_{start_video_id + seq_idx:03d}"
        v_type = np.random.choice(vehicle_types, p=[0.7, 0.15, 0.08, 0.07])
        tid = 1

        # Khởi tạo thông số theo từng kịch bản
        if scenario_type == "normal_highway":
            base_speed = np.random.uniform(22.0, 32.0)  # 80 - 115 km/h
            speed = base_speed
            direction = np.random.uniform(85.0, 95.0)
            x, y = np.random.uniform(10.0, 30.0), 10.0
            neighbor_dist = np.random.uniform(20.0, 50.0)
            has_neigh = 1 if neighbor_dist <= 25.0 else 0
            is_accident = False

        elif scenario_type == "normal_urban":
            base_speed = np.random.uniform(8.0, 16.0)  # 30 - 60 km/h
            speed = base_speed
            direction = np.random.uniform(0.0, 360.0)
            x, y = np.random.uniform(10.0, 50.0), np.random.uniform(10.0, 50.0)
            neighbor_dist = np.random.uniform(5.0, 25.0)
            has_neigh = 1
            is_accident = False

        elif scenario_type == "normal_stop_and_go":
            base_speed = np.random.uniform(3.0, 8.0)
            speed = base_speed
            direction = 90.0
            x, y = 20.0, 10.0
            neighbor_dist = np.random.uniform(3.0, 12.0)
            has_neigh = 1
            is_accident = False

        elif scenario_type == "normal_intersection_turn":
            base_speed = np.random.uniform(6.0, 10.0)
            speed = base_speed
            direction = 0.0
            x, y = 25.0, 25.0
            neighbor_dist = np.random.uniform(15.0, 35.0)
            has_neigh = 1 if neighbor_dist <= 25.0 else 0
            is_accident = False

        elif scenario_type == "accident_rear_end":
            base_speed = np.random.uniform(15.0, 25.0)
            speed = base_speed
            direction = 90.0
            x, y = 20.0, 15.0
            neighbor_dist = np.random.uniform(10.0, 20.0)
            has_neigh = 1
            is_accident = True
            impact_frame = int(seq_length * 0.45)

        elif scenario_type == "accident_head_on":
            base_speed = np.random.uniform(16.0, 24.0)
            speed = base_speed
            direction = 180.0
            x, y = 30.0, 20.0
            neighbor_dist = np.random.uniform(15.0, 30.0)
            has_neigh = 1
            is_accident = True
            impact_frame = int(seq_length * 0.40)

        elif scenario_type == "accident_side_impact":
            base_speed = np.random.uniform(12.0, 20.0)
            speed = base_speed
            direction = 90.0
            x, y = 25.0, 25.0
            neighbor_dist = np.random.uniform(10.0, 18.0)
            has_neigh = 1
            is_accident = True
            impact_frame = int(seq_length * 0.40)

        elif scenario_type == "accident_single_vehicle_spin":
            base_speed = np.random.uniform(18.0, 28.0)
            speed = base_speed
            direction = 45.0
            x, y = 20.0, 10.0
            neighbor_dist = 50.0  # Xe đơn độc, không có xe gần
            has_neigh = 0
            is_accident = True
            impact_frame = int(seq_length * 0.35)

        elif scenario_type == "accident_single_vehicle_barrier":
            base_speed = np.random.uniform(16.0, 26.0)
            speed = base_speed
            direction = 90.0
            x, y = 15.0, 10.0
            neighbor_dist = 50.0  # Xe đơn lẻ tự đâm chướng ngại vật
            has_neigh = 0
            is_accident = True
            impact_frame = int(seq_length * 0.40)

        else:
            continue

        prev_speed = speed
        prev_accel = 0.0
        prev_dir = direction
        speed_window: List[float] = []
        dir_window: List[float] = []

        for f in range(1, seq_length + 1):
            timestamp = f * dt

            # Cập nhật động học từng frame
            if not is_accident:
                # Giao thông bình thường
                if scenario_type == "normal_highway":
                    accel = np.random.normal(0.0, 0.4)
                    dir_change = np.random.normal(0.0, 0.5)
                    neighbor_dist = float(np.clip(neighbor_dist + np.random.normal(0, 0.1), 15.0, 50.0))
                    closing_speed = np.random.normal(0.0, 0.5)

                elif scenario_type == "normal_urban":
                    accel = np.random.normal(0.0, 0.8)
                    dir_change = np.random.normal(0.0, 1.5)
                    neighbor_dist = float(np.clip(neighbor_dist + np.random.normal(0, 0.2), 4.0, 30.0))
                    closing_speed = np.random.normal(0.0, 1.0)

                elif scenario_type == "normal_stop_and_go":
                    accel = -1.2 if f > seq_length // 2 else 0.8
                    dir_change = np.random.normal(0.0, 0.3)
                    speed = max(0.0, speed + accel * dt)
                    neighbor_dist = float(np.clip(neighbor_dist + accel * dt * 0.5, 2.0, 15.0))
                    closing_speed = -accel * 0.5

                elif scenario_type == "normal_intersection_turn":
                    accel = -1.0 if f < 10 else (1.0 if f > 20 else 0.0)
                    dir_change = 3.0  # Rẽ 90 độ từ từ trong 30 frame
                    neighbor_dist = float(np.clip(neighbor_dist + np.random.normal(0, 0.1), 10.0, 40.0))
                    closing_speed = 0.0

                speed = float(np.clip(speed + accel * dt, 0.0, 40.0))
                direction = float((direction + dir_change) % 360.0)
                status = "NORMAL"
                prob = float(np.clip(np.random.normal(0.03, 0.02), 0.001, 0.25))

            else:
                # Kịch bản tai nạn
                if f < impact_frame:
                    accel = np.random.normal(0.0, 0.5)
                    dir_change = np.random.normal(0.0, 0.8)
                    speed = float(np.clip(speed + accel * dt, 10.0, 35.0))
                    if has_neigh:
                        neighbor_dist = float(max(1.0, neighbor_dist - (speed * 0.3 * dt)))
                        closing_speed = float(speed * 0.5)
                    else:
                        closing_speed = 0.0
                    status = "NORMAL"
                    prob = float(np.clip(np.random.normal(0.05, 0.03), 0.001, 0.25))

                elif impact_frame <= f <= impact_frame + 6:
                    # Khoảnh khắc va chạm (Impact phase)
                    if scenario_type == "accident_rear_end":
                        accel = -np.random.uniform(9.0, 16.0)  # Phanh khẩn cấp & va chạm mạnh
                        dir_change = np.random.uniform(5.0, 18.0)
                        neighbor_dist = float(np.random.uniform(0.3, 1.5))
                        closing_speed = float(np.random.uniform(8.0, 15.0))

                    elif scenario_type == "accident_head_on":
                        accel = -np.random.uniform(14.0, 22.0)
                        dir_change = np.random.uniform(15.0, 35.0)
                        neighbor_dist = float(np.random.uniform(0.2, 1.2))
                        closing_speed = float(np.random.uniform(15.0, 25.0))

                    elif scenario_type == "accident_side_impact":
                        accel = -np.random.uniform(8.0, 14.0)
                        dir_change = np.random.uniform(40.0, 75.0)  # Bị hất chệch hướng mạnh
                        neighbor_dist = float(np.random.uniform(0.4, 2.0))
                        closing_speed = float(np.random.uniform(6.0, 12.0))

                    elif scenario_type == "accident_single_vehicle_spin":
                        accel = -np.random.uniform(5.0, 10.0)
                        dir_change = np.random.uniform(50.0, 95.0)  # Quay vòng mất lái
                        neighbor_dist = 50.0
                        closing_speed = 0.0

                    elif scenario_type == "accident_single_vehicle_barrier":
                        accel = -np.random.uniform(12.0, 20.0)  # Đâm dải phân cách/vật cản
                        dir_change = np.random.uniform(20.0, 45.0)
                        neighbor_dist = 50.0
                        closing_speed = 0.0

                    speed = float(max(0.0, speed + accel * dt))
                    direction = float((direction + dir_change) % 360.0)
                    status = "POSSIBLE ACCIDENT"
                    prob = float(np.clip(np.random.uniform(0.78, 0.98), 0.70, 0.99))

                else:
                    # Sau va chạm (Post-impact settled phase)
                    accel = -np.random.uniform(0.5, 2.0) if speed > 0.5 else 0.0
                    dir_change = np.random.normal(0.0, 1.0)
                    speed = float(max(0.0, speed + accel * dt))
                    if speed < 0.5:
                        speed = 0.0
                    if has_neigh:
                        neighbor_dist = float(np.clip(neighbor_dist + np.random.normal(0, 0.05), 0.5, 3.0))
                        closing_speed = 0.0
                    status = "POSSIBLE ACCIDENT" if f <= impact_frame + 12 else "NORMAL"
                    prob = float(np.clip(0.85 - (f - impact_frame) * 0.05, 0.15, 0.90))

            decel = float(max(0.0, -accel))
            speed_delta = float(speed - prev_speed)
            yaw_rate = float(abs(dir_change) / dt)
            lateral_accel = float(speed * np.radians(yaw_rate))
            jerk = float((accel - prev_accel) / dt)

            # Cập nhật cửa sổ trượt
            speed_window.append(speed)
            dir_window.append(abs(dir_change))
            if len(speed_window) > 15:
                speed_window.pop(0)
                dir_window.pop(0)

            speed_drop_window = float(max(speed_window) - speed)
            stopped_after_motion = 1 if (max(speed_window) >= 3.5 and speed < 1.0 and decel >= 2.0) else 0

            # Cập nhật vị trí
            x += float(speed * np.cos(np.radians(direction)) * dt)
            y += float(speed * np.sin(np.radians(direction)) * dt)

            row = {
                "video_id": video_name,
                "frame_id": f,
                "timestamp": timestamp,
                "track_id": tid,
                "class_name": v_type,
                "object_type": "vehicle",
                "x_m": round(x, 2),
                "y_m": round(y, 2),
                "speed_mps": round(speed, 2),
                "acceleration_mps2": round(accel, 2),
                "deceleration_mps2": round(decel, 2),
                "speed_delta_mps": round(speed_delta, 2),
                "direction_deg": round(direction, 2),
                "direction_change_deg": round(abs(dir_change), 2),
                "yaw_rate_dps": round(yaw_rate, 2),
                "lateral_acceleration_mps2": round(lateral_accel, 2),
                "jerk_mps3": round(jerk, 2),
                "nearest_distance_m": round(neighbor_dist, 2),
                "neighbor_count": 1 if has_neigh else 0,
                "has_neighbor": has_neigh,
                "nearest_distance_valid": 1 if has_neigh else 0,
                "closing_speed_mps": round(closing_speed, 2),
                "interaction_count": 1 if has_neigh else 0,
                "sudden_stop": 1 if decel >= 3.5 else 0,
                "trajectory_change": 1 if abs(dir_change) >= 35.0 else 0,
                "speed_drop_window_mps": round(speed_drop_window, 2),
                "direction_change_max_deg": round(float(max(dir_window)), 2),
                "direction_change_sum_deg": round(float(sum(dir_window)), 2),
                "speed_mean_mps": round(float(np.mean(speed_window)), 2),
                "speed_std_mps": round(float(np.std(speed_window)), 2),
                "acceleration_std_mps2": round(float(abs(accel) * 0.4), 2),
                "stopped_after_motion": stopped_after_motion,
                "temporal_window": len(speed_window),
                "interpolated": 0,
                "accident_probability": round(prob, 4),
                "status": status,
            }
            rows.append(row)

            prev_speed = speed
            prev_accel = accel
            prev_dir = direction

    return rows


def build_datasets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        # Giao thông bình thường
        ("normal_highway", 40),
        ("normal_urban", 40),
        ("normal_stop_and_go", 30),
        ("normal_intersection_turn", 30),
        # Tai nạn đa xe
        ("accident_rear_end", 30),
        ("accident_head_on", 25),
        ("accident_side_impact", 25),
        # Tai nạn đơn xe
        ("accident_single_vehicle_spin", 25),
        ("accident_single_vehicle_barrier", 25),
    ]

    all_rows: List[Dict] = []
    print("Generating synthetic-realistic diverse traffic & accident scenarios...")

    for scen, count in scenarios:
        rows = generate_scenario_samples(scenario_type=scen, num_sequences=count, seq_length=35)
        all_rows.extend(rows)
        print(f"  + {scen:<35}: {len(rows)} samples")

    df_all = pd.DataFrame(all_rows)
    print(f"\nTotal generated samples: {len(df_all)}")
    print(f"Class distribution: {df_all['status'].value_counts().to_dict()}")

    # Trộn ngẫu nhiên theo video_id để tránh rò rỉ dữ liệu giữa train/val/test
    unique_videos = df_all["video_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_videos)

    n_vids = len(unique_videos)
    train_vids = set(unique_videos[: int(n_vids * 0.70)])
    val_vids = set(unique_videos[int(n_vids * 0.70) : int(n_vids * 0.85)])
    test_vids = set(unique_videos[int(n_vids * 0.85) :])

    df_train = df_all[df_all["video_id"].isin(train_vids)].reset_index(drop=True)
    df_val = df_all[df_all["video_id"].isin(val_vids)].reset_index(drop=True)
    df_test = df_all[df_all["video_id"].isin(test_vids)].reset_index(drop=True)

    # Lưu các file dataset
    train_path = output_dir / "train_diverse_features.csv"
    val_path = output_dir / "val_diverse_features.csv"
    test_path = output_dir / "test_diverse_features.csv"
    benchmark_path = output_dir / "benchmark_diverse_dataset.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_all.to_csv(benchmark_path, index=False)

    print("\nSaved datasets:")
    print(f"  1. Train set:      {train_path} ({len(df_train)} rows)")
    print(f"  2. Val set:        {val_path} ({len(df_val)} rows)")
    print(f"  3. Test set:       {test_path} ({len(df_test)} rows)")
    print(f"  4. Full Benchmark: {benchmark_path} ({len(df_all)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diverse traffic accident scenarios dataset.")
    parser.add_argument("--output-dir", default=None, help="Target directory to save datasets")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    build_datasets(out_dir)
