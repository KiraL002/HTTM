"""
Công cụ chuẩn hóa, phân loại chi tiết và gán nhãn toàn bộ dữ liệu huấn luyện (Dataset Cleaner & Multi-Class Labeler).

Thực hiện:
1. Gán nhãn chuẩn xác cho từng dòng dữ liệu trong tất cả các file CSV:
   - status: "NORMAL" hoặc "POSSIBLE ACCIDENT"
   - label: 0 (Bình thường) hoặc 1 (Tai nạn)
   - accident_type: Mã loại va chạm chuẩn (normal, rear_end_collision, head_on_collision,
     side_impact_collision, car_motorcycle_collision, vehicle_pedestrian_collision,
     single_vehicle_spin_or_rollover, single_vehicle_crash)
   - accident_label_vn: Mô tả nhãn tiếng Việt tường minh cho từng loại tai nạn & phương tiện.
2. Làm sạch 17.766 dòng nhãn rác trong combined_train_dataset.csv (loại bỏ xe rẽ cua/chuyển làn bị gán nhầm).
3. Bổ sung đầy đủ các mẫu tai nạn Xe máy (motorcycle), Xe đạp (bicycle), Người đi bộ (person),
   Ô tô (car), Xe tải (truck), Xe buýt (bus) và 14.000 mẫu biên bình thường (Edge Cases).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


ACCIDENT_TYPE_VN_MAP: Dict[str, str] = {
    "normal": "Bình thường (An toàn)",
    "rear_end_collision": "Va chạm phía sau (Tông đuôi)",
    "head_on_collision": "Va chạm đối đầu (Trực diện)",
    "side_impact_collision": "Va chạm sườn xe / Giao lộ (T-bone)",
    "car_motorcycle_collision": "Va chạm xe máy với ô tô / xe lớn",
    "motorcycle_fall_or_crash": "Tai nạn xe máy tự ngã / trượt đường",
    "vehicle_pedestrian_collision": "Va chạm người đi bộ / xe đạp",
    "single_vehicle_spin_or_rollover": "Xe mất lái quay vòng / Lật xe",
    "single_vehicle_crash": "Xe đâm vật cản / Dải phân cách",
}


def assign_detailed_labels(df: pd.DataFrame, is_accident_mask: pd.Series) -> pd.DataFrame:
    """
    Gán đầy đủ 4 cột nhãn tường minh cho DataFrame:
    - status: 'NORMAL' | 'POSSIBLE ACCIDENT'
    - label: 0 | 1
    - accident_type: mã loại tai nạn cụ thể
    - accident_label_vn: tên loại tai nạn tiếng Việt
    """
    df = df.copy()
    df["status"] = np.where(is_accident_mask, "POSSIBLE ACCIDENT", "NORMAL")
    df["label"] = is_accident_mask.astype(int)

    acc_types: List[str] = []
    vid_series = df["video_id"].astype(str).str.lower()
    cls_series = df["class_name"].astype(str).str.lower()
    has_neigh_series = df["has_neighbor"].fillna(0).astype(int)
    dir_chg_series = df["direction_change_deg"].fillna(0.0).astype(float)
    dir_sum_series = df["direction_change_sum_deg"].fillna(0.0).astype(float)
    yaw_series = df["yaw_rate_dps"].fillna(0.0).astype(float).abs()
    closing_series = df["closing_speed_mps"].fillna(0.0).astype(float)

    for idx in range(len(df)):
        if not is_accident_mask.iloc[idx]:
            acc_types.append("normal")
            continue

        vid = vid_series.iloc[idx]
        cls_name = cls_series.iloc[idx]
        has_n = has_neigh_series.iloc[idx]
        d_chg = dir_chg_series.iloc[idx]
        d_sum = dir_sum_series.iloc[idx]
        yaw = yaw_series.iloc[idx]
        closing = closing_series.iloc[idx]

        # Ưu tiên nhận diện từ tên kịch bản nếu có
        if "pedestrian" in vid or "bicycle" in vid or cls_name in ("person", "pedestrian", "bicycle", "bike"):
            acc_types.append("vehicle_pedestrian_collision")
        elif "motorcycle_fall" in vid or (cls_name == "motorcycle" and has_n == 0):
            acc_types.append("motorcycle_fall_or_crash")
        elif "motorcycle" in vid or cls_name == "motorcycle":
            acc_types.append("car_motorcycle_collision")
        elif "rear_end" in vid:
            acc_types.append("rear_end_collision")
        elif "head_on" in vid:
            acc_types.append("head_on_collision")
        elif "side_impact" in vid:
            acc_types.append("side_impact_collision")
        elif "spin" in vid:
            acc_types.append("single_vehicle_spin_or_rollover")
        elif "barrier" in vid:
            acc_types.append("single_vehicle_crash")
        else:
            # Phân loại dựa trên động học vật lý thực tế
            if has_n == 1:
                if closing >= 12.0 and d_chg < 30.0:
                    acc_types.append("head_on_collision")
                elif d_chg >= 28.0 or d_sum >= 55.0:
                    acc_types.append("side_impact_collision")
                else:
                    acc_types.append("rear_end_collision")
            else:
                if yaw >= 80.0 or d_sum >= 70.0:
                    acc_types.append("single_vehicle_spin_or_rollover")
                else:
                    acc_types.append("single_vehicle_crash")

    df["accident_type"] = acc_types
    df["accident_label_vn"] = [ACCIDENT_TYPE_VN_MAP.get(t, "Sự cố giao thông") for t in acc_types]
    return df


def clean_real_dataset(input_csv: Path, backup: bool = True) -> pd.DataFrame:
    """Làm sạch và gán lại nhãn chi tiết cho tập dữ liệu thực tế."""
    backup_orig = input_csv.with_name(f"{input_csv.stem}_original_backup.csv")
    source_to_read = backup_orig if backup_orig.exists() else input_csv

    print(f"[1/5] Đang nạp tập dữ liệu gốc: {source_to_read} ...")
    if not source_to_read.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {source_to_read}")

    if backup and not backup_orig.exists():
        print(f"      Sao lưu bản gốc sang: {backup_orig}")
        shutil.copy2(input_csv, backup_orig)

    df = pd.read_csv(source_to_read)
    original_total = len(df)
    original_accident_count = (df["status"] == "POSSIBLE ACCIDENT").sum()
    print(f"      Tổng số mẫu: {original_total:,} | Nhãn tai nạn cũ: {original_accident_count:,} mẫu")

    # 1. Các video giao thông bình thường kiểm chứng (traffic.mp4, normal_*)
    is_known_normal = df["video_id"].astype(str).str.contains("normal", case=False, na=False) | (
        df["video_id"] == "traffic"
    )

    # Bổ sung tính lại stopped_after_motion cho các dòng video thực tế bị thiếu cờ này
    recomputed_stop = (
        (df["speed_mps"] < 1.5)
        & (df["speed_drop_window_mps"] >= 3.5)
        & (df["deceleration_mps2"] >= 2.5)
    ).astype(int)
    df["stopped_after_motion"] = np.maximum(df["stopped_after_motion"].fillna(0).astype(int), recomputed_stop)

    # 2. Điều kiện va chạm đa phương tiện vật lý chuẩn:
    is_two_wheeler = df["class_name"].astype(str).str.lower().isin(["motorcycle", "bicycle", "bike", "person"])
    dist_limit = np.where(is_two_wheeler, 4.5, 3.5)

    cond_multi_vehicle = (
        (df["has_neighbor"] == 1)
        & (df["nearest_distance_m"] <= dist_limit)
        & ((df["closing_speed_mps"] >= 2.0) | (df["speed_drop_window_mps"] >= 3.0))
        & (
            (df["deceleration_mps2"] >= 3.0)
            | (df["direction_change_deg"] >= 25.0)
            | (df["speed_mps"] <= 3.0)
            | (df["stopped_after_motion"] == 1)
        )
    )

    # 3. Điều kiện sự cố đơn phương tiện vật lý chuẩn:
    cond_single_barrier = (
        (df["has_neighbor"] == 0)
        & (df["stopped_after_motion"] == 1)
        & (df["deceleration_mps2"] >= 4.5)
        & (df["speed_drop_window_mps"] >= 4.0)
    )
    cond_single_spin = (
        (df["has_neighbor"] == 0)
        & (df["yaw_rate_dps"].abs() >= 80.0)
        & (df["direction_change_sum_deg"] >= 70.0)
        & (df["speed_mps"] <= 3.5)
    )
    cond_single_vehicle = cond_single_barrier | cond_single_spin

    # Tổng hợp nhãn tai nạn thực sự
    is_true_accident = (~is_known_normal) & (cond_multi_vehicle | cond_single_vehicle)

    # Cắt trần các giá trị vô cực / nhiễu pixel đo lường
    cleaned_df = df.copy()
    cleaned_df["nearest_distance_m"] = cleaned_df["nearest_distance_m"].replace([np.inf, -np.inf], 50.0).clip(0.0, 50.0)
    cleaned_df["deceleration_mps2"] = cleaned_df["deceleration_mps2"].clip(0.0, 20.0)
    cleaned_df["speed_mps"] = cleaned_df["speed_mps"].clip(0.0, 50.0)
    cleaned_df["closing_speed_mps"] = cleaned_df["closing_speed_mps"].clip(-30.0, 30.0)
    cleaned_df["direction_change_deg"] = cleaned_df["direction_change_deg"].clip(0.0, 180.0)

    np.random.seed(42)
    accident_probs = np.random.uniform(0.85, 0.98, size=len(cleaned_df))
    normal_probs = np.random.uniform(0.01, 0.08, size=len(cleaned_df))
    cleaned_df["accident_probability"] = np.where(is_true_accident, accident_probs, normal_probs)

    cleaned_df = assign_detailed_labels(cleaned_df, is_true_accident)

    new_accidents = int(cleaned_df["label"].sum())
    print(f"      Kết quả làm sạch & gán nhãn:")
    print(f"      - Số mẫu tai nạn thực sự xác thực: {new_accidents:,} mẫu (Đã lọc bỏ {original_accident_count - new_accidents:,} mẫu nhãn sai)")
    print(f"      - Số mẫu bình thường: {(cleaned_df['label'] == 0).sum():,} mẫu")

    return cleaned_df


def generate_motorcycle_and_vulnerable_user_accidents() -> pd.DataFrame:
    """
    Tạo bộ mẫu động học có gán nhãn chi tiết cho các vụ tai nạn đặc thù tại Việt Nam:
    1. Tai nạn xe máy va chạm ô tô / xe tải tại giao lộ & tông đuôi (car_motorcycle_collision)
    2. Tai nạn xe máy tự trượt ngã / mất lái trên đường (motorcycle_fall_or_crash)
    3. Tai nạn va chạm với người đi bộ & xe đạp (vehicle_pedestrian_collision)
    """
    print("[2/5] Đang bổ sung dữ liệu có gán nhãn cho Tai nạn Xe máy, Xe đạp & Người đi bộ...")
    np.random.seed(101)

    # A. Va chạm Xe máy vs Ô tô / Xe tải (1200 mẫu tai nạn + 1800 mẫu xe máy lưu thông bình thường)
    n_mc_acc = 1200
    mc_acc = {
        "video_id": [f"vn_motorcycle_collision_{i//15:03d}" for i in range(n_mc_acc)],
        "frame_id": [(i % 15) + 10 for i in range(n_mc_acc)],
        "timestamp": [((i % 15) + 10) * 0.033 for i in range(n_mc_acc)],
        "track_id": [(i // 15) + 1 for i in range(n_mc_acc)],
        "class_name": "motorcycle",
        "object_type": "vehicle",
        "x_m": np.random.uniform(12.0, 35.0, n_mc_acc),
        "y_m": np.random.uniform(12.0, 35.0, n_mc_acc),
        "speed_mps": np.random.uniform(0.0, 3.2, n_mc_acc),
        "acceleration_mps2": np.random.uniform(-16.0, -5.0, n_mc_acc),
        "deceleration_mps2": np.random.uniform(5.0, 16.0, n_mc_acc),
        "speed_delta_mps": np.random.uniform(-9.0, -3.0, n_mc_acc),
        "direction_deg": np.random.uniform(0.0, 360.0, n_mc_acc),
        "direction_change_deg": np.random.uniform(25.0, 95.0, n_mc_acc),
        "yaw_rate_dps": np.random.uniform(75.0, 180.0, n_mc_acc),
        "lateral_acceleration_mps2": np.random.uniform(8.0, 28.0, n_mc_acc),
        "jerk_mps3": np.random.uniform(-55.0, -20.0, n_mc_acc),
        "nearest_distance_m": np.random.uniform(0.3, 3.2, n_mc_acc),
        "neighbor_count": np.random.randint(1, 4, n_mc_acc),
        "has_neighbor": 1,
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(3.5, 16.0, n_mc_acc),
        "interaction_count": np.random.randint(1, 4, n_mc_acc),
        "sudden_stop": 1,
        "trajectory_change": 1,
        "speed_drop_window_mps": np.random.uniform(4.5, 14.0, n_mc_acc),
        "direction_change_max_deg": np.random.uniform(35.0, 110.0, n_mc_acc),
        "direction_change_sum_deg": np.random.uniform(55.0, 180.0, n_mc_acc),
        "speed_mean_mps": np.random.uniform(5.0, 13.0, n_mc_acc),
        "speed_std_mps": np.random.uniform(2.5, 6.0, n_mc_acc),
        "acceleration_std_mps2": np.random.uniform(4.5, 11.0, n_mc_acc),
        "stopped_after_motion": np.random.choice([0, 1], p=[0.2, 0.8], size=n_mc_acc),
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.88, 0.99, n_mc_acc),
        "status": "POSSIBLE ACCIDENT",
    }
    df_mc_acc = pd.DataFrame(mc_acc)

    # B. Xe máy tự trượt ngã / mất lái đơn phương tiện (450 mẫu)
    n_mc_fall = 450
    mc_fall = {
        "video_id": [f"vn_motorcycle_fall_{i//15:03d}" for i in range(n_mc_fall)],
        "frame_id": [(i % 15) + 12 for i in range(n_mc_fall)],
        "timestamp": [((i % 15) + 12) * 0.033 for i in range(n_mc_fall)],
        "track_id": [(i // 15) + 100 for i in range(n_mc_fall)],
        "class_name": "motorcycle",
        "object_type": "vehicle",
        "x_m": np.random.uniform(10.0, 40.0, n_mc_fall),
        "y_m": np.random.uniform(10.0, 40.0, n_mc_fall),
        "speed_mps": np.random.uniform(0.0, 2.0, n_mc_fall),
        "acceleration_mps2": np.random.uniform(-14.0, -5.5, n_mc_fall),
        "deceleration_mps2": np.random.uniform(5.5, 14.0, n_mc_fall),
        "speed_delta_mps": np.random.uniform(-8.0, -3.5, n_mc_fall),
        "direction_deg": np.random.uniform(0.0, 360.0, n_mc_fall),
        "direction_change_deg": np.random.uniform(45.0, 120.0, n_mc_fall),
        "yaw_rate_dps": np.random.uniform(95.0, 180.0, n_mc_fall),
        "lateral_acceleration_mps2": np.random.uniform(10.0, 26.0, n_mc_fall),
        "jerk_mps3": np.random.uniform(-50.0, -22.0, n_mc_fall),
        "nearest_distance_m": np.random.uniform(12.0, 50.0, n_mc_fall),
        "neighbor_count": 0,
        "has_neighbor": 0,
        "nearest_distance_valid": 1,
        "closing_speed_mps": 0.0,
        "interaction_count": 0,
        "sudden_stop": 1,
        "trajectory_change": 1,
        "speed_drop_window_mps": np.random.uniform(5.0, 13.0, n_mc_fall),
        "direction_change_max_deg": np.random.uniform(50.0, 130.0, n_mc_fall),
        "direction_change_sum_deg": np.random.uniform(80.0, 210.0, n_mc_fall),
        "speed_mean_mps": np.random.uniform(4.5, 12.0, n_mc_fall),
        "speed_std_mps": np.random.uniform(2.5, 5.5, n_mc_fall),
        "acceleration_std_mps2": np.random.uniform(4.0, 10.0, n_mc_fall),
        "stopped_after_motion": 1,
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.86, 0.98, n_mc_fall),
        "status": "POSSIBLE ACCIDENT",
    }
    df_mc_fall = pd.DataFrame(mc_fall)

    # C. Va chạm với Người đi bộ (person) và Xe đạp (bicycle) (450 mẫu tai nạn + 1200 mẫu bình thường)
    n_ped_acc = 450
    ped_acc_classes = np.random.choice(["person", "bicycle"], p=[0.6, 0.4], size=n_ped_acc)
    ped_acc_obj_types = np.where(ped_acc_classes == "person", "pedestrian", "two_wheeler")
    ped_acc = {
        "video_id": [f"vn_pedestrian_bicycle_crash_{i//15:03d}" for i in range(n_ped_acc)],
        "frame_id": [(i % 15) + 10 for i in range(n_ped_acc)],
        "timestamp": [((i % 15) + 10) * 0.033 for i in range(n_ped_acc)],
        "track_id": [(i // 15) + 200 for i in range(n_ped_acc)],
        "class_name": ped_acc_classes,
        "object_type": ped_acc_obj_types,
        "x_m": np.random.uniform(10.0, 35.0, n_ped_acc),
        "y_m": np.random.uniform(10.0, 35.0, n_ped_acc),
        "speed_mps": np.random.uniform(0.0, 2.2, n_ped_acc),
        "acceleration_mps2": np.random.uniform(-14.0, -4.5, n_ped_acc),
        "deceleration_mps2": np.random.uniform(4.5, 14.0, n_ped_acc),
        "speed_delta_mps": np.random.uniform(-7.0, -2.5, n_ped_acc),
        "direction_deg": np.random.uniform(0.0, 360.0, n_ped_acc),
        "direction_change_deg": np.random.uniform(30.0, 100.0, n_ped_acc),
        "yaw_rate_dps": np.random.uniform(70.0, 180.0, n_ped_acc),
        "lateral_acceleration_mps2": np.random.uniform(6.0, 22.0, n_ped_acc),
        "jerk_mps3": np.random.uniform(-45.0, -18.0, n_ped_acc),
        "nearest_distance_m": np.random.uniform(0.2, 2.5, n_ped_acc),
        "neighbor_count": np.random.randint(1, 3, n_ped_acc),
        "has_neighbor": 1,
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(3.5, 14.0, n_ped_acc),
        "interaction_count": np.random.randint(1, 3, n_ped_acc),
        "sudden_stop": 1,
        "trajectory_change": 1,
        "speed_drop_window_mps": np.random.uniform(3.5, 10.0, n_ped_acc),
        "direction_change_max_deg": np.random.uniform(35.0, 110.0, n_ped_acc),
        "direction_change_sum_deg": np.random.uniform(55.0, 160.0, n_ped_acc),
        "speed_mean_mps": np.random.uniform(2.0, 7.0, n_ped_acc),
        "speed_std_mps": np.random.uniform(1.5, 4.5, n_ped_acc),
        "acceleration_std_mps2": np.random.uniform(3.5, 9.0, n_ped_acc),
        "stopped_after_motion": 1,
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.88, 0.99, n_ped_acc),
        "status": "POSSIBLE ACCIDENT",
    }
    df_ped_acc = pd.DataFrame(ped_acc)

    # D. Người đi bộ & Xe đạp di chuyển bình thường trên đường (1500 mẫu NORMAL)
    n_ped_norm = 1500
    ped_norm_classes = np.random.choice(["person", "bicycle"], p=[0.55, 0.45], size=n_ped_norm)
    ped_norm_obj_types = np.where(ped_norm_classes == "person", "pedestrian", "two_wheeler")
    ped_norm = {
        "video_id": [f"vn_pedestrian_bicycle_normal_{i//30:03d}" for i in range(n_ped_norm)],
        "frame_id": [(i % 30) + 1 for i in range(n_ped_norm)],
        "timestamp": [(i % 30) * 0.033 for i in range(n_ped_norm)],
        "track_id": [(i // 30) + 300 for i in range(n_ped_norm)],
        "class_name": ped_norm_classes,
        "object_type": ped_norm_obj_types,
        "x_m": np.random.uniform(5.0, 45.0, n_ped_norm),
        "y_m": np.random.uniform(5.0, 45.0, n_ped_norm),
        "speed_mps": np.random.uniform(0.8, 5.5, n_ped_norm),
        "acceleration_mps2": np.random.uniform(-1.5, 1.5, n_ped_norm),
        "deceleration_mps2": np.random.uniform(0.0, 1.8, n_ped_norm),
        "speed_delta_mps": np.random.uniform(-0.6, 0.6, n_ped_norm),
        "direction_deg": np.random.uniform(0.0, 360.0, n_ped_norm),
        "direction_change_deg": np.random.uniform(0.0, 18.0, n_ped_norm),
        "yaw_rate_dps": np.random.uniform(0.0, 45.0, n_ped_norm),
        "lateral_acceleration_mps2": np.random.uniform(0.0, 3.5, n_ped_norm),
        "jerk_mps3": np.random.uniform(-8.0, 8.0, n_ped_norm),
        "nearest_distance_m": np.random.uniform(3.0, 40.0, n_ped_norm),
        "neighbor_count": np.random.randint(0, 3, n_ped_norm),
        "has_neighbor": np.random.choice([0, 1], p=[0.4, 0.6], size=n_ped_norm),
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(-1.5, 1.2, n_ped_norm),
        "interaction_count": np.random.randint(0, 2, n_ped_norm),
        "sudden_stop": 0,
        "trajectory_change": 0,
        "speed_drop_window_mps": np.random.uniform(0.0, 1.8, n_ped_norm),
        "direction_change_max_deg": np.random.uniform(0.0, 20.0, n_ped_norm),
        "direction_change_sum_deg": np.random.uniform(0.0, 45.0, n_ped_norm),
        "speed_mean_mps": np.random.uniform(1.2, 5.0, n_ped_norm),
        "speed_std_mps": np.random.uniform(0.1, 0.8, n_ped_norm),
        "acceleration_std_mps2": np.random.uniform(0.1, 1.5, n_ped_norm),
        "stopped_after_motion": 0,
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.01, 0.05, n_ped_norm),
        "status": "NORMAL",
    }
    df_ped_norm = pd.DataFrame(ped_norm)

    combined = pd.concat([df_mc_acc, df_mc_fall, df_ped_acc, df_ped_norm], ignore_index=True)
    is_acc = combined["status"] == "POSSIBLE ACCIDENT"
    combined = assign_detailed_labels(combined, is_acc)
    print(f"      Đã bổ sung {len(combined):,} mẫu Xe máy, Xe đạp, Người đi bộ (Tai nạn: {int(is_acc.sum()):,} mẫu).")
    return combined


def generate_edge_case_normal_samples(n_per_case: int = 3500) -> pd.DataFrame:
    """
    Tạo các mẫu tình huống bình thường đa dạng (Edge Cases) chống báo động giả:
    Rẽ cua ngã tư, kẹt xe dừng đèn đỏ sát nhau, phanh dừng an toàn, nhiễu rung tracking.
    """
    print(f"[3/5] Đang tổng hợp các kịch bản giao thông biên (Edge Cases: Rẽ cua, kẹt xe, phanh dừng)...")
    np.random.seed(42)

    # 1. Kẹt xe / Dừng đèn đỏ nối đuôi (Cự ly rất gần 0.05m - 3.2m, nhưng closing speed thấp, không phanh sốc)
    n1 = n_per_case
    c1 = {
        "video_id": [f"edge_queue_{i//30:03d}" for i in range(n1)],
        "frame_id": [(i % 30) + 1 for i in range(n1)],
        "timestamp": [(i % 30) * 0.033 for i in range(n1)],
        "track_id": [(i // 30) + 1 for i in range(n1)],
        "class_name": np.random.choice(["car", "motorcycle", "truck", "bus", "bicycle"], p=[0.4, 0.35, 0.1, 0.08, 0.07], size=n1),
        "object_type": "vehicle",
        "x_m": np.random.uniform(10.0, 40.0, n1),
        "y_m": np.random.uniform(10.0, 40.0, n1),
        "speed_mps": np.random.uniform(0.0, 2.5, n1),
        "acceleration_mps2": np.random.uniform(-1.0, 1.0, n1),
        "deceleration_mps2": np.random.uniform(0.0, 1.5, n1),
        "speed_delta_mps": np.random.uniform(-0.5, 0.5, n1),
        "direction_deg": np.random.uniform(0.0, 360.0, n1),
        "direction_change_deg": np.random.uniform(0.0, 5.0, n1),
        "yaw_rate_dps": np.random.uniform(0.0, 20.0, n1),
        "lateral_acceleration_mps2": np.random.uniform(0.0, 2.0, n1),
        "jerk_mps3": np.random.uniform(-10.0, 10.0, n1),
        "nearest_distance_m": np.random.uniform(0.05, 3.2, n1),
        "neighbor_count": np.random.randint(1, 6, n1),
        "has_neighbor": 1,
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(-1.0, 1.2, n1),
        "interaction_count": np.random.randint(1, 6, n1),
        "sudden_stop": np.random.choice([0, 1], p=[0.7, 0.3], size=n1),
        "trajectory_change": 0,
        "speed_drop_window_mps": np.random.uniform(0.0, 5.5, n1),
        "direction_change_max_deg": np.random.uniform(0.0, 8.0, n1),
        "direction_change_sum_deg": np.random.uniform(0.0, 20.0, n1),
        "speed_mean_mps": np.random.uniform(0.5, 3.0, n1),
        "speed_std_mps": np.random.uniform(0.1, 1.0, n1),
        "acceleration_std_mps2": np.random.uniform(0.1, 2.0, n1),
        "stopped_after_motion": np.random.choice([0, 1], p=[0.6, 0.4], size=n1),
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.01, 0.05, n1),
        "status": "NORMAL",
    }
    df1 = pd.DataFrame(c1)

    # 2. Vào cua rẽ ngã tư / Quay đầu chữ U (Góc đổi hướng lớn 10 - 38 độ, nhưng khoảng cách an toàn > 3.8m)
    n2 = n_per_case
    c2 = {
        "video_id": [f"edge_turn_{i//30:03d}" for i in range(n2)],
        "frame_id": [(i % 30) + 1 for i in range(n2)],
        "timestamp": [(i % 30) * 0.033 for i in range(n2)],
        "track_id": [(i // 30) + 1 for i in range(n2)],
        "class_name": np.random.choice(["car", "motorcycle", "bus", "truck", "bicycle"], p=[0.4, 0.35, 0.08, 0.1, 0.07], size=n2),
        "object_type": "vehicle",
        "x_m": np.random.uniform(15.0, 50.0, n2),
        "y_m": np.random.uniform(15.0, 50.0, n2),
        "speed_mps": np.random.uniform(4.0, 14.0, n2),
        "acceleration_mps2": np.random.uniform(-2.5, 1.5, n2),
        "deceleration_mps2": np.random.uniform(0.0, 3.0, n2),
        "speed_delta_mps": np.random.uniform(-1.0, 1.0, n2),
        "direction_deg": np.random.uniform(0.0, 360.0, n2),
        "direction_change_deg": np.random.uniform(10.0, 38.0, n2),
        "yaw_rate_dps": np.random.uniform(25.0, 110.0, n2),
        "lateral_acceleration_mps2": np.random.uniform(2.5, 18.0, n2),
        "jerk_mps3": np.random.uniform(-15.0, 15.0, n2),
        "nearest_distance_m": np.random.uniform(3.8, 50.0, n2),
        "neighbor_count": np.random.randint(0, 3, n2),
        "has_neighbor": np.random.choice([0, 1], p=[0.4, 0.6], size=n2),
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(-2.0, 1.8, n2),
        "interaction_count": 1,
        "sudden_stop": 0,
        "trajectory_change": 1,
        "speed_drop_window_mps": np.random.uniform(0.5, 4.0, n2),
        "direction_change_max_deg": np.random.uniform(15.0, 45.0, n2),
        "direction_change_sum_deg": np.random.uniform(25.0, 130.0, n2),
        "speed_mean_mps": np.random.uniform(5.0, 14.0, n2),
        "speed_std_mps": np.random.uniform(0.5, 2.5, n2),
        "acceleration_std_mps2": np.random.uniform(0.5, 3.0, n2),
        "stopped_after_motion": 0,
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.01, 0.06, n2),
        "status": "NORMAL",
    }
    df2 = pd.DataFrame(c2)

    # 3. Phanh gấp an toàn trước vạch dừng đèn đỏ / chướng ngại vật từ xa
    n3 = n_per_case
    c3 = {
        "video_id": [f"edge_hardbrake_{i//30:03d}" for i in range(n3)],
        "frame_id": [(i % 30) + 1 for i in range(n3)],
        "timestamp": [(i % 30) * 0.033 for i in range(n3)],
        "track_id": [(i // 30) + 1 for i in range(n3)],
        "class_name": np.random.choice(["car", "motorcycle", "truck", "bus"], p=[0.45, 0.35, 0.1, 0.1], size=n3),
        "object_type": "vehicle",
        "x_m": np.random.uniform(20.0, 60.0, n3),
        "y_m": np.random.uniform(20.0, 60.0, n3),
        "speed_mps": np.random.uniform(0.0, 8.0, n3),
        "acceleration_mps2": np.random.uniform(-6.5, -2.0, n3),
        "deceleration_mps2": np.random.uniform(3.5, 18.0, n3),
        "speed_delta_mps": np.random.uniform(-3.5, -1.0, n3),
        "direction_deg": np.random.uniform(0.0, 360.0, n3),
        "direction_change_deg": np.random.uniform(0.0, 6.0, n3),
        "yaw_rate_dps": np.random.uniform(0.0, 20.0, n3),
        "lateral_acceleration_mps2": np.random.uniform(0.0, 2.0, n3),
        "jerk_mps3": np.random.uniform(-25.0, 10.0, n3),
        "nearest_distance_m": np.random.uniform(1.5, 50.0, n3),
        "neighbor_count": np.random.randint(0, 3, n3),
        "has_neighbor": np.random.choice([0, 1], p=[0.5, 0.5], size=n3),
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(-5.0, 1.2, n3),
        "interaction_count": 0,
        "sudden_stop": 1,
        "trajectory_change": 0,
        "speed_drop_window_mps": np.random.uniform(4.0, 12.0, n3),
        "direction_change_max_deg": np.random.uniform(0.0, 8.0, n3),
        "direction_change_sum_deg": np.random.uniform(0.0, 15.0, n3),
        "speed_mean_mps": np.random.uniform(2.0, 10.0, n3),
        "speed_std_mps": np.random.uniform(1.0, 3.5, n3),
        "acceleration_std_mps2": np.random.uniform(1.0, 4.0, n3),
        "stopped_after_motion": np.random.choice([0, 1], p=[0.3, 0.7], size=n3),
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.02, 0.08, n3),
        "status": "NORMAL",
    }
    df3 = pd.DataFrame(c3)

    # 4. Nhiễu rung bounding box tracking
    n4 = n_per_case
    c4 = {
        "video_id": [f"edge_jitter_{i//30:03d}" for i in range(n4)],
        "frame_id": [(i % 30) + 1 for i in range(n4)],
        "timestamp": [(i % 30) * 0.033 for i in range(n4)],
        "track_id": [(i // 30) + 1 for i in range(n4)],
        "class_name": np.random.choice(["car", "motorcycle"], size=n4),
        "object_type": "vehicle",
        "x_m": np.random.uniform(10.0, 50.0, n4),
        "y_m": np.random.uniform(10.0, 50.0, n4),
        "speed_mps": np.random.uniform(8.0, 22.0, n4),
        "acceleration_mps2": np.random.uniform(-4.0, 4.0, n4),
        "deceleration_mps2": np.random.uniform(2.0, 5.0, n4),
        "speed_delta_mps": np.random.uniform(-2.0, 2.0, n4),
        "direction_deg": np.random.uniform(0.0, 360.0, n4),
        "direction_change_deg": np.random.uniform(5.0, 22.0, n4),
        "yaw_rate_dps": np.random.uniform(15.0, 65.0, n4),
        "lateral_acceleration_mps2": np.random.uniform(2.0, 12.0, n4),
        "jerk_mps3": np.random.uniform(-20.0, 20.0, n4),
        "nearest_distance_m": np.random.uniform(5.0, 50.0, n4),
        "neighbor_count": np.random.randint(0, 2, n4),
        "has_neighbor": np.random.choice([0, 1], p=[0.7, 0.3], size=n4),
        "nearest_distance_valid": 1,
        "closing_speed_mps": np.random.uniform(-1.0, 1.0, n4),
        "interaction_count": 0,
        "sudden_stop": 0,
        "trajectory_change": 0,
        "speed_drop_window_mps": np.random.uniform(0.5, 3.0, n4),
        "direction_change_max_deg": np.random.uniform(5.0, 22.0, n4),
        "direction_change_sum_deg": np.random.uniform(5.0, 25.0, n4),
        "speed_mean_mps": np.random.uniform(8.0, 22.0, n4),
        "speed_std_mps": np.random.uniform(0.5, 2.0, n4),
        "acceleration_std_mps2": np.random.uniform(0.5, 2.5, n4),
        "stopped_after_motion": 0,
        "temporal_window": 15,
        "interpolated": 0,
        "accident_probability": np.random.uniform(0.01, 0.05, n4),
        "status": "NORMAL",
    }
    df4 = pd.DataFrame(c4)

    augmented_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    is_acc = augmented_df["status"] == "POSSIBLE ACCIDENT"
    augmented_df = assign_detailed_labels(augmented_df, is_acc)
    print(f"      Đã sinh thành công {len(augmented_df):,} mẫu bình thường biên có gán nhãn.")
    return augmented_df


def build_unified_cleaned_dataset(
    project_root: Path,
) -> Path:
    """Xây dựng và lưu tập dữ liệu chuẩn hóa toàn diện cleaned_train_dataset.csv."""
    dataset_dir = project_root / "dataset"
    combined_csv = dataset_dir / "combined_train_dataset.csv"
    diverse_csv = dataset_dir / "train_diverse_features.csv"
    benchmark_csv = dataset_dir / "benchmark_diverse_dataset.csv"
    output_csv = dataset_dir / "cleaned_train_dataset.csv"

    # 1. Làm sạch & gán nhãn chi tiết cho dữ liệu thực tế
    cleaned_real_df = clean_real_dataset(combined_csv, backup=True)
    cleaned_real_df.to_csv(combined_csv, index=False)
    print(f"      Đã lưu nhãn chi tiết vào: {combined_csv}")

    # 2. Gán nhãn chi tiết cho train_diverse_features.csv và benchmark_diverse_dataset.csv
    print(f"[4/5] Đang cập nhật cột nhãn chi tiết cho {diverse_csv.name} & {benchmark_csv.name} ...")
    df_diverse = pd.read_csv(diverse_csv)
    df_diverse = assign_detailed_labels(df_diverse, df_diverse["status"] == "POSSIBLE ACCIDENT")
    df_diverse.to_csv(diverse_csv, index=False)

    if benchmark_csv.exists():
        df_bench = pd.read_csv(benchmark_csv)
        df_bench = assign_detailed_labels(df_bench, df_bench["status"] == "POSSIBLE ACCIDENT")
        df_bench.to_csv(benchmark_csv, index=False)

    # 3. Bổ sung tai nạn Xe máy, Xe đạp, Người đi bộ & Mẫu biên
    df_vn_mc = generate_motorcycle_and_vulnerable_user_accidents()
    df_edge = generate_edge_case_normal_samples(n_per_case=3500)

    # 4. Hợp nhất toàn bộ dữ liệu đã gán nhãn
    print(f"[5/5] Hợp nhất và lưu bộ dữ liệu sạch vào: {output_csv} ...")
    real_vru_csv = dataset_dir / "real_vru_motorcycle_dataset.csv"
    parts = [cleaned_real_df, df_diverse, df_vn_mc, df_edge]
    if real_vru_csv.exists():
        df_real_vru = pd.read_csv(real_vru_csv)
        print(f"      Tích hợp thêm {len(df_real_vru):,} mẫu thực tế từ {real_vru_csv.name}")
        parts.append(df_real_vru)
    unified_df = pd.concat(parts, ignore_index=True)

    # Đồng bộ 100% cột object_type theo chuẩn OBJECT_TYPE_MAP trong core/tracker.py
    obj_type_map = {
        "car": "vehicle",
        "motorcycle": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "train": "vehicle",
        "bicycle": "two_wheeler",
        "bike": "vehicle",
        "person": "pedestrian",
        "human": "pedestrian",
    }
    unified_df["object_type"] = unified_df["class_name"].astype(str).str.lower().map(obj_type_map).fillna("vehicle")

    num_cols = unified_df.select_dtypes(include=[np.number]).columns
    unified_df[num_cols] = unified_df[num_cols].replace([np.inf, -np.inf], 50.0).fillna(0.0)
    unified_df.to_csv(output_csv, index=False)

    total_rows = len(unified_df)
    accidents = int((unified_df["label"] == 1).sum())
    normals = int((unified_df["label"] == 0).sum())

    print("=" * 70)
    print("HOÀN TẤT GÁN NHÃN CHI TIẾT TOÀN BỘ BỘ DỮ LIỆU:")
    print(f"  - Tổng số mẫu đã gán nhãn : {total_rows:,} mẫu")
    print(f"  - Nhãn 0 (NORMAL)         : {normals:,} mẫu ({normals/total_rows*100:.1f}%)")
    print(f"  - Nhãn 1 (ACCIDENT)       : {accidents:,} mẫu ({accidents/total_rows*100:.1f}%)")
    print("\nPhân bố theo Phương tiện (Nhãn 1 - Tai nạn):")
    print(unified_df[unified_df["label"] == 1]["class_name"].value_counts().to_string())
    print("\nPhân bố theo Loại tai nạn (accident_label_vn):")
    print(unified_df["accident_label_vn"].value_counts().to_string())
    print("=" * 70)

    return output_csv


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    build_unified_cleaned_dataset(root)
