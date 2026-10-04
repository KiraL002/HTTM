"""
AcciVision — Huấn Luyện Mô Hình Phân Loại Tai Nạn (Random Forest Trainer)

Script huấn luyện mô hình Random Forest từ dữ liệu đặc trưng thực tế,
hỗ trợ:
- Nạp dữ liệu gán nhãn (CSV)
- Chuẩn hóa đặc trưng
- Huấn luyện với class_weight="balanced" cho dữ liệu mất cân bằng
- Đánh giá toàn diện: Confusion Matrix, Classification Report, ROC-AUC
- Xuất mô hình .pkl đã huấn luyện
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


FEATURE_COLUMNS: List[str] = [
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
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa và bù đắp các đặc trưng nếu file CSV đầu vào thiếu một số cột mới.
    Đồng thời giới hạn trần khoảng cách và triệt tiêu NaN/inf.
    """
    df = df.copy()

    # Bù đắp semantics nếu thiếu
    if "deceleration_mps2" not in df.columns and "acceleration_mps2" in df.columns:
        df["deceleration_mps2"] = df["acceleration_mps2"].apply(lambda a: max(0.0, -float(a)))

    if "speed_delta_mps" not in df.columns and "acceleration_mps2" in df.columns:
        df["speed_delta_mps"] = df["acceleration_mps2"] * 0.033

    if "has_neighbor" not in df.columns:
        if "neighbor_count" in df.columns:
            df["has_neighbor"] = (df["neighbor_count"] > 0).astype(int)
        else:
            df["has_neighbor"] = 0

    if "nearest_distance_m" not in df.columns and "nearest_distance" in df.columns:
        df["nearest_distance_m"] = df["nearest_distance"]

    if "nearest_distance_m" in df.columns:
        # Cắt trần tại 50.0m để triệt tiêu bias multi-object từ các sentinel 1000m / inf
        df["nearest_distance_m"] = df["nearest_distance_m"].replace([np.inf, -np.inf], 50.0).clip(0.0, 50.0)

    # Các đặc trưng single-vehicle & temporal bổ sung
    if "yaw_rate_dps" not in df.columns:
        if "direction_change_deg" in df.columns:
            df["yaw_rate_dps"] = df["direction_change_deg"] / 0.033
        else:
            df["yaw_rate_dps"] = 0.0

    if "lateral_acceleration_mps2" not in df.columns:
        if "speed_mps" in df.columns and "yaw_rate_dps" in df.columns:
            df["lateral_acceleration_mps2"] = df["speed_mps"] * np.radians(df["yaw_rate_dps"])
        else:
            df["lateral_acceleration_mps2"] = 0.0

    if "jerk_mps3" not in df.columns:
        df["jerk_mps3"] = 0.0

    if "closing_speed_mps" not in df.columns:
        df["closing_speed_mps"] = 0.0

    if "speed_drop_window_mps" not in df.columns:
        if "speed_std_mps" in df.columns:
            df["speed_drop_window_mps"] = df["speed_std_mps"] * 2.0
        else:
            df["speed_drop_window_mps"] = 0.0

    if "direction_change_max_deg" not in df.columns:
        df["direction_change_max_deg"] = df.get("direction_change_deg", 0.0)

    if "direction_change_sum_deg" not in df.columns:
        df["direction_change_sum_deg"] = df.get("direction_change_deg", 0.0)

    if "stopped_after_motion" not in df.columns:
        if "speed_mps" in df.columns and "sudden_stop" in df.columns:
            df["stopped_after_motion"] = ((df["speed_mps"] < 1.0) & (df["sudden_stop"] == 1)).astype(int)
        else:
            df["stopped_after_motion"] = 0

    # Điền giá trị 0 cho các cột chưa có
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    # Xử lý NaN và inf an toàn
    X = df[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], 50.0).fillna(0.0)
    return X


def load_labeled_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Nạp dữ liệu gán nhãn thực tế từ file CSV hoặc thư mục CSV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {p}")

    if p.is_dir():
        csv_files = list(p.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {p}")
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(p)

    print(f"Loaded dataset: {len(df)} rows from {p}")

    # Tìm cột nhãn (target / label / accident / status)
    label_col = None
    for cand in ("label", "target", "accident", "is_accident", "status"):
        if cand in df.columns:
            label_col = cand
            break

    if label_col is None:
        raise ValueError(
            f"CSV must contain a label column (candidates: 'label', 'target', 'accident', 'is_accident', 'status'). "
            f"Columns found: {list(df.columns)}"
        )

    print(f"Target column: '{label_col}'")

    # Chuẩn hóa nhãn thành 0 (Normal) và 1 (Accident)
    y_raw = df[label_col]
    if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
        # Chuyển đổi chuỗi: NORMAL -> 0, POSSIBLE ACCIDENT / ACCIDENT -> 1
        y = y_raw.astype(str).str.upper().str.strip().apply(
            lambda val: 1 if val in ("ACCIDENT", "POSSIBLE ACCIDENT", "CRASH", "1", "TRUE") else 0
        )
    else:
        y = (y_raw > 0).astype(int)

    X = prepare_features(df)
    return X, y


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    n_estimators: int = 300,
    max_depth: int = 12,
) -> RandomForestClassifier:
    """
    Huấn luyện Random Forest trên dữ liệu thật với phân tầng (stratified split),
    xử lý imbalanced dataset và in báo cáo đánh giá toàn diện.
    """
    class_counts = y.value_counts().to_dict()
    print("\nClass distribution:")
    print(f"  - Class 0 (NORMAL):   {class_counts.get(0, 0)} samples")
    print(f"  - Class 1 (ACCIDENT): {class_counts.get(1, 0)} samples")

    if class_counts.get(1, 0) == 0:
        raise ValueError("Dataset does not contain any ACCIDENT (class 1) samples!")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",  # Rất quan trọng khi tai nạn là sự kiện hiếm
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Đánh giá trên tập test độc lập
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ (DỮ LIỆU THỰC)")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=["BÌNH THƯỜNG", "TAI NẠN"], digits=4))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
    except Exception:
        pass

    # Top Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 10 đặc trưng quan trọng nhất:")
    for rank in range(min(10, len(indices))):
        idx = indices[rank]
        print(f"  {rank + 1:2d}. {FEATURE_COLUMNS[idx]:<28}: {importances[idx]:.4f}")

    # Huấn luyện lại trên toàn bộ dữ liệu thật trước khi lưu (optional production pattern)
    # hoặc lưu model đã kiểm chứng
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nMô hình đã lưu thành công tại: {output_path}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest accident classifier from real labeled data."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV file or directory containing real features and labels.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the output .pkl model file.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in Random Forest (default: 300).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Maximum depth of trees (default: 12).",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    csv_path = args.csv
    if csv_path is None:
        default_csv = project_root / "dataset" / "output_features.csv"
        if default_csv.exists():
            csv_path = str(default_csv)
            print(f"Using default real features CSV: {csv_path}")
        else:
            raise FileNotFoundError(
                "Please provide --csv path to real labeled features CSV file."
            )

    output_path = (
        Path(args.output)
        if args.output
        else project_root / "models" / "accident_rf_model.pkl"
    )

    print("=" * 60)
    print("ACCIVISION — HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI TAI NẠN")
    print("=" * 60)

    X, y = load_labeled_dataset(csv_path)
    train_and_evaluate(
        X=X,
        y=y,
        output_path=output_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
