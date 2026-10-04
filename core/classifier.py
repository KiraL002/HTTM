"""
AcciVision — Module Phân Loại Tai Nạn (Accident Classifier)

Module này chứa lớp AccidentClassifier sử dụng mô hình Random Forest
để phân loại xem các đặc trưng động học của phương tiện có phản ánh
tình huống tai nạn giao thông hay không.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


class AccidentClassifier:
    """
    Bộ phân loại tai nạn giao thông sử dụng Random Forest.

    Hỗ trợ schema feature mới (bao gồm single-vehicle và temporal features),
    đồng thời tương thích ngược với các model đã huấn luyện trước đó.
    """

    DEFAULT_FEATURE_NAMES = [
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

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.70,
        feature_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {self.model_path}"
            )

        self.threshold = float(threshold)
        self.model = joblib.load(self.model_path)

        # Tự động đồng bộ feature_names theo model đã huấn luyện nếu có
        model_columns = getattr(self.model, "feature_names_in_", None)
        if model_columns is not None:
            self.feature_names = [str(name) for name in model_columns]
        elif feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = list(self.DEFAULT_FEATURE_NAMES)

    def _prepare_input(
        self,
        features: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Chuẩn hóa dictionary features thành DataFrame đúng thứ tự cột của model,
        xử lý missing features và giá trị NaN/inf an toàn.
        """
        values: List[float] = []

        for name in self.feature_names:
            if name in features:
                val = features[name]
            # Hỗ trợ tương thích tên cũ
            elif name == "nearest_distance" and "nearest_distance_m" in features:
                val = features["nearest_distance_m"]
            elif name == "nearest_distance_m" and "nearest_distance" in features:
                val = features["nearest_distance"]
            elif name == "has_neighbor" and "neighbor_count" in features:
                val = 1.0 if float(features["neighbor_count"]) > 0 else 0.0
            elif "distance" in name:
                val = 50.0  # safe default cho khoảng cách
            else:
                val = 0.0

            try:
                num_val = float(val)
            except (ValueError, TypeError):
                num_val = 0.0

            # Thay thế NaN hoặc inf bằng giá trị hữu hạn
            if not np.isfinite(num_val):
                num_val = 50.0 if "distance" in name else 0.0

            values.append(num_val)

        return pd.DataFrame([values], columns=self.feature_names)

    def predict_probability(
        self,
        features: Dict[str, Any],
    ) -> float:
        """
        Tính xác suất xảy ra tai nạn (từ 0.0 đến 1.0).
        """
        X = self._prepare_input(features)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            classes = list(self.model.classes_)

            # 1. Tìm class ACCIDENT hoặc POSSIBLE ACCIDENT
            for index, cls in enumerate(classes):
                cls_str = str(cls).upper().strip()
                if cls_str in ("ACCIDENT", "POSSIBLE ACCIDENT", "CRASH", "1", "TRUE"):
                    return float(probabilities[0, index])

            # 2. Tìm nhãn 1 (int)
            if 1 in classes:
                index = classes.index(1)
                return float(probabilities[0, index])

            # 3. Nếu là nhãn binary [0, 1] dạng int/bool
            if len(classes) == 2:
                return float(probabilities[0, 1])

        # Fallback nếu model chỉ có predict()
        prediction = self.model.predict(X)
        pred_val = prediction[0]
        pred_str = str(pred_val).upper().strip()

        if pred_str in ("ACCIDENT", "POSSIBLE ACCIDENT", "1", "TRUE"):
            return 1.0
        if pred_val == 1:
            return 1.0

        return 0.0

    def predict(
        self,
        features: Dict[str, Any],
    ) -> str:
        probability = self.predict_probability(features)
        if probability >= self.threshold:
            return "ACCIDENT"
        return "NORMAL"

    def predict_with_probability(
        self,
        features: Dict[str, Any],
    ) -> Tuple[str, float]:
        probability = self.predict_probability(features)
        status = "ACCIDENT" if probability >= self.threshold else "NORMAL"
        return status, probability
