"""
AcciVision — Module Phân Loại Tai Nạn (Accident Classifier)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .feature_pipeline import RF_FEATURE_COLUMNS


class AccidentClassifier:
    DEFAULT_FEATURE_NAMES = list(RF_FEATURE_COLUMNS)

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.75,
        feature_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {self.model_path}"
            )

        self.threshold = float(threshold)
        self.model = joblib.load(self.model_path)

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
        values: List[float] = []

        for name in self.feature_names:
            if name in features:
                val = features[name]
            elif name == "nearest_distance" and "nearest_distance_m" in features:
                val = features["nearest_distance_m"]
            elif name == "nearest_distance_m" and "nearest_distance" in features:
                val = features["nearest_distance"]
            elif name == "has_neighbor" and "neighbor_count" in features:
                val = 1.0 if float(features["neighbor_count"]) > 0 else 0.0
            elif "distance" in name:
                val = 50.0
            else:
                val = 0.0

            try:
                num_val = float(val)
            except (ValueError, TypeError):
                num_val = 0.0

            if not np.isfinite(num_val):
                num_val = 50.0 if "distance" in name else 0.0

            values.append(num_val)

        return pd.DataFrame([values], columns=self.feature_names)

    def predict_probability(
        self,
        features: Dict[str, Any],
    ) -> float:
        X = self._prepare_input(features)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            classes = list(self.model.classes_)

            for index, cls in enumerate(classes):
                cls_str = str(cls).upper().strip()
                if cls_str in ("ACCIDENT", "POSSIBLE ACCIDENT", "CRASH", "1", "TRUE"):
                    return float(probabilities[0, index])

            if 1 in classes:
                index = classes.index(1)
                return float(probabilities[0, index])

            if len(classes) == 2:
                return float(probabilities[0, 1])

        prediction = self.model.predict(X)
        pred_val = prediction[0]
        pred_str = str(pred_val).upper().strip()

        if pred_str in ("ACCIDENT", "POSSIBLE ACCIDENT", "1", "TRUE") or pred_val == 1:
            return 1.0

        return 0.0

    def predict(
        self,
        features: Dict[str, Any],
    ) -> str:
        probability = self.predict_probability(features)
        return "ACCIDENT" if probability >= self.threshold else "NORMAL"

    def predict_with_probability(
        self,
        features: Dict[str, Any],
    ) -> Tuple[str, float]:
        probability = self.predict_probability(features)
        status = "ACCIDENT" if probability >= self.threshold else "NORMAL"
        return status, probability

    def predict_many_with_probability(self, features):
        """Same predictions in track order, with a single forest call per frame."""
        if not features:
            return []
        if not hasattr(self.model, "predict_proba"):
            return [self.predict_with_probability(row) for row in features]
        frame = pd.concat([self._prepare_input(row) for row in features], ignore_index=True)
        values = self.model.predict_proba(frame)
        classes = list(self.model.classes_)
        positive = next((i for i, label in enumerate(classes)
                         if str(label).upper().strip() in
                         ("ACCIDENT", "POSSIBLE ACCIDENT", "CRASH", "1", "TRUE")), None)
        if positive is None and len(classes) == 2:
            positive = 1
        if positive is None:
            return [self.predict_with_probability(row) for row in features]
        return [("ACCIDENT" if float(p) >= self.threshold else "NORMAL", float(p))
                for p in values[:, positive]]
