from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np


class AccidentClassifier:
    """
    Wrapper cho Random Forest classifier.

    Input:
        feature vector dạng 1D.

    Output:
        NORMAL / ACCIDENT
        probability
    """

    DEFAULT_FEATURE_NAMES = [
        "speed_mps",
        "acceleration_mps2",
        "direction_deg",
        "direction_change_deg",
        "nearest_distance_m",
        "sudden_stop",
        "trajectory_change",
    ]

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.70,
        feature_names: Optional[
            Iterable[str]
        ] = None,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: "
                f"{self.model_path}"
            )

        self.threshold = float(threshold)

        self.feature_names = list(
            feature_names
            if feature_names is not None
            else self.DEFAULT_FEATURE_NAMES
        )

        self.model = joblib.load(
            self.model_path
        )

    def _prepare_input(
        self,
        features: dict[str, float],
    ) -> np.ndarray:

        values = []

        for name in self.feature_names:

            if name not in features:
                raise KeyError(
                    f"Missing feature: {name}"
                )

            values.append(
                float(features[name])
            )

        return np.asarray(
            values,
            dtype=np.float32,
        ).reshape(1, -1)

    def predict_probability(
        self,
        features: dict[str, float],
    ) -> float:

        X = self._prepare_input(
            features
        )

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model.predict_proba(X)
            )

            classes = list(
                self.model.classes_
            )

            # Tìm probability của class ACCIDENT.
            for index, cls in enumerate(
                classes
            ):

                if str(cls).upper() == "ACCIDENT":

                    return float(
                        probabilities[
                            0,
                            index
                        ]
                    )

            # Trường hợp model dùng label 1
            if 1 in classes:

                index = classes.index(1)

                return float(
                    probabilities[
                        0,
                        index
                    ]
                )

        # Fallback nếu model không có predict_proba
        prediction = self.model.predict(X)

        predicted = prediction[0]

        if str(predicted).upper() == "ACCIDENT":
            return 1.0

        if predicted == 1:
            return 1.0

        return 0.0

    def predict(
        self,
        features: dict[str, float],
    ) -> str:

        probability = (
            self.predict_probability(
                features
            )
        )

        if probability >= self.threshold:
            return "POSSIBLE ACCIDENT"

        return "NORMAL"

    def predict_with_probability(
        self,
        features: dict[str, float],
    ) -> tuple[str, float]:

        probability = (
            self.predict_probability(
                features
            )
        )

        status = (
            "POSSIBLE ACCIDENT"
            if probability >= self.threshold
            else "NORMAL"
        )

        return status, probability