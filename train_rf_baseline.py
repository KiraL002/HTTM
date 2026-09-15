from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "speed_mps",
    "acceleration_mps2",
    "direction_deg",
    "direction_change_deg",
    "nearest_distance_m",
    "sudden_stop",
    "trajectory_change",
]


def create_synthetic_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:

    rng = np.random.default_rng(
        random_state
    )

    speed = rng.uniform(
        0.0,
        35.0,
        n_samples,
    )

    acceleration = rng.normal(
        0.0,
        2.5,
        n_samples,
    )

    direction = rng.uniform(
        -180.0,
        180.0,
        n_samples,
    )

    direction_change = np.abs(
        rng.normal(
            5.0,
            15.0,
            n_samples,
        )
    )

    nearest_distance = rng.uniform(
        1.0,
        60.0,
        n_samples,
    )

    sudden_stop = rng.binomial(
        1,
        0.08,
        n_samples,
    )

    trajectory_change = rng.binomial(
        1,
        0.10,
        n_samples,
    )

    # ---------------------------------------------------------
    # CHỈ LÀ RULE TẠO DATA GIẢ ĐỂ TEST PIPELINE.
    # KHÔNG PHẢI ground truth thực tế.
    # ---------------------------------------------------------

    risk_score = (
        0.8 * (speed > 20.0)
        + 1.3 * (acceleration < -4.0)
        + 1.5 * (nearest_distance < 5.0)
        + 1.7 * sudden_stop
        + 1.0 * trajectory_change
        + 1.0 * (direction_change > 45.0)
        + rng.normal(
            0.0,
            0.25,
            n_samples,
        )
    )

    labels = (
        risk_score >= 2.0
    ).astype(int)

    X = pd.DataFrame(
        {
            "speed_mps": speed,
            "acceleration_mps2": acceleration,
            "direction_deg": direction,
            "direction_change_deg": direction_change,
            "nearest_distance_m": nearest_distance,
            "sudden_stop": sudden_stop,
            "trajectory_change": trajectory_change,
        }
    )

    return X, labels


def main() -> None:

    X, y = create_synthetic_dataset()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X[FEATURE_COLUMNS],
        y,
    )

    output_path = (
        Path(__file__).resolve().parent
        / "models"
        / "accident_rf_model.pkl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )

    print(
        f"Model saved to: {output_path}"
    )

    print(
        "Features:"
    )

    for name in FEATURE_COLUMNS:
        print(
            f"  - {name}"
        )

    print(
        "\nWARNING:"
    )

    print(
        "This is a synthetic baseline model "
        "for pipeline testing only."
    )

    print(
        "It must NOT be used as the final "
        "accident detector."
    )


if __name__ == "__main__":
    main()