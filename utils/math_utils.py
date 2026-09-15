from __future__ import annotations

from typing import Tuple

import numpy as np


EPSILON = 1e-8


def euclidean_distance(
    point_a: np.ndarray | Tuple[float, float],
    point_b: np.ndarray | Tuple[float, float],
) -> float:
    """
    Khoảng cách Euclidean giữa 2 điểm.

    Ví dụ:
        A = (10, 20)
        B = (13, 24)

        distance = 5
    """

    a = np.asarray(
        point_a,
        dtype=np.float64,
    )

    b = np.asarray(
        point_b,
        dtype=np.float64,
    )

    if a.shape != b.shape:
        raise ValueError(
            "point_a and point_b "
            "must have the same shape."
        )

    return float(
        np.linalg.norm(a - b)
    )


def displacement(
    previous_position: np.ndarray,
    current_position: np.ndarray,
) -> np.ndarray:
    """
    Vector dịch chuyển.
    """

    previous_position = np.asarray(
        previous_position,
        dtype=np.float64,
    )

    current_position = np.asarray(
        current_position,
        dtype=np.float64,
    )

    return (
        current_position
        - previous_position
    )


def calculate_velocity(
    previous_position: np.ndarray,
    current_position: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    v = dx / dt

    Position nên ở đơn vị meter.

    Kết quả:
        m/s
    """

    if dt <= 0:
        raise ValueError(
            "dt must be > 0."
        )

    dx = displacement(
        previous_position,
        current_position,
    )

    return dx / dt


def calculate_speed(
    velocity: np.ndarray,
) -> float:
    """
    |v|
    """

    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(velocity)
    )


def calculate_acceleration(
    previous_velocity: np.ndarray,
    current_velocity: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    a = dv / dt

    Kết quả:
        m/s^2
    """

    if dt <= 0:
        raise ValueError(
            "dt must be > 0."
        )

    previous_velocity = np.asarray(
        previous_velocity,
        dtype=np.float64,
    )

    current_velocity = np.asarray(
        current_velocity,
        dtype=np.float64,
    )

    return (
        current_velocity
        - previous_velocity
    ) / dt


def calculate_acceleration_magnitude(
    previous_velocity: np.ndarray,
    current_velocity: np.ndarray,
    dt: float,
) -> float:
    """
    Độ lớn vector gia tốc.
    """

    acceleration = calculate_acceleration(
        previous_velocity,
        current_velocity,
        dt,
    )

    return float(
        np.linalg.norm(acceleration)
    )


def vector_to_angle(
    vector: np.ndarray,
) -> float:
    """
    Chuyển vector 2D thành góc độ.

    atan2(y, x)

    Giá trị:
        [-180, 180]
    """

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    if vector.shape != (2,):
        raise ValueError(
            "vector must have shape (2,)."
        )

    if (
        np.linalg.norm(vector)
        < EPSILON
    ):
        return 0.0

    angle = np.degrees(
        np.arctan2(
            vector[1],
            vector[0],
        )
    )

    return float(angle)


def normalize_angle(
    angle_deg: float,
) -> float:
    """
    Đưa góc về [-180, 180].
    """

    return float(
        (angle_deg + 180.0)
        % 360.0
        - 180.0
    )


def angle_difference(
    angle_a: float,
    angle_b: float,
) -> float:
    """
    Sai khác góc nhỏ nhất.

    Ví dụ:

        angle_a = 179
        angle_b = -179

        difference = 2 độ

    thay vì 358 độ.
    """

    difference = (
        angle_a
        - angle_b
    )

    return abs(
        normalize_angle(
            difference
        )
    )


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Chia an toàn tránh ZeroDivisionError.
    """

    if abs(denominator) < EPSILON:
        return default

    return float(
        numerator / denominator
    )


def moving_average(
    values: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Moving average đơn giản.

    Dùng để giảm nhiễu speed/acceleration.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be 1D."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be > 0."
        )

    if len(values) == 0:
        return values.copy()

    kernel = np.ones(
        window_size
    ) / window_size

    padded = np.pad(
        values,
        (
            window_size // 2,
            window_size // 2,
        ),
        mode="edge",
    )

    result = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    return result[:len(values)]


def angle_between_vectors(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Góc giữa 2 vector theo đơn vị degree.
    """

    a = np.asarray(
        vector_a,
        dtype=np.float64,
    )

    b = np.asarray(
        vector_b,
        dtype=np.float64,
    )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if (
        norm_a < EPSILON
        or norm_b < EPSILON
    ):
        return 0.0

    cosine = np.dot(a, b) / (
        norm_a * norm_b
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )