from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


def draw_detection(
    frame: np.ndarray,
    bbox: Iterable[float],
    label: str,
    confidence: Optional[float] = None,
    track_id: Optional[int] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Vẽ bounding box + label + track ID.
    """

    x1, y1, x2, y2 = map(
        int,
        bbox,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        thickness,
    )

    parts = []

    if track_id is not None:
        parts.append(
            f"ID:{track_id}"
        )

    parts.append(label)

    if confidence is not None:
        parts.append(
            f"{confidence:.2f}"
        )

    text = " | ".join(parts)

    text_y = max(
        y1 - 10,
        20,
    )

    cv2.putText(
        frame,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_track_point(
    frame: np.ndarray,
    point: tuple[float, float],
    radius: int = 4,
) -> np.ndarray:
    """
    Vẽ bottom-center point.
    """

    x, y = map(
        int,
        point,
    )

    cv2.circle(
        frame,
        (x, y),
        radius,
        (0, 0, 255),
        -1,
    )

    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """
    Vẽ trajectory.

    points:
        shape (T, 2)

    Có thể chứa NaN.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2:
        raise ValueError(
            "points must be 2D."
        )

    if points.shape[1] != 2:
        raise ValueError(
            "points must have shape (T, 2)."
        )

    previous = None

    for point in points:

        if not np.all(
            np.isfinite(point)
        ):
            previous = None
            continue

        current = tuple(
            map(
                int,
                point,
            )
        )

        if previous is not None:

            cv2.line(
                frame,
                previous,
                current,
                (255, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        previous = current

    return frame


def draw_speed(
    frame: np.ndarray,
    position: tuple[float, float],
    speed_mps: float,
) -> np.ndarray:
    """
    Vẽ tốc độ.
    """

    x, y = map(
        int,
        position,
    )

    text = (
        f"Speed: "
        f"{speed_mps:.2f} m/s"
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_world_position(
    frame: np.ndarray,
    position: tuple[float, float],
    world_x: float,
    world_y: float,
) -> np.ndarray:
    """
    Hiển thị tọa độ thực tế trong BEV.
    """

    x, y = map(
        int,
        position,
    )

    text = (
        f"X={world_x:.2f}m "
        f"Y={world_y:.2f}m"
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_status(
    frame: np.ndarray,
    status: str,
    probability: Optional[float] = None,
) -> np.ndarray:
    """
    Vẽ trạng thái tổng thể.
    """

    if probability is None:

        text = f"STATUS: {status}"

    else:

        text = (
            f"STATUS: {status} | "
            f"ACCIDENT: "
            f"{probability * 100:.1f}%"
        )

    # OpenCV BGR.
    # Màu đỏ cho cảnh báo.
    color = (
        (0, 0, 255)
        if "ACCIDENT" in status.upper()
        else (0, 255, 0)
    )

    cv2.rectangle(
        frame,
        (20, 20),
        (550, 65),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_fps(
    frame: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Vẽ FPS thực tế.
    """

    text = f"FPS: {fps:.1f}"

    cv2.putText(
        frame,
        text,
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame