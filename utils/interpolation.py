"""
AcciVision — Module Nội Suy Quỹ Đạo (Trajectory Interpolation)

Module này cung cấp các hàm nội suy tuyến tính cho quỹ đạo
chuyển động khi phương tiện bị mất phát hiện tạm thời.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def interpolate_nan_1d(
    values: np.ndarray,
    max_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Nội suy tuyến tính các giá trị NaN trong mảng 1 chiều.

    Ví dụ:

        [100, NaN, NaN, 130]

    =>

        [100, 110, 120, 130]

    Parameters
    ----------
    values:
        Mảng 1 chiều.

    max_gap:
        Số lượng phần tử NaN liên tiếp tối đa được phép nội suy.
        Nếu None: nội suy tất cả các gap nằm giữa 2 giá trị hợp lệ.

    Returns
    -------
    np.ndarray
        Mảng sau nội suy.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be a 1D array."
        )

    result = values.copy()

    valid_mask = np.isfinite(result)

    # Không có dữ liệu hợp lệ
    if not np.any(valid_mask):
        return result

    valid_indices = np.flatnonzero(
        valid_mask
    )

    # Chỉ có đúng một điểm hợp lệ
    if len(valid_indices) == 1:
        return result

    all_indices = np.arange(
        len(result)
    )

    missing_indices = np.flatnonzero(
        ~valid_mask
    )

    # Nội suy tạm thời toàn bộ
    interpolated = np.interp(
        missing_indices,
        valid_indices,
        result[valid_indices],
    )

    for index, value in zip(
        missing_indices,
        interpolated,
    ):

        # Không cho phép nội suy ngoài khoảng
        # có dữ liệu thật.
        if (
            index < valid_indices[0]
            or index > valid_indices[-1]
        ):
            continue

        left_valid = (
            valid_indices[
                valid_indices < index
            ]
        )

        right_valid = (
            valid_indices[
                valid_indices > index
            ]
        )

        if (
            len(left_valid) == 0
            or len(right_valid) == 0
        ):
            continue

        left_index = left_valid[-1]
        right_index = right_valid[0]

        gap = (
            right_index
            - left_index
            - 1
        )

        if (
            max_gap is not None
            and gap > max_gap
        ):
            continue

        result[index] = value

    return result


def interpolate_positions(
    positions: np.ndarray,
    max_gap: Optional[int] = 15,
) -> np.ndarray:
    """
    Nội suy trajectory dạng:

        shape = (T, 2)

    với:

        [:, 0] = x
        [:, 1] = y

    Ví dụ:

        [
            [100, 200],
            [nan, nan],
            [nan, nan],
            [130, 200]
        ]

    =>

        [
            [100, 200],
            [110, 200],
            [120, 200],
            [130, 200]
        ]

    Parameters
    ----------
    positions:
        Array shape (T, 2).

    max_gap:
        Gap tối đa được phép nội suy.
    """

    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    if positions.ndim != 2:
        raise ValueError(
            "positions must be a 2D array."
        )

    if positions.shape[1] != 2:
        raise ValueError(
            "positions must have shape (T, 2)."
        )

    result = positions.copy()

    result[:, 0] = interpolate_nan_1d(
        result[:, 0],
        max_gap=max_gap,
    )

    result[:, 1] = interpolate_nan_1d(
        result[:, 1],
        max_gap=max_gap,
    )

    return result


def interpolate_trajectory(
    frame_ids: np.ndarray,
    positions: np.ndarray,
    max_gap: int = 15,
) -> np.ndarray:
    """
    Nội suy trajectory khi frame_id có thể không liên tục.

    Parameters
    ----------
    frame_ids:
        Mảng frame IDs, shape (T,).

    positions:
        Tọa độ x/y, shape (T, 2).

    max_gap:
        Khoảng cách frame tối đa được nội suy.

    Returns
    -------
    np.ndarray
        positions sau nội suy.
    """

    frame_ids = np.asarray(
        frame_ids,
        dtype=np.int64,
    )

    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    if len(frame_ids) != len(positions):
        raise ValueError(
            "frame_ids and positions "
            "must have the same length."
        )

    result = positions.copy()

    for axis in range(2):

        values = result[:, axis]

        valid = np.isfinite(values)

        valid_indices = np.flatnonzero(valid)

        if len(valid_indices) < 2:
            continue

        for i in range(
            len(valid_indices) - 1
        ):

            left = valid_indices[i]
            right = valid_indices[i + 1]

            frame_gap = (
                frame_ids[right]
                - frame_ids[left]
                - 1
            )

            if frame_gap <= 0:
                continue

            if frame_gap > max_gap:
                continue

            y1 = values[left]
            y2 = values[right]

            f1 = frame_ids[left]
            f2 = frame_ids[right]

            missing = np.arange(
                f1 + 1,
                f2,
            )

            alpha = (
                missing - f1
            ) / (
                f2 - f1
            )

            result[
                left + 1:right,
                axis
            ] = (
                y1
                + alpha * (y2 - y1)
            )

    return result


def count_missing(
    values: np.ndarray,
) -> int:
    """
    Đếm số lượng NaN / non-finite.
    """

    values = np.asarray(values)

    return int(
        np.count_nonzero(
            ~np.isfinite(values)
        )
    )


def has_large_gap(
    values: np.ndarray,
    max_gap: int,
) -> bool:
    """
    Kiểm tra trajectory có gap lớn hơn
    max_gap hay không.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be 1D."
        )

    valid = np.isfinite(values)

    missing_indices = np.flatnonzero(
        ~valid
    )

    if len(missing_indices) == 0:
        return False

    groups = np.split(
        missing_indices,
        np.where(
            np.diff(missing_indices) != 1
        )[0] + 1,
    )

    for group in groups:

        if len(group) > max_gap:
            return True

    return False