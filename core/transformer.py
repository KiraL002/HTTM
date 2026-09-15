from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class PerspectiveTransformer:
    """
    Camera image -> Bird's Eye View -> Real-world meters
    """

    def __init__(
        self,
        source_points: np.ndarray,
        destination_points: np.ndarray,
        meters_per_pixel_x: float = 0.05,
        meters_per_pixel_y: float = 0.05,
    ) -> None:

        self.source_points = self._validate_points(
            source_points
        )

        self.destination_points = self._validate_points(
            destination_points
        )

        self.meters_per_pixel_x = (
            float(meters_per_pixel_x)
        )

        self.meters_per_pixel_y = (
            float(meters_per_pixel_y)
        )

        self.matrix = cv2.getPerspectiveTransform(
            self.source_points,
            self.destination_points,
        )

    @staticmethod
    def _validate_points(
        points: np.ndarray,
    ) -> np.ndarray:

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if points.shape != (4, 2):
            raise ValueError(
                "Perspective points must have shape (4, 2)."
            )

        return points

    def transform_point(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:

        point = np.array(
            [[[x, y]]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.matrix,
        )

        new_x = float(
            transformed[0, 0, 0]
        )

        new_y = float(
            transformed[0, 0, 1]
        )

        return new_x, new_y

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "points must have shape (N, 2)."
            )

        reshaped = points.reshape(
            -1,
            1,
            2,
        )

        transformed = cv2.perspectiveTransform(
            reshaped,
            self.matrix,
        )

        return transformed.reshape(
            -1,
            2,
        )

    def pixel_to_meter(
        self,
        bev_x: float,
        bev_y: float,
    ) -> Tuple[float, float]:

        real_x = (
            bev_x
            * self.meters_per_pixel_x
        )

        real_y = (
            bev_y
            * self.meters_per_pixel_y
        )

        return real_x, real_y

    def image_to_world(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:

        bev_x, bev_y = self.transform_point(
            x,
            y,
        )

        return self.pixel_to_meter(
            bev_x,
            bev_y,
        )

    def warp_frame(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp toàn bộ frame sang Bird's Eye View.

        output_size:
            (width, height)
        """

        width, height = output_size

        return cv2.warpPerspective(
            frame,
            self.matrix,
            (width, height),
        )

    @classmethod
    def from_real_dimensions(
        cls,
        source_points: np.ndarray,
        width_meters: float,
        height_meters: float,
        output_width_pixels: int = 1000,
        output_height_pixels: int = 1000,
    ) -> "PerspectiveTransformer":
        """
        Tạo BEV từ kích thước thực tế.

        Ví dụ:
            đoạn đường:
                width = 20 m
                height = 40 m
        """

        meters_per_pixel_x = (
            width_meters
            / output_width_pixels
        )

        meters_per_pixel_y = (
            height_meters
            / output_height_pixels
        )

        destination_points = np.array(
            [
                [0, 0],
                [output_width_pixels, 0],
                [
                    output_width_pixels,
                    output_height_pixels,
                ],
                [0, output_height_pixels],
            ],
            dtype=np.float32,
        )

        return cls(
            source_points=source_points,
            destination_points=destination_points,
            meters_per_pixel_x=meters_per_pixel_x,
            meters_per_pixel_y=meters_per_pixel_y,
        )