"""
AcciVision — Module Chuyển Đổi Phối Cảnh & Tự Động Hiệu Chỉnh Tỷ Lệ (Perspective Transformer & Auto-Scale)

Module này chuyển đổi tọa độ pixel từ ảnh camera sang tọa độ thực tế (mét)
thông qua Bird's Eye View (BEV) sử dụng phép biến đổi phối cảnh OpenCV,
kết hợp cơ chế tự động hiệu chỉnh tỷ lệ (Auto-Scale) dựa trên độ phân giải
khung hình và kích thước vật lý thực tế của phương tiện.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# Kích thước chiều rộng quy chiếu trung bình (mét) trên mặt phẳng đường của các lớp phương tiện
REFERENCE_VEHICLE_WIDTH_M: Dict[str, float] = {
    "car": 2.05,
    "motorcycle": 0.90,
    "bicycle": 0.75,
    "bus": 2.75,
    "truck": 2.65,
    "person": 0.60,
}


class PerspectiveTransformer:
    """
    Camera image -> Bird's Eye View -> Real-world meters.
    Hỗ trợ:
    1. Tự động co giãn 4 điểm phối cảnh (source_points) theo độ phân giải thực của video.
    2. Chống suy biến điểm tụ (horizon singularity clamping) khi xe ở xa gần đường chân trời.
    3. Tự động chuẩn hóa tỷ lệ mét/pixel (Auto-Scale) dựa trên kích thước thực tế của phương tiện.
    """

    def __init__(
        self,
        source_points: np.ndarray,
        destination_points: np.ndarray,
        meters_per_pixel_x: float = 0.05,
        meters_per_pixel_y: float = 0.05,
        auto_scale: bool = True,
        reference_width: int = 1280,
        reference_height: int = 720,
    ) -> None:
        self.base_source_points = self._validate_points(source_points)
        self.source_points = self.base_source_points.copy()
        self.destination_points = self._validate_points(destination_points)

        self.meters_per_pixel_x = float(meters_per_pixel_x)
        self.meters_per_pixel_y = float(meters_per_pixel_y)

        self.auto_scale = bool(auto_scale)
        self.reference_width = int(max(320, reference_width))
        self.reference_height = int(max(240, reference_height))
        self._adapted_resolution: Optional[Tuple[int, int]] = None

        # Hệ số hiệu chỉnh tỷ lệ tự động (EMA)
        self.auto_scale_factor: float = 1.0
        self._scale_ema_alpha: float = 0.08
        self._scale_samples_count: int = 0

        self.matrix = cv2.getPerspectiveTransform(
            self.source_points,
            self.destination_points,
        )

    @staticmethod
    def _validate_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("Perspective points must have shape (4, 2).")
        return points

    def adapt_to_frame_size(self, frame_width: int, frame_height: int) -> None:
        """
        Tự động điều chỉnh 4 điểm source_points khi độ phân giải khung hình thực tế
        khác với độ phân giải gốc lúc cấu hình (mặc định 1280x720).
        """
        if frame_width <= 0 or frame_height <= 0:
            return
        res = (int(frame_width), int(frame_height))
        if self._adapted_resolution == res:
            return

        # Nếu tọa độ cấu hình vượt ngoài kích thước frame hiện tại hoặc chênh lệch độ phân giải > 10%
        max_cfg_x = float(np.max(self.base_source_points[:, 0]))
        max_cfg_y = float(np.max(self.base_source_points[:, 1]))

        ref_w = float(self.reference_width)
        ref_h = float(self.reference_height)
        base_w = max(ref_w, max_cfg_x)
        base_h = max(ref_h, max_cfg_y)
        if (
            abs(float(frame_width) - base_w) / base_w > 0.03
            or abs(float(frame_height) - base_h) / base_h > 0.03
            or max_cfg_x > frame_width * 1.02
            or max_cfg_y > frame_height * 1.02
        ):
            scale_x = float(frame_width) / base_w
            scale_y = float(frame_height) / base_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        scaled_src = self.base_source_points.copy()
        scaled_src[:, 0] *= scale_x
        scaled_src[:, 1] *= scale_y

        self.source_points = scaled_src.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(
            self.source_points,
            self.destination_points,
        )
        self._adapted_resolution = res

    def _transform_point_safe(self, x: float, y: float) -> Tuple[float, float]:
        """
        Biến đổi phối cảnh an toàn có chống suy biến gần đường chân trời (w -> 0).
        """
        H = self.matrix
        u = float(H[0, 0] * x + H[0, 1] * y + H[0, 2])
        v = float(H[1, 0] * x + H[1, 1] * y + H[1, 2])
        w = float(H[2, 0] * x + H[2, 1] * y + H[2, 2])

        # Tránh chia cho số gần 0 hoặc âm khi xe ở phía trên đường chân trời của ma trận H
        if w < 0.18:
            w = 0.18

        bev_x = float(np.clip(u / w, -2500.0, 2500.0))
        bev_y = float(np.clip(v / w, -2500.0, 2500.0))
        return bev_x, bev_y

    def transform_point(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        return self._transform_point_safe(x, y)

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2).")

        out = np.empty_like(points, dtype=np.float32)
        for i in range(points.shape[0]):
            out[i, 0], out[i, 1] = self._transform_point_safe(
                float(points[i, 0]),
                float(points[i, 1]),
            )
        return out

    def pixel_to_meter(
        self,
        bev_x: float,
        bev_y: float,
        apply_auto_scale: bool = True,
    ) -> Tuple[float, float]:
        scale = self.auto_scale_factor if (self.auto_scale and apply_auto_scale) else 1.0
        real_x = bev_x * self.meters_per_pixel_x * scale
        real_y = bev_y * self.meters_per_pixel_y * scale
        return float(real_x), float(real_y)

    def image_to_world(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        bev_x, bev_y = self.transform_point(x, y)
        return self.pixel_to_meter(bev_x, bev_y, apply_auto_scale=True)

    def calibrate_from_tracks(
        self,
        tracks: Iterable,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> float:
        """
        Tự động ước lượng và cập nhật hệ số tỷ lệ `auto_scale_factor` từ kích thước
        bounding box của các phương tiện trong khung hình.
        Giúp hệ thống tính toán khoảng cách/vận tốc ổn định khi đổi góc camera.
        """
        if frame_width > 0 and frame_height > 0:
            self.adapt_to_frame_size(frame_width, frame_height)

        if not self.auto_scale:
            return self.auto_scale_factor

        ratios: List[float] = []
        for track in tracks:
            cls_name = str(getattr(track, "class_name", "")).lower().strip()
            obj_type = str(getattr(track, "object_type", "")).lower().strip()
            ref_width = REFERENCE_VEHICLE_WIDTH_M.get(
                cls_name,
                REFERENCE_VEHICLE_WIDTH_M.get(obj_type, 0.0),
            )
            if ref_width <= 0.0:
                continue

            bbox = getattr(track, "bbox", None)
            if bbox is None or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            w_px = x2 - x1
            h_px = y2 - y1
            if w_px < 22.0 or h_px < 18.0:
                continue

            # Chỉ lấy các phương tiện nằm trong vùng mặt đường tin cậy (không bị cắt ở sát mép)
            if frame_height > 0 and (y2 < frame_height * 0.22 or y2 > frame_height * 0.96):
                continue

            bx1, by1 = self._transform_point_safe(x1, y2)
            bx2, by2 = self._transform_point_safe(x2, y2)
            mx1, my1 = self.pixel_to_meter(bx1, by1, apply_auto_scale=False)
            mx2, my2 = self.pixel_to_meter(bx2, by2, apply_auto_scale=False)

            bev_span_m = float(np.hypot(mx2 - mx1, my2 - my1))
            if 0.25 <= bev_span_m <= 15.0:
                ratio = ref_width / bev_span_m
                # Chỉ điều chỉnh khi tỷ lệ BEV lệch đáng kể ngoài dải chuẩn [0.65, 1.45]
                if 0.65 <= ratio <= 1.45:
                    ratios.append(1.0)
                else:
                    ratios.append(float(np.clip(ratio, 0.45, 2.20)))

        if ratios:
            median_ratio = float(np.median(ratios))
            if self._scale_samples_count == 0:
                self.auto_scale_factor = float(np.clip(median_ratio, 0.50, 2.00))
            elif self._scale_samples_count < 3:
                self.auto_scale_factor = float(
                    np.clip(
                        (1.0 - self._scale_ema_alpha) * self.auto_scale_factor
                        + self._scale_ema_alpha * median_ratio,
                        0.50,
                        2.00,
                    )
                )
            self._scale_samples_count += 1

        return self.auto_scale_factor

    def warp_frame(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp toàn bộ frame sang Bird's Eye View.
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
        """
        meters_per_pixel_x = width_meters / output_width_pixels
        meters_per_pixel_y = height_meters / output_height_pixels

        destination_points = np.array(
            [
                [0, 0],
                [output_width_pixels, 0],
                [output_width_pixels, output_height_pixels],
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
