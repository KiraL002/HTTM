"""
AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh
Module chính: Pipeline xử lý đa tầng & Giao diện dòng lệnh (CLI)

Pipeline xử lý:
    OpenCV (đọc video)
      ↓
    YOLOv8 (phát hiện phương tiện) + ByteTrack (theo dõi đa đối tượng)
      ↓
    Quản lý quỹ đạo (Trajectory Management)
      ↓
    Chuyển đổi phối cảnh (Perspective Transform → Bird's Eye View)
      ↓
    Trích xuất đặc trưng động học (Kinematic Feature Extraction)
      ↓
    Phân loại tai nạn (Random Forest Classifier)
      ↓
    Phát hiện & quản lý sự kiện (Event Detection & Lifecycle)
      ↓
    Video kết quả + CSV đặc trưng
"""

from __future__ import annotations

import argparse
import base64
import csv
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import yaml

from core import (
    AccidentClassifier,
    ByteTrackTracker,
    FeatureExtractor,
    PerspectiveTransformer,
    TrajectoryManager,
)
from core.event_detector import EventDetector
from utils import (
    draw_detection,
    draw_fps,
    draw_speed,
    draw_status,
    draw_track_point,
    draw_trajectory,
    draw_world_position,
)


# ============================================================
# NẠP CẤU HÌNH (Configuration Loader)
# ============================================================

def load_config(config_path: str) -> dict:
    """
    Nạp tệp cấu hình YAML.

    Parameters
    ----------
    config_path : str
        Đường dẫn đến tệp config.yaml

    Returns
    -------
    dict
        Dictionary chứa cấu hình hệ thống
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            "Config file is empty."
        )

    return config


# ============================================================
# XỬ LÝ ĐƯỜNG DẪN (Path Resolution)
# ============================================================

def resolve_path(
    project_root: Path,
    configured_path: str,
) -> Path:
    """
    Giải quyết đường dẫn tương đối từ thư mục gốc dự án.
    """

    path = Path(configured_path)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_parent_directory(
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PIPELINE XỬ LÝ CHÍNH (Main Processing Pipeline)
# ============================================================

class TrafficAccidentPipeline:
    """
    Pipeline phát hiện tai nạn giao thông đầu-cuối (end-to-end).

    Quy trình xử lý:
        [1] Đọc video (OpenCV)
        [2] Phát hiện phương tiện (YOLOv8) + Theo dõi (ByteTrack)
        [3] Quản lý quỹ đạo chuyển động (Trajectory)
        [4] Chuyển đổi phối cảnh (Perspective Transform → BEV)
        [5] Trích xuất đặc trưng động học (Feature Extraction)
        [6] Phân loại tai nạn (Random Forest Classifier)
        [7] Phát hiện & quản lý sự kiện (Event Detection)
        [8] Xuất video kết quả + CSV đặc trưng
    """

    def __init__(
        self,
        config: dict,
        project_root: Optional[Path] = None,
    ) -> None:

        self.config = config

        self.project_root = (
            project_root
            if project_root is not None
            else Path(__file__).resolve().parent
        )
        # ----------------------------------------------------
        # Bộ phát hiện sự kiện tai nạn (Accident Event Detector)
        # ----------------------------------------------------

        classifier_thresh = float(
            config.get("classifier", {}).get("threshold", 0.75)
        )
        confirmation_frames = int(
            config.get("classifier", {}).get("confirmation_frames", 8)
        )

        self.event_detector = EventDetector(
            fps=30.0,
            start_threshold=classifier_thresh,
            confirmation_frames=confirmation_frames,
            release_threshold=0.35,
            release_frames=12,
            spatial_merge_distance=6.0,
            min_track_age=int(config.get("classifier", {}).get("min_track_age", 2)),
        )

        self.accident_detected = False
        self.accident_start_frame = None
        self.accident_probability = 0.0
        self.accident_track_ids = set()
        # ----------------------------------------------------
        # Cấu hình Video đầu vào/đầu ra
        # ----------------------------------------------------

        video_cfg = config["video"]

        self.video_input = resolve_path(
            self.project_root,
            video_cfg["input"],
        )

        self.video_output = resolve_path(
            self.project_root,
            video_cfg["output"],
        )

        self.features_csv = resolve_path(
            self.project_root,
            video_cfg["features_csv"],
        )

        # ----------------------------------------------------
        # Mô hình phát hiện (YOLO) & Theo dõi (ByteTrack)
        # ----------------------------------------------------

        yolo_cfg = config["yolo"]

        self.yolo_model_path = resolve_path(
            self.project_root,
            yolo_cfg["model"],
        )

        self.yolo_confidence = float(
            yolo_cfg.get(
                "confidence",
                0.25,
            )
        )

        self.yolo_iou = float(
            yolo_cfg.get(
                "iou",
                0.50,
            )
        )

        self.yolo_device = yolo_cfg.get(
            "device"
        )

        self.vehicle_classes = yolo_cfg.get(
            "classes"
        )

        tracker_cfg = config["tracker"]

        self.tracker_config = tracker_cfg.get(
            "config",
            "bytetrack.yaml",
        )

        # ----------------------------------------------------
        # Quản lý quỹ đạo (Trajectory Manager)
        # ----------------------------------------------------

        self.trajectory_manager = (
            TrajectoryManager(
                max_history=int(
                    tracker_cfg.get(
                        "max_history",
                        300,
                    )
                ),
                max_missed_frames=int(
                    tracker_cfg.get(
                        "max_missed_frames",
                        15,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # Chuyển đổi phối cảnh (Perspective Transform)
        # ----------------------------------------------------

        perspective_cfg = config["perspective"]

        self.perspective_enabled = bool(
            perspective_cfg.get(
                "enabled",
                True,
            )
        )

        self.transformer = None

        if self.perspective_enabled:

            source_points = np.asarray(
                perspective_cfg[
                    "source_points"
                ],
                dtype=np.float32,
            )

            destination_points = np.asarray(
                perspective_cfg[
                    "destination_points"
                ],
                dtype=np.float32,
            )

            self.transformer = (
                PerspectiveTransformer(
                    source_points=source_points,
                    destination_points=destination_points,
                    meters_per_pixel_x=float(
                        perspective_cfg[
                            "meters_per_pixel_x"
                        ]
                    ),
                    meters_per_pixel_y=float(
                        perspective_cfg[
                            "meters_per_pixel_y"
                        ]
                    ),
                    auto_scale=bool(
                        perspective_cfg.get(
                            "auto_scale",
                            True,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Trích xuất đặc trưng động học (Feature Extractor)
        # ----------------------------------------------------

        feature_cfg = config["features"]

        self.feature_extractor = (
            FeatureExtractor(
                fps=30.0,
                sudden_stop_deceleration=float(
                    feature_cfg.get(
                        "sudden_stop_deceleration",
                        4.5,
                    )
                ),
                direction_change_threshold=float(
                    feature_cfg.get(
                        "direction_change_threshold",
                        45.0,
                    )
                ),
                temporal_window=int(feature_cfg.get("temporal_window", 8)),
                velocity_smoothing_alpha=float(
                    feature_cfg.get("velocity_smoothing_alpha", 0.30)
                ),
            )
        )

        # ----------------------------------------------------
        # Mô hình phân loại tai nạn (Accident Classifier)
        # ----------------------------------------------------

        classifier_cfg = config[
            "classifier"
        ]

        self.classifier = None

        classifier_enabled = bool(
            classifier_cfg.get(
                "enabled",
                True,
            )
        )

        if classifier_enabled:

            classifier_path = resolve_path(
                self.project_root,
                classifier_cfg["model"],
            )

            if classifier_path.exists():

                try:

                    self.classifier = (
                        AccidentClassifier(
                            model_path=str(
                                classifier_path
                            ),
                            threshold=float(
                                classifier_cfg.get(
                                    "threshold",
                                    0.70,
                                )
                            ),
                        )
                    )

                except Exception as exc:

                    print(
                        "[WARNING] "
                        "Cannot load classifier: "
                        f"{exc}"
                    )

                    self.classifier = None

            else:

                print(
                    "[WARNING] "
                    f"Classifier not found: "
                    f"{classifier_path}"
                )

        # ----------------------------------------------------
        # Mô hình xác thực thị giác kép (Visual Verifier - MobileNetV3)
        # ----------------------------------------------------
        self.visual_verifier = None
        visual_cfg = config.get("visual_verification", {})
        if visual_cfg.get("enabled", True):
            try:
                from core.visual_classifier import VisualAccidentVerifier
                v_model_path = resolve_path(self.project_root, visual_cfg.get("model", "models/visual_accident_classifier.pth"))
                self.visual_verifier = VisualAccidentVerifier(
                    model_path=str(v_model_path),
                    threshold=float(visual_cfg.get("threshold", 0.40)),
                    device=self.yolo_device,
                )
            except Exception as e:
                print(f"[WARNING] Cannot init visual verifier: {e}")

        # ----------------------------------------------------
        # Trạng thái trước đó (Previous States)
        # Dùng để tính toán: vận tốc, gia tốc, hướng di chuyển
        # ----------------------------------------------------

        self.previous_positions: Dict[
            int,
            np.ndarray
        ] = {}

        self.previous_velocities: Dict[
            int,
            np.ndarray
        ] = {}

        self.previous_directions: Dict[
            int,
            float
        ] = {}

        self.previous_speeds: Dict[
            int,
            float
        ] = {}
        self.previous_bottom_centers: Dict[int, tuple] = {}

        # Lưu lịch sử tiếp xúc vật lý 2D+BEV đã xác thực gần nhất của từng phương tiện
        self.recent_contacts: Dict[int, Dict[str, Any]] = {}
        self._recent_raw_frames: deque = deque(maxlen=8)

        # Đếm frame hiện tại
        self.frame_id = 0
        self.fps = 30.0

        # Trạng thái phát hiện tai nạn tổng thể
        self.global_status = "BÌNH THƯỜNG"
        self.global_probability = 0.0

        # Các trường dữ liệu xuất CSV
        self.csv_fields = [
            "video_id",
            "frame_id",
            "timestamp",
            "track_id",
            "class_name",
            "object_type",
            "x_m",
            "y_m",
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
            "interpolated",
            "bbox_iou",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_aspect_ratio",
            "bbox_aspect_change_max",
            "bbox_area_change_ratio",
            "bbox_deformation",
            "accident_probability",
            "status",
        ]

    # ========================================================
    # KHỞI TẠO MÔ HÌNH (Model Initialization)
    # ========================================================

    def _initialize_models(self) -> None:

        self.tracker = ByteTrackTracker(
            model_path=str(
                self.yolo_model_path
            ),
            confidence=self.yolo_confidence,
            iou_threshold=self.yolo_iou,
            tracker_config=self.tracker_config,
            device=self.yolo_device,
            classes=self.vehicle_classes,
            vehicle_only=True,
        )

    # ========================================================
    # ĐỌC VIDEO ĐẦU VÀO (Video Input)
    # ========================================================

    def _open_video(self):

        if not self.video_input.exists():
            raise FileNotFoundError(
                f"Input video not found: "
                f"{self.video_input}"
            )

        cap = cv2.VideoCapture(
            str(self.video_input)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: "
                f"{self.video_input}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0 or not np.isfinite(fps):
            fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        return (
            cap,
            float(fps),
            width,
            height,
        )

    # ========================================================
    # XUẤT CSV ĐẶC TRƯNG (Feature CSV Output)
    # ========================================================

    def _open_csv(self):

        ensure_parent_directory(
            self.features_csv
        )

        file = self.features_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            file,
            fieldnames=self.csv_fields,
        )

        writer.writeheader()

        return file, writer

    # ========================================================
    # TỌA ĐỘ THỰC TẾ (World Positions)
    # ========================================================

    def _get_world_positions(
        self,
        tracks,
        frame_shape: Optional[tuple] = None,
    ) -> Dict[int, np.ndarray]:

        positions = {}

        if self.transformer is not None:
            fh = int(frame_shape[0]) if (frame_shape is not None and len(frame_shape) >= 2) else 0
            fw = int(frame_shape[1]) if (frame_shape is not None and len(frame_shape) >= 2) else 0
            self.transformer.calibrate_from_tracks(
                tracks=tracks,
                frame_width=fw,
                frame_height=fh,
            )

        for track in tracks:

            pixel_x, pixel_y = (
                track.bottom_center
            )

            if self.transformer is not None:

                world_x, world_y = (
                    self.transformer.image_to_world(
                        pixel_x,
                        pixel_y,
                    )
                )

            else:

                # Fallback chỉ để debug.
                # Đây là PIXEL chứ không phải meter.
                world_x = pixel_x
                world_y = pixel_y

            positions[
                track.track_id
            ] = np.asarray(
                [world_x, world_y],
                dtype=np.float64,
            )

        return positions

    # ========================================================
    # TRÍCH XUẤT ĐẶC TRƯNG & PHÂN LOẠI (Feature + Classification)
    # ========================================================

    @staticmethod
    def _infer_accident_type(feature) -> str:
        """Phân loại sơ bộ loại tai nạn dựa trên đặc trưng động học."""
        if feature.neighbor_count == 0:
            return "single_vehicle_loss_of_control" if feature.trajectory_change else "single_vehicle_sudden_stop"
        if feature.trajectory_change:
            return "side_collision"
        if feature.deceleration_mps2 >= 3.0:
            return "rear_end_collision"
        return "vehicle_collision"

    @staticmethod
    def _compute_2d_bbox_interactions(tracks):
        """
        Tính toán chi tiết tương tác hình học 2D giữa các phương tiện:
        - `iou`: IoU toàn khung bao 2D
        - `inner_iou`: IoU vùng thân trong (co lề 3% mỗi cạnh để loại nhiễu góc rỗng khung bao YOLO)
        - `footprint_iou`: IoU nửa dưới thân xe chạm mặt đường (50% chiều cao đáy)
        - `ground_y_diff_ratio`: Độ chênh đáy `|y2_a - y2_b| / mean_h` phát hiện chồng lấn phối cảnh khác làn
        """
        def _box_iou(a, b) -> float:
            ix1 = max(float(a[0]), float(b[0]))
            iy1 = max(float(a[1]), float(b[1]))
            ix2 = min(float(a[2]), float(b[2]))
            iy2 = min(float(a[3]), float(b[3]))
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0.0:
                return 0.0
            aa = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
            ab = max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
            denom = aa + ab - inter
            return float(inter / denom) if denom > 0.0 else 0.0

        interactions = {}
        for i, t1 in enumerate(tracks):
            b1 = t1.bbox
            w1 = max(1.0, float(b1[2] - b1[0]))
            h1 = max(1.0, float(b1[3] - b1[1]))
            ib1 = (
                float(b1[0]) + 0.03 * w1,
                float(b1[1]) + 0.03 * h1,
                float(b1[2]) - 0.03 * w1,
                float(b1[3]) - 0.03 * h1,
            )
            fb1 = (
                float(b1[0]) + 0.03 * w1,
                float(b1[3]) - 0.50 * h1,
                float(b1[2]) - 0.03 * w1,
                float(b1[3]),
            )
            c1 = ((float(b1[0]) + float(b1[2])) * 0.5, (float(b1[1]) + float(b1[3])) * 0.5)

            max_iou = 0.0
            best_inner_iou = 0.0
            best_footprint_iou = 0.0
            best_gy_diff = 0.0
            min_dist_px = 9999.0
            touching_tid = None
            partners = []

            for j, t2 in enumerate(tracks):
                if i == j:
                    continue
                b2 = t2.bbox
                w2 = max(1.0, float(b2[2] - b2[0]))
                h2 = max(1.0, float(b2[3] - b2[1]))
                iou = _box_iou(b1, b2)

                c2 = ((float(b2[0]) + float(b2[2])) * 0.5, (float(b2[1]) + float(b2[3])) * 0.5)
                d_px = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if d_px < min_dist_px:
                    min_dist_px = d_px

                dx_edge = max(0.0, max(float(b1[0]), float(b2[0])) - min(float(b1[2]), float(b2[2])))
                dy_edge = max(0.0, max(float(b1[1]), float(b2[1])) - min(float(b1[3]), float(b2[3])))
                edge_dist_px = (dx_edge ** 2 + dy_edge ** 2) ** 0.5

                cls1_str = str(t1.class_name).lower().strip()
                cls2_str = str(t2.class_name).lower().strip()
                is_pair_vru = any(
                    c in ("motorcycle", "motorbike", "bicycle", "bike", "person", "pedestrian")
                    for c in (cls1_str, cls2_str)
                )

                if iou > 0.0 or (is_pair_vru and edge_dist_px <= 18.0):
                    ib2 = (
                        float(b2[0]) + 0.03 * w2,
                        float(b2[1]) + 0.03 * h2,
                        float(b2[2]) - 0.03 * w2,
                        float(b2[3]) - 0.03 * h2,
                    )
                    fb2 = (
                        float(b2[0]) + 0.03 * w2,
                        float(b2[3]) - 0.50 * h2,
                        float(b2[2]) - 0.03 * w2,
                        float(b2[3]),
                    )
                    in_iou = _box_iou(ib1, ib2)
                    fp_iou = _box_iou(fb1, fb2)
                    gy_diff = abs(float(b1[3]) - float(b2[3])) / max(1.0, 0.5 * (h1 + h2))

                    t2_is_vru = 1 if cls2_str in (
                        "motorcycle", "motorbike", "bicycle", "bike", "person", "pedestrian"
                    ) and (iou >= 0.012 or edge_dist_px <= 18.0) else 0
                    partners.append(
                        {
                            "tid": t2.track_id,
                            "iou": float(iou),
                            "inner_iou": float(in_iou),
                            "footprint_iou": float(fp_iou),
                            "ground_y_diff_ratio": float(gy_diff),
                            "dist_px": float(d_px),
                            "edge_dist_px": float(edge_dist_px),
                            "is_vru": t2_is_vru,
                        }
                    )
                    if (in_iou, iou) > (best_inner_iou, max_iou):
                        max_iou = iou
                        best_inner_iou = in_iou
                        best_footprint_iou = fp_iou
                        best_gy_diff = gy_diff
                        touching_tid = t2.track_id

            partners.sort(key=lambda item: (item.get("is_vru", 0), item["inner_iou"], item["iou"]), reverse=True)
            interactions[t1.track_id] = {
                "max_iou": float(max_iou),
                "inner_iou": float(best_inner_iou),
                "footprint_iou": float(best_footprint_iou),
                "ground_y_diff_ratio": float(best_gy_diff),
                "min_dist_px": float(min_dist_px),
                "touching_tid": touching_tid,
                "partners": partners,
            }
        return interactions

    @staticmethod
    def _pair_closing_speed(
        pos1: np.ndarray,
        pos2: np.ndarray,
        vel1: Optional[np.ndarray],
        vel2: Optional[np.ndarray],
    ) -> float:
        """Tính vận tốc tiếp cận dọc trục nối tâm giữa 2 phương tiện."""
        if vel1 is None or vel2 is None:
            return 0.0
        rel_pos = np.asarray(pos2, dtype=np.float32) - np.asarray(pos1, dtype=np.float32)
        dist = float(np.linalg.norm(rel_pos))
        if dist <= 1e-4:
            return 0.0
        unit_vec = rel_pos / dist
        rel_vel = np.asarray(vel1, dtype=np.float32) - np.asarray(vel2, dtype=np.float32)
        return float(np.dot(rel_vel, unit_vec))

    def _process_track_features(
        self,
        tracks,
        world_positions,
        frame=None,
    ):

        frame_features = []

        max_probability = 0.0
        frame_status = "NORMAL"

        bbox_interactions = self._compute_2d_bbox_interactions(tracks)
        prev_velocities_snapshot = {
            tid: v.copy() for tid, v in self.previous_velocities.items()
        }

        # --------------------------------------------------------
        # BƯỚC 5A (Pass 1): Trích xuất động học & dự đoán RF thô cho toàn bộ phương tiện
        # --------------------------------------------------------
        track_by_id = {}
        feat_by_id = {}
        fdict_by_id = {}
        rf_prob_by_id = {}
        rf_status_by_id = {}
        cont_age_by_id = {}
        curr_vel_by_id = {}

        for track in tracks:
            tid = track.track_id
            track_by_id[tid] = track
            current_position = world_positions[tid]
            previous_position = self.previous_positions.get(tid)
            if tid in self.previous_bottom_centers:
                prev_px, prev_py = self.previous_bottom_centers[tid]
                curr_px, curr_py = track.bottom_center
                px_shift = ((float(curr_px) - float(prev_px)) ** 2 + (float(curr_py) - float(prev_py)) ** 2) ** 0.5
                if px_shift < max(1.6, 0.016 * float(track.width)) and self.previous_speeds.get(tid, 0.0) < 1.8:
                    # Ngưỡng khử rung sub-pixel: xe đứng yên không bị gán vận tốc ảo trên BEV
                    previous_position = current_position.copy()
                elif self.transformer is not None:
                    # Chiếu vị trí frame trước trên cùng hệ quy chiếu ma trận + tỷ lệ hiện tại để triệt tiêu trôi hệ trục
                    wx_prev, wy_prev = self.transformer.image_to_world(float(prev_px), float(prev_py))
                    previous_position = np.asarray([wx_prev, wy_prev], dtype=np.float64)
                    # Chặn phóng đại sai số Homography ở vùng xa gần đường chân trời
                    world_step = float(np.linalg.norm(current_position - previous_position))
                    max_phys_step = max(0.35, (px_shift / max(28.0, float(track.width))) * 2.6)
                    if world_step > max_phys_step and world_step > 1e-4:
                        direction_unit = (current_position - previous_position) / world_step
                        previous_position = current_position - direction_unit * max_phys_step
            previous_velocity = prev_velocities_snapshot.get(tid)
            previous_direction = self.previous_directions.get(tid)
            previous_speed = self.previous_speeds.get(tid, 0.0)

            feature = self.feature_extractor.extract(
                track_id=tid,
                current_position=current_position,
                previous_position=previous_position,
                previous_velocity=previous_velocity,
                previous_direction=previous_direction,
                previous_speed=previous_speed,
                all_vehicle_positions=world_positions,
                all_vehicle_velocities=prev_velocities_snapshot,
                object_type=track.object_type,
                bbox=track.bbox,
                frame_id=self.frame_id,
            )
            feat_by_id[tid] = feature
            cont_age_by_id[tid] = self.feature_extractor.get_continuous_age(tid)
            curr_vel_by_id[tid] = (
                self.feature_extractor.get_smoothed_velocity(tid)
                if previous_position is not None
                else np.zeros(2, dtype=np.float64)
            )

            inter = bbox_interactions.get(tid, {})
            fdict = feature.to_dict()
            fdict["bbox_iou"] = float(inter.get("max_iou", 0.0))
            fdict["inner_iou"] = float(inter.get("inner_iou", 0.0))
            fdict["footprint_iou"] = float(inter.get("footprint_iou", 0.0))
            fdict["bbox_x1"] = float(track.bbox[0])
            fdict["bbox_y1"] = float(track.bbox[1])
            fdict["bbox_x2"] = float(track.bbox[2])
            fdict["bbox_y2"] = float(track.bbox[3])
            fdict["continuous_age"] = int(cont_age_by_id[tid])
            fdict["track_confidence"] = float(track.confidence)
            fdict["visual_accident_tag"] = float(getattr(track, "visual_accident_tag", 0.0))
            fdict_by_id[tid] = fdict

            raw_prob = 0.0
            raw_status = "NORMAL"
            if self.classifier is not None:
                try:
                    raw_status, raw_prob = self.classifier.predict_with_probability(fdict)
                except Exception as exc:
                    print(f"\n[WARNING] Classifier error for track {tid}: {exc}")
                    raw_status = "CLASSIFIER_ERROR"
                    raw_prob = 0.0
            rf_prob_by_id[tid] = float(raw_prob)
            rf_status_by_id[tid] = raw_status

        # --------------------------------------------------------
        # BƯỚC 5B: Xác thực tiếp xúc vật lý 2D + BEV (Loại bỏ xe chưa chạm & xe khác làn)
        # --------------------------------------------------------
        recent_window = int(max(12, round(self.fps * 1.25)))
        for stale_tid in [
            k for k, v in self.recent_contacts.items()
            if (self.frame_id - int(v.get("last_contact_frame", 0))) > recent_window
        ]:
            self.recent_contacts.pop(stale_tid, None)

        verified_contacts_now: Dict[int, Dict[str, Any]] = {}
        for track in tracks:
            tid = track.track_id
            feat1 = feat_by_id[tid]
            pos1 = world_positions[tid]
            cls1 = str(track.class_name).lower().strip()
            max_spd1 = float(feat1.speed_mps + feat1.speed_drop_window_mps)

            for p_info in bbox_interactions.get(tid, {}).get("partners", []):
                ptid = p_info["tid"]
                if ptid not in track_by_id:
                    continue
                p_track = track_by_id[ptid]
                feat2 = feat_by_id[ptid]
                pos2 = world_positions[ptid]
                cls2 = str(p_track.class_name).lower().strip()

                iou_2d = float(p_info["iou"])
                in_iou = float(p_info["inner_iou"])
                fp_iou = float(p_info["footprint_iou"])
                gy_diff = float(p_info["ground_y_diff_ratio"])
                edge_dist_px = float(p_info.get("edge_dist_px", 0.0))
                pair_dist_m = float(np.linalg.norm(pos2 - pos1))

                if cls1 in ("person", "pedestrian") and cls2 in ("person", "pedestrian"):
                    continue

                any_vru = any(
                    c in ("person", "pedestrian", "bicycle", "bike", "motorcycle", "motorbike")
                    for c in (cls1, cls2)
                )
                any_large = any(c in ("truck", "bus", "lorry") for c in (cls1, cls2))

                closing_curr = self._pair_closing_speed(
                    pos1, pos2, curr_vel_by_id.get(tid), curr_vel_by_id.get(ptid)
                )
                closing_prev = self._pair_closing_speed(
                    pos1, pos2, prev_velocities_snapshot.get(tid), prev_velocities_snapshot.get(ptid)
                )
                pair_closing_mps = max(closing_curr, closing_prev)
                if pair_dist_m <= feat1.nearest_distance_m + 0.5:
                    pair_closing_mps = max(pair_closing_mps, float(feat1.closing_speed_mps))
                if pair_dist_m <= feat2.nearest_distance_m + 0.5:
                    pair_closing_mps = max(pair_closing_mps, float(feat2.closing_speed_mps))

                raw_2d_touch = (
                    (in_iou >= 0.005 and iou_2d >= 0.012)
                    or (any_vru and (iou_2d >= 0.008 or (edge_dist_px <= 18.0 and pair_closing_mps >= 2.5)))
                )
                if not raw_2d_touch:
                    continue

                max_spd2 = float(feat2.speed_mps + feat2.speed_drop_window_mps)
                peak_pair_speed = max(max_spd1, max_spd2)

                # Kiểm tra chồng lấn phối cảnh 2D giữa 2 xe đi song song khác làn hoặc xe chạy ngang qua xe đỗ
                angle_diff = None
                lat_off = 0.0
                if max_spd1 >= 1.2 and max_spd2 >= 1.2:
                    angle_diff = abs((feat1.direction_deg - feat2.direction_deg + 180.0) % 360.0 - 180.0)
                    ref_dir = feat1.direction_deg if max_spd1 >= max_spd2 else feat2.direction_deg
                    rad = np.radians(float(ref_dir))
                    ux, uy = float(np.cos(rad)), float(np.sin(rad))
                    dx, dy = float(pos2[0] - pos1[0]), float(pos2[1] - pos1[1])
                    lat_off = abs(-dx * uy + dy * ux)

                max_decel_now = max(float(feat1.deceleration_mps2), float(feat2.deceleration_mps2))
                max_drop_now = max(float(feat1.speed_drop_window_mps), float(feat2.speed_drop_window_mps))
                one_is_stationary = (
                    (feat1.speed_mean_mps < 0.85)
                    or (feat2.speed_mean_mps < 0.85)
                    or (feat1.speed_mps < 0.8 and feat1.speed_drop_window_mps < 1.0 and feat1.deceleration_mps2 < 1.5)
                    or (feat2.speed_mps < 0.8 and feat2.speed_drop_window_mps < 1.0 and feat2.deceleration_mps2 < 1.5)
                )

                is_adj_lane_overlap = False
                if one_is_stationary and (gy_diff > 0.35 or fp_iou < 0.05) and max_decel_now < 3.5 and max_drop_now < 2.5:
                    # Xe di chuyển lướt ngang qua xe đang đỗ bên đường ở hai lớp chiều sâu khác nhau
                    is_adj_lane_overlap = True
                elif gy_diff > 0.55 and fp_iou == 0.0 and max_decel_now < 3.5:
                    # Chồng lấn phối cảnh theo chiều sâu dọc đường (đáy xe xa chỉ chạm nóc xe gần, 2 nửa thân dưới không chạm nhau)
                    is_adj_lane_overlap = True
                elif angle_diff is not None and angle_diff <= 42.0:
                    max_dir_chg = max(feat1.direction_change_max_deg, feat2.direction_change_max_deg)
                    if lat_off > 1.85 and fp_iou < 0.03 and pair_closing_mps < 3.0 and max_dir_chg < 25.0:
                        is_adj_lane_overlap = True
                    elif gy_diff > 0.72 and fp_iou == 0.0 and pair_closing_mps < 2.2 and max_decel_now < 3.5:
                        is_adj_lane_overlap = True

                max_contact_dist = 6.2 if any_large else (3.8 if any_vru else 4.8)
                if is_adj_lane_overlap or pair_dist_m > max_contact_dist:
                    continue
                if max(feat1.speed_mps, feat2.speed_mps) < 0.8 and pair_closing_mps < 1.5 and peak_pair_speed < 2.2:
                    continue

                p_bbox_tuple = (
                    tuple(p_track.bbox.tolist())
                    if hasattr(p_track.bbox, "tolist")
                    else tuple(p_track.bbox)
                )
                contact_info = {
                    "partner_tid": ptid,
                    "partner_class": p_track.class_name,
                    "partner_dir": float(feat2.direction_deg),
                    "partner_pos": pos2.copy(),
                    "partner_bbox": p_bbox_tuple,
                    "pair_dist_m": pair_dist_m,
                    "pair_closing_mps": pair_closing_mps,
                    "peak_pair_speed": peak_pair_speed,
                    "partner_rf_prob": rf_prob_by_id.get(ptid, 0.0),
                    "any_vru": any_vru,
                }
                verified_contacts_now[tid] = contact_info

                if peak_pair_speed >= (1.8 if any_vru else 2.4):
                    prev_rc = self.recent_contacts.get(tid)
                    first_cf = (
                        int(prev_rc["first_contact_frame"])
                        if (prev_rc is not None and prev_rc.get("partner_tid") == ptid)
                        else self.frame_id
                    )
                    self.recent_contacts[tid] = {
                        "first_contact_frame": first_cf,
                        "last_contact_frame": self.frame_id,
                        "own_dir": (
                            float(prev_rc["own_dir"])
                            if (prev_rc is not None and prev_rc.get("partner_tid") == ptid)
                            else float(feat1.direction_deg)
                        ),
                        "partner_tid": ptid,
                        "partner_class": p_track.class_name,
                        "partner_dir": (
                            float(feat2.direction_deg)
                            if feat2.speed_mps >= 1.2 or prev_rc is None
                            else float(prev_rc.get("partner_dir", feat2.direction_deg))
                        ),
                        "partner_pos": pos2.copy(),
                        "partner_bbox": p_bbox_tuple,
                        "min_dist_m": min(
                            pair_dist_m,
                            float(prev_rc["min_dist_m"]) if prev_rc is not None else pair_dist_m,
                        ),
                        "peak_closing_spd": max(
                            pair_closing_mps,
                            float(prev_rc["peak_closing_spd"]) if prev_rc is not None else 0.0,
                        ),
                        "peak_speed": max(
                            peak_pair_speed,
                            float(prev_rc["peak_speed"]) if prev_rc is not None else 0.0,
                        ),
                        "partner_rf_prob": max(
                            rf_prob_by_id.get(ptid, 0.0),
                            float(prev_rc["partner_rf_prob"]) if prev_rc is not None else 0.0,
                        ),
                        "any_vru": any_vru,
                    }
                break

        # --------------------------------------------------------
        # BƯỚC 5C (Pass 2): Hợp nhất Vật lý - Động học - Thị giác & Phát hiện sự kiện
        # --------------------------------------------------------
        for track in tracks:
            track_id = track.track_id
            current_position = world_positions[track_id]
            previous_position = self.previous_positions.get(track_id)
            feature = feat_by_id[track_id]
            features_dict = fdict_by_id[track_id]
            cont_age = cont_age_by_id[track_id]

            raw_rf_prob = rf_prob_by_id[track_id]
            probability = raw_rf_prob
            status = rf_status_by_id[track_id]

            if self.classifier is not None:
                cls_lower = str(track.class_name).lower().strip()
                vis_acc_tag = float(getattr(track, "visual_accident_tag", 0.0))
                vc_now = verified_contacts_now.get(track_id)
                rc = self.recent_contacts.get(track_id)

                has_physical_contact = (vc_now is not None) or (rc is not None)
                has_kinematic_impact = False
                eff_closing_spd = float(feature.closing_speed_mps)
                partner_bbox_for_crop = None

                if has_physical_contact:
                    p_tid = vc_now["partner_tid"] if vc_now is not None else rc["partner_tid"]
                    p_feat = feat_by_id.get(p_tid)
                    any_vru = bool(
                        (vc_now and vc_now.get("any_vru"))
                        or (rc and rc.get("any_vru"))
                        or cls_lower in ("person", "pedestrian", "bicycle", "bike", "motorcycle", "motorbike")
                    )

                    own_max_spd = float(feature.speed_mps + feature.speed_drop_window_mps)
                    peak_pair_speed = max(
                        own_max_spd,
                        float(vc_now["peak_pair_speed"]) if vc_now is not None else 0.0,
                        float(rc["peak_speed"]) if rc is not None else 0.0,
                    )
                    eff_closing_spd = max(
                        float(vc_now["pair_closing_mps"]) if vc_now is not None else 0.0,
                        float(rc["peak_closing_spd"]) if rc is not None else 0.0,
                    )
                    partner_rf_prob = max(
                        float(p_feat and rf_prob_by_id.get(p_tid, 0.0) or 0.0),
                        float(vc_now["partner_rf_prob"]) if vc_now is not None else 0.0,
                        float(rc["partner_rf_prob"]) if rc is not None else 0.0,
                    )

                    max_decel = max(
                        float(feature.deceleration_mps2),
                        float(p_feat.deceleration_mps2) if p_feat is not None else 0.0,
                    )
                    max_speed_drop = max(
                        float(feature.speed_drop_window_mps),
                        float(p_feat.speed_drop_window_mps) if p_feat is not None else 0.0,
                    )
                    max_dir_change = max(
                        float(feature.direction_change_max_deg),
                        float(p_feat.direction_change_max_deg) if p_feat is not None else 0.0,
                    )
                    max_yaw_rate = max(
                        abs(float(feature.yaw_rate_dps)),
                        abs(float(p_feat.yaw_rate_dps)) if p_feat is not None else 0.0,
                    )
                    any_stopped = (
                        int(feature.stopped_after_motion) == 1
                        or (p_feat is not None and int(p_feat.stopped_after_motion) == 1)
                    )
                    any_deform = (
                        int(feature.bbox_deformation) == 1
                        or (p_feat is not None and int(p_feat.bbox_deformation) == 1)
                    )

                    min_motion_req = 1.8 if any_vru else 2.5
                    if peak_pair_speed >= min_motion_req:
                        has_kinematic_impact = (
                            (max_decel >= 3.5 and max_speed_drop >= 2.2)
                            or (any_stopped and max_speed_drop >= 2.2)
                            or (
                                eff_closing_spd >= 3.5
                                and (
                                    max_decel >= 2.5
                                    or max_speed_drop >= 1.8
                                    or max_dir_change >= 20.0
                                    or max_yaw_rate >= 45.0
                                    or raw_rf_prob >= 0.70
                                    or partner_rf_prob >= 0.70
                                )
                            )
                            or (any_deform and (max_speed_drop >= 1.8 or max_decel >= 2.5 or eff_closing_spd >= 2.5))
                            or (max_yaw_rate >= 65.0 and max_speed_drop >= 2.2)
                            or (
                                vis_acc_tag >= (0.42 if any_vru else 0.55)
                                and vc_now is not None
                                and (
                                    eff_closing_spd >= (1.8 if any_vru else 2.2)
                                    or max_decel >= (2.2 if any_vru else 2.8)
                                    or max_speed_drop >= (1.5 if any_vru else 2.0)
                                    or any_stopped
                                    or any_deform
                                )
                            )
                            or (
                                any_vru
                                and vc_now is not None
                                and (eff_closing_spd >= 2.2 or max_speed_drop >= 2.0 or max_decel >= 2.8)
                                and (max_dir_change >= 18.0 or max_yaw_rate >= 40.0 or any_stopped or any_deform or raw_rf_prob >= 0.65 or partner_rf_prob >= 0.65)
                            )
                        )

                    src_contact = vc_now if vc_now is not None else rc
                    partner_bbox_for_crop = src_contact.get("partner_bbox")
                    features_dict["contact_partner_tid"] = int(src_contact["partner_tid"])
                    features_dict["contact_partner_class"] = str(src_contact["partner_class"])
                    features_dict["contact_partner_dir"] = float(src_contact["partner_dir"])
                    features_dict["own_contact_dir"] = float(
                        rc["own_dir"] if (rc is not None and "own_dir" in rc) else feature.direction_deg
                    )
                    features_dict["contact_partner_pos"] = src_contact["partner_pos"]
                    features_dict["contact_partner_bbox"] = partner_bbox_for_crop
                    features_dict["first_contact_frame"] = int(
                        rc["first_contact_frame"] if rc is not None else self.frame_id
                    )
                    features_dict["last_contact_frame"] = int(
                        rc["last_contact_frame"] if rc is not None else self.frame_id
                    )

                features_dict["has_physical_contact"] = 1 if has_physical_contact else 0
                features_dict["has_kinematic_impact"] = 1 if has_kinematic_impact else 0

                is_two_wheeler = cls_lower in ("motorcycle", "motorbike", "bicycle", "bike")
                is_vru_cls = is_two_wheeler or cls_lower in ("person", "pedestrian")
                w_box = max(1.0, float(track.bbox[2] - track.bbox[0]))
                h_box = max(1.0, float(track.bbox[3] - track.bbox[1]))
                box_ar = w_box / h_box

                near_recent_crash_site = False
                if is_vru_cls and feature.speed_mps < 2.0:
                    cur_t_s = self.frame_id / max(1.0, self.fps)
                    for ev in self.event_detector.all_events():
                        if (cur_t_s - ev.end_time_s) <= 5.0:
                            crash_p = self.event_detector._event_crash_positions.get(ev.event_id)
                            if crash_p is not None:
                                if float(np.linalg.norm(current_position - crash_p)) <= 5.0:
                                    near_recent_crash_site = True
                                    break
                            for ex_tid, ex_box in ev.track_bboxes.items():
                                if ex_box is not None:
                                    dx_b = max(0.0, max(float(track.bbox[0]), float(ex_box[0])) - min(float(track.bbox[2]), float(ex_box[2])))
                                    dy_b = max(0.0, max(float(track.bbox[1]), float(ex_box[1])) - min(float(track.bbox[3]), float(ex_box[3])))
                                    if (dx_b**2 + dy_b**2)**0.5 <= 60.0:
                                        near_recent_crash_site = True
                                        break

                is_fallen_vru = (
                    (
                        (is_two_wheeler and box_ar >= 2.0 and feature.speed_mps < 1.8)
                        or (cls_lower in ("person", "pedestrian") and box_ar >= 0.72 and feature.speed_mps < 1.8)
                    )
                    and (
                        vis_acc_tag >= 0.35
                        or self.accident_detected
                        or feature.bbox_deformation == 1
                        or near_recent_crash_site
                    )
                ) or (
                    near_recent_crash_site
                    and (
                        vis_acc_tag >= 0.35
                        or is_two_wheeler
                        or (cls_lower in ("person", "pedestrian") and box_ar >= 0.65)
                    )
                )
                features_dict["is_fallen_vru"] = 1 if is_fallen_vru else 0

                is_pedestrian_cls = cls_lower in ("person", "pedestrian")
                is_severe_single_crash = (
                    not has_physical_contact
                    and (feature.has_neighbor == 0 or feature.nearest_distance_m > 4.5 or (is_vru_cls and vis_acc_tag >= 0.45))
                    and cont_age >= 6
                    and (
                        (
                            feature.stopped_after_motion == 1
                            and feature.deceleration_mps2 >= 5.2
                            and feature.speed_drop_window_mps >= 4.2
                            and feature.speed_mps < 1.5
                        )
                        or (
                            is_vru_cls
                            and feature.deceleration_mps2 >= 8.0
                            and feature.speed_drop_window_mps >= 4.5
                            and feature.speed_mps < 2.5
                        )
                        or (
                            not is_pedestrian_cls
                            and abs(feature.yaw_rate_dps) >= 95.0
                            and feature.speed_drop_window_mps >= 4.5
                            and feature.speed_mps < 1.8
                        )
                        or (
                            feature.bbox_deformation == 1
                            and is_two_wheeler
                            and feature.speed_drop_window_mps >= 2.8
                            and feature.speed_mps < 2.2
                        )
                        or (
                            is_vru_cls
                            and vis_acc_tag >= 0.45
                            and (feature.speed_mps < 2.2 or feature.stopped_after_motion == 1)
                            and (
                                feature.deceleration_mps2 >= 2.8
                                or feature.speed_drop_window_mps >= 2.0
                                or feature.bbox_deformation == 1
                                or abs(feature.yaw_rate_dps) >= 60.0
                            )
                        )
                    )
                )

                # 1. Nhãn tai nạn độc lập độ tự tin rất cao từ YOLO
                if (
                    cls_lower in ("accident", "car_car_accident", "car_person_accident", "injured")
                    and float(track.confidence) >= 0.78
                ):
                    probability = max(raw_rf_prob, 0.94)
                    status = "ACCIDENT"
                # 2. Hiện trường hậu quả: phương tiện / người ngã đổ tại điểm va chạm
                elif is_fallen_vru:
                    probability = max(raw_rf_prob, 0.89)
                    status = "ACCIDENT"
                # 3. Va chạm đa phương tiện ĐÃ TIẾP XÚC VẬT LÝ + CÓ XUNG LỰC ĐỘNG HỌC THỰC TẾ
                elif has_physical_contact and has_kinematic_impact:
                    probability = max(raw_rf_prob, 0.92 if vc_now is not None else 0.90)
                    status = "ACCIDENT"
                # 4. Sự cố đơn phương tiện nghiêm trọng (kể cả khi có xe khác không liên quan ở xa > 4.5m)
                elif is_severe_single_crash:
                    probability = max(raw_rf_prob, 0.86)
                    status = "ACCIDENT"
                # 5. Chặn 100% dự đoán sớm khi 2 xe CHƯA CHẠM, chạy song song khác làn, hoặc dừng đèn đỏ
                else:
                    probability = min(raw_rf_prob, 0.25)
                    status = "NORMAL"

                # 5. Xác thực kép kết hợp Thị giác (MobileNetV3 Visual Verifier)
                if (
                    cls_lower not in ("accident", "car_car_accident", "car_person_accident", "injured")
                    and self.visual_verifier is not None
                    and self.visual_verifier.enabled
                    and frame is not None
                    and probability >= 0.45
                ):
                    try:
                        crop_box = track.bbox
                        if partner_bbox_for_crop is not None:
                            crop_box = (
                                min(float(track.bbox[0]), float(partner_bbox_for_crop[0])),
                                min(float(track.bbox[1]), float(partner_bbox_for_crop[1])),
                                max(float(track.bbox[2]), float(partner_bbox_for_crop[2])),
                                max(float(track.bbox[3]), float(partner_bbox_for_crop[3])),
                            )
                        visual_prob = self.visual_verifier.verify_crop(frame, crop_box)
                        if has_physical_contact and has_kinematic_impact:
                            if visual_prob < 0.08 and eff_closing_spd < 4.5 and vis_acc_tag < 0.50:
                                probability = probability * 0.45
                                if probability < float(self.config.get("classifier", {}).get("threshold", 0.75)):
                                    status = "NORMAL"
                            elif visual_prob >= 0.65:
                                probability = max(probability, visual_prob)
                        else:
                            if (
                                not is_fallen_vru
                                and feature.deceleration_mps2 < 10.0
                                and visual_prob < 0.22
                                and vis_acc_tag < 0.45
                            ):
                                probability = probability * 0.35
                                if probability < float(self.config.get("classifier", {}).get("threshold", 0.75)):
                                    status = "NORMAL"
                            elif visual_prob >= 0.65:
                                probability = max(probability, visual_prob)
                    except Exception:
                        pass

                # ------------------------------------------------
                # Phát hiện sự kiện tai nạn (Event Detection)
                # ------------------------------------------------
                track_bbox_tuple = (
                    tuple(track.bbox.tolist())
                    if hasattr(track.bbox, "tolist")
                    else tuple(track.bbox)
                )

                event = self.event_detector.update(
                    track_id=track_id,
                    frame_id=self.frame_id,
                    probability=probability,
                    position=current_position,
                    speed=feature.speed_mps,
                    direction=feature.direction_deg,
                    vehicle_class=track.class_name,
                    feature_dict=features_dict,
                    bbox=track_bbox_tuple,
                )

                if event is not None:
                    self.accident_detected = True
                    self.accident_start_frame = event.start_frame
                    self.accident_probability = event.probability
                    self.accident_track_ids.update(event.track_ids)

            # ------------------------------------------------
            # Trạng thái tổng thể (Global Status)
            # ------------------------------------------------
            if probability > max_probability:
                max_probability = probability

            if status in ("ACCIDENT", "POSSIBLE ACCIDENT"):
                frame_status = "ACCIDENT"

            # ------------------------------------------------
            # Cập nhật trạng thái trước đó
            # ------------------------------------------------
            velocity = curr_vel_by_id[track_id]
            self.previous_positions[track_id] = current_position.copy()
            self.previous_bottom_centers[track_id] = tuple(track.bottom_center)
            self.previous_velocities[track_id] = velocity.copy()
            self.previous_directions[track_id] = feature.direction_deg
            self.previous_speeds[track_id] = feature.speed_mps

            # ------------------------------------------------
            # Dòng dữ liệu CSV
            # ------------------------------------------------

            row = {
                "video_id": self.video_input.stem,
                "frame_id": self.frame_id,
                "timestamp": (
                    self.frame_id
                    / self.fps
                ),
                "track_id": track_id,
                "class_name": track.class_name,
                "object_type": track.object_type,
                "x_m": feature.x_m,
                "y_m": feature.y_m,
                "speed_mps": (
                    feature.speed_mps
                ),
                "acceleration_mps2": (
                    feature.acceleration_mps2
                ),
                "deceleration_mps2": feature.deceleration_mps2,
                "speed_delta_mps": feature.speed_delta_mps,
                "direction_deg": (
                    feature.direction_deg
                ),
                "direction_change_deg": (
                    feature.direction_change_deg
                ),
                "yaw_rate_dps": feature.yaw_rate_dps,
                "lateral_acceleration_mps2": feature.lateral_acceleration_mps2,
                "jerk_mps3": feature.jerk_mps3,
                "nearest_distance_m": (
                    feature.nearest_distance_m
                ),
                "neighbor_count": feature.neighbor_count,
                "has_neighbor": feature.has_neighbor,
                "nearest_distance_valid": feature.nearest_distance_valid,
                "closing_speed_mps": feature.closing_speed_mps,
                "interaction_count": feature.interaction_count,
                "sudden_stop": (
                    feature.sudden_stop
                ),
                "trajectory_change": (
                    feature.trajectory_change
                ),
                "speed_drop_window_mps": feature.speed_drop_window_mps,
                "direction_change_max_deg": feature.direction_change_max_deg,
                "direction_change_sum_deg": feature.direction_change_sum_deg,
                "speed_mean_mps": feature.speed_mean_mps,
                "speed_std_mps": feature.speed_std_mps,
                "acceleration_std_mps2": feature.acceleration_std_mps2,
                "stopped_after_motion": feature.stopped_after_motion,
                "temporal_window": feature.temporal_window,
                "interpolated": 0,
                "bbox_iou": features_dict.get("bbox_iou", 0.0),
                "bbox_x1": features_dict.get("bbox_x1", 0.0),
                "bbox_y1": features_dict.get("bbox_y1", 0.0),
                "bbox_x2": features_dict.get("bbox_x2", 0.0),
                "bbox_y2": features_dict.get("bbox_y2", 0.0),
                "bbox_aspect_ratio": feature.bbox_aspect_ratio,
                "bbox_aspect_change_max": feature.bbox_aspect_change_max,
                "bbox_area_change_ratio": feature.bbox_area_change_ratio,
                "bbox_deformation": feature.bbox_deformation,
                "accident_probability": (
                    probability
                ),
                "status": status,
            }

            frame_features.append(
                (
                    track,
                    feature,
                    status,
                    probability,
                    row,
                )
            )

        active_events = self.event_detector.get_active_events()
        all_events = self.event_detector.all_events()
        if active_events or all_events:
            self.global_status = "ACCIDENT DETECTED"
            self.global_probability = max((e.probability for e in (active_events or all_events)), default=0.0)
        else:
            self.global_status = "NORMAL"
            self.global_probability = max_probability

        return frame_features

    # ========================================================
    # VẼ KẾT QUẢ LÊN FRAME (Drawing / Visualization)
    # ========================================================

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks,
        frame_features,
        processing_fps: float,
    ) -> np.ndarray:

        feature_by_id = {
            item[0].track_id: item
            for item in frame_features
        }

        output_cfg = self.config[
            "output"
        ]

        for track in tracks:

            item = feature_by_id.get(
                track.track_id
            )

            # Safety
            if item is None:
                continue

            (
                _track,
                feature,
                status,
                probability,
                _row,
            ) = item

            # ------------------------------------------------
            # Vẽ bounding box
            # ------------------------------------------------

            draw_detection(
                frame,
                bbox=track.bbox,
                label=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
                status=status,
            )

            # ------------------------------------------------
            # Vẽ điểm đáy (bottom center)
            # ------------------------------------------------

            draw_track_point(
                frame,
                track.bottom_center,
            )

            # ------------------------------------------------
            # Hiển thị tốc độ
            # ------------------------------------------------

            if output_cfg.get(
                "draw_speed",
                True,
            ):

                draw_speed(
                    frame,
                    position=(
                        int(track.x1),
                        int(track.y2 + 20),
                    ),
                    speed_mps=feature.speed_mps,
                )

            # ------------------------------------------------
            # Hiển thị tọa độ thực tế (mét)
            # ------------------------------------------------

            if (
                output_cfg.get(
                    "draw_world_position",
                    True,
                )
                and self.transformer is not None
            ):

                draw_world_position(
                    frame,
                    position=(
                        int(track.x1),
                        int(track.y2 + 40),
                    ),
                    world_x=feature.x_m,
                    world_y=feature.y_m,
                )

            # ------------------------------------------------
            # Vẽ quỹ đạo chuyển động
            # ------------------------------------------------

            if output_cfg.get(
                "draw_trajectory",
                True,
            ):

                trajectory = (
                    self.trajectory_manager
                    .get_points(
                        track.track_id
                    )
                )

                trajectory_length = (
                    int(
                        output_cfg.get(
                            "trajectory_length",
                            50,
                        )
                    )
                )

                if (
                    len(trajectory)
                    > trajectory_length
                ):

                    trajectory = (
                        trajectory[
                            -trajectory_length:
                        ]
                    )

                # Chỉ vẽ nếu đang dùng pixel.
                #
                # World coordinates không thể vẽ trực tiếp
                # lên ảnh camera.
                #
                # Do đó trajectory visualization ở đây
                # lấy lại pixel bottom-center từ bbox.
                #
                # Ta xây trajectory từ bbox points ở video
                # trong bản MVP này.

        # ----------------------------------------------------
        # Hiển thị trạng thái phát hiện
        # ----------------------------------------------------

        draw_status(
            frame,
            self.global_status,
            self.global_probability,
        )

        if output_cfg.get(
            "draw_fps",
            True,
        ):

            draw_fps(
                frame,
                processing_fps,
            )

        return frame

    # ========================================================
    # CHẠY PIPELINE (Run Pipeline)
    # ========================================================

    def run(
        self,
        progress_callback: Optional[
            Callable[[float], None]
        ] = None,
    ) -> dict:

        self._initialize_models()

        (
            cap,
            self.fps,
            width,
            height,
        ) = self._open_video()

        # Cập nhật FPS thực tế cho FeatureExtractor & EventDetector
        self.feature_extractor.fps = (
            self.fps
        )

        self.feature_extractor.dt = (
            1.0 / self.fps
        )

        self.event_detector.fps = (
            self.fps
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        output_cfg = self.config[
            "output"
        ]

        video_cfg = self.config[
            "video"
        ]

        writer = None

        if (
            output_cfg.get(
                "save_video",
                True,
            )
            and video_cfg.get(
                "save_output",
                True,
            )
        ):

            ensure_parent_directory(
                self.video_output
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                str(self.video_output),
                fourcc,
                self.fps,
                (width, height),
            )

            if not writer.isOpened():
                writer.release()

                raise RuntimeError(
                    "Cannot create output video."
                )

        csv_file = None
        csv_writer = None

        if output_cfg.get(
            "save_csv",
            True,
        ):

            csv_file, csv_writer = (
                self._open_csv()
            )

        processing_times = []

        try:

            while True:

                start_time = (
                    time.perf_counter()
                )

                success, frame = cap.read()

                if not success:
                    break

                self.frame_id += 1

                # ------------------------------------------------
                # Bước 1: Phát hiện & Theo dõi (YOLO + ByteTrack)
                # ------------------------------------------------

                tracks = self.tracker.update(
                    frame
                )

                # Lọc chỉ giữ phương tiện giao thông (vehicle filtering)
                tracks = [t for t in tracks if t.is_vehicle]

                active_ids = {
                    track.track_id
                    for track in tracks
                }

                # ------------------------------------------------
                # Bước 2: Tính tọa độ thực tế (World Coordinates)
                # ------------------------------------------------

                world_positions = (
                    self._get_world_positions(
                        tracks,
                        frame_shape=frame.shape,
                    )
                )

                # ------------------------------------------------
                # Bước 3: Cập nhật quỹ đạo (Trajectory Update)
                # ------------------------------------------------

                timestamp = (
                    self.frame_id
                    / self.fps
                )

                for track in tracks:

                    x, y = (
                        world_positions[
                            track.track_id
                        ]
                    )

                    self.trajectory_manager.update_track(
                        track_id=track.track_id,
                        frame_id=self.frame_id,
                        timestamp=timestamp,
                        x=float(x),
                        y=float(y),
                        confidence=track.confidence,
                    )

                # ------------------------------------------------
                # Bước 4: Đánh dấu track mất tích
                # ------------------------------------------------

                self.trajectory_manager.mark_missed(
                    active_track_ids=active_ids,
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                )

                # Lưu bản sao frame gốc chưa vẽ overlay để cắt ảnh hiện trường đúng thời điểm va chạm
                if frame is not None:
                    self._recent_raw_frames.append((self.frame_id, frame.copy()))

                # ------------------------------------------------
                # Bước 5: Trích xuất đặc trưng động học
                # ------------------------------------------------

                frame_features = (
                    self._process_track_features(
                        tracks,
                        world_positions,
                        frame=frame,
                    )
                )
                self.event_detector.end_frame(self.frame_id, set(active_ids))

                # ------------------------------------------------
                # Chụp & cập nhật ảnh cận cảnh hiện trường tai nạn
                # ------------------------------------------------
                active_events = self.event_detector.get_active_events()
                for ev in active_events:
                    ev_boxes = [t.bbox for t in tracks if t.track_id in ev.track_ids]
                    saved_boxes = [b for b in ev.track_bboxes.values() if b is not None]
                    union_boxes = saved_boxes if len(saved_boxes) > len(ev_boxes) else (ev_boxes or saved_boxes)
                    if union_boxes and frame is not None:
                        ux1 = min(float(b[0]) for b in union_boxes)
                        uy1 = min(float(b[1]) for b in union_boxes)
                        ux2 = max(float(b[2]) for b in union_boxes)
                        uy2 = max(float(b[3]) for b in union_boxes)

                        pad_x = max(50, int((ux2 - ux1) * 0.40))
                        pad_y = max(50, int((uy2 - uy1) * 0.45))
                        cx1 = max(0, int(ux1 - pad_x))
                        cy1 = max(0, int(uy1 - pad_y))
                        cx2 = min(frame.shape[1], int(ux2 + pad_x))
                        cy2 = min(frame.shape[0], int(uy2 + pad_y))

                        if (cx2 - cx1) > 20 and (cy2 - cy1) > 20:
                            if ev.snapshot_base64 is None or self.frame_id == ev.peak_frame:
                                snap_frame = frame
                                if ev.snapshot_base64 is None and ev.peak_frame != self.frame_id:
                                    for cached_fid, cached_img in self._recent_raw_frames:
                                        if cached_fid == ev.peak_frame:
                                            snap_frame = cached_img
                                            break
                                crop_img = snap_frame[cy1:cy2, cx1:cx2]
                                _, buf = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                ev.snapshot_base64 = base64.b64encode(buf).decode("utf-8")
                                ev.peak_bbox = (cx1, cy1, cx2, cy2)

                                snap_dir = self.project_root / "outputs" / "snapshots"
                                snap_dir.mkdir(parents=True, exist_ok=True)
                                snap_file = snap_dir / f"accident_{self.video_input.stem}_event_{ev.event_id}.jpg"
                                cv2.imwrite(str(snap_file), crop_img)
                                ev.crop_path = str(snap_file)

                # ------------------------------------------------
                # Bước 6: Ghi dữ liệu CSV
                # ------------------------------------------------

                if csv_writer is not None:

                    for item in frame_features:

                        row = item[4]

                        csv_writer.writerow(row)

                # ------------------------------------------------
                # Bước 7: Tính FPS xử lý
                # ------------------------------------------------

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                processing_fps = (
                    1.0 / elapsed
                    if elapsed > 0
                    else 0.0
                )

                processing_times.append(
                    elapsed
                )

                # ------------------------------------------------
                # Bước 8: Vẽ kết quả lên frame
                # ------------------------------------------------

                frame = self._draw_frame(
                    frame=frame,
                    tracks=tracks,
                    frame_features=frame_features,
                    processing_fps=processing_fps,
                )

                # ------------------------------------------------
                # Bước 9: Ghi video kết quả
                # ------------------------------------------------

                if writer is not None:
                    writer.write(frame)

                # ------------------------------------------------
                # Bước 10: Hiển thị cửa sổ OpenCV
                # ------------------------------------------------

                if video_cfg.get(
                    "display",
                    True,
                ):

                    cv2.imshow(
                        "AcciVision - Phat Hien Tai Nan Giao Thong",
                        frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                # ------------------------------------------------
                # Bước 11: Cập nhật tiến trình (Progress Callback)
                # ------------------------------------------------

                if (
                    progress_callback
                    is not None
                    and total_frames > 0
                    and (self.frame_id % 5 == 0 or self.frame_id == total_frames)
                ):

                    progress = min(
                        self.frame_id
                        / total_frames,
                        1.0,
                    )

                    progress_callback(
                        progress
                    )

        finally:

            cap.release()

            if writer is not None:
                writer.release()

            if csv_file is not None:
                csv_file.close()

            cv2.destroyAllWindows()

        self.event_detector.finish(self.frame_id)

        average_fps = 0.0

        if processing_times:

            average_time = (
                sum(processing_times)
                / len(processing_times)
            )

            if average_time > 0:
                average_fps = (
                    1.0 / average_time
                )

        confirmed_events = self.event_detector.all_events()
        has_accident = len(confirmed_events) > 0
        overall_status = "ACCIDENT DETECTED" if has_accident else "NORMAL"
        overall_verdict = "CÓ TAI NẠN" if has_accident else "KHÔNG CÓ TAI NẠN"
        overall_probability = max((e.probability for e in confirmed_events), default=0.0)

        return {
            "video_input": str(
                self.video_input
            ),
            "video_output": str(
                self.video_output
            ),
            "features_csv": str(
                self.features_csv
            ),
            "frames_processed": (
                self.frame_id
            ),
            "average_fps": average_fps,
            # Kết luận video có tai nạn hay không (Nhị phân)
            "has_accident": has_accident,
            "verdict": overall_verdict,
            "status": overall_status,
            "accident_probability": overall_probability,
            "events_count": len(confirmed_events),
            "events": [
                {
                    "event_id": e.event_id,
                    "accident_type": e.accident_type,
                    "type_title": getattr(e, "type_title", "💥 Va chạm giao thông"),
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "peak_frame": e.peak_frame,
                    "peak_time_s": round(e.peak_frame / self.fps, 2) if self.fps > 0 else 0.0,
                    "start_time_s": round(e.start_time_s, 2),
                    "end_time_s": round(e.end_time_s, 2),
                    "duration_s": round(e.duration_s, 2),
                    "track_ids": sorted(list(e.track_ids)),
                    "vehicle_types": e.vehicle_types,
                    "probability": round(e.probability, 4),
                    "active": e.active,
                    "description": e.description,
                    "snapshot_base64": getattr(e, "snapshot_base64", None),
                    "crop_path": getattr(e, "crop_path", None),
                }
                for e in confirmed_events
            ],
        }


# ============================================================
# GIAO DIỆN DÒNG LỆNH (Command Line Interface)
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AcciVision — Hệ Thống Phát Hiện "
            "Tai Nạn Giao Thông Thông Minh"
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Đường dẫn đến tệp cấu hình config.yaml",
    )

    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Đường dẫn video đầu vào (ghi đè video.input trong config.yaml)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Đường dẫn video đầu ra (ghi đè video.output trong config.yaml)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Tắt cửa sổ hiển thị OpenCV (chế độ server)",
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        args.config
    )

    if args.input:
        config["video"]["input"] = args.input

    if args.output:
        config["video"]["output"] = args.output

    if args.no_display:
        config["video"]["display"] = False

    pipeline = (
        TrafficAccidentPipeline(
            config=config,
            project_root=project_root,
        )
    )

    print("=" * 60)
    print(
        "ACCIVISION — PHÁT HIỆN TAI NẠN GIAO THÔNG THÔNG MINH"
    )
    print("=" * 60)

    print(
        f"Video đầu vào : {pipeline.video_input}"
    )

    print(
        f"Mô hình YOLO  : {pipeline.yolo_model_path}"
    )

    print("=" * 60)

    result = pipeline.run()

    print()
    print("=" * 60)
    print("KẾT QUẢ PHÂN TÍCH VIDEO")
    print("=" * 60)
    verdict_str = "CÓ TAI NẠN (ACCIDENT DETECTED)" if result["has_accident"] else "KHÔNG CÓ TAI NẠN (NORMAL)"
    print(f"📌 KẾT LUẬN VIDEO   : {verdict_str}")
    print(f"🚨 Số vụ tai nạn    : {result['events_count']}")
    print(f"🎞️ Tổng số frame    : {result['frames_processed']}")
    print(f"⚡ Tốc độ xử lý     : {result['average_fps']:.1f} FPS")
    print(f"📁 Video đầu ra     : {result['video_output']}")
    print(f"📊 CSV đặc trưng    : {result['features_csv']}")
    if result["has_accident"]:
        print("-" * 60)
        print("CHI TIẾT CÁC VỤ TAI NẠN:")
        for idx, ev in enumerate(result.get("events", []), 1):
            print(
                f"  [{idx}] Loại: {ev['accident_type']} | "
                f"Thời gian: {ev['start_time_s']}s -> {ev['end_time_s']}s "
                f"(Thời lượng: {ev['duration_s']}s) | Xe ID: {list(ev['vehicle_types'].keys())}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
