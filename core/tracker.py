"""
AcciVision — Module Theo Dõi Đa Đối Tượng (Multi-Object Tracker)

Module này kết hợp YOLO11/YOLOv8 và ByteTrack để theo dõi liên tục
các phương tiện giao thông qua nhiều frame video, gán mã định danh
phiên theo dõi (Track ID) cho từng phương tiện, đồng thời khử trùng lặp
khung bao đa lớp (Cross-Class NMS) để tránh tạo 2 track ảo trên cùng 1 xe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ultralytics import YOLO


# Bảng chuẩn hóa loại đối tượng
OBJECT_TYPE_MAP: Dict[str, str] = {
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "bicycle": "two_wheeler",
    "bike": "vehicle",
    "person": "pedestrian",
    "human": "pedestrian",
    "accident": "accident",
    "car_car_accident": "accident",
    "car_person_accident": "accident",
    "injured": "accident",
}

DEFAULT_VEHICLE_CLASSES: Set[str] = {
    "car", "motorcycle", "bus", "truck", "bike", "bicycle", "vehicle",
    "accident", "car_car_accident", "car_person_accident", "injured",
    "person", "human",
}

PHYSICAL_VEHICLE_NAMES: Set[str] = {
    "car", "motorcycle", "bus", "truck", "bike", "bicycle", "person", "human",
}

ACCIDENT_CLASS_NAMES: Set[str] = {
    "accident", "car_car_accident", "car_person_accident", "injured",
}


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Tính IoU và tỷ lệ bao hàm giữa 2 khung bao 2D."""
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(1.0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    return float(inter / (area_a + area_b - inter))


def _bbox_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Tỷ lệ diện tích của `inner` nằm bên trong `outer`."""
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_inner = max(1.0, (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1])))
    return float(inter / area_inner)


@dataclass
class Track:
    """
    Một đối tượng (object) đang được tracking.
    """

    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    object_type: str = "vehicle"
    visual_accident_tag: float = 0.0

    @property
    def is_vehicle(self) -> bool:
        """Kiểm tra đối tượng có phải là đối tượng giao thông cần theo dõi."""
        return self.object_type in ("vehicle", "two_wheeler", "pedestrian", "accident")

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Điểm giữa cạnh dưới bounding box (tiếp xúc mặt đường)."""
        x = (self.x1 + self.x2) / 2.0
        y = self.y2
        return x, y

    @property
    def center(self) -> Tuple[float, float]:
        """Tâm bounding box."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class ByteTrackTracker:
    """
    YOLO + ByteTrack có tích hợp bộ lọc khử trùng lặp đa lớp (Cross-Class Deduplication).
    Ngăn chặn hiện tượng 1 phương tiện vừa bị gán nhãn `car`, vừa bị gán nhãn `motorcycle`
    hoặc `accident` đè lên nhau tạo ra va chạm ảo.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        vehicle_class_names: Optional[List[str]] = None,
        vehicle_only: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.tracker_config = tracker_config
        self.device = device
        self.classes = classes
        self.vehicle_only = bool(vehicle_only)

        if vehicle_class_names:
            self.vehicle_class_names = {x.lower().strip() for x in vehicle_class_names}
        else:
            self.vehicle_class_names = set(DEFAULT_VEHICLE_CLASSES)

        self.model = YOLO(model_path)
        self._track_class_scores: Dict[int, Dict[str, float]] = {}
        self._aux_to_primary_tid: Dict[int, int] = {}
        self._primary_to_aux_tid: Dict[int, int] = {}
        self._vru_aux_model: Optional[YOLO] = None

        model_names_lower = {str(v).lower().strip() for v in getattr(self.model, "names", {}).values()}
        if "accident" in model_names_lower or any("accident" in n for n in model_names_lower):
            self.classes = None
            self.vehicle_only = False
            aux_candidate = Path(model_path).resolve().parent / "yolo11n.pt"
            if aux_candidate.exists() and aux_candidate.name.lower() != Path(model_path).name.lower():
                try:
                    self._vru_aux_model = YOLO(str(aux_candidate))
                except Exception:
                    self._vru_aux_model = None

    @staticmethod
    def _deduplicate_tracks(raw_tracks: List[Track]) -> List[Track]:
        """
        Khử trùng lặp giữa các khung bao trên cùng một đối tượng:
        1. Giữa các phương tiện vật lý (ví dụ 1 xe ô tô bị nhận diện kép thành cả `car` và `motorcycle`
           với IoU >= 0.55 hoặc độ bao hàm >= 0.80): chỉ giữ lại track có độ tin cậy cao hơn.
        2. Đối với lớp `accident` (khung bao vùng tai nạn của YOLO):
           - Nếu khung `accident` đè lên các xe vật lý đã được nhận diện (`IoU >= 0.25`),
             gắn `visual_accident_tag` vào xe đó và KHÔNG tạo thêm track xe ảo thứ 2
             (tránh làm `nearest_distance_m = 0m` trên xe đang chạy bình thường).
           - Chỉ giữ lại track `accident` độc lập khi hiện trường biến dạng nặng không còn
             nhận diện được khung xe thông thường (`max_iou < 0.25`) và `confidence >= 0.55`.
        """
        phys_tracks: List[Track] = []
        acc_tracks: List[Track] = []

        for t in raw_tracks:
            cname = t.class_name.lower().strip()
            if cname in ACCIDENT_CLASS_NAMES:
                acc_tracks.append(t)
            else:
                phys_tracks.append(t)

        # Bước 1: Khử trùng lặp Cross-Class giữa các phương tiện vật lý (sắp xếp theo confidence giảm dần)
        phys_sorted = sorted(phys_tracks, key=lambda x: x.confidence, reverse=True)
        kept_phys: List[Track] = []
        for cand in phys_sorted:
            is_dup = False
            for existing in kept_phys:
                if cand.track_id == existing.track_id:
                    is_dup = True
                    break
                iou = _bbox_iou(cand.bbox, existing.bbox)
                cont_1 = _bbox_containment(cand.bbox, existing.bbox)
                cont_2 = _bbox_containment(existing.bbox, cand.bbox)
                # Nếu 2 box khác lớp (vd car & motorcycle) nằm đè khít lên nhau trên cùng 1 xe
                if iou >= 0.55 or (cand.class_name != existing.class_name and max(cont_1, cont_2) >= 0.82):
                    is_dup = True
                    break
            if not is_dup:
                kept_phys.append(cand)

        # Bước 2: Xử lý các khung bao `accident`
        standalone_acc: List[Track] = []
        for acc_t in acc_tracks:
            acc_area = max(1.0, acc_t.width * acc_t.height)
            overlapping_vehicles: List[Track] = []
            sum_veh_area = 0.0
            for pt in kept_phys:
                iou = _bbox_iou(acc_t.bbox, pt.bbox)
                cont = _bbox_containment(pt.bbox, acc_t.bbox)
                if iou >= 0.25 or cont >= 0.65:
                    overlapping_vehicles.append(pt)
                    sum_veh_area += max(1.0, pt.width * pt.height)

            if overlapping_vehicles:
                has_vru_overlap = any(
                    pt.class_name.lower() in ("motorcycle", "bicycle", "bike", "person", "pedestrian")
                    for pt in overlapping_vehicles
                )
                area_mult = 6.0 if has_vru_overlap else 3.0
                min_acc_conf = 0.42 if has_vru_overlap else 0.52
                # Bỏ qua các khung `accident` quá rộng bao trùm cả đoàn xe chờ đèn đỏ
                if acc_t.confidence >= min_acc_conf and acc_area <= sum_veh_area * area_mult and len(overlapping_vehicles) <= 3:
                    for pt in overlapping_vehicles:
                        pt.visual_accident_tag = max(pt.visual_accident_tag, float(acc_t.confidence))
            else:
                # Hiện trường tai nạn mà YOLO không bắt được khung xe rời (yêu cầu độ tin cậy rất cao để tránh bắt nhầm cảnh vật)
                if acc_t.confidence >= 0.78:
                    standalone_acc.append(acc_t)

        return kept_phys + standalone_acc

    def update(
        self,
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Nhận một frame và trả về danh sách tracks đã được chuẩn hóa object_type
        và khử trùng lặp đa lớp.
        """
        kwargs = {
            "source": frame,
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        if self.classes is not None:
            kwargs["classes"] = self.classes

        results = self.model.track(**kwargs)

        raw_tracks: List[Track] = []
        frame_h, frame_w = frame.shape[:2]

        if results and results[0].boxes is not None and results[0].boxes.is_track:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox, confidence, class_id, track_id in zip(
                boxes,
                confidences,
                class_ids,
                track_ids,
            ):
                tid_int = int(track_id)
                raw_class_name = str(result.names[int(class_id)])
                lower_name = raw_class_name.lower().strip()

                bw = float(bbox[2] - bbox[0])
                bh = float(bbox[3] - bbox[1])
                # Chống nhận diện nhầm xe ô tô lật ngửa / cột đèn lớn thành xe máy ở góc camera giao lộ
                if lower_name in ("motorcycle", "bicycle", "bike"):
                    if bw >= frame_w * 0.12 and (bw / max(1.0, bh)) >= 1.38 and (bw * bh >= frame_w * frame_h * 0.015):
                        raw_class_name = "car"
                        lower_name = "car"

                if lower_name in PHYSICAL_VEHICLE_NAMES:
                    scores = self._track_class_scores.setdefault(tid_int, {})
                    weight = float(confidence) * (1.35 if lower_name in ("car", "truck", "bus") else 1.0)
                    scores[lower_name] = scores.get(lower_name, 0.0) + weight
                    best_cls = max(scores.items(), key=lambda kv: kv[1])[0]
                    raw_class_name = best_cls
                    lower_name = best_cls

                object_type = OBJECT_TYPE_MAP.get(lower_name, "other")
                if lower_name in self.vehicle_class_names and object_type == "other":
                    object_type = "vehicle"

                if self.vehicle_only and object_type not in ("vehicle", "two_wheeler", "pedestrian", "accident"):
                    continue

                raw_tracks.append(
                    Track(
                        track_id=int(track_id),
                        bbox=np.asarray(bbox, dtype=np.float32),
                        confidence=float(confidence),
                        class_id=int(class_id),
                        class_name=raw_class_name,
                        object_type=object_type,
                    )
                )

        # Bổ sung nhận diện đầy đủ phương tiện (car, bus, truck, motorcycle, bicycle, person) từ mô hình COCO chuẩn
        if self._vru_aux_model is not None:
            aux_kwargs = {
                "source": frame,
                "persist": True,
                "tracker": self.tracker_config,
                "conf": max(0.26, min(self.confidence, 0.32)),
                "iou": 0.50,
                "classes": [0, 1, 2, 3, 5, 7],
                "imgsz": 640,
                "verbose": False,
            }
            if self.device is not None:
                aux_kwargs["device"] = self.device
            aux_res = self._vru_aux_model.track(**aux_kwargs)
            if aux_res and aux_res[0].boxes is not None and aux_res[0].boxes.is_track:
                a_boxes = aux_res[0].boxes.xyxy.cpu().numpy()
                a_confs = aux_res[0].boxes.conf.cpu().numpy()
                a_clss = aux_res[0].boxes.cls.cpu().numpy().astype(int)
                a_tids = aux_res[0].boxes.id.cpu().numpy().astype(int)

                aux_items = []
                for abox, aconf, acls, atid in zip(a_boxes, a_confs, a_clss, a_tids):
                    cname = str(aux_res[0].names[int(acls)]).lower().strip()
                    aux_items.append([np.asarray(abox, dtype=np.float32), float(aconf), int(acls), cname, int(atid)])

                # Gộp người lái (person) đang ngồi trên xe máy/xe đạp thành một thực thể xe 2 bánh duy nhất
                merged_rider_indices: Set[int] = set()
                for i, item_i in enumerate(aux_items):
                    if item_i[3] == "person":
                        b_p = item_i[0]
                        pw = max(1.0, float(b_p[2] - b_p[0]))
                        ph = max(1.0, float(b_p[3] - b_p[1]))
                        for j, item_j in enumerate(aux_items):
                            if i != j and item_j[3] in ("motorcycle", "bicycle"):
                                b_m = item_j[0]
                                mw = max(1.0, float(b_m[2] - b_m[0]))
                                mh = max(1.0, float(b_m[3] - b_m[1]))
                                x_ov = max(0.0, min(float(b_p[2]), float(b_m[2])) - max(float(b_p[0]), float(b_m[0]))) / min(pw, mw)
                                y_gap = abs(float(b_p[3]) - float(b_m[3])) / max(ph, mh)
                                if (
                                    _bbox_iou(b_p, b_m) >= 0.04
                                    or _bbox_containment(b_p, b_m) >= 0.25
                                    or _bbox_containment(b_m, b_p) >= 0.25
                                    or (x_ov >= 0.55 and y_gap <= 0.45)
                                ):
                                    item_j[0] = np.array([
                                        min(b_p[0], b_m[0]),
                                        min(b_p[1], b_m[1]),
                                        max(b_p[2], b_m[2]),
                                        max(b_p[3], b_m[3]),
                                    ], dtype=np.float32)
                                    item_j[1] = max(item_i[1], item_j[1])
                                    merged_rider_indices.add(i)
                                    break
                        if i not in merged_rider_indices:
                            for pt in raw_tracks:
                                if pt.class_name.lower() in ("motorcycle", "bicycle", "bike"):
                                    b_m = pt.bbox
                                    mw = max(1.0, float(b_m[2] - b_m[0]))
                                    mh = max(1.0, float(b_m[3] - b_m[1]))
                                    x_ov = max(0.0, min(float(b_p[2]), float(b_m[2])) - max(float(b_p[0]), float(b_m[0]))) / min(pw, mw)
                                    y_gap = abs(float(b_p[3]) - float(b_m[3])) / max(ph, mh)
                                    if _bbox_iou(b_p, b_m) >= 0.04 or _bbox_containment(b_p, b_m) >= 0.25 or (x_ov >= 0.55 and y_gap <= 0.45):
                                        merged_rider_indices.add(i)
                                        break

                for idx_a, (abox, aconf, acls, cname, atid) in enumerate(aux_items):
                    if idx_a in merged_rider_indices:
                        continue
                    bw = float(abox[2] - abox[0])
                    bh = float(abox[3] - abox[1])
                    is_vru_aux = cname in ("motorcycle", "bicycle", "bike", "person", "pedestrian")
                    if is_vru_aux:
                        if aconf < 0.34:
                            continue
                        if bh < frame_h * 0.04 or (bw >= frame_w * 0.12 and (bw / max(1.0, bh)) >= 1.38 and (bw * bh >= frame_w * frame_h * 0.015)):
                            continue
                    else:
                        # Bổ sung các xe ô tô, xe buýt, xe tải rõ ràng bị mô hình tai nạn bỏ sót,
                        # đồng thời lọc bỏ các chấm xe quá nhỏ ở sát đường chân trời (tránh nhiễu BEV)
                        if (
                            aconf < 0.46
                            or bh < frame_h * 0.055
                            or bw < frame_w * 0.045
                            or (bw * bh) < (frame_w * frame_h * 0.0035)
                        ):
                            continue

                    aux_tid = 10000 + atid
                    overlapped_existing = False
                    for pt in raw_tracks:
                        if pt.class_name.lower() in ACCIDENT_CLASS_NAMES:
                            continue
                        iou = _bbox_iou(abox, pt.bbox)
                        cont_in_pt = _bbox_containment(abox, pt.bbox)
                        cont_pt_in_a = _bbox_containment(pt.bbox, abox)
                        if not is_vru_aux:
                            if iou >= 0.45 or max(cont_in_pt, cont_pt_in_a) >= 0.72:
                                overlapped_existing = True
                                break
                        elif pt.class_name.lower() in ("motorcycle", "bicycle", "bike", "person", "pedestrian"):
                            if iou >= 0.25 or cont_in_pt >= 0.55:
                                overlapped_existing = True
                                break
                        else:
                            # Nếu mô hình chính gán nhầm xe 2 bánh/người đi bộ dáng đứng thành 'car'
                            pt_ar = pt.width / max(1.0, pt.height)
                            if iou >= 0.45 and pt_ar < 1.12 and aconf >= max(0.48, pt.confidence - 0.05) and (pt.width * pt.height) < frame_w * frame_h * 0.025:
                                pt.class_name = cname
                                pt.object_type = OBJECT_TYPE_MAP.get(cname, "vehicle")
                                overlapped_existing = True
                                break
                            if iou >= 0.42 or cont_in_pt >= 0.72:
                                overlapped_existing = True
                                break

                    if not overlapped_existing:
                        scores = self._track_class_scores.setdefault(aux_tid, {})
                        scores[cname] = scores.get(cname, 0.0) + aconf
                        best_aux_cls = max(scores.items(), key=lambda kv: kv[1])[0]
                        raw_tracks.append(
                            Track(
                                track_id=aux_tid,
                                bbox=abox,
                                confidence=aconf,
                                class_id=acls,
                                class_name=best_aux_cls,
                                object_type=OBJECT_TYPE_MAP.get(best_aux_cls, "vehicle"),
                            )
                        )

        return self._deduplicate_tracks(raw_tracks)

    def reset(self) -> None:
        """Reset tracker state khi chuyển video mới."""
        self.model = YOLO(self.model_path)
        self._track_class_scores.clear()
        self._aux_to_primary_tid.clear()
        self._primary_to_aux_tid.clear()
        if self._vru_aux_model is not None:
            aux_candidate = Path(self.model_path).resolve().parent / "yolo11n.pt"
            if aux_candidate.exists():
                self._vru_aux_model = YOLO(str(aux_candidate))

    def __call__(
        self,
        frame: np.ndarray,
    ) -> List[Track]:
        return self.update(frame)
