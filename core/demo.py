"""Inference-only analysis: an ACCIDENT verdict requires a localized vehicle event."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import train_temporal_accident as engine
from utils.video_writer import BrowserVideoWriter

VERSION = "demo-2026-10-28.4-event-gate"
MODEL_SHA = "a3afa3a07b380cdbd3dff8cd16220eecbeb2b884f5b25f6389157f3924756b9c"


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Không mở được video. Hãy thử video MP4 khác.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(3)), int(cap.get(4))
        if not math.isfinite(fps) or fps <= 0 or count <= 0 or min(width, height) <= 0:
            raise ValueError("Video không có thông tin khung hình hợp lệ.")
        return dict(fps=fps, frames=count, width=width, height=height, duration=count / fps)
    finally:
        cap.release()


def timeline_from_logits(logits, starts, targets, fps, threshold, duration):
    """Same max-view/mean-adjacent-logits pooling as the saved validation."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    paired = (logits[:-1] + logits[1:]) / 2 if len(starts) > 1 else logits
    scores = paired.max(dim=1).values.sigmoid().tolist()
    rows = []
    for i, score in enumerate(scores):
        last = starts[min(i + 1, len(starts) - 1)] + engine.CONFIG["frames"] - 1
        rows.append(dict(start=targets[starts[i]] / fps,
                         end=min(duration, (targets[min(last, len(targets) - 1)] + 1) / fps),
                         score=float(score), flagged=bool(score >= threshold)))
    intervals = []
    for row in rows:
        if not row["flagged"]:
            continue
        if intervals and row["start"] <= intervals[-1]["end"] + 1e-6:
            intervals[-1]["end"] = max(intervals[-1]["end"], row["end"])
            intervals[-1]["score"] = max(intervals[-1]["score"], row["score"])
        else:
            intervals.append({k: row[k] for k in ("start", "end", "score")})
    return max(scores), rows, intervals


def warning_at(timeline, when):
    """A window score exists only after its final source frame has arrived."""
    for row in reversed(timeline):
        if row["end"] <= when:
            return bool(row["flagged"])
    return False


class TemporalVideoClassifier:
    def __init__(self, model_path, threshold=None):
        self.path = Path(model_path)
        if file_sha(self.path) != MODEL_SHA:
            raise ValueError("Model demo không khớp bản Temporal AI City v3 đã chọn.")
        package = torch.load(self.path, map_location="cpu", weights_only=True)
        if package["kind"] != "x3d_s_partial_temporal_finetune":
            raise ValueError("Sai loại model temporal.")
        for key in ("backbone", "frames", "sample_fps", "image_size", "views", "video_stride_samples", "video_pool"):
            if json.dumps(package["config"][key]) != json.dumps(engine.CONFIG[key]):
                raise ValueError(f"Cấu hình suy luận khác model: {key}")
        self.threshold = float(package["threshold"])
        if not math.isclose(self.threshold, 0.9297188818454742, abs_tol=1e-12):
            raise ValueError("Ngưỡng demo khác ngưỡng đã lưu khi validation.")
        self.trained_threshold = self.threshold
        if threshold is not None:
            if not math.isfinite(float(threshold)) or not 0 < float(threshold) < 1:
                raise ValueError("Ngưỡng phân loại phải nằm giữa 0 và 1.")
            self.threshold = float(threshold)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        self.body, self.tail = engine.build_parts()
        self.body.load_state_dict(package["body_state"], strict=True)
        self.tail.load_state_dict(package["tail_state"], strict=True)
        self.body.requires_grad_(False).eval().to(self.device)
        self.tail.requires_grad_(False).eval().to(self.device)
        self.lock = threading.RLock()

    def analyze(self, path, progress=None):
        with self.lock, torch.inference_mode():
            info = video_info(path)
            if info["duration"] > 120.5:
                raise ValueError("Bản demo nhận video tối đa 2 phút. Hãy cắt video rồi tải lại.")
            fps, count = info["fps"], info["frames"]
            targets = np.clip(np.rint(np.arange(0, count / fps, 1 / engine.CONFIG["sample_fps"]) * fps).astype(int), 0, count - 1)
            last = max(0, len(targets) - engine.CONFIG["frames"])
            starts = sorted(set(range(0, last + 1, engine.CONFIG["video_stride_samples"])) | {last})
            # Retain only the last window, rather than all five views of the whole video.
            recent, logits = deque(maxlen=13), []
            pointer = 0
            cap = cv2.VideoCapture(str(path))
            try:
                for frame_id in range(count):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Video bị ngắt tại khung hình {frame_id}; chưa có kết luận.")
                    if pointer < len(targets) and frame_id == targets[pointer]:
                        tiles = np.stack([engine.crop_letterbox(image, v) for v in range(5)])
                        while pointer < len(targets) and frame_id == targets[pointer]:
                            recent.append((pointer, tiles))
                            pointer += 1
                            while len(logits) < len(starts):
                                start = starts[len(logits)]
                                end = min(start + 12, len(targets) - 1)
                                if pointer - 1 < end:
                                    break
                                frames = dict(recent)
                                indices = np.minimum(np.arange(start, start + 13), len(targets) - 1)
                                clips = np.stack([frames[int(i)] for i in indices]).transpose(1, 0, 2, 3, 4)
                                chunks = []
                                for offset in range(0, 5, 3):
                                    with engine.amp(self.device):
                                        value = self.tail(self.body(engine.normalized_tensor(clips[offset:offset + 3], self.device))).flatten().float()
                                    if not torch.isfinite(value).all():
                                        raise ValueError("Model trả về điểm không hợp lệ; chưa có kết luận.")
                                    chunks.append(value.cpu())
                                logits.append(torch.cat(chunks))
                                if progress:
                                    progress(.05 + .7 * len(logits) / len(starts), "Đang phân tích diễn biến")
                if pointer != len(targets) or len(logits) != len(starts):
                    raise ValueError("Chưa phân tích đủ toàn bộ video.")
            finally:
                cap.release()
            score, rows, intervals = timeline_from_logits(torch.stack(logits), starts, targets, fps, self.threshold, info["duration"])
            return {**info, "score": score, "threshold": self.threshold,
                    "trained_threshold": self.trained_threshold,
                    "threshold_selection_scope": "custom_threshold_override" if self.threshold != self.trained_threshold else "original_validation",
                    "label": "ACCIDENT" if score >= self.threshold else "NORMAL",
                    "timeline": rows, "input_window_ranges": intervals,
                    "first_alert_time": next((r["end"] for r in rows if r["flagged"]), None),
                    "temporal_localization": "not_estimated",
                    "timing_scope": "scores_available_at_window_end_not_collision_boundaries",
                    "model_name": "Temporal AI City v3", "model_sha256": MODEL_SHA,
                    "device": str(self.device), "analysis_version": VERSION,
                    "scope": "research_demo", "windows": len(starts)}


def cache_key(source_sha, vehicles, yolo_sha="", threshold=0.9297188818454742):
    return hashlib.sha256(f"{VERSION}:{MODEL_SHA}:{source_sha}:{vehicles}:{yolo_sha}:{threshold:.12g}".encode()).hexdigest()


def render_video(source, destination, result, project_root, vehicles=False, progress=None):
    """Vehicle IDs are observations, never asserted to be collision participants."""
    tracker = None
    if vehicles:
        from core.tracker import ByteTrackTracker
        tracker = ByteTrackTracker(
            str(project_root / "models/yolov8m.pt"), confidence=.10,
            iou_threshold=.50, tracker_config=str(project_root / "configs/bytetrack_detail.yaml"),
            device=result["device"], classes=[2, 3, 5, 7], imgsz=640,
            max_det=300, agnostic_nms=True,
        )
    cap = cv2.VideoCapture(str(source))
    writer = None
    ids = set()
    try:
        writer = BrowserVideoWriter(destination, result["fps"], (result["width"], result["height"]))
        for index in range(result["frames"]):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Không thể xuất đủ video kết quả.")
            if tracker is not None:
                for track in tracker.update(frame):
                    ids.add(track.track_id)
                    x1, y1, x2, y2 = map(int, track.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 155, 35), 2)
                    cv2.putText(frame, f"#{track.track_id} {track.class_name}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 155, 35), 1, cv2.LINE_AA)
            when = index / result["fps"]
            flagged = warning_at(result["timeline"], when)
            # ASCII overlay for OpenCV; the Vietnamese explanation is in the UI.
            w = min(result["width"], 560)
            cv2.rectangle(frame, (0, 0), (w, 55), (47, 26, 12), -1)
            text = "MODEL CANH BAO - CAN XEM LAI" if flagged else "CHUA CO CANH BAO TU MODEL"
            cv2.putText(frame, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (110, 180, 255) if flagged else (245, 225, 210), 1, cv2.LINE_AA)
            cv2.putText(frame, f"AcciVision | {when:.1f}s | Temporal AI City v3", (12, 43), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 210, 195), 1, cv2.LINE_AA)
            writer.write(frame)
            if progress and index % 15 == 0:
                progress(.76 + .23 * (index + 1) / result["frames"], "Đang dựng video kết quả")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return len(ids)


def localized_events(result, info):
    """Only completed per-vehicle analysis with usable event times is evidence.

    A temporal score, file name, known demo label, or input window is not an event.
    Incomplete/failed analysis raises instead of reporting a completed NORMAL run.
    """
    if result.get("frames_processed") != info["frames"]:
        raise ValueError("Chưa phân tích đủ video; chưa có kết luận.")
    events = []
    for event in result.get("events", []):
        start, end = event.get("start_time_s"), event.get("end_time_s")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end <= info["duration"] + 1 / info["fps"]):
            continue
        participants = [v for v in event.get("vehicles", [])
                        if isinstance(v.get("impact_track_id", v.get("vehicle_id")), int)]
        if participants:
            events.append({**event, "vehicles": participants})
    return events


def run_analysis(classifier, source, output, project_root, vehicles=False, progress=None,
                 localization=None):
    started = time.monotonic()
    result = classifier.analyze(source, progress)
    source_sha = file_sha(source)
    # Scan all videos, including low temporal scores. No demo-label shortcut.
    if localization is None:
        local = localize_vehicles(source, output, project_root,
            (lambda value: progress(.75 + .24 * value, "Đang xác định sự kiện và xe liên quan"))
            if progress else None)
    else:
        local = localization["result"]
        video = Path(localization["video_path"])
        if local.get("source_sha256") != source_sha or file_sha(video) != localization["video_sha256"]:
            raise ValueError("Kết quả theo dõi không khớp video đầu vào.")
        Path(output).write_bytes(video.read_bytes())
    events = localized_events(local, result)
    result["temporal_label"] = result["label"]
    result["label"] = "ACCIDENT" if events else "NORMAL"
    result["events"] = events
    result["verdict_rule"] = "localized_vehicle_event_v1"
    result["event_detection_complete"] = True
    result["tracking_enabled"] = True
    result["temporal_localization"] = "rf_motion_event_estimate" if events else "no_event_detected"
    result["timing_scope"] = "localized_event_estimate_not_human_verified"
    result["first_alert_time"] = min((e["start_time_s"] for e in events), default=None)
    result["track_count"] = len(local.get("tracking_observations", []))
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["source_sha256"] = source_sha
    if progress:
        progress(1., "Hoàn tất phân tích")
    return result


def vehicle_analysis_key(source_sha, project_root):
    paths = ["models/accident_rf_model.pkl", "models/yolov8m.pt",
             "models/visual_accident_classifier.pth", "config.yaml", "main.py",
             "configs/bytetrack_detail.yaml"]
    paths += [str(p.relative_to(project_root)) for p in sorted((project_root / "core").glob("*.py"))]
    return hashlib.sha256(("vehicle-localization-v1:" + source_sha +
        ":".join(file_sha(project_root / p) for p in paths)).encode()).hexdigest()


def localize_vehicles(source, output, project_root, progress=None):
    """Run the existing per-track RF/motion/event pipeline, without fitting.

    Candidate participants are separate from the temporal whole-video verdict.
    IDs are observations and do not identify fault or a persistent vehicle.
    """
    import copy
    from main import TrafficAccidentPipeline, load_config
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    settings = copy.deepcopy(load_config(str(project_root / "config.yaml")))
    settings["video"].update(input=str(source), output=str(output), display=False,
                             save_output=True)
    settings["output"].update(save_csv=False, save_video=True,
                              draw_speed=False, draw_world_position=False, draw_fps=False)
    pipeline = TrafficAccidentPipeline(settings, project_root=project_root)
    # Tiny per-frame batches are faster without starting a forest worker pool.
    # This changes execution only; serialized model weights remain untouched.
    if hasattr(pipeline.classifier.model, "n_jobs"):
        pipeline.classifier.model.n_jobs = 1
    result = pipeline.run(progress_callback=progress)
    expected = video_info(source)["frames"]
    if result["frames_processed"] != expected:
        raise ValueError("Chưa phân tích đủ khung hình để khoanh vùng phương tiện.")
    result["localization_scope"] = "rf_motion_candidate_participants_not_human_verified"
    result["source_sha256"] = file_sha(source)
    return result
