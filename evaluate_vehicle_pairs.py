"""Run YOLO/ByteTrack pair proposals without accident labels or reference times.

This is a diagnostic command. Candidates are NOT accident verdicts. Complete
detection traces are cached with source/model/settings identities for replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from core.tracker import ByteTrackTracker
from core.feature_pipeline import tracker_parameters

COLUMNS = ("frame", "id", "class_id", "confidence", "x1", "y1", "x2", "y2")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(source, output, root, config, progress=None):
    params = tracker_parameters(config, root)
    identity = {"source_sha256": sha(source), "model_sha256": sha(params["model_path"]),
                "tracker_sha256": sha(params["tracker_config"]),
                "tracker_code_sha256": sha(root / "core" / "tracker.py"),
                "ultralytics": importlib.metadata.version("ultralytics"),
                "parameters": {k: v for k, v in params.items() if k not in {"model_path", "tracker_config"}}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = output / f"{source.stem}_{key}_tracks.npz"
    # Upload staging names and original filenames may differ. The full identity
    # already binds video bytes, model, tracker code, settings and library version.
    matches = [cache] if cache.is_file() else sorted(output.glob(f"*_{key}_tracks.npz"))
    for existing in matches:
        with np.load(existing, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity or not bool(saved["complete"]):
                raise ValueError("Invalid trace cache")
        print(f"Using complete trace: {existing}", flush=True)
        return existing
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or expected < 1:
        cap.release()
        raise ValueError("Invalid video header")
    tracker = ByteTrackTracker(**params)
    records, index, started = [], 0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tracks = tracker.update(frame)
            for track in tracks:
                records.append([index, track.track_id, track.class_id, track.confidence, *track.bbox.tolist()])
            index += 1
            if index % 50 == 0 or index == expected:
                print(f"{source.name}: {index}/{expected} frames | {time.monotonic()-started:.1f}s", flush=True)
                if progress:
                    progress(index / expected)
    finally:
        cap.release()
    if index != expected:
        raise ValueError(f"Incomplete decoding {index}/{expected}; no complete cache saved")
    output.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(temporary, columns=np.array(COLUMNS),
                            observations=np.asarray(records, dtype=np.float64).reshape(-1, len(COLUMNS)),
                            fps=fps, frames=index, width=width, height=height,
                            identity=json.dumps(identity, sort_keys=True), complete=True)
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return cache


def candidate_preview(result, path):
    """Automatically display distinct high-approach proposals, with no GT input."""
    chosen, seen = [], set()
    for window in sorted(result["windows"], key=lambda w: w["approach_priority"], reverse=True):
        key = tuple(window["track_ids"])
        if key not in seen:
            chosen.append(window)
            seen.add(key)
        if len(chosen) == 8:
            break
    if not chosen:
        return
    canvas = np.zeros((math.ceil(len(chosen) / 4) * 260 + 32, 1280, 3), dtype=np.uint8)
    cv2.putText(canvas, "PAIR CANDIDATES ONLY - NOT ACCIDENT VERDICTS", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    cap = cv2.VideoCapture(result["source"])
    try:
        for n, window in enumerate(chosen):
            visible = [i for i, gap in enumerate(window["sample_normalized_box_gaps"]) if gap is not None]
            sample = min(visible, key=lambda i: window["sample_normalized_box_gaps"][i]) if visible else 6
            index = window["sample_frame_indices"][sample]
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode preview frame {index}")
            for actor, box in zip(window["track_ids"], window["sample_pair_boxes"][sample]):
                if box is not None:
                    x1, y1, x2, y2 = map(round, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 3)
                    cv2.putText(frame, f"ID {actor}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 200, 0), 2)
            x1, y1, x2, y2 = window["crop_xyxy"]
            crop = frame[y1:y2, x1:x2]
            scale = min(320 / crop.shape[1], 210 / crop.shape[0])
            tile = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
            x, y = (n % 4) * 320, (n // 4) * 260 + 32
            title = f"IDs {window['track_ids']} | frame {index}"
            cv2.putText(canvas, title, (x+3, y+20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
            canvas[y+32:y+32+tile.shape[0], x:x+tile.shape[1]] = tile
    finally:
        cap.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError("Cannot save candidate preview")


def proposals(cache, output, source, preview_dir=None):
    from utils.vehicle_pairs import PairCandidateTracker
    with np.load(cache, allow_pickle=False) as saved:
        observations = saved["observations"]
        fps, frames = float(saved["fps"]), int(saved["frames"])
        tracker = PairCandidateTracker(fps=fps, width=int(saved["width"]), height=int(saved["height"]))
        cursor = 0
        for index in range(frames):
            end = cursor
            while end < len(observations) and int(observations[end, 0]) == index:
                end += 1
            tracker.update(index, observations[cursor:end, 1:])
            cursor = end
        if cursor != len(observations):
            raise ValueError("Unordered trace")
        result = tracker.finish(frames)
        result.update(source=str(source.resolve()), source_sha256=json.loads(str(saved["identity"]))["source_sha256"],
                      detection_trace=str(cache.resolve()), trace_sha256=sha(cache),
                      module_sha256=sha(Path(__file__).parent / "utils" / "vehicle_pairs.py"))
    path = output / f"{cache.stem.removesuffix('_tracks')}_pairs.json"
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if preview_dir is not None:
        candidate_preview(result, preview_dir / f"{cache.stem.removesuffix('_tracks')}_candidates.jpg")
    print(json.dumps({"video": source.name, "frames": frames, "pair_encounters": len(result["encounters"]),
                      "candidate_windows": len(result["windows"]), "output": str(path)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/vehicle_pair_tracking"))
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    for source in args.videos:
        cache = collect(source, args.output, root, config)
        if not args.collect_only:
            proposals(cache, args.output, source, args.preview_dir)


if __name__ == "__main__":
    main()
