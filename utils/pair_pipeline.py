"""Detector-only vehicle pairs, shared by evaluation and the local application.

Native-rate detector observations; missing observations are never invented.
The event timestamp is a box-contact transition, not an accident duration.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import torch
import yaml

from core.tracker import ByteTrackTracker
from core.feature_pipeline import tracker_parameters
from .vehicle_pairs import PairCandidateTracker
from .pair_events import POLICY, contact_proposals, decide
from .pair_verifier import INPUT, PairHead, build_encoder, encode, geometry_features, iter_clips

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/vehicle_pair_training/accident_pair_verifier.pt"
GEOMETRY = ROOT / "outputs/vehicle_pair_training/validated/geometry/geometry_baseline.joblib"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    params = tracker_parameters(config, ROOT)
    identity = {"policy": POLICY, "parameters": params,
                "detector_sha256": sha(params["model_path"]),
                "tracker_config_sha256": sha(params["tracker_config"]),
                "code": {p.name: sha(p) for p in [Path(__file__),
                         ROOT / "utils/pair_events.py", ROOT / "utils/vehicle_pairs.py",
                         ROOT / "utils/pair_verifier.py", ROOT / "core/tracker.py",
                         ROOT / "evaluate_vehicle_pairs.py"]}}
    return identity, params


def collect(source, output, progress=None, trace_output=None):
    from evaluate_vehicle_pairs import collect as collect_trace, proposals
    source, output = Path(source).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity, _ = fingerprint()
    identity["source_sha256"] = sha(source)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = output / (source.stem + "_" + signature[:12] + "_pairs.json")
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity"] != identity or sha(result["detection_trace"]) != result["trace_sha256"]:
            raise ValueError("Pair trace identity mismatch")
        # Uploads are staged in a new temporary directory on each attempt.
        # The verified content/trace can be reused; its old input path cannot.
        result["source"] = str(source)
        return result, path
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cache = collect_trace(source, Path(trace_output) if trace_output is not None else ROOT / "outputs/vehicle_pair_tracking", ROOT, config,
                          progress=(lambda value: progress(.65*value,"Đang theo dõi từng khung hình")) if progress else None)
    proposals(cache, output, source, None)
    raw = output / f"{cache.stem.removesuffix('_tracks')}_pairs.json"
    result = json.loads(raw.read_text(encoding="utf-8"))
    with np.load(cache, allow_pickle=False) as trace:
        observations = trace["observations"]
        contacts = contact_proposals(observations, result["fps"])
    result.update(identity=identity, signature=signature,
                  detector_observations=result["frames_processed"], contacts=contacts)
    atomic_json(result,path)
    # Raw proposal file is an intermediate regenerated from the persistent trace.
    if raw != path:
        raw.unlink()
    return result, path


class Verifier:
    def __init__(self):
        self.geometry = joblib.load(GEOMETRY)["model"]
        self.model = None
        self.encoder = None
        self.head = None

    def load_rgb(self):
        if self.model is not None:
            return
        self.model = torch.load(MODEL, map_location="cpu", weights_only=True)
        if self.model["input"] != INPUT or self.model["kind"] != "x3d_s_pair_rgb_geometry":
            raise ValueError("Pair verifier protocol mismatch")
        self.encoder = build_encoder(self.model["encoder_state"])
        self.head = PairHead().eval()
        self.head.load_state_dict(self.model["head_state"])
        self.stats = {k: v.numpy() for k, v in self.model["normalization"].items()}

    def score(self, result, path, progress=None):
        identity = {"proposal_sha256": sha(path), "model_sha256": sha(MODEL),
                    "geometry_sha256": sha(GEOMETRY), "pipeline_sha256": sha(__file__)}
        score_path = path.with_name(path.name.replace("_pairs.json", "_scores.json"))
        if score_path.is_file():
            saved = json.loads(score_path.read_text(encoding="utf-8"))
            if saved["identity"] == identity and saved["complete"]:
                saved["source"] = result["source"]
                return saved
        contacts = result["contacts"]
        eligible = [(i, w) for i, w in enumerate(result["windows"]) if any(
            w["track_ids"] == c["track_ids"] and w["context_start_seconds"] <= c["time_s"] <= w["context_end_seconds"]
            for c in contacts)]
        rows, rgb_windows, geometries = [], [], {}
        descriptor = {"video": Path(result["source"]).name, "video_bytes": Path(result["source"]).stat().st_size,
                      "fps_metadata": result["fps"], "height": result["height"], "width": result["width"]}
        # Compute cheap geometry first, then decode overlapping RGB windows once.
        # The sampled indices and attended pixels are identical to read_clip's
        # exact sequential path, including fractional-FPS source videos.
        for index, window in eligible:
            g = geometry_features(window["sample_pair_boxes"], window["sample_frame_indices"], result["fps"])[None]
            geometry_score = float(self.geometry.predict_proba(g)[0, 1])
            rgb_score = None
            if geometry_score >= POLICY["score_floor"]:
                rgb_windows.append((index, window))
                geometries[index] = g
            row = {k: window[k] for k in ("track_ids", "context_start_seconds", "context_end_seconds")}
            row.update(window_index=index, geometry_score=geometry_score, rgb_score=rgb_score,
                       joint_score=geometry_score, rgb_evaluated=False)
            rows.append(row)
        by_index = {row["window_index"]: row for row in rows}
        if rgb_windows:
            self.load_rgb()
            for done, (index, clip) in enumerate(iter_clips(
                    Path(result["source"]).parent, descriptor, rgb_windows), 1):
                g = geometries[index]
                v = encode(self.encoder, clip[None], torch.device("cpu"))
                vn = torch.tensor((v-self.stats["video_mean"])/self.stats["video_std"])
                gn = torch.tensor((g-self.stats["geometry_mean"])/self.stats["geometry_std"])
                with torch.inference_mode():
                    rgb_score = float(self.head(vn, gn).sigmoid()[0])
                row = by_index[index]
                row.update(rgb_score=rgb_score, joint_score=min(row["geometry_score"], rgb_score),
                           rgb_evaluated=True)
                if progress:
                    progress(.65+.3*done/len(rgb_windows), "Đang kiểm tra chuyển động của từng cặp xe")
        saved = {"identity": identity, "source": result["source"], "complete": True,
                 "scope": "development_vehicle_pair_diagnostic", "deployment_approved": False,
                 "contacts": contacts, "scores": rows,
                 "events_at_floor": decide(contacts, rows, POLICY["score_floor"])}
        atomic_json(saved, score_path)
        return saved


def analyze(source, output, progress=None, trace_output=None):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    result, path = collect(source, output, progress, trace_output=trace_output)
    verifier = Verifier()
    return result, verifier.score(result, path, progress)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/vehicle_pair_runtime")
    args = parser.parse_args()
    for source in args.videos:
        result, scored = analyze(source, args.output)
        print(json.dumps({"video": source.name, "contacts": len(result["contacts"]),
                          "events": scored["events_at_floor"]}, ensure_ascii=False), flush=True)
