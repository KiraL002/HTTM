"""Optional, offline Simple-TAD review; a score is NOT a collision timestamp.

Architecture/checkpoint compatibility: Orlova et al., Simple-TAD (ICCVW 2025),
https://github.com/tue-mps/simple-tad, CC BY-NC 4.0; see
licenses/simple-tad-LICENSE.txt and the attribution in README.md.
This small inference adapter uses PyTorch SDPA instead of FlashAttention and
does not import the reference project's training environment or install packages.
The application event decisions and their trained cutoffs are unchanged.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(384, 1152, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(384))
        self.v_bias = nn.Parameter(torch.zeros(384))
        self.proj = nn.Linear(384, 384)

    def forward(self, x):
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = F.linear(x, self.qkv.weight, bias).reshape(
            x.shape[0], x.shape[1], 3, 6, 64).permute(2, 0, 3, 1, 4).unbind(0)
        value = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(value.transpose(1, 2).reshape(x.shape))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 384, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VideoMAESmall(nn.Module):
    """Exact released 16-frame, 224px, two-class inference architecture."""
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.fc_norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Linear(384, 2)
        position = np.arange(1568)[:, None] / np.power(
            10000, 2 * (np.arange(384)[None, :] // 2) / 384)
        position[:, 0::2] = np.sin(position[:, 0::2])
        position[:, 1::2] = np.cos(position[:, 1::2])
        self.register_buffer("pos_embed", torch.tensor(position, dtype=torch.float32)[None],
                             persistent=False)

    def forward(self, clip):
        if tuple(clip.shape[1:]) != (3, 16, 224, 224):
            raise ValueError("Simple-TAD needs B,C,T,H,W = B,3,16,224,224")
        x = self.patch_embed(clip) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.fc_norm(x.mean(1))).softmax(-1)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name in ("model", "state_dict", "module"):
        if isinstance(state, dict) and isinstance(state.get(name), dict):
            state = state[name]
            break
    if not isinstance(state, dict):
        raise ValueError("Unsupported Simple-TAD checkpoint")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = VideoMAESmall()
    # No fallback to a random head, generic action model or another architecture.
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def prepare_frame(frame):
    rgb = cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC),
                       cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray((x - np.array([.485, .456, .406], np.float32)[:, None, None])
                               / np.array([.229, .224, .225], np.float32)[:, None, None])


def annotate_scores(rows, *, enter=.75, release=.5, minimum_samples=2, max_gap=.81):
    """Separate raw risk, causal scene change and review intervals.

    Sustained high background is exposed in raw scores, not presented as a
    confirmed accident. A window's beginning is context, never impact onset.
    Numerical defaults are diagnostic settings, NOT CCTV-calibrated cutoffs.
    """
    if not (0 <= release < enter <= 1 and minimum_samples >= 2 and max_gap > 0):
        raise ValueError("Invalid diagnostic interval settings")
    timeline, episodes, active = [], [], []
    baseline = None

    def finish():
        if len(active) >= minimum_samples:
            peak = max(active, key=lambda r: r["score"])
            episodes.append({"context_start_s": active[0]["window_start_s"],
                             "context_end_s": active[-1]["window_end_s"],
                             "first_high_window_available_s": active[0]["available_at_s"],
                             "confirmed_available_s": active[minimum_samples-1]["available_at_s"],
                             "peak_score": peak["score"],
                             "peak_window_end_s": peak["window_end_s"],
                             "samples": len(active), "collision_time_s": None,
                             "kind": "temporal_anomaly_review_only"})
        active.clear()

    previous = -math.inf
    for raw in rows:
        row = dict(raw)
        score, when = float(row["score"]), float(row["available_at_s"])
        start, end = float(row["window_start_s"]), float(row["window_end_s"])
        if not all(map(math.isfinite, [score, when, start, end])) or not 0 <= score <= 1:
            raise ValueError("Non-finite or invalid temporal score")
        if not 0 <= start <= end <= when or when <= previous:
            raise ValueError("Invalid temporal score ordering or window availability")
        if when - previous > max_gap:
            finish()
        row["scene_baseline"] = score if baseline is None else baseline
        row["change_score"] = (0.0 if baseline is None or len(timeline) < 4 else
                               min(1., max(0., (score-baseline-.10)/.35)))
        baseline = score if baseline is None else baseline + (
            .02 if score > baseline else .12) * (score-baseline)
        timeline.append(row)
        if score >= (release if active else enter):
            active.append(row)
        else:
            finish()
        previous = when
    finish()
    return timeline, episodes


@torch.inference_mode()
def scan_video(source, models, *, sample_fps=10., stride=4, progress=None):
    """Decode once and score all supplied models on identical native frames.

    Returns evidence for review, never an application ACCIDENT/NORMAL verdict.
    Source SHA and native frame indices make saved diagnostics auditable.
    """
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 60 or stride < 1:
        raise ValueError("Invalid sample FPS or inference stride")
    source = Path(source).resolve()
    if not source.is_file() or not models:
        raise ValueError("Need a readable video and at least one loaded model")
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(source))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
        cap.release()
        raise ValueError("Cannot determine video timing; no temporal conclusion")
    if count/fps > 120.5:
        cap.release()
        raise ValueError("Temporal review currently accepts at most two minutes")
    period = fps/min(sample_fps, fps)
    samples = deque(maxlen=16)
    rows = {name: [] for name in models}
    index, sample_index = 0, 0
    next_frame = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= next_frame:
                samples.append((index, prepare_frame(frame)))
                if len(samples) == 16 and (sample_index-15) % stride == 0:
                    ids = [i for i, _ in samples]
                    clip = torch.from_numpy(np.stack([x for _, x in samples], axis=1))[None]
                    for name, model in models.items():
                        value = model(clip)
                        if value.shape != (1, 2) or not torch.isfinite(value).all():
                            raise ValueError(f"Invalid model output: {name}")
                        rows[name].append({"window_start_s": ids[0]/fps,
                                            "window_end_s": ids[-1]/fps,
                                            "available_at_s": ids[-1]/fps,
                                            "source_frame_indices": ids,
                                            "score": float(value[0, 1])})
                sample_index += 1
                next_frame = max(index+1, round(sample_index*period))
            index += 1
            if progress and index % max(1, round(fps*5)) == 0:
                progress(index/count)
        if index != count:
            raise ValueError(f"Incomplete video: decoded {index}, expected {count}; no conclusion")
    finally:
        cap.release()
    if len(samples) < 16:
        raise ValueError("Fewer than 16 distinct sampled source frames; no conclusion")
    outputs = {}
    for name, values in rows.items():
        timeline, candidates = annotate_scores(values, max_gap=max(.81, stride/min(sample_fps, fps)*1.8))
        peak = max(timeline, key=lambda r: r["score"])
        outputs[name] = {"peak_score": peak["score"],
                         "peak_window_end_s": peak["window_end_s"],
                         "timeline": timeline, "review_candidates": candidates}
    return {"source": str(source), "source_sha256": sha(source), "complete": True,
            "fps": fps, "frames_decoded": index, "duration_s": index/fps,
            "sample_fps": sample_fps, "stride": stride,
            "timestamp_basis": "native_frame_index_divided_by_metadata_fps",
            "scope": "development_temporal_anomaly_not_accident_classification",
            "thresholds": {"enter": .75, "release": .5, "minimum_samples": 2,
                           "selected_on": "diagnostic_defaults_not_CCTV_calibration"},
            "collision_time_estimated": False, "application_verdict_changed": False,
            "models": outputs, "elapsed_seconds": round(time.perf_counter()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    checkpoints = {"simple_tad": args.checkpoint}
    if args.compare_checkpoint:
        checkpoints["traffic_checker"] = args.compare_checkpoint
    identities = {name: {"path": str(p.resolve()), "sha256": sha(p)} for name, p in checkpoints.items()}
    models = {name: load_model(p) for name, p in checkpoints.items()}
    results = []
    for video in args.videos:
        print(f"Scanning {video.name} (fresh inference, no application cache)", flush=True)
        value = scan_video(video, models)
        results.append(value)
        print(json.dumps({"video": video.name, "seconds": value["elapsed_seconds"],
                          "models": {k: {a: b for a, b in v.items() if a != "timeline"}
                                     for k, v in value["models"].items()}}, ensure_ascii=False), flush=True)
    payload = {"complete": True, "implementation_sha256": sha(__file__),
               "checkpoints": identities, "results": results,
               "application_changed": False, "training_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
