"""Shared, actor-symmetric inputs and model for pair-contact verification.

No labels, publisher contact times, actor IDs or scene names enter the model.
Inference supplies detector boxes; supervised preparation supplies GT boxes.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch import nn

from .vehicle_pairs import normalized_box_gap, box_iou

INPUT = {"version": 1, "frames": 13, "sample_fps": 5, "image_size": 320,
         "rgb_attention": "union_of_observed_pair_boxes_with_dimmed_context",
         "context_brightness": .35, "mask_box_padding": .15,
         "geometry_features_per_frame": 20, "geometry_dimension": 260,
         "geometry_units": "image-box-diagonals per second; NOT metres",
         "actor_order": "symmetric", "encoder": "frozen_kinetics_x3d_s_2048",
         "training_augmentation": "box_jitter_short_missing_spans_temporal_shift_with_edge_padding",
         "temporal_shift_max_samples": 2}


def safe_video(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe video path")
    path = root.joinpath(*parts.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Video outside dataset")
    return path


def augment_boxes(boxes, seed):
    rng = np.random.default_rng(seed)
    value = [[None if b is None else list(b) for b in frame] for frame in boxes]
    for frame in value:
        for i, box in enumerate(frame):
            if box is not None:
                box = np.array(box, dtype=float)
                size = box[2:] - box[:2]
                offset = rng.normal(0, .025, 2) * size
                box += np.tile(offset, 2)
                frame[i] = box.tolist()
    if rng.random() < .65:
        actor = int(rng.integers(2))
        start = int(rng.integers(2, 10))
        for frame in value[start:start + int(rng.integers(1, 4))]:
            frame[actor] = None
    return value


def geometry_features(boxes, indices, fps):
    if len(boxes) != INPUT["frames"] or len(indices) != len(boxes) or fps <= 0:
        raise ValueError("Invalid temporal geometry input")
    result = np.zeros((len(boxes), INPUT["geometry_features_per_frame"]), np.float32)
    previous, previous_speeds, previous_gap, previous_iou = None, None, 0., 0.
    for t, pair in enumerate(boxes):
        if len(pair) != 2:
            raise ValueError("Need exactly two actors")
        present = [b is not None for b in pair]
        result[t, 1] = sum(present) / 2
        if not all(present):
            previous, previous_speeds = None, None
            continue
        a, b = np.asarray(pair, dtype=float)
        if not np.isfinite([a, b]).all() or np.any(a[2:] <= a[:2]) or np.any(b[2:] <= b[:2]):
            raise ValueError("Invalid observed box")
        sizes = np.array([a[2:] - a[:2], b[2:] - b[:2]])
        centers = np.array([(a[:2]+a[2:])/2, (b[:2]+b[2:])/2])
        scale = max(1., min(np.linalg.norm(sizes[0]), np.linalg.norm(sizes[1])))
        gap, overlap = normalized_box_gap(a, b), box_iou(a, b)
        delta = np.abs(centers[0]-centers[1]) / scale
        areas = sizes.prod(axis=1)
        aspects = np.log(sizes[:, 0] / sizes[:, 1])
        result[t, :12] = [1., 1., min(gap, 5)/5, overlap,
                          min(delta[0], 5)/5, min(delta[1], 5)/5,
                          min(abs(a[3]-b[3])/scale, 5)/5,
                          min(abs(math.log(areas[0]/areas[1])), 3)/3,
                          np.clip(aspects.mean(), -3, 3)/3,
                          min(abs(aspects[0]-aspects[1]), 3)/3,
                          sizes[:, 1].min()/sizes[:, 1].max(), min(np.linalg.norm(delta), 5)/5]
        if previous is not None:
            dt = (indices[t] - indices[t-1]) / fps
            if dt <= 0:
                raise ValueError("Frame samples must be strictly increasing")
            velocities = (centers - previous) / dt / scale
            speeds = np.linalg.norm(velocities, axis=1)
            cosine = (velocities[0] @ velocities[1]) / max(1e-8, speeds.prod())
            result[t, 12:] = [np.clip((gap-previous_gap)/dt, -5, 5)/5,
                              min(np.linalg.norm(velocities[0]-velocities[1]), 10)/10,
                              min(speeds.mean(), 10)/10, min(abs(speeds[0]-speeds[1]), 10)/10,
                              np.clip(cosine, -1, 1),
                              min(np.abs(speeds-previous_speeds).mean()/dt, 10)/10 if previous_speeds is not None else 0.,
                              1., np.clip((overlap-previous_iou)/dt, -5, 5)/5]
            previous_speeds = speeds
        previous, previous_gap, previous_iou = centers, gap, overlap
    if not np.isfinite(result).all():
        raise ValueError("Non-finite geometry")
    return result.flatten()


def attended_frame(frame, boxes, crop_xyxy):
    x1,y1,x2,y2=crop_xyxy
    crop=frame[y1:y2,x1:x2].copy()
    h,w=crop.shape[:2]
    if h<=0 or w<=0:raise ValueError("Empty pair crop")
    mask=np.zeros((h,w),np.float32)
    for box in boxes:
        if box is not None:
            a,b,c,d=box
            px,py=(c-a)*INPUT["mask_box_padding"],(d-b)*INPUT["mask_box_padding"]
            left,top=max(0,math.floor(a-px-x1)),max(0,math.floor(b-py-y1))
            right,bottom=min(w,math.ceil(c+px-x1)),min(h,math.ceil(d+py-y1))
            if right>left and bottom>top:mask[top:bottom,left:right]=1
    alpha=(INPUT["context_brightness"]+(1-INPUT["context_brightness"])*mask)[...,None]
    crop=np.clip(crop*alpha+115*(1-alpha),0,255).astype(np.uint8)
    size=INPUT["image_size"];scale=size/max(h,w)
    rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
    canvas=np.full((size,size,3),115,np.uint8)
    canvas[(size-rh)//2:(size-rh)//2+rh,(size-rw)//2:(size-rw)//2+rw]=cv2.cvtColor(cv2.resize(crop,(rw,rh),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    return canvas


def read_clip(root, pair, window, boxes=None, _sequential=False, native_frames=None):
    """Decode real pixels with supplied pair attention; never draw annotation text."""
    boxes = boxes if boxes is not None else window["sample_pair_boxes"]
    indices = window["sample_frame_indices"]
    if len(indices) != INPUT["frames"] or len(set(indices)) != len(indices):
        raise ValueError("Need 13 distinct source frames")
    path = safe_video(root, pair["video"])
    if path.stat().st_size != pair["video_bytes"]:
        raise ValueError(f"Video differs from manifest: {path}")
    if native_frames is not None:
        return np.stack([attended_frame(native_frames[i],b,window["crop_xyxy"]) for i,b in zip(indices,boxes)])
    # Fractional average FPS can make OpenCV's timestamp-based seek land on a
    # different frame than its sequential decoder. Preserve the detector index.
    _sequential = _sequential or abs(pair["fps_metadata"]-round(pair["fps_metadata"]))>.001
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        if abs(cap.get(cv2.CAP_PROP_FPS)-pair["fps_metadata"]) > .01:
            raise ValueError("Source FPS differs from preparation")
        if not _sequential:
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        pointer = 0
        for index in range(0 if _sequential else indices[0], indices[-1]+1):
            ok, frame = cap.read()
            if not ok:
                if not _sequential:
                    cap.release()
                    return read_clip(root,pair,window,boxes,_sequential=True)
                raise ValueError(f"Cannot decode {path}, frame {index}")
            if index != indices[pointer]:
                continue
            if frame.shape[:2] != (pair["height"], pair["width"]):
                raise ValueError("Source dimensions differ from preparation")
            frames.append(attended_frame(frame,boxes[pointer],window["crop_xyxy"]))
            pointer += 1
            if pointer == len(indices):
                break
    finally:
        cap.release()
    if len(frames) != INPUT["frames"]:
        raise ValueError("Incomplete pair clip")
    return np.stack(frames)


def build_encoder(state):
    from pytorchvideo.models.hub import x3d_s
    encoder = x3d_s(pretrained=False)
    encoder.blocks[-1].proj = nn.Identity()
    encoder.blocks[-1].activation = None
    encoder.load_state_dict(state, strict=True)
    return encoder.eval().requires_grad_(False)


@torch.inference_mode()
def encode(encoder, clips, device):
    x = torch.from_numpy(np.ascontiguousarray(clips)).to(device=device, dtype=torch.float32)
    x = (x.permute(0, 4, 1, 2, 3)/255 - .45)/.225
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        features = encoder(x).flatten(1).float()
    if features.shape[1] != 2048 or not torch.isfinite(features).all():
        raise ValueError("Invalid video embedding")
    return features.cpu().numpy()


class PairHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2048+INPUT["geometry_dimension"], 128),
                                     nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))

    def forward(self, video, geometry):
        return self.network(torch.cat((video, geometry), dim=1)).flatten()
