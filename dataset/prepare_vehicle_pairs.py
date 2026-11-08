"""Prepare synthetic vehicle-pair supervision; never writes to raw ACCIDENT data.

GT boxes are for supervised crops, not proposals at inference. Validation here
is a synthetic unseen-map diagnostic, not a real-video or end-to-end test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from utils.vehicle_pairs import normalized_box_gap

FOUR_WHEEL = {14, 15, 16, 29}
VALIDATION_MAPS = {"Town03", "Town04"}
POLICY = {"frames": 13, "sample_fps": 5, "image_size": 320,
          "duration_seconds": 2.4, "context_fraction": 0.25,
          "negative_clearance_seconds": 1.0,
          "minimum_joint_visible_samples": 9,
          "positive": "first annotated contact of this four-wheel pair",
          "negative": "same pair, window ends >=1s before ANY annotated collision",
          "box_policy": "fixed union of visible pair boxes over the window, with context"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe dataset path: {relative}")
    path = root.joinpath(*parts.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path outside source dataset")
    return path


def annotation_path(root, relative):
    path = source_path(root, relative)
    unpacked = Path(str(path).removesuffix(".gz"))
    for candidate in (path, unpacked, unpacked / unpacked.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(relative)


def load_annotation(path):
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def box_of(obj, width, height):
    box = np.asarray(obj.get("2d_bbox", []), dtype=float)
    if box.shape != (2, 2) or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box.reshape(-1)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def make_window(anchor, label, pair, frames, fps, count, width, height, first_collision, precontact_negative=True):
    span = POLICY["duration_seconds"]
    last_time = (count - 1) / fps
    if last_time < span:
        return None, "video_too_short"
    start = min(max(0., anchor - span / 2), last_time - span)
    end = start + span
    if label == 0 and precontact_negative and end > first_collision - POLICY["negative_clearance_seconds"] + 1e-8:
        return None, "no_safe_precontact_window"
    indices = np.rint((start + np.arange(POLICY["frames"]) / POLICY["sample_fps"]) * fps).astype(int)
    boxes, joint = [], 0
    for index in indices:
        objects = frames.get(int(index), {})
        at_frame = [box_of(objects[actor], width, height) if actor in objects else None for actor in pair]
        joint += all(box is not None for box in at_frame)
        boxes.append(at_frame)
    if joint < POLICY["minimum_joint_visible_samples"]:
        return None, "insufficient_joint_visibility"
    if label == 1:
        contact_objects = frames.get(round(anchor * fps), {})
        if any(actor not in contact_objects or box_of(contact_objects[actor], width, height) is None for actor in pair):
            return None, "pair_not_visible_at_contact"
    valid = np.array([box for sample in boxes for box in sample if box is not None])
    low, high = valid[:, :2].min(axis=0), valid[:, 2:].max(axis=0)
    padding = (high - low) * POLICY["context_fraction"]
    low = np.maximum(0, np.floor(low - padding)).astype(int)
    high = np.minimum([width, height], np.ceil(high + padding)).astype(int)
    crop = [*low.tolist(), *high.tolist()]
    return {"label": label, "label_source": "synthetic_collision_annotation",
            "human_verified": False, "start_seconds": round(start, 6),
            "end_seconds": round(end, 6), "sample_frame_indices": indices.tolist(),
            "joint_visible_samples": int(joint), "crop_xyxy": crop,
            "sample_pair_boxes": boxes}, None


def prepare_video(row, root):
    video_path = source_path(root, row["rgb_path"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    annotation = annotation_path(root, row["annotations_path"])
    data = load_annotation(annotation)
    offset, count = int(row["annotations_start_offset"]), int(row["no_frames"])
    fps = count / float(row["duration"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid metadata fps")
    frames, tags = {}, defaultdict(set)
    for frame in data["base"]:
        index = int(frame["iteration"]) - offset
        if not 0 <= index < count or index in frames:
            raise ValueError(f"Duplicate or out-of-range base frame: {index}")
        objects = {}
        for obj in frame["objects"]:
            actor = int(obj["id"])
            if actor in objects:
                raise ValueError("Duplicate actor in one frame")
            objects[actor] = obj
            tags[actor].add(int(obj["tag"]))
        frames[index] = objects
    collisions = data["collision"]
    first_by_pair = {}
    possible_contact_pairs = set()
    collision_actor_ids = set()
    for collision in collisions:
        index = int(collision["iteration"]) - offset
        if not 0 <= index < count:
            raise ValueError("Out-of-range collision frame")
        ids = tuple(sorted(set(int(actor) for actor in collision["ids"])))
        collision_actor_ids.update(ids)
        # Some frames list several colliding IDs without identifying edges.
        # Any pair within that group is ambiguous, so must never be a negative.
        possible_contact_pairs.update(combinations(ids, 2))
        if len(ids) == 2:
            first_by_pair[ids] = min(index, first_by_pair.get(ids, count))
    if not collisions:
        return [], {"no_collision": 1}
    first_frame = min(int(c["iteration"]) - offset for c in collisions)
    if first_frame != int(row["accident_frame"]):
        raise ValueError("Collision frame does not match metadata")
    width, height = int(row["width"]), int(row["height"])
    results, rejected = [], Counter()
    annotation_sha = digest(annotation)
    def entry(pair, contact, windows, role):
        return {"video": row["rgb_path"], "annotation": annotation.relative_to(root).as_posix(),
                "annotation_metadata_path": row["annotations_path"], "annotation_sha256": annotation_sha,
                "video_bytes": video_path.stat().st_size, "map": row["map"], "weather": row["weather"],
                "camera_position": int(row["camera_position"]), "event_type": row["type"],
                "actor_ids": list(pair), "actor_tags": [next(iter(tags[actor])) for actor in pair],
                "fps_metadata": fps, "frame_count_metadata": count, "width": width, "height": height,
                "first_pair_contact_frame": contact, "pair_role": role,
                "split": "validation" if row["map"] in VALIDATION_MAPS else "train",
                "source_group": f"synthetic:{row['map']}", "windows": windows}
    for pair, contact in sorted(first_by_pair.items()):
        if not all(len(tags[actor]) == 1 and next(iter(tags[actor])) in FOUR_WHEEL for actor in pair):
            rejected["not_two_consistent_four_wheel_actors"] += 1
            continue
        windows = []
        for label, anchor in ((1, contact / fps),
                              (0, first_frame / fps - POLICY["negative_clearance_seconds"] - 1.2)):
            window, reason = make_window(anchor, label, pair, frames, fps, count,
                                         width, height, first_frame / fps)
            if window is None:
                rejected[f"{'positive' if label else 'negative'}_{reason}"] += 1
            else:
                windows.append(window)
        # Only paired examples with a usable contact crop enter this manifest.
        if not any(window["label"] == 1 for window in windows):
            continue
        for window in windows:
            window["negative_kind"] = "precontact_same_pair" if window["label"] == 0 else None
        results.append(entry(pair, contact, windows, "annotated_contact_pair"))
    # Hard negatives include a crash participant paired with a bystander, and
    # bystander/bystander pairs. The same RGB scene may contain a real collision;
    # the label concerns contact BETWEEN THESE TWO IDs, never whole-scene NORMAL.
    contact_objects = frames.get(first_frame, {})
    eligible = [actor for actor, obj in contact_objects.items()
                if len(tags[actor]) == 1 and next(iter(tags[actor])) in FOUR_WHEEL
                and box_of(obj, width, height) is not None]
    involved = collision_actor_ids
    candidates = []
    for pair in combinations(sorted(eligible), 2):
        if pair in possible_contact_pairs:
            continue
        gap = normalized_box_gap(*(box_of(contact_objects[actor], width, height) for actor in pair))
        if gap <= 1.5:
            candidates.append((bool(set(pair) & involved), gap, pair))
    selected = 0
    for has_participant in (True, False):
        for _, gap, pair in sorted(c for c in candidates if c[0] == has_participant):
            window, reason = make_window(first_frame / fps, 0, pair, frames, fps, count,
                                         width, height, first_frame / fps, precontact_negative=False)
            if window is None:
                rejected[f"hard_negative_{reason}"] += 1
                continue
            window.update(label_source="synthetic_no_contact_between_these_ids",
                          negative_kind="other_partner_in_crash_scene" if has_participant else "nearby_bystanders")
            results.append(entry(pair, None, [window], "noncontact_hard_negative"))
            selected += 1
            break
    return results, rejected


def verify_preview(root, pairs, preview_path):
    """Decode 13 frames of each selected crop; render start/contact/aftermath."""
    chosen = []
    for event_type in ("head-on", "rear-end", "sideswipe", "t-bone"):
        for split in ("train", "validation"):
            candidates = [p for p in pairs if p["event_type"] == event_type and p["split"] == split
                          and any(w["label"] == 1 for w in p["windows"])]
            if candidates:
                chosen.append(candidates[0])
    canvas = np.zeros((len(chosen) * 205, 960, 3), dtype=np.uint8)
    checked = []
    for row_index, pair in enumerate(chosen):
        window = next(w for w in pair["windows"] if w["label"] == 1)
        cap = cv2.VideoCapture(str(source_path(root, pair["video"])))
        try:
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or abs(actual_fps - pair["fps_metadata"]) > .01 or actual_count != pair["frame_count_metadata"]:
                raise ValueError(f"Video header mismatch: {pair['video']}")
            decoded = []
            for index in window["sample_frame_indices"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (pair["height"], pair["width"]):
                    raise ValueError(f"Cannot decode frame {index}: {pair['video']}")
                decoded.append(frame)
            for column, sample in enumerate((0, 6, 12)):
                frame = decoded[sample]
                for actor, box in zip(pair["actor_ids"], window["sample_pair_boxes"][sample]):
                    if box is not None:
                        x1, y1, x2, y2 = map(round, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
                        cv2.putText(frame, f"GT {actor}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
                x1, y1, x2, y2 = window["crop_xyxy"]
                crop = frame[y1:y2, x1:x2]
                h, w = crop.shape[:2]
                scale = min(320 / w, 170 / h)
                tile = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))))
                y, x = row_index * 205 + 30, column * 320
                canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
                title = f"{pair['map']} {pair['event_type']} {window['sample_frame_indices'][sample] / actual_fps:.1f}s"
                cv2.putText(canvas, title, (x + 3, row_index * 205 + 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
            checked.append({"video": pair["video"], "decoded_frames": len(decoded)})
        finally:
            cap.release()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), canvas):
        raise OSError("Could not write preview")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    if args.output.resolve().is_relative_to(root) or (args.preview and args.preview.resolve().is_relative_to(root)):
        raise ValueError("Output must be outside the raw dataset")
    cv2.setNumThreads(1)
    metadata = root / "metadata-synthetic.csv"
    rows = pd.read_csv(metadata).to_dict("records")
    if len({r["rgb_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate metadata video path")
    pairs, rejected, errors = [], Counter(), []
    for i, row in enumerate(rows, 1):
        try:
            prepared, reasons = prepare_video(row, root)
            pairs.extend(prepared)
            rejected.update(reasons)
        except Exception as error:
            errors.append({"video": row["rgb_path"], "error": str(error)})
        if i % 250 == 0 or i == len(rows):
            print(f"Checked {i}/{len(rows)} | usable pairs {len(pairs)} | errors {len(errors)}", flush=True)
    if errors:
        raise ValueError(f"Source errors: {errors[:8]}")
    train_sources = {p["video"] for p in pairs if p["split"] == "train"}
    validation_sources = {p["video"] for p in pairs if p["split"] == "validation"}
    if train_sources & validation_sources:
        raise ValueError("Source leakage between splits")
    counts = {}
    for split in ("train", "validation"):
        subset = [p for p in pairs if p["split"] == split]
        counts[split] = {"videos": len({p["video"] for p in subset}), "pairs": len(subset),
                         "positive_windows": sum(w["label"] == 1 for p in subset for w in p["windows"]),
                         "negative_windows": sum(w["label"] == 0 for p in subset for w in p["windows"]),
                         "event_types": dict(Counter(p["event_type"] for p in subset))}
    checked = verify_preview(root, pairs, args.preview) if args.preview else []
    result = {"schema_version": 2, "task": "four_wheel_pair_contact_supervision",
              "source_root": str(root), "metadata_sha256": digest(metadata),
              "policy": POLICY, "validation_maps": sorted(VALIDATION_MAPS),
              "split_scope": "synthetic held-out maps; NOT independent real-world evaluation",
              "evaluation_requirement": "Full-video candidates must come from detector/tracker without GT time, boxes or actor IDs",
              "negative_label_scope": "No contact between the selected IDs; crash-scene hard negatives may contain other accidents. NOT whole-video NORMAL",
              "preparation_code_sha256": digest(__file__),
              "official_real_test_used": False, "training_started": False,
              "deployment_approved": False, "source_errors": errors,
              "counts": counts, "rejections": dict(rejected),
              "decoded_sample_checks": checked, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"counts": counts, "rejections": dict(rejected), "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
