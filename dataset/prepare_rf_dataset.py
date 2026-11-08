"""Join explicitly reviewed vehicle-track intervals to one raw feature export."""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import tempfile
from collections import defaultdict
from pathlib import Path

from core.feature_pipeline import RF_FEATURE_COLUMNS


KEY_FIELDS = ("extraction_id", "source_sha256", "feature_signature", "track_id")
REVIEW_FIELDS = (*KEY_FIELDS, "start_frame", "end_frame", "label", "label_scope",
                 "human_verified", "label_source")


def integer(value, name, minimum=0):
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be an integer") from None
    if str(result) != str(value).strip() or result < minimum:
        raise ValueError(f"Invalid {name}: {value}")
    return result


def identity(row):
    for name in KEY_FIELDS:
        if not str(row.get(name, "")).strip():
            raise ValueError(f"Missing {name}; labels must refer to the exact source and extraction")
    return tuple(str(row[name]).strip() for name in KEY_FIELDS[:-1]) + (integer(row["track_id"], "track_id"),)


def read_reviews(path: Path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not set(REVIEW_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError(f"Reviewed labels require columns: {REVIEW_FIELDS}")
        for row in reader:
            if row["label_scope"] != "vehicle_track_interval":
                raise ValueError("Scene/video labels cannot label every vehicle; review vehicle_track_interval labels")
            if row["human_verified"].strip().lower() not in ("1", "true"):
                raise ValueError("Only independently reviewed human_verified=1 labels may be joined")
            if not row["label_source"].strip() or row["label_source"].lower().strip() in ("status", "heuristic", "model_prediction"):
                raise ValueError("label_source must describe independent annotation/review")
            start = integer(row["start_frame"], "start_frame", 1)
            end = integer(row["end_frame"], "end_frame", start)
            try:
                label = int(row["label"])
            except (ValueError, TypeError):
                raise ValueError("label must be -1, 0 or 1") from None
            if str(label) != row["label"].strip() or label not in (-1, 0, 1):
                raise ValueError("label must be -1, 0 or 1")
            groups[identity(row)].append({"start": start, "end": end, "label": label,
                                         "source": row["label_source"].strip(), "matched": 0})
    if not groups:
        raise ValueError("No reviewed label intervals")
    for intervals in groups.values():
        intervals.sort(key=lambda item: item["start"])
        for previous, current in zip(intervals, intervals[1:]):
            if current["start"] <= previous["end"]:
                raise ValueError("Overlapping reviewed intervals for the same vehicle/extraction")
    return groups


def prepare_dataset(features: Path, reviews: Path, output: Path, *, overwrite=False):
    features, reviews, output = map(lambda path: Path(path).resolve(), (features, reviews, output))
    if output in (features, reviews):
        raise ValueError("Output must not replace source features or reviewed labels")
    if output.suffix.lower() != ".csv":
        raise ValueError("Select an explicit output CSV")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    groups = read_reviews(reviews)
    starts = {key: [item["start"] for item in intervals] for key, intervals in groups.items()}
    total = {"raw_rows": 0, "labeled_rows": 0, "normal": 0, "accident": 0, "ignored": 0, "unreviewed": 0}
    temporary = None
    source_splits, frame_state = {}, {}
    try:
        with features.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames or []
            required = {*KEY_FIELDS, *RF_FEATURE_COLUMNS, "frame_id", "video_id", "split_group"}
            if not required <= set(fields):
                raise ValueError(f"Raw feature export is missing columns: {sorted(required - set(fields))}")
            if any(name in fields for name in ("label", "status", "accident_probability")):
                raise ValueError("Use raw unlabeled features, not inference statuses or existing labels")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=output.parent,
                                             prefix=".labeled-", suffix=".tmp", delete=False) as destination:
                temporary = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=[*fields, "label", "label_scope", "human_verified", "label_source"])
                writer.writeheader()
                for row in reader:
                    total["raw_rows"] += 1
                    key, frame = identity(row), integer(row["frame_id"], "frame_id", 1)
                    split = row["split_group"].strip()
                    if not split:
                        raise ValueError("split_group is required to prevent source/camera leakage")
                    sha = row["source_sha256"]
                    if sha in source_splits and source_splits[sha] != split:
                        raise ValueError("The same source video cannot belong to different split groups")
                    source_splits[sha] = split
                    # Bounded per-video state catches duplicate/out-of-order rows.
                    video_key = key[:-1]
                    previous, seen_tracks = frame_state.get(video_key, (0, set()))
                    if frame < previous or (frame == previous and key[-1] in seen_tracks):
                        raise ValueError("Duplicate or out-of-order source/frame/track feature rows")
                    if frame != previous:
                        seen_tracks = set()
                    seen_tracks.add(key[-1])
                    frame_state[video_key] = (frame, seen_tracks)
                    if key not in groups:
                        total["unreviewed"] += 1
                        continue
                    index = bisect.bisect_right(starts[key], frame) - 1
                    interval = groups[key][index] if index >= 0 else None
                    if interval is None or frame > interval["end"]:
                        total["unreviewed"] += 1
                        continue
                    interval["matched"] += 1
                    if interval["label"] == -1:
                        total["ignored"] += 1
                        continue
                    if not all(math.isfinite(float(row[name])) for name in RF_FEATURE_COLUMNS):
                        raise ValueError("Labeled rows must contain all finite runtime features")
                    writer.writerow({**row, "label": interval["label"], "label_scope": "vehicle_track_interval",
                                     "human_verified": 1, "label_source": interval["source"]})
                    total["labeled_rows"] += 1
                    total["accident" if interval["label"] else "normal"] += 1
        if any(not interval["matched"] for intervals in groups.values() for interval in intervals):
            raise ValueError("Some reviewed intervals match no feature rows; check source, extraction ID, signature, track and sampling")
        if not total["labeled_rows"]:
            raise ValueError("No reviewed non-ignored samples; no training CSV saved")
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        temporary.replace(output)
        return total
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(prepare_dataset(args.features, args.reviews, args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
