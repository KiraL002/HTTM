"""Image-space pair proposals, independent of accident labels and classifiers.

Proximity creates candidates only. Perspective overlap, queues and ordinary
passing traffic also create candidates. Missing detections remain missing;
there is no claim that two different tracker IDs are the same vehicle.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import numpy as np

PAIR_POLICY = {"version": 2, "four_wheel_coco_classes": [2, 5, 7],
               "near_normalized_box_gap": .60, "missing_or_separated_grace_seconds": 1.2,
               "minimum_near_observations": 2, "minimum_near_span_seconds": .10,
               "window_seconds": 2.4, "window_stride_seconds": 1.2,
               "sample_frames": 13, "sample_fps": 5, "crop_context_fraction": .25,
               "lost_actor_crop_envelope_seconds": 1.2,
               "lost_actor_crop_shift_limit_box_sizes": 2.0}


def normalized_box_gap(a, b):
    """Image-space edge distance / smaller box diagonal; never metres/TTC."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.)
    dy = max(a[1] - b[3], b[1] - a[3], 0.)
    scale = min(math.hypot(a[2] - a[0], a[3] - a[1]),
                math.hypot(b[2] - b[0], b[3] - b[1]))
    return math.hypot(dx, dy) / max(1., scale)


def box_iou(a, b):
    area = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / max(union, 1.)


class PairCandidateTracker:
    """Consume every frame's detector tracks, then export fixed-length windows.

Input rows: id, COCO class, detector confidence, x1, y1, x2, y2.
Frame indices are zero based. Whole-video replay permits the same symmetric
pre/post context as supervised crops. A live consumer must wait for post-context.
"""

    def __init__(self, fps, width, height):
        if not math.isfinite(fps) or fps < 5 or width < 1 or height < 1:
            raise ValueError("Need valid image dimensions and source fps >= sample fps (5)")
        self.fps, self.width, self.height = float(fps), int(width), int(height)
        self.frames, self.active, self.episodes = [], {}, []
        self.classes, self.rejections = {}, Counter()
        self.finalized = False

    def _close(self, key):
        state = self.active.pop(key)
        if (state["near_observations"] >= PAIR_POLICY["minimum_near_observations"] and
                (state["last_near_frame"] - state["first_near_frame"]) / self.fps >= PAIR_POLICY["minimum_near_span_seconds"] - 1e-8):
            self.episodes.append(state)
        else:
            self.rejections["insufficient_temporal_support"] += 1

    def update(self, frame_index, tracks):
        if self.finalized or frame_index != len(self.frames):
            raise ValueError("Frames must be consumed once in consecutive zero-based order")
        rows = np.asarray(tracks, dtype=float).reshape(-1, 7)
        current = {}
        for row in rows:
            if not np.isfinite(row).all():
                self.rejections["nonfinite_detection"] += 1
                continue
            actor, cls, confidence, x1, y1, x2, y2 = row.tolist()
            if actor != int(actor) or actor < 0 or cls != int(cls) or not 0 <= confidence <= 1:
                raise ValueError("Invalid detector ID/class/confidence")
            actor, cls = int(actor), int(cls)
            if cls not in PAIR_POLICY["four_wheel_coco_classes"]:
                continue
            box = [max(0., min(self.width, x1)), max(0., min(self.height, y1)),
                   max(0., min(self.width, x2)), max(0., min(self.height, y2))]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                self.rejections["tiny_or_invalid_box"] += 1
                continue
            if actor in current:
                raise ValueError("Duplicate tracker ID within one frame")
            current[actor] = box
            self.classes.setdefault(actor, Counter())[cls] += 1
        self.frames.append(current)
        for key in list(self.active):
            if (frame_index - self.active[key]["last_near_frame"]) / self.fps > PAIR_POLICY["missing_or_separated_grace_seconds"]:
                self._close(key)
        for first, second in combinations(sorted(current), 2):
            key = (first, second)
            a, b = current[first], current[second]
            gap = normalized_box_gap(a, b)
            if gap > PAIR_POLICY["near_normalized_box_gap"]:
                continue
            state = self.active.get(key)
            if state is None:
                state = {"track_ids": list(key), "first_near_frame": frame_index,
                         "last_near_frame": frame_index, "near_observations": 0,
                         "closest_observed_frame": frame_index, "minimum_normalized_gap": gap,
                         "initial_normalized_gap": gap, "maximum_observed_iou": 0.}
                self.active[key] = state
            state["last_near_frame"] = frame_index
            state["near_observations"] += 1
            if gap < state["minimum_normalized_gap"]:
                state["minimum_normalized_gap"] = gap
                state["closest_observed_frame"] = frame_index
            state["maximum_observed_iou"] = max(state["maximum_observed_iou"], box_iou(a, b))

    def _window(self, episode, anchor):
        last_time = (len(self.frames) - 1) / self.fps
        span = PAIR_POLICY["window_seconds"]
        if last_time < span:
            return None
        start = min(max(0., anchor / self.fps - span / 2), last_time - span)
        indices = np.rint((start + np.arange(PAIR_POLICY["sample_frames"]) / PAIR_POLICY["sample_fps"]) * self.fps).astype(int)
        actors = episode["track_ids"]
        samples = [[self.frames[i].get(actor) for actor in actors] for i in indices]
        # Use actual observations throughout the window for one stable crop.
        # No predicted/held-last boxes count as evidence during an occlusion.
        boxes = [self.frames[i][actor] for i in range(int(indices[0]), int(indices[-1]) + 1)
                 for actor in actors if actor in self.frames[i]]
        if not boxes:
            return None
        envelopes = []
        for actor in actors:
            history = [(i, self.frames[i][actor]) for i in range(int(indices[0]), int(indices[-1]) + 1)
                       if actor in self.frames[i]]
            if len(history) < 2:
                continue
            last_index, last_box = history[-1]
            if last_index == indices[-1]:
                continue
            recent = [(i, b) for i, b in history if i >= last_index - .4 * self.fps]
            if len(recent) < 2:
                continue
            earlier_index, earlier_box = recent[0]
            last_box, earlier_box = np.asarray(last_box), np.asarray(earlier_box)
            velocity = ((last_box[:2] + last_box[2:] - earlier_box[:2] - earlier_box[2:]) / 2
                        / (last_index - earlier_index))
            horizon = min(int(indices[-1]) - last_index,
                          round(PAIR_POLICY["lost_actor_crop_envelope_seconds"] * self.fps))
            size = last_box[2:] - last_box[:2]
            shift = np.clip(velocity * horizon,
                            -size * PAIR_POLICY["lost_actor_crop_shift_limit_box_sizes"],
                            size * PAIR_POLICY["lost_actor_crop_shift_limit_box_sizes"])
            # Only broaden the image crop when an actor disappears during impact.
            # This envelope is NOT a detection, ID association or contact evidence.
            envelope = last_box + np.tile(shift, 2)
            envelope[[0, 2]] = np.clip(envelope[[0, 2]], 0, self.width)
            envelope[[1, 3]] = np.clip(envelope[[1, 3]], 0, self.height)
            if envelope[2] > envelope[0] and envelope[3] > envelope[1]:
                boxes.append(envelope.tolist())
                envelopes.append({"track_id": actor, "last_observed_frame": last_index,
                                  "projection_frame": last_index + horizon,
                                  "xyxy": envelope.tolist(), "scope": "crop_context_only_not_detection"})
        array = np.asarray(boxes)
        low, high = array[:, :2].min(axis=0), array[:, 2:].max(axis=0)
        padding = (high - low) * PAIR_POLICY["crop_context_fraction"]
        low = np.maximum(0, np.floor(low - padding)).astype(int)
        high = np.minimum([self.width, self.height], np.ceil(high + padding)).astype(int)
        joint = sum(all(box is not None for box in sample) for sample in samples)
        gaps = [normalized_box_gap(*sample) if all(box is not None for box in sample) else None for sample in samples]
        present = [gap for gap in gaps if gap is not None]
        # Scheduling hint only: approach is more useful than a permanent
        # perspective overlap. This value is not a probability or a verdict.
        early = [gap for gap in gaps[:5] if gap is not None]
        approach = max(0., max(early) - min(present)) if early and present else 0.
        return {"track_ids": actors, "context_start_seconds": round(start, 6),
                "context_end_seconds": round(start + span, 6),
                "sample_frame_indices": indices.tolist(), "sample_pair_boxes": samples,
                "joint_visible_samples": joint, "crop_xyxy": [*low.tolist(), *high.tolist()],
                "requires_occlusion_handling": joint < len(samples),
                "sample_normalized_box_gaps": gaps,
                "lost_actor_crop_envelopes": envelopes,
                "approach_priority": round(approach, 6)}

    def finish(self, frame_count):
        if self.finalized or frame_count != len(self.frames):
            raise ValueError("Finish once, with the complete consumed frame count")
        for key in list(self.active):
            self._close(key)
        self.finalized = True
        encounters, windows = [], []
        for episode in sorted(self.episodes, key=lambda e: (e["first_near_frame"], e["track_ids"])):
            encounter_id = len(encounters)
            episode = dict(episode, encounter_id=encounter_id,
                           near_start_seconds=episode["first_near_frame"] / self.fps,
                           near_end_seconds=episode["last_near_frame"] / self.fps,
                           dominant_coco_classes=[self.classes[actor].most_common(1)[0][0] for actor in episode["track_ids"]])
            encounters.append(episode)
            stride = max(1, round(PAIR_POLICY["window_stride_seconds"] * self.fps))
            anchors = sorted({*range(episode["first_near_frame"], episode["last_near_frame"] + 1, stride),
                              episode["closest_observed_frame"], episode["last_near_frame"]})
            seen = set()
            for anchor in anchors:
                window = self._window(episode, anchor)
                if window is not None:
                    key = window["sample_frame_indices"][0]
                    if key not in seen:
                        windows.append(dict(window, encounter_id=encounter_id))
                        seen.add(key)
        return {"schema_version": 1, "scope": "detector_pair_proposals_only",
                "accident_classification_performed": False, "ground_truth_used_for_proposals": False,
                "time_scope": "near intervals and context windows; NOT accident start/end",
                "identity_scope": "temporary ByteTrack IDs; no automatic cross-ID identity merge",
                "policy": dict(PAIR_POLICY), "fps": self.fps, "frames_processed": frame_count,
                "width": self.width, "height": self.height, "rejections": dict(self.rejections),
                "encounters": encounters, "windows": windows}
