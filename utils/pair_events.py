"""Pair-contact evidence and consensus decisions, with no demo identities.

Birth near another box is not proof of a second independent vehicle. A pair
must have been observed apart before a new contact transition can be proposed.
Context-window bounds never become accident-duration estimates.
"""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import numpy as np

from .vehicle_pairs import normalized_box_gap

POLICY = {"version": 1, "detector_sampling": "every_native_frame", "independent_gap": .75,
          "contact_gap": .15, "history_seconds": 4., "minimum_history_samples": 3,
          "minimum_history_span_seconds": .4, "score_floor": .5,
          "consensus": "minimum_rgb_and_geometry_score",
          "threshold_selection": "max_0_5_and_nextafter_max_calibration_normal_video_score",
          "timing": "first_observed_contact_after_separation_not_full_accident_duration"}


def contact_proposals(observations, fps):
    """Input Nx8: native frame, temporary ID, class, confidence, xyxy."""
    a=np.asarray(observations,dtype=float).reshape(-1,8)
    histories=defaultdict(deque);armed={};contacts=[]
    if not np.isfinite(a).all() or fps<=0:raise ValueError("Invalid observations")
    for frame in np.unique(a[:,0]).astype(int):
        rows=a[a[:,0]==frame]
        current={int(r[1]):r[4:8] for r in rows if int(r[2]) in {2,5,7}}
        for pair in combinations(sorted(current),2):
            gap=normalized_box_gap(current[pair[0]],current[pair[1]])
            history=histories[pair]
            while history and (frame-history[0][0])/fps>POLICY["history_seconds"]:history.popleft()
            if gap>=POLICY["independent_gap"]:
                armed[pair]=True
            if gap<=POLICY["contact_gap"] and armed.get(pair,False):
                independent=[r for r in history if r[1]>=POLICY["independent_gap"]]
                if (independent and len(history)>=POLICY["minimum_history_samples"] and
                        (frame-history[0][0])/fps>=POLICY["minimum_history_span_seconds"]-1e-8):
                    witness=max(independent,key=lambda r:r[1])
                    contacts.append({"track_ids":list(pair),"frame_index":int(frame),"time_s":float(frame/fps),
                                     "witness_frame":int(witness[0]),"witness_normalized_gap":float(witness[1]),
                                     "contact_normalized_gap":float(gap)})
                    armed[pair]=False
            history.append((frame,gap))
    return contacts


def decide(contacts, scored_windows, threshold):
    if not .5<=threshold<=1.000001:raise ValueError("Uncalibrated threshold")
    events=[]
    for contact in contacts:
        candidates=[s for s in scored_windows if s["track_ids"]==contact["track_ids"]
                    and s["context_start_seconds"]<=contact["time_s"]<=s["context_end_seconds"]]
        if not candidates:continue
        strongest=max(candidates,key=lambda s:s["joint_score"])
        if strongest["joint_score"]>=threshold:
            events.append(dict(contact,score=strongest["joint_score"],
                               rgb_score=strongest["rgb_score"],geometry_score=strongest["geometry_score"],
                               timing_scope=POLICY["timing"]))
    return events
