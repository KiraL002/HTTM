"""Temporal anomaly proposals followed by the existing learned verifiers.

No reference labels, file names or precomputed predictions enter this path.
An anomaly opens a search interval; it never creates an accident or an actor ID.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def intervals(scan, padding=1.2):
    """Merge high-score episodes plus pre/post verification context."""
    duration = scan['duration_s']
    merged = []
    for item in scan['models']['temporal']['review_candidates']:
        lo = max(0., item['context_start_s']-padding)
        hi = min(duration, item['context_end_s']+padding)
        if merged and lo <= merged[-1]['end_s']:
            merged[-1]['end_s'] = max(hi, merged[-1]['end_s'])
            merged[-1]['peak_score'] = max(item['peak_score'], merged[-1]['peak_score'])
        else:
            merged.append(dict(start_s=lo, end_s=hi, peak_score=item['peak_score']))
    return merged


def scan(source, settings, root, progress=None):
    from core.temporal_anomaly import load_model, scan_video, sha
    import torch
    checkpoint = (Path(root)/settings['model']).resolve()
    if not checkpoint.is_relative_to(Path(root).resolve()) or sha(checkpoint) != settings['model_sha256']:
        raise ValueError('Temporal checkpoint differs from the configured model')
    torch.set_num_threads(4)
    model = load_model(checkpoint)
    value = scan_video(source, {'temporal': model}, progress=progress)
    value.update(model_sha256=settings['model_sha256'],
                 model_source=settings['source'], inference_cache_reused=False)
    value['intervals'] = intervals(value)
    value['verification'] = []
    return value


def pair_windows(proposal, observations, regions):
    """Densify windows around genuine observed contacts within anomaly regions.

    Contact gates and the trained 13-frame crop/feature protocol stay intact.
    This uses the normal candidate builder's exact crop implementation.
    """
    from .vehicle_pairs import PairCandidateTracker
    fps = proposal['fps']
    builder = PairCandidateTracker(fps, proposal['width'], proposal['height'])
    builder.frames = [{} for _ in range(proposal['frames_processed'])]
    for row in observations:
        if int(row[2]) in {2, 5, 7}:
            builder.frames[int(row[0])][int(row[1])] = row[4:8].tolist()
    seen = {(tuple(w['track_ids']), tuple(w['sample_frame_indices'])) for w in proposal['windows']}
    extra = []
    for region in regions:
        for contact in proposal['contacts']:
            if not region['start_s'] <= contact['time_s'] <= region['end_s']:
                continue
            for shift in (-.4, -.2, 0., .2, .4):
                w = builder._window({'track_ids': contact['track_ids']},
                                    round((contact['time_s']+shift)*fps))
                if w is None:
                    continue
                key = (tuple(w['track_ids']), tuple(w['sample_frame_indices']))
                if key not in seen and w['context_start_seconds'] <= contact['time_s'] <= w['context_end_seconds']:
                    extra.append(dict(w, proposal_origin='temporal_anomaly'))
                    seen.add(key)
    return extra


def single_windows(proposal, observations, regions, existing):
    """Search a 0.2s grid inside anomaly regions with unchanged motion gates."""
    from .single_vehicle import sample_actor, geometry, crop_bounds, motion_supported
    from .single_vehicle_runtime import coherent_identity
    fps, last = proposal['fps'], (proposal['frames_processed']-1)/proposal['fps']
    seen = {(w['actor_id'], tuple(w['indices'])) for w in existing}
    extra = []
    for region in regions:
        lo, hi = max(1.2, region['start_s']), min(last-1.2, region['end_s'])
        for actor in np.unique(observations[:, 1]).astype(int):
            for centre in np.arange(lo, hi+1e-6, .2):
                indices = np.rint((centre-1.2+np.arange(13)/5)*fps).astype(int).tolist()
                key = (int(actor), tuple(indices))
                if key in seen:
                    continue
                boxes = sample_actor(observations, actor, indices, fps,
                                     tolerance_frames=max(1, round(.15*fps)))
                if boxes is None:
                    continue
                _, motion = geometry(boxes, indices, fps)
                if not motion_supported(motion):
                    continue
                w = dict(actor_id=int(actor), indices=indices, boxes=boxes, motion=motion,
                         anchor_frame=indices[6], time_s=indices[6]/fps,
                         crop=crop_bounds(boxes, proposal['width'], proposal['height']),
                         proposal_origin='temporal_anomaly')
                if coherent_identity(w, observations, fps):
                    extra.append(w)
                    seen.add(key)
    return extra


def merge_events(established, additions):
    """Keep established event IDs, times and scores; suppress overlapping extras."""
    merged = list(established)
    for event in sorted(additions, key=lambda e: e['time_s']):
        if any(set(event['track_ids']) & set(old['track_ids']) and
               abs(event['time_s']-old['time_s']) <= 2.4 for old in merged):
            continue
        merged.append(dict(event, candidate_source='temporal_anomaly_then_existing_verifier'))
    return sorted(merged, key=lambda e: e['time_s'])


def unexplained_regions(regions, events):
    """A global scene episode already explained by an event adds no actors.

    Otherwise a real collision can make every ordinary nearby vehicle a new
    candidate. Preserve pair-first / single-fallback priority per episode.
    """
    return [r for r in regions if not any(r['start_s'] <= e['time_s'] <= r['end_s']
                                         for e in events)]


def verify(review, proposal, scored, policy, output, established, progress=None):
    """Only the original pair/single event rules can admit an extra event."""
    from .pair_pipeline import Verifier, atomic_json
    from .pair_events import decide
    from .single_vehicle_runtime import analyze, candidate_windows
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(proposal['detection_trace'], allow_pickle=False) as data:
        observations = data['observations']
    observations = observations[np.isin(observations[:, 2], [2, 5, 7])]
    regions = review['intervals']
    if not regions:
        review['extra_pair_windows'] = review['extra_single_windows'] = 0
        return established, False
    pending = unexplained_regions(regions, established)
    extra_pairs = pair_windows(proposal, observations, pending)
    review.update(extra_pair_windows=len(extra_pairs), extra_single_windows=0,
                  regions_already_explained=len(regions)-len(pending))
    additions = []
    extra_scores = []
    if extra_pairs:
        additional = dict(proposal, windows=extra_pairs)
        path = output/'temporal_pairs.json'
        atomic_json(additional, path)
        extra_scores = Verifier().score(additional, path)['scores']
        additions.extend(decide(proposal['contacts'], extra_scores, policy['threshold']))
    remaining = unexplained_regions(pending, additions)
    base_single = candidate_windows(observations, proposal) if remaining else []
    extra_single = single_windows(proposal, observations, remaining, base_single)
    review['extra_single_windows'] = len(extra_single)
    single_evaluated = False
    single_result = None
    if extra_single:
        single_result = analyze(proposal, output=output/'single', threshold=policy['single_vehicle']['threshold'],
                                additional_windows=extra_single)
        if single_result['identity']['model_sha256'] != policy['single_vehicle']['model_sha256']:
            raise ValueError('Temporal search used a different single-vehicle model')
        single_evaluated = True
        extra_keys = {(w['actor_id'], tuple(w['indices'])) for w in extra_single}
        # Require at least one extra grid anchor to support an added event.
        extra_anchors = {(actor, indices[6]) for actor, indices in extra_keys}
        additions.extend(e for e in single_result['events'] if any(
            r['start_s'] <= e['time_s'] <= r['end_s'] for r in remaining) and any(
            (actor, anchor) in extra_anchors for actor in e['track_ids']
            for anchor in e.get('support_window_anchor_frames', [])))
    merged = merge_events(established, additions)
    for region in regions:
        inside = [e for e in merged if region['start_s'] <= e['time_s'] <= region['end_s']]
        review['verification'].append(dict(region,
            state='incident_evidence' if inside else 'not_confirmed',
            event_times_s=[e['time_s'] for e in inside],
            observed_track_ids=sorted({i for e in inside for i in e['track_ids']})))
    review['added_events'] = sum(e.get('candidate_source') is not None for e in merged)
    if progress:
        progress(.93, 'Đã xác minh các đoạn bất thường')
    return merged, single_evaluated
