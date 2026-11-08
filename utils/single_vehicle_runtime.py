"""Experimental one-actor verifier, isolated from the established pair decision."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import joblib,numpy as np,torch
from .single_vehicle import PROTOCOL,proposals,geometry,read_clip,sample_actor,crop_bounds,motion_supported
from .pair_verifier import build_encoder,encode

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'outputs/single_vehicle_training/single_vehicle_head.joblib'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def coherent_identity(window,observations,fps):
    sizes=np.array([np.array(b)[2:]-np.array(b)[:2] for b in window['boxes'] if b is not None])
    if np.median(sizes[:,0])<14 or np.median(sizes[:,1])<12:return False
    rows=observations[(observations[:,1]==window['actor_id']) &
                      (observations[:,0]>=window['indices'][0]) & (observations[:,0]<=window['indices'][-1])]
    if len(rows)<5:return False
    centres=(rows[:,4:6]+rows[:,6:8])/2;wh=rows[:,6:8]-rows[:,4:6]
    distance=np.linalg.norm(np.diff(centres,axis=0),axis=1)/np.maximum(1.,np.linalg.norm((wh[1:]+wh[:-1])/2,axis=1))
    dt=np.maximum(np.diff(rows[:,0])/fps,.001)
    # Large image-space teleportation is an ID association failure, not impact evidence.
    return not bool(np.any((distance>.75) & (distance/dt>8.)))


def candidate_windows(observations, proposal):
    """Confirm an arrest on a staggered grid, without altering trained features.

    A short track can have just one 2.4-second window on the original .4s
    grid. Neighbouring *observed* windows prevent a grid boundary from hiding
    corroboration. They use the same model, sampling and probability cutoff.
    """
    fps=proposal['fps']
    base=proposals(observations,fps,proposal['frames_processed'],proposal['width'],proposal['height'])
    windows={(w['actor_id'],tuple(w['indices'])):w for w in base}
    for w in base:
        m=w['motion']
        if not (m['speed_before']>.6 and m['speed_after']<.3*m['speed_before']):continue
        for offset in (-.2,.2):
            indices=np.rint(np.asarray(w['indices'])+offset*fps).astype(int).tolist()
            if indices[0]<0 or indices[-1]>=proposal['frames_processed']:continue
            boxes=sample_actor(observations,w['actor_id'],indices,fps,tolerance_frames=max(1,round(.15*fps)))
            if boxes is None:continue
            _,motion=geometry(boxes,indices,fps)
            if not motion_supported(motion):continue
            extra=dict(w,indices=indices,boxes=boxes,motion=motion,anchor_frame=indices[6],time_s=indices[6]/fps,
                       crop=crop_bounds(boxes,proposal['width'],proposal['height']))
            windows[(extra['actor_id'],tuple(indices))]=extra
    return [w for w in windows.values() if coherent_identity(w,observations,fps)]

def analyze(proposal,output=None,progress=None,threshold=None):
    output=Path(output or ROOT/'outputs/single_vehicle_runtime');output.mkdir(parents=True,exist_ok=True)
    package=joblib.load(MODEL)
    if package['protocol']!=PROTOCOL or package['input_code_sha256']!=sha(ROOT/'utils/single_vehicle.py'):
        raise ValueError('Single-vehicle input protocol changed')
    encoder_source=(ROOT/package['encoder_source']).resolve()
    if not encoder_source.is_relative_to(ROOT) or sha(encoder_source)!=package['encoder_source_sha256']:
        raise ValueError('Single-vehicle encoder changed')
    identity={'model_sha256':sha(MODEL),'trace_sha256':sha(proposal['detection_trace']),
              'source_sha256':sha(proposal['source']),'input_code_sha256':package['input_code_sha256'],
              'runtime_sha256':sha(__file__),'protocol':PROTOCOL}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();cache=output/(key+'.json')
    if cache.exists():
        value=json.loads(cache.read_text(encoding='utf-8'))
        if value['identity']!=identity or value['complete'] is not True:raise ValueError('Invalid single-vehicle score cache')
    else:
        with np.load(proposal['detection_trace'],allow_pickle=False) as data:observations=data['observations']
        observations=observations[np.isin(observations[:,2],[2,5,7])]
        windows=candidate_windows(observations,proposal)
        encoder=None;rows=[];torch.set_num_threads(4)
        for i,w in enumerate(windows):
            rotation=(w['motion']['course_change_degrees']>=60 and w['motion']['shape_change']>=.4)
            needs_rgb=package.get('requires_rgb',True) or (rotation and 'rotation_classifier' in package)
            if needs_rgb and encoder is None:
                original=torch.load(encoder_source,map_location='cpu',weights_only=True)
                encoder=build_encoder(original['encoder_state'])
            g,_=geometry(w['boxes'],w['indices'],proposal['fps'])
            rgb=(encode(encoder,read_clip(proposal['source'],w)[None],torch.device('cpu'))[0]
                 if needs_rgb else np.zeros(2048,np.float32))
            score=float(package['classifier'].predict_proba(np.r_[rgb,g][None])[0,1])
            rotation_score=(float(package['rotation_classifier'].predict_proba(np.r_[rgb,g][None])[0,1])
                            if rotation and 'rotation_classifier' in package else None)
            rows.append(dict(w,score=score,rgb_evaluated=needs_rgb,rotation_score=rotation_score))
            if progress:progress(.80+.14*(i+1)/max(1,len(windows)),'Đang xác minh sự cố từng xe')
        value={'identity':identity,'complete':True,'source':proposal['source'],'scores':rows,
               'model_validation_threshold':package['threshold'],'rotation_threshold':package.get('rotation_threshold'),'deployment_approved':False}
        temporary=cache.with_suffix('.tmp');temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf-8');temporary.replace(cache)
    cutoff=float(package['threshold'] if threshold is None else threshold)
    value=dict(value,threshold=cutoff,events=events(value['scores'],proposal,cutoff,package.get('rotation_threshold')))
    return value

def events(scores,proposal,threshold,rotation_threshold=None):
    if not .5<=threshold<=1.000001:raise ValueError('Invalid single-vehicle threshold')
    with np.load(proposal['detection_trace'],allow_pickle=False) as data:observations=data['observations']
    # Reject isolated score spikes: require support in two successive scan windows.
    supported=[]
    # Starting to move and turning out of a driveway is not established loss of
    # control. The geometry-only fallback needs the same pre-motion baseline
    # (.6 box diagonals/s) as its arrest gate. RGB rotation has its own gate.
    passing=[r for r in scores if r['score']>=threshold and r['motion']['speed_before']>.6
             and coherent_identity(r,observations,proposal['fps'])]
    for row in passing:
        nearby=[other for other in passing if other['actor_id']==row['actor_id'] and .19<=abs(other['time_s']-row['time_s'])<=.81]
        if nearby:
            supported.append(dict(row,verifier='single_geometry',event_threshold=threshold,
                                  evidence_bounds=[min(r['indices'][0] for r in [row,*nearby]),
                                                   max(r['indices'][-1] for r in [row,*nearby])],
                                  support_window_anchor_frames=sorted({r['anchor_frame'] for r in [row,*nearby]})))
    if rotation_threshold is not None:
        severe=[r for r in scores if r.get('rotation_score') is not None
                and r['rotation_score']>=rotation_threshold and coherent_identity(r,observations,proposal['fps'])]
        for row in severe:
            nearby=[r for r in severe if r['actor_id']==row['actor_id'] and .19<=abs(r['time_s']-row['time_s'])<=.81]
            if not nearby:continue
            # Timestamp the corroborated turn, never the beginning of its context crop.
            witness=max([row,*nearby],key=lambda r:r['time_s'])
            supported.append(dict(witness,score=min(r['rotation_score'] for r in [row,*nearby]),verifier='single_rgb_rotation',event_threshold=rotation_threshold,
                                  evidence_bounds=[min(r['indices'][0] for r in [row,*nearby]),max(r['indices'][-1] for r in [row,*nearby])],
                                  support_window_anchor_frames=sorted({r['anchor_frame'] for r in [row,*nearby]})))
    selected={}
    for row in sorted(supported,key=lambda r:r['time_s']):
        actor=row['actor_id'];at=observations[observations[:,1]==actor]
        closest=at[np.argmin(abs(at[:,0]-row['anchor_frame']))]
        if abs(closest[0]-row['anchor_frame'])>max(1,round(.3*proposal['fps'])):continue
        frame=int(closest[0]);when=frame/proposal['fps']
        previous=selected.get(actor)
        if previous is not None and when-previous['time_s']<=2.4:
            previous['supporting_windows']+=1
            previous['last_support_time_s']=when
            previous['support_window_anchor_frames']=sorted(set(previous['support_window_anchor_frames']+row['support_window_anchor_frames']))
            bounds=row.get('evidence_bounds',[row['indices'][0],row['indices'][-1]])
            previous['evidence_window_seconds'][0]=min(previous['evidence_window_seconds'][0],bounds[0]/proposal['fps'])
            previous['evidence_window_seconds'][1]=max(previous['evidence_window_seconds'][1],bounds[-1]/proposal['fps'])
            previous['evidence_available_time_s']=previous['evidence_window_seconds'][1]
            continue
        # The timestamp is a motion-change observation, not the whole context window.
        m=row['motion'];event={'kind':'single_vehicle_incident','track_ids':[actor],
              'frame_index':frame,'time_s':when,'score':row['score'],'threshold':row['event_threshold'],'verifier':row['verifier'],
              'support_window_anchor_frames':row['support_window_anchor_frames'],
              'motion':m,'supporting_windows':1,'last_support_time_s':when,
              'evidence_window_seconds':[i/proposal['fps'] for i in row.get('evidence_bounds',[row['indices'][0],row['indices'][-1]])],
              'evidence_available_time_s':row.get('evidence_bounds',row['indices'])[-1]/proposal['fps'],
              'timing_scope':'observed_motion_change_estimate_not_exact_physical_contact_or_duration',
              'incident_type':'abrupt_arrest' if m['speed_after']<.3*m['speed_before'] else 'loss_of_control'}
        # Keep one first supported episode per actor. Tracking loss does not authorize ID stitching.
        if previous is None:selected[actor]=event
    return sorted([attribute_contact(e,observations,proposal) for e in selected.values()],key=lambda e:e['time_s'])


def attribute_contact(event, observations, proposal):
    """Attribute an observed interaction, never claim the pair head passed.

    Require a learned, corroborated arrest plus separation/contact and an
    independently observed partner's slowdown AND direction change. A nearby
    box alone, a duplicate box or a track born at overlap cannot add a vehicle.
    """
    from .pair_events import contact_proposals
    m=event['motion'];actor=event['track_ids'][0];fps=proposal['fps']
    if not (m['speed_before']>.6 and m['speed_after']<.3*m['speed_before']):return event
    contacts=[c for c in contact_proposals(observations,fps) if actor in c['track_ids']
              and event['time_s']-1.2<=c['time_s']<=event['time_s']+.2]
    accepted=[]
    for contact in contacts:
        partner=next(i for i in contact['track_ids'] if i!=actor)
        indices=np.rint(event['frame_index']+np.arange(-6,7)*fps/5).astype(int).tolist()
        if indices[0]<0 or indices[-1]>=proposal['frames_processed']:continue
        boxes=sample_actor(observations,partner,indices,fps,tolerance_frames=max(1,round(.15*fps)))
        if boxes is None:continue
        window={'actor_id':partner,'indices':indices,'boxes':boxes}
        if not coherent_identity(window,observations,fps):continue
        _,pm=geometry(boxes,indices,fps)
        if not (pm['speed_before']>.6 and pm['speed_after']<.75*pm['speed_before']
                and pm['course_change_degrees']>=25):continue
        current=observations[observations[:,0]==contact['frame_index']]
        if not all(np.any(current[:,1]==i) for i in contact['track_ids']):continue
        accepted.append((contact,pm))
    # Several plausible partners are ambiguous. Never choose by ID or proximity.
    if len({tuple(c['track_ids']) for c,_ in accepted})!=1:return event
    contact,pm=min(accepted,key=lambda p:abs(event['time_s']-p[0]['time_s']))
    return dict(event,kind='motion_supported_pair_contact',track_ids=contact['track_ids'],
                frame_index=contact['frame_index'],time_s=contact['time_s'],
                motion_anchor_time_s=event['time_s'],contact_evidence=contact,partner_motion=pm,
                attribution_rule='learned_arrest_plus_observed_contact_and_partner_response_v1',
                pair_classifier_confirmed=False,
                timing_scope='observed_box_contact_supported_by_two_vehicle_motion_not_exact_physical_impact')
