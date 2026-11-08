"""Actor-centred single-vehicle inputs; no annotation or demo identity at inference."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

PROTOCOL = {"version": 1, "frames": 13, "sample_fps": 5, "image_size": 320,
            "crop_padding": .5, "background_brightness": .35,
            "geometry_units": "image_box_diagonals_per_second_not_metres",
            "minimum_observed_samples": 9, "minimum_distinct_observations": 5}


def sample_actor(observations, actor, indices, fps, tolerance_frames=None):
    rows = observations[observations[:, 1] == actor]
    if not len(rows):
        return None
    tolerance = max(1, round(fps * .3)) if tolerance_frames is None else tolerance_frames
    boxes, actual = [], []
    for index in indices:
        at = int(np.argmin(abs(rows[:, 0] - index)))
        if abs(rows[at, 0] - index) > tolerance:
            boxes.append(None)
        else:
            boxes.append(rows[at, 4:8].tolist())
            actual.append(int(rows[at, 0]))
    if (sum(b is not None for b in boxes) < PROTOCOL["minimum_observed_samples"]
            or len(set(actual)) < PROTOCOL["minimum_distinct_observations"]):
        return None
    return boxes


def geometry(boxes, indices, fps):
    """Relative motion/shape only. Excludes scene coordinates, labels and time of impact."""
    indices = np.asarray(indices, float)
    a = np.array([b if b is not None else [np.nan]*4 for b in boxes], float)
    valid = np.isfinite(a).all(1)
    if len(a) != 13 or valid.sum() < 5 or fps <= 0:
        raise ValueError("Insufficient actor observations")
    wh = a[:, 2:] - a[:, :2]
    if np.any(wh[valid] <= 0):
        raise ValueError("Invalid actor boxes")
    scale = max(1., float(np.median(np.linalg.norm(wh[valid], axis=1))))
    centres = (a[:, :2] + a[:, 2:]) / 2
    # Interpolation is used ONLY for numerical descriptors, never drawn as a detection.
    for k in range(2):
        centres[:, k] = np.interp(indices, indices[valid], centres[valid, k])
        wh[:, k] = np.interp(indices, indices[valid], wh[valid, k])
    centres = np.array([np.median(centres[max(0,i-1):min(13,i+2)],axis=0) for i in range(13)])
    dt = np.maximum(np.diff(indices)/fps, .001)
    velocity = np.diff(centres, axis=0)/dt[:, None]/scale
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0)/np.maximum((dt[1:]+dt[:-1])/2,.001)[:,None]
    relative = (centres-centres[0])/scale
    shape = np.log(wh/np.median(wh,axis=0))
    rows = np.column_stack((valid,relative,shape,np.vstack(([0.,0.],velocity)),
                            np.r_[0.,speed],np.vstack((np.zeros((2,2)),acceleration))))
    before = (np.median(centres[4:8],axis=0)-np.median(centres[:4],axis=0))/max(.01,(np.median(indices[4:8])-np.median(indices[:4]))/fps)/scale
    after = (np.median(centres[-4:],axis=0)-np.median(centres[4:8],axis=0))/max(.01,(np.median(indices[-4:])-np.median(indices[4:8]))/fps)/scale
    v,w = np.linalg.norm(before),np.linalg.norm(after)
    cosine = float(np.dot(before,after)/max(1e-6,v*w))
    lateral = abs(before[0]*after[1]-before[1]*after[0])/max(v,1e-6)
    aspects = np.log(wh[:,0]/wh[:,1])
    yaw = abs(float(np.median(aspects[-4:])-np.median(aspects[:4])))
    path = np.linalg.norm(np.diff(centres,axis=0),axis=1).sum()/scale
    summary = np.r_[v,w,w/max(v,.01),cosine,lateral,yaw,np.ptp(aspects),path,
                    np.linalg.norm(relative[-1])/max(path,.01),np.quantile(speed,[.1,.5,.9]),
                    np.quantile(np.linalg.norm(acceleration,axis=1),[.5,.9]),valid.mean()]
    features = np.r_[rows.ravel(),summary].astype(np.float32)
    return np.clip(features,-30,30), {"speed_before":float(v),"speed_after":float(w),
                 "lateral_change":float(lateral),"shape_change":yaw,
                 "course_change_degrees":float(np.degrees(np.arccos(np.clip(cosine,-1,1))))}


def motion_supported(m):
    turning = (m["speed_before"] > .15 and m["lateral_change"] > .15
               and m["shape_change"] > .18)
    arrest = (m["speed_before"] > .6 and m["speed_after"] < .3*m["speed_before"])
    return bool(turning or arrest)


def crop_bounds(boxes, width, height):
    a=np.asarray([b for b in boxes if b is not None],float)
    lo,hi=a[:,:2].min(0),a[:,2:].max(0)
    padding=(hi-lo)*PROTOCOL["crop_padding"]
    return np.r_[np.maximum(0,np.floor(lo-padding)),np.minimum([width,height],np.ceil(hi+padding))].astype(int).tolist()


def frame_input(frame, box, bounds):
    x1,y1,x2,y2=bounds
    crop=frame[y1:y2,x1:x2].copy()
    if not crop.size:
        raise ValueError("Empty single-vehicle crop")
    attended=(crop.astype(float)*PROTOCOL["background_brightness"]).astype(np.uint8)
    if box is not None:
        b=np.asarray(box,float);pad=(b[2:]-b[:2])*.2
        lo=np.maximum([x1,y1],np.floor(b[:2]-pad)).astype(int)-[x1,y1]
        hi=np.minimum([x2,y2],np.ceil(b[2:]+pad)).astype(int)-[x1,y1]
        attended[lo[1]:hi[1],lo[0]:hi[0]]=crop[lo[1]:hi[1],lo[0]:hi[0]]
    h,w=crop.shape[:2];s=320/max(h,w);nw,nh=max(1,round(w*s)),max(1,round(h*s))
    result=np.full((320,320,3),114,np.uint8)
    result[(320-nh)//2:(320-nh)//2+nh,(320-nw)//2:(320-nw)//2+nw]=cv2.resize(attended,(nw,nh))
    return cv2.cvtColor(result,cv2.COLOR_BGR2RGB)


def read_clip(source, window):
    indices=window["indices"];cap=cv2.VideoCapture(str(source));frames={};want=set(indices)
    try:
        for i in range(max(indices)+1):
            ok,frame=cap.read()
            if not ok:raise ValueError(f"Cannot decode {Path(source).name} frame {i}")
            if i in want:frames[i]=frame
    finally:cap.release()
    return np.stack([frame_input(frames[i],b,window["crop"]) for i,b in zip(indices,window["boxes"])])


def proposals(observations, fps, frames, width, height):
    """Scan every observed actor; no GT timing, reference boxes or sample names."""
    result=[]
    for actor in np.unique(observations[:,1]).astype(int):
        rows=observations[observations[:,1]==actor]
        if len(rows)<9:continue
        low=max(0.,float(rows[0,0]/fps));high=min((frames-1)/fps,float(rows[-1,0]/fps))
        for start in np.arange(low,max(low,high-2.4)+1e-6,.4):
            if start+2.4>high+1e-6:continue
            indices=np.rint((start+np.arange(13)/5)*fps).astype(int).tolist()
            boxes=sample_actor(observations,actor,indices,fps,tolerance_frames=max(1,round(.15*fps)))
            if boxes is None:continue
            sizes=np.asarray([np.asarray(b)[2:]-np.asarray(b)[:2] for b in boxes if b is not None])
            if np.median(sizes[:,0])<10 or np.median(sizes[:,1])<8:continue
            _,motion=geometry(boxes,indices,fps)
            if not motion_supported(motion):continue
            result.append({"actor_id":int(actor),"indices":indices,"boxes":boxes,
                           "crop":crop_bounds(boxes,width,height),"motion":motion,
                           "anchor_frame":int(indices[6]),"time_s":indices[6]/fps})
    return result
