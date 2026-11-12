"""Application adapter: established pair decisions, then one-vehicle fallback."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import tempfile

import cv2
import numpy as np

from .video_writer import BrowserVideoWriter
from .event_explanations import describe_event

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "outputs/vehicle_pair_runtime/runtime_policy.json"
CACHE = ROOT / "outputs/pair_demo"
VERSION = "vehicle-incident-demo-2026-11-11-temporal-proposals"
DECISION_RULE = "observed_pair_or_single_vehicle_fallback_v2"
CLASSES = {2: "car", 5: "bus", 7: "truck"}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def profile():
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if (value.get("demo_enabled") is not True or value.get("production_deployment_approved") is not False
            or value.get("decision_rule") != DECISION_RULE
            or not .5 <= value["threshold"] <= 1):
        raise ValueError("Chưa có cấu hình cặp xe đã kiểm tra cho bản demo.")
    for key in ("model", "geometry_model"):
        path = (ROOT / value[key]).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path) != value[key+"_sha256"]:
            raise ValueError("Tệp mô hình đã thay đổi so với lần kiểm tra.")
    single=value.get("single_vehicle",{})
    single_path=(ROOT/single.get("model","")).resolve()
    if (single.get("enabled") is not True or not single_path.is_relative_to(ROOT)
            or not .5<=single.get("threshold",0)<=1 or not .5<=single.get("rotation_threshold",0)<=1
            or file_sha(single_path)!=single.get("model_sha256")):
        raise ValueError("Nhánh sự cố một xe chưa khớp bản đã kiểm tra.")
    temporal=value.get("temporal_proposals",{})
    if temporal.get("enabled"):
        path=(ROOT/temporal["model"]).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path)!=temporal["model_sha256"]:
            raise ValueError("Mô hình tìm đoạn bất thường không khớp cấu hình.")
    for relative,digest in value["checked_code"].items():
        if file_sha(ROOT/relative) != digest:
            raise ValueError("Mã suy luận đã thay đổi; cần kiểm tra lại trước khi dùng kết quả lưu.")
    for package,version in value["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Phiên bản {package} khác môi trường đã kiểm tra ({version}).")
    return value


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = round(cap.get(3)),round(cap.get(4))
        if not cap.isOpened() or not math.isfinite(fps) or fps < 5 or frames < 13 or min(width,height)<=0:
            raise ValueError("Không đọc được video hợp lệ từ 5 khung hình/giây trở lên.")
        return dict(fps=fps,frames=frames,width=width,height=height,duration=frames/fps)
    finally:
        cap.release()


def cache_key(source_sha):
    value = profile()
    signature = {"version": VERSION, "source": source_sha, "policy":file_sha(POLICY_PATH),
                 "adapter":file_sha(__file__),"config":file_sha(ROOT/"config.yaml")}
    return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()


def checked_events(result):
    if (result.get("event_detection_complete") is not True
            or result.get("verdict_rule") != DECISION_RULE):
        raise ValueError("Phân tích chưa hoàn tất.")
    events = result["events"]
    if result["label"] != ("ACCIDENT" if events else "NORMAL"):
        raise ValueError("Nhãn không khớp bằng chứng cặp xe.")
    for e in events:
        single=e.get("kind")=="single_vehicle_incident"
        attributed=e.get("kind")=="motion_supported_pair_contact"
        count=1 if single else 2
        cutoff=result["threshold"]
        if single or attributed:
            if result.get("single_vehicle_evaluated") is not True:
                raise ValueError("Thiếu lượt xác minh sự cố một xe.")
            cutoffs={"single_geometry":result["single_vehicle_threshold"],
                     "single_rgb_rotation":result["single_rotation_threshold"]}
            if e.get("verifier") not in cutoffs or e.get("threshold")!=cutoffs[e["verifier"]]:
                raise ValueError("Ngưỡng sự cố một xe không khớp mô hình.")
            cutoff=cutoffs[e["verifier"]]
            anchors=e.get("support_window_anchor_frames",[])
            if len(set(anchors))<2 or any(not 0<=i<result["frames"] for i in anchors):
                raise ValueError("Thiếu bằng chứng chuyển động lặp lại.")
            if attributed:
                contact=e.get('contact_evidence',{})
                response=e.get('partner_motion',{})
                if (e.get('attribution_rule')!='learned_arrest_plus_observed_contact_and_partner_response_v1'
                        or contact.get('track_ids')!=e['track_ids']
                        or contact.get('frame_index')!=e['frame_index']
                        or contact.get('witness_normalized_gap',0)<.75
                        or contact.get('contact_normalized_gap',1)>.15
                        or response.get('speed_before',0)<=.6
                        or response.get('speed_after',1)>=.75*response.get('speed_before',0)
                        or response.get('course_change_degrees',0)<25
                        or e.get('pair_classifier_confirmed') is not False):
                    raise ValueError("Thiếu bằng chứng tương tác của hai xe.")
        if (not math.isfinite(e["time_s"]) or not 0 <= e["frame_index"] < result["frames"]
                or abs(e["time_s"] - e["frame_index"]/result["fps"]) > 1e-5
                or len(set(e["track_ids"])) != count or len(e["vehicles"]) != count
                or {v["vehicle_id"] for v in e["vehicles"]} != set(e["track_ids"])
                or not math.isfinite(e["score"]) or e["score"] < cutoff):
            raise ValueError("Bằng chứng thời điểm hoặc xe liên quan không hợp lệ.")
    return events


def save_result_cache(key, result, video_path):
    if cache_key(result["source_sha256"]) != key:
        raise ValueError("Video/cấu hình không khớp kết quả cần lưu.")
    checked_events(result)
    folder = CACHE/key
    folder.mkdir(parents=True,exist_ok=True)
    temporary = folder/"result.tmp"
    temporary.write_bytes(Path(video_path).read_bytes())
    temporary.replace(folder/"result.mp4")
    images = {name for e in result["events"] for name in [e["image"],*[v["image"] for v in e["vehicles"]]]}
    for name in images:
        (folder/name).write_bytes((Path(video_path).parent/name).read_bytes())
    payload = dict(key=key,result=result,video_sha256=file_sha(folder/"result.mp4"),
                   image_sha256={name:file_sha(folder/name) for name in images})
    temporary = folder/"metadata.tmp"
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(folder/"result.json")


def read_result_cache(key, source_sha):
    folder = CACHE/key
    try:
        policy = profile()
        payload = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        result = payload["result"]
        if (payload["key"] != key or key != cache_key(source_sha)
                or result["model_sha256"] != policy["model_sha256"]
                or result.get("single_model_sha256") != policy["single_vehicle"]["model_sha256"]
                or result["source_sha256"] != source_sha or result["analysis_version"] != VERSION
                or result["threshold"] != policy["threshold"] or result["tracking_enabled"] is not True
                or result["policy_sha256"] != file_sha(POLICY_PATH)
                or file_sha(folder/"result.mp4") != payload["video_sha256"]):
            return None
        checked_events(result)
        images = {name for e in result["events"] for name in [e["image"],*[v["image"] for v in e["vehicles"]]]}
        if any(file_sha(folder/name) != payload["image_sha256"][name] for name in images):
            return None
        return result,(folder/"result.mp4").read_bytes()
    except (OSError,ValueError,KeyError,TypeError):
        return None


def render(source, destination, proposal, result, progress=None):
    import yaml
    from .scene_objects import SceneObjects, NAMES
    config = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    scene = SceneObjects(ROOT/config['yolo']['model'], ROOT/config['tracker']['config'],
                         config['yolo'].get('device', 'cpu'))
    with np.load(proposal["detection_trace"],allow_pickle=False) as data:
        observations = data["observations"]
    groups = {int(f):observations[observations[:,0]==f] for f in np.unique(observations[:,0])}
    events_at = defaultdict(list)
    for event in result["events"]:
        events_at[event["frame_index"]].append(event)
    cap = cv2.VideoCapture(str(source))
    scale = min(1.,1280/result["width"])
    size = (round(result["width"]*scale),round(result["height"]*scale))
    writer = BrowserVideoWriter(destination,result["fps"],size)
    first_contact = None
    alerted_actors = set()
    try:
        for index in range(result["frames"]):
            ok,frame = cap.read()
            if not ok:
                raise ValueError("Video bị ngắt khi tạo bản đánh dấu.")
            frame_rows = groups.get(index, np.empty((0,8)))
            auxiliary = scene.update(frame)
            scene.observe(frame_rows, auxiliary)
            current_events = events_at.get(index, [])
            event = current_events[0] if current_events else None
            for event in current_events:
                if first_contact is None:
                    first_contact = event
                alerted_actors.update(event["track_ids"])
                still = frame.copy()
                for vehicle in event["vehicles"]:
                    row = next((r for r in frame_rows if int(r[1])==vehicle["vehicle_id"]), None)
                    if row is None:
                        raise ValueError("Thiếu một xe trong khung bằng chứng.")
                    x1,y1,x2,y2 = np.rint(row[4:8]).astype(int)
                    x1,y1 = max(0,x1),max(0,y1)
                    x2,y2 = min(result["width"],x2),min(result["height"],y2)
                    crop = frame[y1:y2,x1:x2]
                    crop_path = Path(destination).parent / vehicle["image"]
                    if not crop.size or not cv2.imwrite(str(crop_path),crop):
                        raise ValueError("Không lưu được ảnh xe liên quan.")
                    cv2.rectangle(still,(x1,y1),(x2,y2),(40,50,240),3)
                    cv2.putText(still,f"ID #{vehicle['vehicle_id']}",(x1,max(22,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(40,50,240),2)
                still = cv2.resize(still,size)
                if not cv2.imwrite(str(Path(destination).parent/event["image"]),still):
                    raise ValueError("Không lưu được khung bằng chứng.")
            frame = cv2.resize(frame,size)
            for row in frame_rows:
                if int(row[2]) not in (2, 5, 7):
                    continue
                actor = int(row[1])
                # Tracking is visible independently of the accident verdict.
                # Never invent a box during an occlusion or after a track ends.
                x1,y1,x2,y2 = np.rint(row[4:8]*scale).astype(int)
                x1,y1 = max(0,x1),max(0,y1)
                x2,y2 = min(size[0]-1,x2),min(size[1]-1,y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                color = (40,50,240) if actor in alerted_actors else (220,135,35)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{NAMES[int(row[2])]} | ID #{actor}",(x1,max(22,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            for track in auxiliary:
                x1,y1,x2,y2 = np.rint(track.bbox*scale).astype(int)
                x1,y1 = max(0,x1),max(0,y1)
                x2,y2 = min(size[0]-1,x2),min(size[1]-1,y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                color = (180,170,35) if track.class_id == 0 else (220,135,35)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                # A-prefix prevents confusing auxiliary IDs with incident actors.
                cv2.putText(frame,f"{NAMES[track.class_id]} | ID A{track.track_id}",
                            (x1,max(22,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            # Latch the warning after the first detected event. This is an alert
            # retained for review, not a claim that impact continues to the end.
            text = describe_event(first_contact)['overlay'] if first_contact else "NORMAL - CHUA PHAT HIEN SU CO"
            background = (35,40,185) if first_contact else (38,30,20)
            cv2.rectangle(frame,(10,10),(min(size[0]-10,610),70),background,-1)
            cv2.putText(frame,text,(22,36),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            detail = (f"Phat hien ~{first_contact['time_s']:.2f}s | " if first_contact else "") + f"t={index/result['fps']:.2f}s"
            cv2.putText(frame,detail,(22,58),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,235,245),1,cv2.LINE_AA)
            writer.write(frame)
            if progress and index % 50 == 0:
                progress(.95+.049*index/result["frames"],"Đang vẽ ID các xe và đánh dấu sự kiện")
        if cap.read()[0]:
            raise ValueError("Số khung hình nguồn đã thay đổi.")
        result['object_summary'] = scene.summary()
        result['auxiliary_tracking'] = {'classes': [0, 1, 3], 'image_size': 640,
            'confidence': .25, 'incident_decisions_used': False, 'id_prefix': 'A'}
    finally:
        cap.release()
        writer.release()


def run_analysis(source, destination, progress=None):
    # Every new application run gets empty caches. Historical experiment traces
    # remain available for research, but never substitute for this inference.
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inference-', dir=CACHE) as directory:
        return _run_analysis(source, destination, progress, Path(directory))


def _run_analysis(source, destination, progress, workspace):
    from .pair_pipeline import analyze
    from .pair_events import decide
    started = time.monotonic()
    policy = profile()
    temporal_review=None
    settings=policy.get("temporal_proposals",{})
    if settings.get("enabled"):
        from .temporal_review import scan as scan_temporal
        if progress: progress(.01,"Đang tìm các đoạn bất thường trong video")
        temporal_review=scan_temporal(source,settings,ROOT,
            (lambda value:progress(.02+.16*value,"Đang tìm các đoạn bất thường trong video")) if progress else None)
    pair_progress=(lambda value,message: progress(.18+.60*value,message)) if progress else None
    proposal, scored = analyze(source, workspace/'pairs', pair_progress,
                               trace_output=workspace/'tracks')
    events = decide(proposal["contacts"],scored["scores"],policy["threshold"])
    # The tested pair branch keeps priority. A new fallback cannot replace its
    # participants, timestamps or scores, nor add unverified actors to that event.
    single_evaluated=False
    if not events:
        from .single_vehicle_runtime import analyze as analyze_single
        single=analyze_single(proposal,output=workspace/'single',progress=progress,threshold=policy["single_vehicle"]["threshold"])
        if single["identity"]["model_sha256"]!=policy["single_vehicle"]["model_sha256"]:
            raise ValueError("Model một xe khác cấu hình đã kiểm tra.")
        events=single["events"]
        single_evaluated=True
    if temporal_review is not None:
        from .temporal_review import verify as verify_temporal
        events,extra_single=verify_temporal(temporal_review,proposal,scored,policy,
            workspace/"temporal",events,progress)
        single_evaluated=single_evaluated or extra_single
    with np.load(proposal["detection_trace"],allow_pickle=False) as data:
        observations = data["observations"]
    for i,event in enumerate(events,1):
        event['explanation'] = describe_event(event)
        event["image"] = f"event_{i}.jpg"
        event["vehicles"] = []
        for actor in event["track_ids"]:
            rows = observations[observations[:,1]==actor]
            cls = Counter(rows[:,2].astype(int)).most_common(1)[0][0]
            event["vehicles"].append({"vehicle_id":actor,"class_name":CLASSES.get(cls,"vehicle"),
                                      "image":f"event_{i}_vehicle_{actor}.jpg"})
    result = dict(video_info(source),label="ACCIDENT" if events else "NORMAL",events=events,
                  threshold=policy["threshold"],source_sha256=file_sha(source),
                  model_sha256=policy["model_sha256"],policy_sha256=file_sha(POLICY_PATH),
                  single_model_sha256=policy["single_vehicle"]["model_sha256"],
                  single_vehicle_evaluated=single_evaluated,
                  single_vehicle_threshold=policy["single_vehicle"]["threshold"],
                  single_rotation_threshold=policy["single_vehicle"]["rotation_threshold"],
                  analysis_version=VERSION,verdict_rule=policy["decision_rule"],
                  event_detection_complete=True,tracking_enabled=True,
                  temporal_review=temporal_review,
                  inference_cache_reused=False,
                  production_deployment_approved=False,
                  completed_at=datetime.now().strftime("%d/%m/%Y %H:%M"))
    checked_events(result)
    render(source,destination,proposal,result,progress)
    result["elapsed_seconds"] = round(time.monotonic()-started,1)
    if progress:
        progress(1.,"Đã hoàn tất")
    return result
