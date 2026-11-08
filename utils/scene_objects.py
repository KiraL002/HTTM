"""Auxiliary traffic participants; these tracks never enter incident decisions."""
from collections import Counter
from core.tracker import ByteTrackTracker

NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
LABELS = {'person': 'Người đi bộ', 'bicycle': 'Xe đạp', 'car': 'Ô tô',
          'motorcycle': 'Xe máy', 'bus': 'Xe buýt', 'truck': 'Xe tải'}


class SceneObjects:
    def __init__(self, model_path, tracker_config, device='cpu'):
        self.tracker = ByteTrackTracker(str(model_path), confidence=.25,
            tracker_config=str(tracker_config), device=device, classes=[0, 1, 3],
            vehicle_only=False, imgsz=640)
        if any(self.tracker.model.names.get(k) != v for k, v in NAMES.items()):
            raise ValueError('Auxiliary object labels require COCO model classes')
        self.seen = {name: set() for name in NAMES.values()}
        self.peaks = Counter()

    def update(self, frame):
        return self.tracker.update(frame)

    def observe(self, rows, auxiliary):
        counts = Counter()
        # Retain the incident branch's original ID namespace for four-wheelers.
        for row in rows:
            cls = int(row[2])
            if cls not in (2, 5, 7):
                continue
            name = NAMES[cls]
            self.seen[name].add(int(row[1]))
            counts[name] += 1
        for track in auxiliary:
            name = NAMES[track.class_id]
            self.seen[name].add(track.track_id)
            counts[name] += 1
        for name, count in counts.items():
            self.peaks[name] = max(self.peaks[name], count)
        return counts

    def summary(self):
        return [{'class_name': name, 'label': LABELS[name],
                 'observed_track_ids': len(self.seen[name]),
                 'maximum_simultaneous': self.peaks[name]} for name in NAMES.values()]
