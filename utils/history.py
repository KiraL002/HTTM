"""Persistent analysis history; predictions and evidence remain immutable snapshots."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import os
import shutil
from uuid import uuid4

from .pair_demo import CACHE, ROOT, checked_events, file_sha


def local_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def display_name(name):
    return PureWindowsPath(str(name)).name[:255] or "video"


class HistoryStore:
    def __init__(self, database=None, cache=None):
        self.database = Path(database or ROOT / "outputs/history.sqlite3")
        self.cache = Path(cache or CACHE)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                    finished_at TEXT, source_name TEXT NOT NULL,
                    source_sha TEXT NOT NULL, cache_key TEXT NOT NULL,
                    origin TEXT NOT NULL CHECK(origin IN ('computed','reused','imported')),
                    status TEXT NOT NULL CHECK(status IN ('processing','completed','failed')),
                    label TEXT, duration REAL, processing_seconds REAL,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT, snapshot TEXT,
                    CHECK(label IS NULL OR label IN ('NORMAL','ACCIDENT')),
                    CHECK(status='completed' OR label IS NULL)
                );
                CREATE TABLE IF NOT EXISTS events (
                    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    event_no INTEGER NOT NULL, kind TEXT NOT NULL,
                    time_s REAL NOT NULL, frame_index INTEGER NOT NULL,
                    score REAL NOT NULL, vehicle_ids TEXT NOT NULL, image TEXT NOT NULL,
                    PRIMARY KEY(analysis_id,event_no)
                );
                CREATE INDEX IF NOT EXISTS analyses_recent ON analyses(started_at DESC);
                CREATE INDEX IF NOT EXISTS analyses_cache ON analyses(cache_key,status);
                CREATE TABLE IF NOT EXISTS reviews (
                    analysis_id TEXT PRIMARY KEY REFERENCES analyses(id) ON DELETE CASCADE,
                    verdict TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def start(self, source_name, source_sha, key):
        identity = uuid4().hex
        with self.connect() as db:
            db.execute("""INSERT INTO analyses
                (id,started_at,source_name,source_sha,cache_key,origin,status)
                VALUES (?,?,?,?,?,'computed','processing')""",
                (identity, local_time(), display_name(source_name), source_sha, key))
        return identity

    def _folder(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Mã kết quả lưu không hợp lệ.")
        folder = (self.cache / key).resolve()
        if not folder.is_relative_to(self.cache.resolve()):
            raise ValueError("Đường dẫn kết quả nằm ngoài thư mục lưu.")
        return folder

    def _verify(self, payload, key, source_sha):
        folder = self._folder(key)
        result = payload["result"]
        if payload["key"] != key or result["source_sha256"] != source_sha:
            raise ValueError("Video không khớp hồ sơ phân tích.")
        checked_events(result)
        video = folder / "result.mp4"
        if file_sha(video) != payload["video_sha256"]:
            raise ValueError("Video kết quả đã thay đổi hoặc bị hỏng.")
        names = {name for event in result["events"]
                 for name in [event["image"], *[v["image"] for v in event["vehicles"]]]}
        for name in names:
            path = (folder / name).resolve()
            if path.parent != folder or file_sha(path) != payload["image_sha256"][name]:
                raise ValueError("Ảnh bằng chứng đã thay đổi hoặc không hợp lệ.")
        return result, video, folder

    def complete(self, key, source_sha, source_name, analysis_id=None,
                 processing_seconds=None, origin="reused"):
        if origin not in ("computed", "reused", "imported"):
            raise ValueError("Nguồn kết quả không hợp lệ.")
        payload = json.loads((self._folder(key) / "result.json").read_text(encoding="utf-8"))
        result, _, _ = self._verify(payload, key, source_sha)
        snapshot = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Opening a saved result does not manufacture another inference run.
            if analysis_id is None:
                previous = db.execute("""SELECT id FROM analyses WHERE cache_key=?
                    AND status='completed' AND snapshot=? LIMIT 1""", (key, snapshot)).fetchone()
                if previous:
                    return previous["id"]
                analysis_id = uuid4().hex
                db.execute("""INSERT INTO analyses
                    (id,started_at,source_name,source_sha,cache_key,origin,status)
                    VALUES (?,?,?,?,?,?,'processing')""",
                    (analysis_id, local_time(), display_name(source_name), source_sha, key, origin))
            current = db.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            if (current is None or current["status"] != "processing"
                    or current["source_sha"] != source_sha or current["cache_key"] != key):
                raise ValueError("Lượt phân tích không khớp hoặc đã kết thúc.")
            db.execute("""UPDATE analyses SET finished_at=?,status='completed',label=?,
                duration=?,processing_seconds=?,event_count=?,snapshot=?,error=NULL WHERE id=?""",
                (local_time(), result["label"], result["duration"], processing_seconds,
                 len(result["events"]), snapshot, analysis_id))
            for number, event in enumerate(result["events"], 1):
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                    (analysis_id, number, event.get("kind", "pair_contact"), event["time_s"],
                     event["frame_index"], event["score"], json.dumps(event["track_ids"]), event["image"]))
        return analysis_id

    def fail(self, analysis_id, error):
        with self.connect() as db:
            db.execute("""UPDATE analyses SET status='failed',finished_at=?,error=?,label=NULL
                WHERE id=? AND status='processing'""", (local_time(), str(error)[:2000], analysis_id))

    def list(self, search="", verdict="all", limit=200):
        clauses, params = [], []
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("source_name LIKE ? ESCAPE '\\'")
            params.append("%" + escaped + "%")
        if verdict in ("NORMAL", "ACCIDENT"):
            clauses.append("status='completed' AND label=?")
            params.append(verdict)
        elif verdict in ("failed", "processing"):
            clauses.append("status=?")
            params.append(verdict)
        elif verdict != "all":
            raise ValueError("Bộ lọc lịch sử không hợp lệ.")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,started_at,finished_at,source_name,origin,status,label,duration,"
                "processing_seconds,event_count,error,"
                "(SELECT verdict FROM reviews r WHERE r.analysis_id=analyses.id) AS review_verdict,"
                "(SELECT note FROM reviews r WHERE r.analysis_id=analyses.id) AS review_note FROM analyses" + where
                + " ORDER BY started_at DESC,id DESC LIMIT ?", (*params, min(max(int(limit), 1), 1000)))]

    def latest(self, source_sha):
        """Find a saved run by video bytes, independently of today's policy.

        This only offers a historical replay. It never authorizes reusing old
        predictions as a fresh inference result. open() still checks evidence.
        """
        with self.connect() as db:
            row = db.execute("""SELECT id,source_name,cache_key,finished_at
                FROM analyses WHERE source_sha=? AND status='completed'
                ORDER BY finished_at DESC,started_at DESC,id DESC LIMIT 1""",
                (source_sha,)).fetchone()
        return dict(row) if row else None

    def open(self, analysis_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if row is None or row["status"] != "completed":
            raise ValueError("Lượt phân tích chưa có kết quả hoàn chỉnh.")
        payload = json.loads(row["snapshot"])
        result, video, folder = self._verify(payload, row["cache_key"], row["source_sha"])
        return dict(key=row["cache_key"], result=result, video=video.read_bytes(),
                    folder=folder, cached=True, name=row["source_name"], historical=True)

    def summary(self):
        with self.connect() as db:
            return dict(db.execute("""SELECT COUNT(*) AS total,
                COALESCE(SUM(status='completed' AND label='ACCIDENT'),0) AS flagged,
                COALESCE(SUM(event_count),0) AS events,
                COALESCE(SUM(status='failed'),0) AS failed FROM analyses""").fetchone())

    def review(self, analysis_id):
        with self.connect() as db:
            row = db.execute('SELECT verdict,note,updated_at FROM reviews WHERE analysis_id=?', (analysis_id,)).fetchone()
        return dict(row) if row else {'verdict': 'unreviewed', 'note': '', 'updated_at': None}

    def save_review(self, analysis_id, verdict, note=''):
        if verdict not in {'unreviewed', 'confirmed', 'false_alarm', 'missed_incident', 'unclear'}:
            raise ValueError('Kết quả duyệt không hợp lệ.')
        with self.connect() as db:
            row = db.execute('SELECT status,label FROM analyses WHERE id=?', (analysis_id,)).fetchone()
            if row is None or row['status'] != 'completed':
                raise ValueError('Chỉ duyệt được lượt phân tích đã hoàn tất.')
            if (verdict in {'confirmed','false_alarm'} and row['label'] != 'ACCIDENT') or (verdict == 'missed_incident' and row['label'] != 'NORMAL'):
                raise ValueError('Lựa chọn duyệt không phù hợp với kết quả AI.')
            db.execute('INSERT INTO reviews VALUES (?,?,?,?) ON CONFLICT(analysis_id) DO UPDATE SET verdict=excluded.verdict,note=excluded.note,updated_at=excluded.updated_at',
                       (analysis_id, verdict, str(note).strip()[:1000], local_time()))

    def _check_delete_folder(self, folder):
        root = self.cache.resolve()
        if (folder.resolve().parent != root or folder.is_symlink()
                or folder.is_junction() or not folder.is_dir()):
            raise ValueError('Thư mục kết quả không an toàn để xóa.')
        for parent, directories, files in os.walk(folder, followlinks=False):
            for name in directories + files:
                child = Path(parent) / name
                if child.is_symlink() or child.is_junction():
                    raise ValueError('Kết quả chứa liên kết ngoài; chưa xóa lịch sử.')

    def cleanup_pending(self, names):
        """Retry only staged deletions, without touching other saved results."""
        failures = []
        root = self.cache.resolve()
        for name in dict.fromkeys(names):
            if not re.fullmatch(r'\.history-delete-[0-9a-f]{32}', name):
                raise ValueError('Mã thư mục chờ xóa không hợp lệ.')
            folder = root / name
            if not folder.exists() and not folder.is_symlink():
                continue
            self._check_delete_folder(folder)
            try:
                shutil.rmtree(folder)
            except OSError:
                failures.append(name)
        return failures

    def delete(self, analysis_id):
        """Delete one run and its evidence only when no other run uses it."""
        moved = []
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM analyses WHERE id=?', (analysis_id,)).fetchone()
            if row is None:
                raise ValueError('Lượt phân tích này không còn trong lịch sử.')
            if row['status'] == 'processing':
                raise ValueError('Chưa thể xóa lượt đang xử lý. Hãy dừng hoặc chờ phân tích kết thúc.')
            key = row['cache_key']
            if not re.fullmatch(r'[0-9a-f]{64}', key):
                raise ValueError('Mã kết quả lưu không hợp lệ; chưa xóa lịch sử.')
            shared = db.execute('SELECT COUNT(*) FROM analyses WHERE cache_key=? AND id<>?',
                                (key, analysis_id)).fetchone()[0] > 0
            folder = self.cache.resolve() / key
            if not shared and (folder.exists() or folder.is_symlink()):
                self._check_delete_folder(folder)
                staged = folder.parent / ('.history-delete-' + uuid4().hex)
                folder.rename(staged)
                moved.append((folder, staged))
            try:
                # Foreign keys remove only this run's events and user review.
                db.execute('DELETE FROM analyses WHERE id=?', (analysis_id,))
                db.commit()
            except Exception:
                db.rollback()
                for original, staged in reversed(moved):
                    staged.rename(original)
                raise
        pending = self.cleanup_pending([staged.name for _, staged in moved])
        return {'analysis_id': analysis_id, 'source_name': row['source_name'],
                'cache_key': key, 'shared_results_retained': shared,
                'result_folders': len(moved), 'pending_cleanup': pending}

    def clear(self):
        """Remove all history and rendered result caches, never source/training data.

        Stage folders on the same volume before committing SQLite changes. A
        staging/SQL failure restores the folders and rolls back the records.
        """
        root = self.cache.resolve()
        root.mkdir(parents=True, exist_ok=True)
        moved = []
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            count = db.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
            # Include orphaned rendered results too, so they cannot reappear
            # through the analysis page after the history is cleared.
            folders = [p for p in root.iterdir() if re.fullmatch(r'[0-9a-f]{64}', p.name)
                       or re.fullmatch(r'\.history-delete-[0-9a-f]{32}', p.name)]
            for folder in folders:
                self._check_delete_folder(folder)
            try:
                for folder in folders:
                    staged = root / ('.history-delete-' + uuid4().hex)
                    folder.rename(staged)
                    moved.append((folder, staged))
                db.execute('DELETE FROM analyses')
                db.commit()
            except Exception:
                db.rollback()
                for original, staged in reversed(moved):
                    staged.rename(original)
                raise
        failures = self.cleanup_pending([staged.name for _, staged in moved])
        return {'analyses': count, 'result_folders': len(moved), 'pending_cleanup': failures}
