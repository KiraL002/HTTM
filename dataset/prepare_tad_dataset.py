"""Import TAD safely and build a source-grouped manifest for a video pilot.

Labels stay at VIDEO level. Accident labels never propagate to every frame.
Camera grouping is a conservative visual heuristic, not verified camera IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_destination(root: Path, member: zipfile.ZipInfo) -> tuple[Path, PurePosixPath]:
    # ZipInfo normalizes backslashes on Windows; inspect the original too.
    if chr(92) in member.orig_filename or ":" in member.orig_filename:
        raise ValueError(f"Unsafe Windows archive entry: {member.orig_filename}")
    relative = PurePosixPath(member.filename).relative_to("TAD-benchmark")
    if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
        raise ValueError(f"Unsafe archive entry: {member.filename}")
    if stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError(f"Symlink in archive: {member.filename}")
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Archive target outside dataset: {destination}")
    return destination, relative


def import_archive(archive: Path, root: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(archive) as source:
        entries = [member for member in source.infolist() if not member.is_dir()]
        destinations = [safe_destination(root, member)[0] for member in entries]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Duplicate destinations in ZIP")
        for index, member in enumerate(entries, 1):
            target, relative = safe_destination(root, member)
            if len(relative.parts) != 3 or target.suffix.lower() != ".mp4":
                raise ValueError(f"Unexpected dataset member: {member.filename}")
            split, category, _ = relative.parts
            label = 0 if category == "normal" else 1 if category.startswith("accident") else None
            if split not in {"train", "test"} or label is None:
                raise ValueError(f"Unknown source label/split: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            sha, crc, size = hashlib.sha256(), 0, 0
            if target.exists():
                # Preserve files from earlier downloads. Verify, never overwrite.
                with target.open("rb") as existing:
                    while block := existing.read(1024 * 1024):
                        sha.update(block)
                        crc = zlib.crc32(block, crc)
                        size += len(block)
                if size != member.file_size or crc != member.CRC:
                    raise ValueError(f"Existing video differs from archive: {target}")
            else:
                temporary = target.with_suffix(".mp4.importing")
                try:
                    with source.open(member) as incoming, temporary.open("xb") as output:
                        # ZipExtFile verifies the source CRC at end of stream.
                        while block := incoming.read(1024 * 1024):
                            output.write(block)
                            sha.update(block)
                            size += len(block)
                    if size != member.file_size:
                        raise ValueError(f"Incomplete extraction: {target}")
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            rows.append({"path": target.relative_to(ROOT).as_posix(),
                         "source_split": split, "label": label,
                         "label_source": "TAD_folder_video_level", "human_verified": False,
                         "size_bytes": size, "sha256": sha.hexdigest(),
                         "zip_crc32": f"{member.CRC:08x}"})
            if index % 25 == 0 or index == len(entries):
                print(f"GIẢI NÉN + CRC: {index}/{len(entries)}", flush=True)
    return rows


def phash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    bits[0] = False
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def dhash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot be opened")
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames <= 0 or not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("Invalid video metadata")
        thumbnails, frame_hashes = [], []
        sampled = sorted({round((frames - 1) * fraction) for fraction in (0, .25, .5, .75, 1)})
        for frame in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode sampled frame {frame + 1}")
            frame_hashes.append(phash(image))
            thumbnails.append(cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA))
        background = np.median(np.stack(thumbnails), axis=0).astype(np.uint8)
        return {"frame_count": frames, "fps": fps, "width": width, "height": height,
                "duration_seconds": frames / fps, "decode_check": "five_sampled_positions",
                "sampled_frames_one_based": [frame + 1 for frame in sampled],
                "frame_phashes": frame_hashes, "background_phash": phash(background),
                "background_dhash": dhash(background), "probe_error": None}
    finally:
        cap.release()


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def related_view(a: dict, b: dict) -> bool:
    if a.get("probe_error") or b.get("probe_error"):
        return False
    aspect_a, aspect_b = a["width"] / a["height"], b["width"] / b["height"]
    return (abs(aspect_a / aspect_b - 1) < .03
            and hamming(a["background_phash"], b["background_phash"]) <= 8
            and hamming(a["background_dhash"], b["background_dhash"]) <= 8)


def source_family(row: dict) -> str | None:
    # Keep numbered cuts together conservatively; this is a filename hint,
    # not a claim that every cut has a verified common camera.
    match = re.fullmatch(r"(videox\d+)_\d+", Path(row["path"]).stem, flags=re.IGNORECASE)
    return match[1].lower() if match else None


def group_and_split(rows: list[dict], debug: list[dict]) -> list[dict]:
    """Keep official test untouched; quarantine potential overlaps on train side."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(j)] = find(i)

    pairs = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            exact = a["sha256"] == b["sha256"]
            visual = not exact and related_view(a, b)
            family = source_family(a) is not None and source_family(a) == source_family(b)
            if exact or visual or family:
                union(i, j)
                pairs.append({"a": a["path"], "b": b["path"],
                              "reason": "exact_bytes" if exact else "filename_cut_family" if family else "possible_same_view",
                              "verified": exact})
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    for members in groups.values():
        group_id = "source_" + min(row["sha256"] for row in members)[:16]
        has_test = any(row["source_split"] == "test" for row in members)
        conflicts = any(len({row["label"] for row in members if row["sha256"] == sha}) > 1
                        for sha in {row["sha256"] for row in members})
        overlaps_debug = any(row["sha256"] == other["sha256"] or related_view(row, other)
                             for row in members for other in debug)
        seen_sha = set()
        for row in sorted(members, key=lambda r: (r["source_split"] != "test", r["path"])):
            row["group_id"] = group_id
            row["camera_group_verified"] = False
            row["debug_view_overlap_candidate"] = overlaps_debug
            reason = ("decode_error" if row.get("probe_error") else
                      "conflicting_source_labels" if conflicts else
                      "duplicate_bytes" if row["sha256"] in seen_sha else
                      "possible_overlap_with_official_test" if has_test and row["source_split"] == "train" else
                      "possible_overlap_with_debug_videos" if overlaps_debug and row["source_split"] == "train" else None)
            row["exclude_reason"] = reason
            row["split"] = "excluded" if reason else row["source_split"]
            seen_sha.add(row["sha256"])
    eligible = [row for row in rows if row["split"] == "train"]
    if len(eligible) < 4:
        raise ValueError("Not enough non-overlapping training videos")
    from sklearn.model_selection import GroupShuffleSplit
    labels = np.array([row["label"] for row in eligible])
    groups_array = [row["group_id"] for row in eligible]
    best = None
    for train, val in GroupShuffleSplit(n_splits=100, test_size=.2, random_state=42).split(eligible, labels, groups_array):
        if set(labels[train]) != {0, 1} or set(labels[val]) != {0, 1}:
            continue
        cost = sum(abs(int((labels[val] == label).sum()) / int((labels == label).sum()) - .2) for label in (0, 1))
        if best is None or cost < best[0]:
            best = cost, val
    if best is None:
        raise ValueError("Cannot create group-separated validation with both classes")
    for index in best[1]:
        eligible[index]["split"] = "validation"
    active = [row for row in rows if row["split"] != "excluded"]
    for i, a in enumerate(active):
        for b in active[:i]:
            if a["split"] != b["split"] and (a["group_id"] == b["group_id"] or a["sha256"] == b["sha256"]):
                raise AssertionError("Known source group crosses split")
    return pairs


def write_report(payload: dict, manifest: Path, report: Path) -> None:
    rows, pairs, counts = payload["videos"], payload["related_pairs"], payload["counts"]
    lines = ["# Kiểm tra bộ TAD đã tải", "", f"Nguồn: `{payload['archive_path']}`.",
             f"SHA-256 ZIP: `{payload['archive_sha256']}`.",
             f"Đã xác minh CRC toàn bộ {len(rows)} file video so với ZIP. "
             "Kiểm tra giải mã ở đầu, 25%, giữa, 75% và cuối mỗi video; chưa giải mã toàn bộ mọi frame.", "",
             "| Tập đề xuất | ACCIDENT | NORMAL | Tổng |", "|---|---:|---:|---:|"]
    for split, count in counts.items():
        lines.append(f"| {split} | {count.get('ACCIDENT', 0)} | {count.get('NORMAL', 0)} | {sum(count.values())} |")
    reasons = Counter(pair["reason"] for pair in pairs)
    lines += ["", "Nhãn theo thư mục tác giả, chưa được duyệt thủ công toàn bộ. "
              "Nhãn ACCIDENT thuộc cả video; không gán cho mọi frame/cửa sổ hoặc mọi xe trong clip. "
              "Mô hình video cần học theo bag (tổng hợp nhiều đoạn thành dự đoán video) hoặc dùng mốc sự kiện đã duyệt.", "",
              "Giữ 32 video official test riêng. Chia validation từ official train theo nhóm nguồn, seed 42. "
              "Các đoạn cắt cùng họ tên videox được giữ chung nhóm; đồng thời đối chiếu SHA-256 và ảnh nền. "
              "File train có khả năng trùng test hoặc năm video dò lỗi được cách ly. "
              "Không có SHA/nhóm đã phát hiện đi qua các tập hoạt động; chưa chứng minh tách camera hoàn toàn.", "",
              f"Cặp byte trùng: {reasons['exact_bytes']}. Cặp cùng họ tên file: {reasons['filename_cut_family']}. "
              f"Cặp ảnh tương tự bổ sung: {reasons['possible_same_view']}. "
              f"Video lỗi tại các vị trí giải mã đã kiểm tra: {sum(bool(r.get('probe_error')) for r in rows)}.", "",
              "## Nhận xét khi xem ảnh mẫu", "",
              "Đã xem ảnh giữa clip của 16 NORMAL train, toàn bộ 16 NORMAL test và 16 ACCIDENT test. "
              "Đây là rà soát ảnh mẫu, chưa phải xác nhận nội dung toàn bộ 404 video.", "",
              "NORMAL chủ yếu là camera đường cao tốc, có ngày/đêm, mưa và ùn tắc. "
              "ACCIDENT test gồm xe hỏng/lật, hiện trường và cứu hộ, có cả góc quay gần. "
              "Vì vậy bộ này phù hợp thử phân loại video có tai nạn/hiện trường tai nạn, "
              "không tự cung cấp nhãn thời điểm va chạm, xe gây tai nạn hay loại va chạm. "
              "Cần đo báo nhầm thêm trên cảnh giao lộ bình thường trước khi áp dụng rộng.", "",
              "## Bước huấn luyện tiếp theo", "",
              "Dùng manifest bên dưới làm nguồn chia tập cố định cho một mô hình video thử nghiệm trên GPU. "
              "Chọn ngưỡng trên validation; giữ test cho đánh giá sau khi đã cố định cách học và ngưỡng. "
              "Báo cáo precision, recall, F1 theo video và số NORMAL bị báo nhầm. "
              "Checkpoint RF 507 video và mô hình đang dùng được giữ nguyên; bước chuẩn bị này chưa huấn luyện mô hình mới.", "",
              f"[Manifest dùng cho huấn luyện]({manifest.as_posix()})", "",
              "## File cách ly", "", "| File | Lý do |", "|---|---|"]
    excluded = [row for row in rows if row["split"] == "excluded"]
    lines += [f"| {row['path']} | {row['exclude_reason']} |" for row in excluded] or ["| Không có | — |"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/tad_video_manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/tad_dataset_review/report.md")
    args = parser.parse_args()
    args.archive, args.manifest, args.report = args.archive.resolve(), args.manifest.resolve(), args.report.resolve()
    if args.manifest.exists() or args.report.exists():
        parser.error("Output exists; choose new manifest/report paths to preserve earlier audit.")
    root = (ROOT / "dataset/TAD-benchmark").resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = import_archive(args.archive, root)
    for index, row in enumerate(rows, 1):
        try:
            row.update(probe(ROOT / row["path"]))
        except Exception as exc:
            row["probe_error"] = str(exc)
        if index % 25 == 0 or index == len(rows):
            print(f"KIỂM TRA VIDEO: {index}/{len(rows)}", flush=True)
    debug = []
    for path in sorted((ROOT / "dataset/input_video").glob("*.mp4")):
        try:
            debug.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest_file(path), **probe(path)})
        except Exception:
            continue
    print("ĐỐI CHIẾU TRÙNG NGUỒN / GÓC CAMERA...", flush=True)
    pairs = group_and_split(rows, debug)
    counts = {split: dict(Counter("ACCIDENT" if row["label"] else "NORMAL" for row in rows if row["split"] == split))
              for split in ("train", "validation", "test", "excluded")}
    payload = {"schema_version": 1, "dataset": "TAD-benchmark",
               "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "archive_path": str(args.archive), "archive_sha256": digest_file(args.archive),
               "archive_crc_verified": True, "source_url": "https://github.com/UnicomAI/UnicomBenchmark/tree/main/TADBench",
               "pilot_training_ready": all({0, 1} <= {r["label"] for r in rows if r["split"] == split}
                                           for split in ("train", "validation", "test")),
               "production_validated": False,
               "label_scope": "whole_video_bag_not_every_frame_or_window",
               "grouping_policy": "Exact SHA-256 + numbered videox source families + candidate camera-view groups (pHash<=8, dHash<=8, aspect within 3%). "
                                  "Groups never cross active splits. Train-side official-test/debug candidates quarantined. "
                                  "Unverified visual grouping does not prove camera-independent testing.",
               "split_seed": 42, "counts": counts, "related_pairs": pairs,
               "debug_references": debug, "videos": rows}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload, args.manifest, args.report)
    print("COUNTS", json.dumps(counts), flush=True)
    print("MANIFEST", args.manifest, flush=True)
    print("REPORT", args.report, flush=True)


if __name__ == "__main__":
    main()
