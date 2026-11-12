"""AcciVision — local, inference-only presentation dashboard."""
from __future__ import annotations

import hashlib
import html
import csv
import io
import sqlite3
import tempfile
import time
from pathlib import Path

import streamlit as st

from utils.pair_demo import (VERSION, POLICY_PATH, profile, cache_key,
                             file_sha, run_analysis, video_info, checked_events,
                             read_result_cache as read_cached, save_result_cache as save_cached)
from utils import history as history_module
# Streamlit can rerun app.py while retaining an imported module from an older
# release. Refresh the history API before registering callbacks using new methods.
if not all(callable(getattr(history_module.HistoryStore, name, None))
           for name in ('delete', 'cleanup_pending')):
    from importlib import reload
    history_module = reload(history_module)
HistoryStore = history_module.HistoryStore
from utils.event_explanations import describe_event

ROOT = Path(__file__).resolve().parent
try:
    PROFILE = profile()
    PROFILE_ERROR = None
except (OSError, ValueError, KeyError, TypeError) as exc:
    PROFILE, PROFILE_ERROR = None, str(exc)
CACHE = ROOT / "outputs/pair_demo"
CACHE.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title="AcciVision · Phân tích giao thông",
                   page_icon=":material/traffic:", layout="wide",
                   initial_sidebar_state="collapsed")
st.html("""<style>
.block-container {max-width:1250px;padding-top:2.5rem;padding-bottom:3rem}
header[data-testid="stHeader"] {background:transparent}
.av-brand {font-size:14px;font-weight:750;letter-spacing:.14em;color:#175eac}
.av-hero {padding:32px 36px;background:#102e55;border-radius:20px;margin:16px 0 28px;position:relative;overflow:hidden}
.av-hero:after {content:"";position:absolute;width:240px;height:240px;border:48px solid #194674;border-radius:100%;right:-60px;top:-85px;opacity:.6}
.av-hero h1 {color:#fff;font-size:2.35rem;font-weight:650;line-height:1.2;letter-spacing:-.035em;margin:0 0 13px;position:relative;z-index:1}
.av-hero p {color:#b7cee8;margin:0;max-width:690px;font-size:1rem;position:relative;z-index:1}
.av-kicker {font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#80bcff;margin-bottom:14px;font-weight:650}
.av-tag {background:#e7f0fc;color:#2160a6;border:1px solid #cadcf4;border-radius:30px;padding:5px 12px;font-size:12px}
.av-result {padding:22px 26px;border-radius:14px;border:1px solid #bfd6f5;background:#eaf3ff;margin:12px 0 20px}
.av-result.alert {border-color:#f1b5bc;background:#fff0f2}
.av-result h2 {font-size:1.45rem;letter-spacing:-.025em;margin:4px 0 6px}
.av-result p {color:#53708f;margin:0;font-size:.94rem}
div[data-testid="stMetric"] {background:white;border:1px solid #dce6f2;border-radius:12px;padding:16px 18px}
div[data-testid="stMetricValue"] {font-size:1.7rem}
div[data-testid="stMetricLabel"] {color:#55708e;font-size:.85rem}
div[data-testid="stVerticalBlockBorderWrapper"]>div {background:rgba(255,255,255,.55)}
.st-key-analyze button {min-height:48px;font-weight:650}
.st-key-source_picker button {font-weight:550}
.av-footer {color:#7187a0;font-size:12px;margin-top:35px;border-top:1px solid #dce6f2;padding-top:15px}
@media(max-width:640px){.block-container{padding-top:1.3rem}.av-hero{padding:25px 22px}.av-hero h1{font-size:1.8rem}}
</style>""")

@st.cache_data(max_entries=6, show_spinner=False)
def uploaded_info(data, suffix):
    with tempfile.TemporaryDirectory(prefix="accivision-metadata-", dir=CACHE) as folder:
        path = Path(folder) / ("input" + suffix)
        path.write_bytes(data)
        return video_info(path)

def clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def csv_data(rows):
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        # Uploaded names may contain spreadsheet formula prefixes.
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                         for k, v in row.items()})
    return stream.getvalue().encode("utf-8-sig")


def event_rows(result):
    return [{"Sự kiện": i, "Loại": describe_event(e)['title'],
             "Dấu hiệu phát hiện": ' '.join(describe_event(e)['evidence']), "Mốc ước lượng (giây)": round(e["time_s"], 2),
             "ID xe": ", ".join(str(actor) for actor in e["track_ids"])}
            for i, e in enumerate(result["events"], 1)]


def show_video(active, start=0):
    st.subheader("Xem lại video đánh dấu", anchor=False)
    st.video(active["video"], start_time=start)
    st.download_button("Tải video kết quả", active["video"],
                       file_name=f"accivision_{Path(active['name']).stem}.mp4",
                       mime="video/mp4", icon=":material/download:", on_click="ignore")


def show_result(active):
    r = active["result"]
    events = checked_events(r)
    st.divider()
    st.caption("KẾT QUẢ PHÂN TÍCH")
    one_vehicle = any(e.get("kind") == "single_vehicle_incident" for e in events)
    title = ("ACCIDENT · Nghi sự cố phương tiện" if one_vehicle else "ACCIDENT · Nghi va chạm hai xe") if events else "NORMAL · Chưa phát hiện sự cố"
    desc = "Xem xe liên quan và mốc nghi sự cố bên dưới." if events else "Chưa tìm thấy bằng chứng sự cố trong phạm vi hỗ trợ."
    st.html(f'<div class="av-result {"alert" if events else ""}"><h2>{title}</h2><p>{desc}</p></div>')
    st.caption(("Kết quả đã lưu" if active["cached"] else "Vừa phân tích")
               + f' · {r["completed_at"]} · {active["name"]}')
    if active.get("historical") and (PROFILE is None or r["policy_sha256"] != file_sha(POLICY_PATH)):
        st.info("Đây là kết quả lịch sử với cấu hình tại thời điểm xử lý; chưa chạy lại bằng cấu hình hiện tại.")
    a, b, c = st.columns(3)
    a.metric("Sự kiện nghi ngờ", len(events))
    b.metric("ID xe được AI đánh dấu", len({actor for e in events for actor in e["track_ids"]}))
    c.metric("Mốc đầu tiên", f'{min(e["time_s"] for e in events):.2f} giây' if events else "—")
    review = r.get('temporal_review')
    if review is not None:
        with st.expander('Các đoạn bất thường đã được kiểm tra', expanded=False):
            st.caption('Điểm bất thường chọn đoạn cần kiểm tra. Mốc trên đường biểu diễn là cuối cửa sổ ngữ cảnh, không phải thời điểm va chạm.')
            timeline = review['models']['temporal']['timeline']
            st.line_chart([{'Giây': row['window_end_s'], 'Điểm bất thường': row['score'],
                            'Ngưỡng chọn đoạn': review['thresholds']['enter']} for row in timeline],
                          x='Giây', y=['Điểm bất thường', 'Ngưỡng chọn đoạn'],
                          color=['#2160a6', '#aab9cc'], height=200)
            checks = review.get('verification', [])
            if checks:
                st.dataframe([{'Đoạn kiểm tra (giây)': f"{v['start_s']:.2f}–{v['end_s']:.2f}",
                               'Kết quả xác minh': 'Có bằng chứng sự cố' if v['state']=='incident_evidence' else 'Chưa đủ bằng chứng tai nạn',
                               'Mốc sự cố được xác minh': ', '.join(f'{t:.2f}s' for t in v['event_times_s']) or '—'} for v in checks],
                             hide_index=True, width='stretch')
            else:
                st.caption('Không có đoạn vượt ngưỡng đủ liên tiếp. Hệ thống vẫn kiểm tra ứng viên chuyển động và tiếp xúc.')
            st.caption('Đoạn chưa đủ bằng chứng không được đưa vào số sự kiện tai nạn. Ngưỡng chọn đoạn đang là cấu hình thử nghiệm trên CCTV.')
    if r.get('object_summary'):
        st.subheader('Đối tượng được quan sát', anchor=False)
        objects = [{'Đối tượng': item['label'], 'Số ID theo dõi': item['observed_track_ids'],
                    'Xuất hiện cùng lúc nhiều nhất': item['maximum_simultaneous']} for item in r['object_summary']]
        st.dataframe(objects, hide_index=True, width='stretch')
        st.caption('Số ID có thể lớn hơn số đối tượng thực khi mất dấu rồi được gán ID mới. ID A thuộc nhánh theo dõi xe máy, xe đạp và người đi bộ; chưa dùng để kết luận tai nạn.')
        st.download_button('Tải thống kê đối tượng (.csv)', csv_data(objects),
                           file_name='accivision_objects.csv', mime='text/csv', on_click='ignore')
    start = 0
    if events:
        st.subheader("Xe liên quan và mốc nghi sự cố", anchor=False)
        labels = [f"Sự kiện {i} · ~{e['time_s']:.2f}s · " + " và ".join(f"ID #{actor}" for actor in e["track_ids"])
                  for i, e in enumerate(events, 1)]
        chosen = st.selectbox("Chọn sự kiện để xem lại", labels, key="event_" + active["key"])
        event = events[labels.index(chosen)]
        explanation = describe_event(event)
        st.markdown('**' + explanation['title'] + '**')
        st.write(' '.join(explanation['evidence']))
        st.caption(explanation['limitation'])
        start = max(0, event["time_s"] - 1)
        folder = active.get("folder", CACHE / active["key"])
        overview, participants = st.columns([2, 1], gap="medium")
        with overview:
            st.image(folder / event["image"], caption=f"Khung quan sát gần mốc {event['time_s']:.2f}s", width="stretch")
        with participants:
            for vehicle in event["vehicles"]:
                name = {"car": "Ô tô", "truck": "Xe tải", "bus": "Xe buýt"}.get(vehicle["class_name"], "Phương tiện")
                st.image(folder / vehicle["image"], caption=f"{name} · ID #{vehicle['vehicle_id']}", width="stretch")
        st.caption("Mốc nghi sự cố là ước lượng, không phải khoảng đầu–cuối tai nạn. Cảnh báo đỏ được giữ đến cuối video; ID chỉ có ý nghĩa trong video này.")
        st.dataframe(event_rows(r), hide_index=True, width="stretch")
        st.download_button("Tải danh sách sự kiện (.csv)", csv_data(event_rows(r)),
                           file_name=f"accivision_{Path(active['name']).stem}_events.csv",
                           mime="text/csv", on_click="ignore")
    show_video(active, start)


try:
    HISTORY = HistoryStore()
except (sqlite3.Error, OSError) as exc:
    HISTORY = None
    st.warning("Chưa mở được lịch sử. Bạn vẫn có thể phân tích và tải kết quả.")


def remember(key, source_sha, name, analysis_id=None, processing_seconds=None):
    if HISTORY is None:
        return
    try:
        HISTORY.complete(key, source_sha, name, analysis_id, processing_seconds,
                         origin="computed" if processing_seconds is not None else "reused")
    except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
        st.warning("Kết quả đã tạo nhưng chưa lưu được vào lịch sử. Hãy tải video để giữ lại.")
        if analysis_id:
            try:
                HISTORY.fail(analysis_id, "Không lưu được hồ sơ kết quả: " + str(exc))
            except (sqlite3.Error, OSError):
                pass


def delete_history_entry(analysis_id=None):
    confirmation = 'history_delete_confirm_' + analysis_id if analysis_id else 'history_delete_confirm'
    if HISTORY is None or not st.session_state.get(confirmation):
        return
    try:
        deleted = HISTORY.delete(analysis_id) if analysis_id else HISTORY.clear()
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.session_state['history_delete_error'] = 'Chưa xóa được lịch sử: ' + str(exc)
        return
    active = st.session_state.get('active_result')
    if not analysis_id or (active and active['key'] == deleted['cache_key']
                           and not deleted['shared_results_retained']):
        st.session_state['active_result'] = None
        for key in list(st.session_state):
            if key.startswith('event_') and (not analysis_id or key == 'event_' + deleted['cache_key']):
                del st.session_state[key]
    st.session_state.pop('history_selected', None)
    st.session_state.pop(confirmation, None)
    st.session_state.pop('history_delete_error', None)
    if analysis_id:
        notice = f"Đã xóa lượt phân tích của {deleted['source_name']}."
        notice += (' Kết quả dùng chung được giữ cho các lượt còn lại.'
                   if deleted['shared_results_retained'] else ' Đã xóa ảnh và video kết quả riêng của lượt này.')
    else:
        notice = f"Đã xóa {deleted['analyses']} lượt lịch sử cùng kết quả, ảnh và video đã lưu."
    pending = list(dict.fromkeys(st.session_state.get('history_pending_cleanup', []) + deleted['pending_cleanup']))
    st.session_state['history_pending_cleanup'] = pending
    if pending:
        notice += ' Một số tệp đang bị khóa; hãy đóng video rồi bấm Dọn tệp chờ xóa.'
    st.session_state['history_delete_pending'] = bool(pending)
    st.session_state['history_delete_notice'] = notice


def cleanup_history_files():
    if HISTORY is None:
        return
    try:
        pending = HISTORY.cleanup_pending(st.session_state.get('history_pending_cleanup', []))
    except (OSError, ValueError) as exc:
        st.session_state['history_delete_error'] = 'Chưa dọn được tệp chờ xóa: ' + str(exc)
        return
    st.session_state['history_pending_cleanup'] = pending
    st.session_state['history_delete_pending'] = bool(pending)
    st.session_state['history_delete_notice'] = ('Một số tệp vẫn đang bị khóa. Hãy đóng video rồi thử lại.'
                                                if pending else 'Đã dọn xong tệp chờ xóa.')


def history_page():
    st.subheader("Lịch sử phân tích", anchor=False)
    st.caption("Lưu trên máy này. Mở lại kết quả không chạy mô hình và không tạo lượt suy luận mới.")
    if HISTORY is None:
        st.info("Chưa truy cập được cơ sở dữ liệu lịch sử.")
        return
    notice = st.session_state.pop('history_delete_notice', None)
    if notice:
        if st.session_state.pop('history_delete_pending', False):
            st.warning(notice)
        else:
            st.success(notice)
    error = st.session_state.pop('history_delete_error', None)
    if error:
        st.error(error)
    if st.session_state.get('history_pending_cleanup'):
        st.button('Dọn tệp chờ xóa', on_click=cleanup_history_files, key='history_cleanup_pending')
    with st.expander('Xóa lịch sử và kết quả đã lưu'):
        st.caption('Xóa toàn bộ lượt phân tích, ảnh bằng chứng và video kết quả trên máy này, kể cả kết quả không còn trong danh sách. Video gốc và dữ liệu training được giữ lại.')
        confirmed = st.checkbox('Tôi muốn xóa toàn bộ lịch sử và kết quả đã lưu', key='history_delete_confirm')
        st.button('Xóa toàn bộ lịch sử', icon=':material/delete:', disabled=not confirmed,
                  key='history_delete_all', on_click=delete_history_entry)
    try:
        totals = HISTORY.summary()
        a, b, c = st.columns(3)
        a.metric("Lượt đã ghi", totals["total"])
        b.metric("Lượt nghi sự cố", totals["flagged"])
        c.metric("Sự kiện đã lưu", totals["events"])
        search_col, verdict_col = st.columns([2, 1])
        search = search_col.text_input("Tìm tên video", key="history_search")
        filters = {"Tất cả": "all", "Nghi sự cố": "ACCIDENT", "Chưa phát hiện": "NORMAL",
                   "Lỗi xử lý": "failed", "Chưa hoàn tất": "processing"}
        verdict = verdict_col.selectbox("Trạng thái", list(filters), key="history_filter")
        rows = HISTORY.list(search, filters[verdict])
    except (sqlite3.Error, OSError):
        st.error("Chưa đọc được lịch sử. Hãy kiểm tra quyền ghi thư mục outputs.")
        return
    if not rows:
        st.info("Chưa có lượt phân tích phù hợp. Chọn Phân tích video để bắt đầu.")
        return
    def status(row):
        return {"processing": "Chưa hoàn tất", "failed": "Lỗi xử lý"}.get(row["status"],
               "Nghi sự cố" if row["label"] == "ACCIDENT" else "Chưa phát hiện")
    table = [{"Video": r["source_name"], "Đã ghi lúc": r["started_at"],
              "Trạng thái AI": status(r), "Sự kiện": r["event_count"],
              "Người dùng duyệt": {'confirmed': 'Xác nhận có sự cố', 'false_alarm': 'Cảnh báo nhầm',
                  'missed_incident': 'AI bỏ sót', 'unclear': 'Không rõ'}.get(r.get('review_verdict'), 'Chưa duyệt'),
              "Ghi chú": r.get('review_note') or ''} for r in rows]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Hiển thị tối đa 200 lượt gần nhất theo bộ lọc.")
    st.download_button("Tải lịch sử đang lọc (.csv)", csv_data(table),
                       file_name="accivision_history.csv", mime="text/csv", on_click="ignore")
    options = {r["id"]: f'{r["source_name"]} · {status(r)} · {r["started_at"]}' for r in rows}
    if st.session_state.get('history_selected') not in options:
        st.session_state.pop('history_selected', None)
    selected = st.selectbox("Chọn lượt để xem lại", list(options), format_func=options.get,
                            key="history_selected")
    row = next(r for r in rows if r["id"] == selected)
    with st.expander('Xóa lượt phân tích đã chọn'):
        st.write(f"Video: {row['source_name']} · {row['started_at']}")
        st.caption('Chỉ xóa lượt đang chọn, đánh giá, ảnh và video kết quả của lượt này. Các lượt khác và video gốc được giữ nguyên. Nếu kết quả dùng chung với lượt khác, tệp kết quả được giữ lại.')
        processing = row['status'] == 'processing'
        if processing:
            st.info('Chưa thể xóa lượt đang xử lý. Hãy dừng hoặc chờ phân tích kết thúc.')
        confirmed_one = st.checkbox('Tôi muốn xóa riêng lượt phân tích này',
                                    key='history_delete_confirm_' + selected, disabled=processing)
        st.button('Xóa lượt này', icon=':material/delete:', disabled=processing or not confirmed_one,
                  key='history_delete_one_' + selected, on_click=delete_history_entry, args=(selected,))
    if row["status"] == "failed":
        st.error("Lượt này không có kết luận NORMAL/ACCIDENT.")
        with st.expander("Chi tiết lỗi"):
            st.code(row["error"] or "Không có chi tiết lỗi.")
    elif row["status"] == "processing":
        st.info("Lượt này chưa hoàn tất, có thể đã bị ngắt. Không có kết luận; bạn có thể chọn lại video ở trang phân tích.")
    else:
        try:
            show_result(HISTORY.open(selected))
            st.subheader('Duyệt kết quả', anchor=False)
            st.caption('Đánh giá của bạn được lưu riêng; không sửa dự đoán AI và không tự đưa vào training.')
            review = HISTORY.review(selected)
            choices = {'Chưa duyệt': 'unreviewed', 'Không rõ': 'unclear'}
            if row['label'] == 'ACCIDENT':
                choices.update({'Xác nhận có sự cố': 'confirmed', 'Cảnh báo nhầm': 'false_alarm'})
            else:
                choices['AI bỏ sót sự cố'] = 'missed_incident'
            with st.form('review_' + selected):
                labels = list(choices)
                index = next((i for i,k in enumerate(labels) if choices[k] == review['verdict']), 0)
                verdict = st.selectbox('Đánh giá sau khi xem video', labels, index=index)
                note = st.text_area('Ghi chú (xe liên quan, mốc giây hoặc lý do)', value=review['note'], max_chars=1000)
                if st.form_submit_button('Lưu đánh giá'):
                    HISTORY.save_review(selected, choices[verdict], note)
                    st.success('Đã lưu đánh giá của bạn.')
        except sqlite3.Error:
            st.error('Chưa lưu hoặc đọc được đánh giá. Hãy thử lại.')
        except (OSError, ValueError, KeyError, TypeError):
            st.error("Không mở được bằng chứng: tệp đã mất hoặc thay đổi. Hồ sơ lịch sử vẫn được giữ; hãy phân tích lại video gốc.")

for key, value in {"active_result": None}.items():
    st.session_state.setdefault(key, value)

top, badge = st.columns([4, 1], vertical_alignment="center")
with top:
    st.html('<div class="av-brand">◈ &nbsp; ACCIVISION</div>')

st.html("""<section class="av-hero">
<div class="av-kicker">Giám sát &amp; phân tích giao thông</div>
<h1>Quan sát giao thông<br>Nhận diện sự cố</h1>
</section>""")

workspace = st.segmented_control("Khu vực làm việc", ["Phân tích video", "Lịch sử"],
                                 default="Phân tích video", key="workspace")
if workspace == "Lịch sử":
    history_page()
    st.stop()

mode = st.segmented_control("Nguồn video", ["Video mẫu", "Tải video"],
                            default="Video mẫu", key="source_picker",
                            label_visibility="collapsed")
samples = {f"{p.stem.replace('_', ' ').title()} · Camera giao thông": p
           for p in sorted((ROOT / "dataset/input_video").glob("*.mp4"),
                           key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem))}
samples = {(f"Video {label}" if label.split(' · ')[0].isdigit() else label): p for label, p in samples.items()}
normal = ROOT / "dataset/TAD-benchmark/test/normal/videox_test_F3.mp4"
if normal.is_file():
    samples["Video mẫu · Giao thông thường ngày"] = normal

source_path, source_data, info, source_name, source_sha = None, None, None, "", ""
if mode == "Video mẫu" and samples:
    selected = st.selectbox("Chọn video có sẵn", list(samples))
    source_path = samples[selected]
    source_name = source_path.name
    try:
        info = video_info(source_path)
        source_sha = file_sha(source_path)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
else:
    uploaded = st.file_uploader("Thả video vào đây hoặc chọn từ máy tính",
                               type=["mp4", "mov", "avi", "mkv"], max_upload_size=200,
                               help="Tối đa 200 MB, thời lượng tối đa 2 phút. Video được xử lý trên máy chạy ứng dụng.")
    if uploaded:
        source_data = uploaded.getvalue()
        source_name = Path(uploaded.name).name
        source_sha = hashlib.sha256(source_data).hexdigest()
        try:
            info = uploaded_info(source_data, Path(source_name).suffix.lower())
        except (ValueError, OSError) as exc:
            st.error(str(exc))

if info:
    preview, controls = st.columns([1.55, 1], gap="large")
    with preview:
        st.video(source_path if source_path else source_data,
                 format="video/mp4" if Path(source_name).suffix.lower() == ".mp4" else "video/quicktime")
    with controls:
        st.subheader("Sẵn sàng phân tích", anchor=False)
        st.caption(source_name)
        a, b = st.columns(2)
        a.metric("Thời lượng", clock(info["duration"]))
        b.metric("Độ phân giải", f'{info["width"]} × {info["height"]}')
        if info["duration"] > 120.5:
            st.error("Video dài hơn 2 phút. Hãy cắt một đoạn ngắn để demo.")
        st.caption("Phát hiện và theo dõi ô tô, xe tải, xe buýt, xe máy, xe đạp và người đi bộ. Nhận diện sự cố hiện áp dụng cho ô tô, xe tải, xe buýt.")
        model_ready = PROFILE is not None
        if not model_ready:
            st.info("Bộ phân tích chưa sẵn sàng. " + (PROFILE_ERROR or ""))
        key = cache_key(source_sha) if model_ready else "unavailable"
        saved = read_cached(key, source_sha) if model_ready else None
        historical = None
        if not saved and HISTORY is not None:
            try:
                historical = HISTORY.latest(source_sha)
            except (sqlite3.Error, OSError):
                st.warning("Chưa tìm được bản lưu trong lịch sử. Bạn vẫn có thể phân tích video.")
        if saved:
            st.caption(":material/inventory_2: Có kết quả đã lưu cho đúng video và cấu hình này.")
        elif historical:
            st.caption(":material/inventory_2: Có bản đã lưu của video này trong lịch sử. Mở lại không chạy mô hình; bản lưu có thể dùng cấu hình trước đây.")
        action, refresh = st.columns([1.5, 1])
        can_open = bool(saved or historical)
        run = action.button("Mở kết quả đã lưu" if can_open else "Bắt đầu phân tích", type="primary",
                            icon=":material/inventory_2:" if can_open else ":material/play_arrow:",
                            width="stretch", key="analyze",
                            disabled=not can_open and (not model_ready or info["duration"] > 120.5))
        rerun = refresh.button("Phân tích lại", icon=":material/refresh:",
                               width="stretch", disabled=not model_ready or info["duration"] > 120.5,
                               help="Chạy lại toàn bộ mô hình từ video gốc, không tái sử dụng dữ liệu phân tích trung gian.")
    if run or rerun:
        st.session_state.active_result = None
        if historical and not rerun:
            try:
                st.session_state.active_result = HISTORY.open(historical['id'])
            except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as exc:
                st.error("Không mở được bản lưu: " + str(exc))
        elif saved and not rerun:
            result, video = saved
            st.session_state.active_result = dict(key=key, result=result, video=video, cached=True, name=source_name)
            remember(key, source_sha, source_name)
        else:
            bar = st.progress(0, text="Đang nạp mô hình")
            status = st.empty()
            started = time.monotonic()
            analysis_id = None
            if HISTORY is not None:
                try:
                    analysis_id = HISTORY.start(source_name, source_sha, key)
                except (sqlite3.Error, OSError):
                    st.warning("Chưa ghi được lượt phân tích vào lịch sử.")
            def progress(value, message):
                bar.progress(min(1., max(0., value)), text=message)
                status.caption(f"Đã xử lý {time.monotonic() - started:.0f} giây · CPU/GPU thực tế của máy")
            try:
                with tempfile.TemporaryDirectory(prefix="accivision-run-", dir=CACHE) as directory:
                    folder = Path(directory)
                    path = source_path
                    if path is None:
                        path = folder / ("input" + Path(source_name).suffix.lower())
                        path.write_bytes(source_data)
                    output = folder / "result.mp4"
                    result = run_analysis(path, output, progress)
                    save_cached(key, result, output)
                    st.session_state.active_result = dict(key=key, result=result,
                                                         video=output.read_bytes(), cached=False, name=source_name)
                    remember(key, source_sha, source_name, analysis_id, round(time.monotonic() - started, 2))
            except Exception as exc:
                if HISTORY is not None and analysis_id:
                    try:
                        HISTORY.fail(analysis_id, exc)
                    except (sqlite3.Error, OSError):
                        pass
                st.error("Chưa hoàn tất phân tích. Không có kết luận cho lần chạy này.")
                with st.expander("Chi tiết lỗi"):
                    st.code(str(exc))
            finally:
                bar.empty()
                status.empty()

    active = st.session_state.active_result
    if active and active["result"]["source_sha256"] == source_sha:
        show_result(active)

else:
    st.info("Chọn tải một video camera giao thông để bắt đầu.")

st.html('<div class="av-footer">AcciVision · Bản thử nghiệm sự cố phương tiện · Phân tích trên máy của bạn.</div>')
