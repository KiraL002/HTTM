"""
AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh
Giao diện Giám sát Trực quan, Tập trung Trọng tâm & Không Chi tiết Thừa
"""

from __future__ import annotations

import base64
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from main import (
    TrafficAccidentPipeline,
    load_config,
)

# ============================================================
# CẤU HÌNH TRANG STREAMLIT (Page Configuration)
# ============================================================

st.set_page_config(
    page_title="AcciVision — Phát Hiện Tai Nạn Giao Thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GIAO DIỆN TẬP TRUNG, TỐI GIẢN & HIỆN ĐẠI (Clean Focused CSS)
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1300px;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.main-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.main-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    margin: 4px 0 0 0;
}

/* Alert Banner Trực diện */
.alert-banner {
    padding: 16px 20px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 14px;
}

.alert-banner.danger {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.45);
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
}

.alert-banner.safe {
    background: rgba(34, 197, 94, 0.15);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.45);
}

/* Thẻ sự kiện va chạm */
.event-item {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.event-item-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.event-item-meta {
    font-size: 0.88rem;
    color: #cbd5e1;
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}

.tag-vehicle {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    padding: 2px 8px;
    border-radius: 6px;
    color: #e2e8f0;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Nút chính */
div.stButton > button[kind="primary"] {
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.65rem 1.4rem;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border: none;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
}

/* Card container */
.clean-panel {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# CẤU HÌNH & TÀI NGUYÊN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEO_DIR = PROJECT_ROOT / "dataset" / "input_video"

ACCIDENT_TYPE_MAP = {
    "car_motorcycle_collision": ("💥", "Va chạm xe máy với ô tô"),
    "vehicle_pedestrian_collision": ("⚠️", "Phương tiện va chạm người đi bộ"),
    "head_on_collision": ("⚡", "Va chạm đối đầu (Trực diện)"),
    "rear_end_collision": ("💥", "Va chạm phía sau (Tông đuôi)"),
    "side_impact_collision": ("📐", "Va chạm sườn xe / Giao lộ"),
    "multi_vehicle_pileup": ("⛓️", "Va chạm liên hoàn (Nhiều xe)"),
    "single_vehicle_spin_or_rollover": ("🔄", "Mất lái quay vòng / Lật xe"),
    "single_vehicle_crash": ("🛑", "Xe đâm vật cản / Dải phân cách"),
    "single_vehicle_loss_of_control": ("〰️", "Mất lái / Chệch làn đường"),
    "single_vehicle_sudden_stop": ("🛑", "Phanh khẩn cấp / Dừng đột ngột"),
    "single_vehicle_incident": ("⚠️", "Sự cố phương tiện"),
    "vehicle_collision": ("💥", "Va chạm phương tiện"),
    "unknown": ("⚠️", "Sự cố giao thông"),
}

try:
    base_config = load_config(str(CONFIG_PATH))
except Exception as exc:
    st.error(f"❌ Không thể nạp tệp config.yaml: {exc}")
    st.stop()


def get_cached_result(video_stem: str) -> Optional[Dict[str, Any]]:
    """Tải kết quả đã lưu sẵn trong outputs/ nếu có."""
    json_path = OUTPUTS_DIR / f"{video_stem}_result.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_result(video_stem: str, res: Dict[str, Any]) -> None:
    """Lưu kết quả phân tích để xem lại nhanh lần sau."""
    json_path = OUTPUTS_DIR / f"{video_stem}_result.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def get_accident_crop_base64(
    video_path_str: str,
    mid_frame: int,
    track_ids: list,
    features_csv_str: str,
    config_path_str: str,
    existing_b64: Optional[str] = None,
    crop_path_str: Optional[str] = None,
) -> Optional[str]:
    """
    Trích xuất hình ảnh cắt ra cận cảnh (cropped snapshot) tại đúng hiện trường va chạm.
    Tập trung chính xác vào phương tiện gặp sự cố, loại bỏ hoàn toàn việc lệch góc.
    """
    # 1. Ưu tiên sử dụng ảnh crop trực tiếp từ pipeline nếu đã có
    if existing_b64:
        return existing_b64
    if crop_path_str and Path(crop_path_str).exists():
        try:
            with open(crop_path_str, "rb") as img_f:
                return base64.b64encode(img_f.read()).decode("utf-8")
        except Exception:
            pass

    if not video_path_str or not Path(video_path_str).exists():
        return None

    try:
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(mid_frame)))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        h, w, _ = frame.shape
        crop_box = None

        # 2. Kiểm tra nếu trong CSV có lưu tọa độ pixel bounding box thực tế
        if features_csv_str and Path(features_csv_str).exists():
            try:
                df = pd.read_csv(features_csv_str)
                req_cols = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
                if req_cols.issubset(set(df.columns)):
                    rows = df[
                        (df["frame_id"].between(mid_frame - 3, mid_frame + 3))
                        & (df["track_id"].isin(track_ids))
                    ]
                    if not rows.empty:
                        ux1 = float(rows["bbox_x1"].min())
                        uy1 = float(rows["bbox_y1"].min())
                        ux2 = float(rows["bbox_x2"].max())
                        uy2 = float(rows["bbox_y2"].max())
                        crop_box = (ux1, uy1, ux2, uy2)
            except Exception:
                pass

        # 3. Nếu chưa có tọa độ pixel từ CSV: Dùng mô hình YOLO phát hiện trực tiếp phương tiện trên frame
        if crop_box is None:
            try:
                from ultralytics import YOLO
                yolo_model_p = PROJECT_ROOT / "models" / "yolo11s_accident_vn.pt"
                if not yolo_model_p.exists():
                    yolo_model_p = PROJECT_ROOT / "models" / "yolov8n.pt"

                det_model = YOLO(str(yolo_model_p))
                preds = det_model(frame, verbose=False)
                if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
                    b_list = preds[0].boxes.xyxy.cpu().numpy()
                    if len(b_list) >= 2:
                        # Tìm 2 xe gần nhau nhất (tiếp xúc va chạm)
                        min_d = 99999.0
                        best_pair = (b_list[0], b_list[1])
                        for i in range(len(b_list)):
                            for j in range(i + 1, len(b_list)):
                                c1 = ((b_list[i][0] + b_list[i][2]) / 2, (b_list[i][1] + b_list[i][3]) / 2)
                                c2 = ((b_list[j][0] + b_list[j][2]) / 2, (b_list[j][1] + b_list[j][3]) / 2)
                                d = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
                                if d < min_d:
                                    min_d = d
                                    best_pair = (b_list[i], b_list[j])
                        ux1 = min(best_pair[0][0], best_pair[1][0])
                        uy1 = min(best_pair[0][1], best_pair[1][1])
                        ux2 = max(best_pair[0][2], best_pair[1][2])
                        uy2 = max(best_pair[0][3], best_pair[1][3])
                        crop_box = (ux1, uy1, ux2, uy2)
                    elif len(b_list) == 1:
                        crop_box = tuple(b_list[0])
            except Exception:
                pass

        # 4. Thực hiện crop với tỷ lệ mở rộng hợp lý
        if crop_box is not None:
            ux1, uy1, ux2, uy2 = crop_box
            pad_x = max(60, int((ux2 - ux1) * 0.45))
            pad_y = max(60, int((uy2 - uy1) * 0.50))
            x1 = max(0, int(ux1 - pad_x))
            y1 = max(0, int(uy1 - pad_y))
            x2 = min(w, int(ux2 + pad_x))
            y2 = min(h, int(uy2 + pad_y))
            display_img = frame[y1:y2, x1:x2]
        else:
            # Fallback an toàn: lấy vùng trung tâm 70% khung hình
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            display_img = frame[margin_y:h - margin_y, margin_x:w - margin_x]

        _, buffer = cv2.imencode(".jpg", display_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return None


# ============================================================
# TIÊU ĐỀ TRANG (Header Gọn gàng)
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div>
            <h1 class="main-title">
                <span>🚦</span> AcciVision — Phát Hiện Tai Nạn Giao Thông
            </h1>
            <p class="main-subtitle">
                Kiểm tra video camera giám sát và phát hiện tức thời các tình huống va chạm giao thông.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CHỌN VIDEO & THAO TÁC (Tập trung trên 1 hàng)
# ============================================================

col_source, col_action = st.columns([1.2, 0.8])

selected_video_path: Optional[Path] = None
selected_video_name: str = ""

with col_source:
    st.markdown("##### 📥 1. Chọn video cần kiểm tra")
    source_type = st.radio(
        "Nguồn video:",
        options=["Chọn video mẫu có sẵn", "Tải video từ máy tính"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_type == "Chọn video mẫu có sẵn":
        sample_videos = sorted(
            list(SAMPLE_VIDEO_DIR.glob("*.mp4")) + list(SAMPLE_VIDEO_DIR.glob("*.avi")),
            key=lambda p: p.name,
        ) if SAMPLE_VIDEO_DIR.exists() else []

        if sample_videos:
            sample_dict = {p.name: p for p in sample_videos}
            chosen_name = st.selectbox(
                "Danh sách video mẫu:",
                options=list(sample_dict.keys()),
                label_visibility="collapsed",
            )
            selected_video_path = sample_dict[chosen_name]
            selected_video_name = chosen_name
        else:
            st.warning("Không có video mẫu trong dataset/input_video/")
    else:
        uploaded_file = st.file_uploader(
            "Tải video (MP4, AVI, MOV):",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                tmp_f.write(uploaded_file.getbuffer())
                selected_video_path = Path(tmp_f.name)
            selected_video_name = uploaded_file.name

# Dọn dẹp kết quả nếu người dùng chuyển sang video khác
if "current_video_name" in st.session_state and st.session_state["current_video_name"] != selected_video_name:
    st.session_state.pop("analysis_result", None)

run_clicked = False
load_cached_clicked = False

with col_action:
    st.markdown("##### ⚡ 2. Bắt đầu xử lý")
    if selected_video_path is not None:
        video_stem = Path(selected_video_name).stem
        cached_result = get_cached_result(video_stem)

        btn_c1, btn_c2 = st.columns([1.2, 0.8])
        with btn_c1:
            run_clicked = st.button(
                "▶ Bắt Đầu Kiểm Tra",
                type="primary",
                use_container_width=True,
            )
        with btn_c2:
            if cached_result is not None:
                load_cached_clicked = st.button(
                    "⚡ Xem Kết Quả",
                    use_container_width=True,
                    help="Tải nhanh kết quả đã phân tích của video này",
                )
    else:
        st.info("Vui lòng chọn hoặc tải video lên để bắt đầu.")

# Cài đặt nâng cao thu gọn trong Sidebar (Không làm phiền người dùng)
with st.sidebar:
    st.markdown("### ⚙️ Cài Đặt Nâng Cao")
    
    model_choices = {
        "YOLO11s Accident VN (Khuyên dùng - Chuẩn giao thông VN)": "models/yolo11s_accident_vn.pt",
        "YOLOv8 Accident Custom (Chuyên phát hiện tai nạn)": "models/accident_yolo_best.pt",
        "YOLO11n Accident VN (Siêu nhẹ - Tốc độ cao)": "models/yolo11n_accident_vn.pt",
        "YOLOv8n Tiêu chuẩn (COCO)": "models/yolov8n.pt",
    }
    selected_model_label = st.selectbox(
        "Mô hình nhận diện phương tiện & tai nạn:",
        options=list(model_choices.keys()),
        index=0,
    )
    chosen_model_path = model_choices[selected_model_label]

    accident_threshold = st.slider(
        "Ngưỡng xác suất tai nạn:",
        min_value=0.50,
        max_value=0.95,
        value=float(base_config.get("classifier", {}).get("threshold", 0.75)),
        step=0.05,
    )
    yolo_confidence = st.slider(
        "Ngưỡng nhận diện xe (YOLO):",
        min_value=0.15,
        max_value=0.70,
        value=float(base_config.get("yolo", {}).get("confidence", 0.30)),
        step=0.05,
    )
    confirmation_frames = st.slider(
        "Số frame xác nhận (Confirmation Frames):",
        min_value=2,
        max_value=10,
        value=int(base_config.get("classifier", {}).get("confirmation_frames", 4)),
        step=1,
    )

# ============================================================
# THỰC THI PHÂN TÍCH HOẶC TẢI KẾT QUẢ
# ============================================================

if load_cached_clicked and selected_video_path is not None:
    video_stem = Path(selected_video_name).stem
    st.session_state["analysis_result"] = get_cached_result(video_stem)
    st.session_state["current_video_name"] = selected_video_name
    st.rerun()

if run_clicked and selected_video_path is not None:
    video_stem = Path(selected_video_name).stem
    run_config = copy.deepcopy(base_config)
    run_config["video"]["input"] = str(selected_video_path)
    run_config["video"]["display"] = False
    run_config["video"]["save_output"] = True
    run_config["output"]["save_video"] = True
    run_config["output"]["save_csv"] = True

    run_config["yolo"]["model"] = chosen_model_path
    run_config["classifier"]["threshold"] = accident_threshold
    run_config["classifier"]["confirmation_frames"] = confirmation_frames
    run_config["yolo"]["confidence"] = yolo_confidence

    output_video_path = OUTPUTS_DIR / f"{video_stem}.mp4"
    run_config["video"]["output"] = str(output_video_path)
    run_config["video"]["features_csv"] = str(OUTPUTS_DIR / "traffic_features.csv")

    pipeline = TrafficAccidentPipeline(
        config=run_config,
        project_root=PROJECT_ROOT,
    )

    progress_bar = st.progress(0, text="Đang phân tích khung hình video...")
    last_pct = [-1]

    def update_prog(val: float) -> None:
        pct = int(val * 100)
        if pct != last_pct[0]:
            last_pct[0] = pct
            progress_bar.progress(val, text=f"Đang phân tích video: {pct}% hoàn tất...")

    try:
        with st.spinner("Đang chạy mô hình AI phát hiện va chạm..."):
            res = pipeline.run(progress_callback=update_prog)

        progress_bar.empty()
        st.session_state["analysis_result"] = res
        st.session_state["current_video_name"] = selected_video_name
        save_result(video_stem, res)
        st.rerun()
    except Exception as err:
        st.error(f"❌ Xử lý video thất bại: {err}")

# ============================================================
# KẾT QUẢ PHÂN TÍCH (TRỌNG TÂM DUY NHẤT - NO FLUFF)
# ============================================================

if "analysis_result" in st.session_state and selected_video_path is not None:
    result = st.session_state["analysis_result"]
    events = result.get("events", [])
    has_accident = result.get("has_accident", len(events) > 0)
    video_stem = Path(selected_video_name).stem

    video_out_raw = result.get("video_output", "")
    video_out_path = Path(video_out_raw) if video_out_raw else OUTPUTS_DIR / f"{video_stem}.mp4"
    if not video_out_path.is_absolute():
        video_out_path = PROJECT_ROOT / video_out_path

    features_csv_path = OUTPUTS_DIR / "traffic_features.csv"

    st.markdown("---")

    # 1. BANNER KẾT LUẬN TRỰC QUAN
    if has_accident:
        st.markdown(
            f"""
            <div class="alert-banner danger">
                <span style="font-size: 1.8rem;">🚨</span>
                <div>
                    CẢNH BÁO: PHÁT HIỆN {len(events)} TÌNH HUỐNG TAI NẠN / VA CHẠM TRONG VIDEO!
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="alert-banner safe">
                <span style="font-size: 1.8rem;">✅</span>
                <div>
                    TÌNH TRẠNG BÌNH THƯỜNG: KHÔNG CÓ TAI NẠN / VA CHẠM TRONG VIDEO NÀY.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. 3 CHỈ SỐ CỐT LÕI (Không rườm rà)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="KẾT LUẬN",
            value="🚨 Có va chạm" if has_accident else "✅ An toàn",
        )
    with m2:
        st.metric(
            label="SỐ VỤ VA CHẠM",
            value=f"{len(events)} vụ việc",
        )
    with m3:
        st.metric(
            label="TỔNG SỐ KHUNG HÌNH (FPS)",
            value=f"{result.get('frames_processed', 0):,} frames ({result.get('average_fps', 0):.1f} FPS)",
        )

    st.write("")

    # 3. BỐ CỤC CHÍNH: VIDEO KẾT QUẢ & DANH SÁCH SỰ CỐ
    col_video_display, col_event_details = st.columns([1.25, 0.75])

    with col_video_display:
        st.markdown("#### 🎬 Video Kết Quả (Đánh dấu xe & vị trí va chạm)")
        if video_out_path.exists():
            st.video(str(video_out_path))

            # Nút tải video kết quả
            with open(video_out_path, "rb") as vf:
                v_data = vf.read()
            st.download_button(
                label="📥 Tải video kết quả (.mp4)",
                data=v_data,
                file_name=video_out_path.name,
                mime="video/mp4",
                type="primary",
                use_container_width=True,
            )
        else:
            st.warning("Chưa tìm thấy video kết quả trên đĩa.")

        with st.expander("▶ Xem video gốc từ camera", expanded=False):
            st.video(str(selected_video_path))

    with col_event_details:
        st.markdown("#### 📋 Chi Tiết Các Vụ Va Chạm")

        if not events:
            st.markdown(
                """
                <div class="clean-panel" style="text-align: center; padding: 30px 15px;">
                    <span style="font-size: 2rem;">🛡️</span>
                    <h4 style="color: #4ade80; margin: 8px 0;">Giao thông an toàn</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
                        Không phát hiện va chạm hoặc xung đột quỹ đạo nguy hiểm.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            video_for_crops = str(video_out_path) if video_out_path.exists() else str(selected_video_path)
            config_str = str(CONFIG_PATH)
            csv_str = str(features_csv_path) if features_csv_path.exists() else ""

            for idx, ev in enumerate(events, start=1):
                start_s = ev.get("start_time_s", 0.0)
                end_s = ev.get("end_time_s", 0.0)
                dur_s = ev.get("duration_s", max(0.0, end_s - start_s))
                prob = ev.get("probability", 0.0) * 100
                tids = ev.get("track_ids", [])
                vtypes = ev.get("vehicle_types", {})

                start_f = ev.get("start_frame", 0)
                end_f = ev.get("end_frame", 0)
                mid_frame = ev.get("peak_frame") or ((start_f + end_f) // 2)
                if ev.get("peak_time_s") is not None:
                    mid_time = float(ev["peak_time_s"])
                elif start_f > 0 and start_s > 0:
                    mid_time = mid_frame * (start_s / start_f)
                else:
                    mid_time = (start_s + end_s) / 2.0

                acc_type = ev.get("accident_type", "unknown")
                icon, fallback_title = ACCIDENT_TYPE_MAP.get(acc_type, ("💥", "Va chạm giao thông"))
                type_title = ev.get("type_title") or fallback_title
                desc = ev.get("description", "")

                def vn_type_name(cls_name: str) -> str:
                    c = str(cls_name).lower()
                    if c in ("car", "sedan", "suv"):
                        return "ô tô con"
                    if c in ("truck", "lorry"):
                        return "xe tải"
                    if c in ("bus",):
                        return "xe buýt"
                    if c in ("motorcycle", "motorbike", "scooter"):
                        return "xe máy"
                    if c in ("bicycle", "bike"):
                        return "xe đạp"
                    if c in ("person", "pedestrian"):
                        return "người đi bộ"
                    return "phương tiện"

                v_badges = " ".join([
                    f'<span class="tag-vehicle">Xe #{tid} ({vn_type_name(vtypes.get(str(tid), vtypes.get(tid, "xe")))})</span>'
                    for tid in tids
                ])

                if not desc:
                    desc = f"Phát hiện tình huống {type_title.lower()} liên quan đến {', '.join([f'Xe #{tid}' for tid in tids])} tại thời điểm {start_s:.2f}s (Frame #{start_f} - #{end_f})."

                # Trích xuất hình ảnh cắt ra của vụ tai nạn
                crop_b64 = get_accident_crop_base64(
                    video_path_str=video_for_crops,
                    mid_frame=mid_frame,
                    track_ids=tids,
                    features_csv_str=csv_str,
                    config_path_str=config_str,
                    existing_b64=ev.get("snapshot_base64"),
                    crop_path_str=ev.get("crop_path"),
                )

                img_html = ""
                if crop_b64:
                    img_html = f"""<div style="margin: 10px 0; border-radius: 8px; overflow: hidden; border: 1px solid rgba(239, 68, 68, 0.45); box-shadow: 0 4px 14px rgba(0,0,0,0.35);">
<img src="data:image/jpeg;base64,{crop_b64}" style="width: 100%; height: auto; display: block; object-fit: cover;" alt="Hiện trường tai nạn #{idx}" />
<div style="background: rgba(15, 23, 42, 0.9); padding: 5px 10px; font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
<span>📸 Ảnh cắt hiện trường va chạm</span>
<span>🎯 Frame #{mid_frame} ({mid_time:.2f}s)</span>
</div>
</div>"""

                card_html = f"""<div class="event-item">
<div class="event-item-title">
<span>{icon} Vụ #{idx}: {type_title}</span>
<span style="color: #f87171; font-size: 0.95rem; font-weight: 700;">{prob:.1f}%</span>
</div>
{img_html}
<div class="event-item-meta">
<span>⏱️ <b>Thời điểm:</b> {start_s:.2f}s ➔ {end_s:.2f}s ({dur_s:.2f}s)</span>
<span>🎯 <b>Frame:</b> #{start_f} - #{end_f}</span>
</div>
<div style="margin-top: 8px;">
<span style="font-size: 0.82rem; color: #94a3b8; font-weight: 600;">🚗 Phương tiện liên quan:</span> {v_badges if v_badges else '<span class="tag-vehicle">Chưa xác định</span>'}
</div>
<div style="margin-top: 10px; padding: 10px 12px; background: rgba(30, 41, 59, 0.75); border-radius: 6px; border-left: 3px solid #f87171;">
<div style="font-size: 0.82rem; color: #fca5a5; font-weight: 700; margin-bottom: 4px;">📝 Mô tả hiện trường & loại tai nạn:</div>
<div style="font-size: 0.88rem; color: #f1f5f9; line-height: 1.45;">{desc}</div>
</div>
</div>"""

                clean_card_html = " ".join(line.strip() for line in card_html.splitlines() if line.strip())
                st.markdown(clean_card_html, unsafe_allow_html=True)

            # 1 nút xuất báo cáo CSV duy nhất nếu cần lưu trữ
            if features_csv_path.exists():
                with open(features_csv_path, "rb") as cf:
                    csv_data = cf.read()
                st.download_button(
                    label="📊 Tải bảng đặc trưng động học (.csv)",
                    data=csv_data,
                    file_name=f"{video_stem}_features.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

else:
    # Trạng thái ban đầu
    st.markdown(
        """
        <div class="clean-panel" style="text-align: center; padding: 45px 20px;">
            <div style="font-size: 2.8rem; margin-bottom: 12px;">📁</div>
            <h3 style="color: #f8fafc; margin-bottom: 6px; font-weight: 700;">
                Sẵn Sàng Phân Tích
            </h3>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 500px; margin: 0 auto;">
                Chọn video mẫu hoặc tải tệp video camera ở trên, sau đó bấm <b>"Bắt Đầu Kiểm Tra"</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
