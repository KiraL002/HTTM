"""
AcciVision — Hệ Thống Giám Sát & Cảnh Báo Tai Nạn Giao Thông
Giao diện Web Trực Quan, Tối Giản Cho Người Dùng
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from main import (
    TrafficAccidentPipeline,
    load_config,
)

# ============================================================
# CẤU HÌNH TRANG (Page Configuration)
# ============================================================

st.set_page_config(
    page_title="Hệ Thống Cảnh Báo Tai Nạn Giao Thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GIAO DIỆN HIỆN ĐẠI & TỐI GIẢN (Custom CSS)
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header Card */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #1e293b12;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        color: #64748b !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 1.4rem;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }

    /* Alert Banner */
    .alert-banner {
        padding: 18px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.15rem;
        margin: 18px 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .alert-banner.danger {
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .alert-banner.safe {
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* Card Box */
    .clean-box {
        background: #1e293b08;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# NẠP CẤU HÌNH (Load Configuration Tự Động)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

try:
    config = load_config(str(CONFIG_PATH))
except Exception as exc:
    st.error(f"❌ Không thể nạp tệp cấu hình: {exc}")
    st.stop()

# ============================================================
# TIÊU ĐỀ ỨNG DỤNG (Header Thân Thiện)
# ============================================================

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">
            <span>🚦</span> Giám Sát & Cảnh Báo Tai Nạn Giao Thông
        </div>
        <p class="hero-subtitle">
            Tải lên video camera giao thông để kiểm tra và phát hiện va chạm tự động.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TẢI VIDEO & PHÂN TÍCH
# ============================================================

uploaded_file = st.file_uploader(
    "Chọn video giao thông cần kiểm tra (MP4, AVI, MOV, MKV)",
    type=["mp4", "avi", "mov", "mkv"],
    help="Hỗ trợ các định dạng video từ camera giám sát giao thông.",
)

if uploaded_file is not None:
    # Xem trước video gốc và nút bắt đầu
    col_video, col_btn = st.columns([1.2, 0.8])
    
    with col_video:
        st.video(uploaded_file)
        
    with col_btn:
        st.markdown("### Thao tác")
        st.caption("Bấm nút bên dưới để hệ thống tự động kiểm tra sự cố trong video:")
        
        run_analysis = st.button(
            "▶ Bắt đầu kiểm tra video",
            type="primary",
            use_container_width=True,
        )

    # Tự động xóa kết quả cũ nếu người dùng đổi video khác
    if "last_analyzed_file" in st.session_state and st.session_state["last_analyzed_file"] != uploaded_file.name:
        st.session_state.pop("analysis_result", None)

    # ========================================================
    # XỬ LÝ VIDEO
    # ========================================================
    if run_analysis:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_video_path = Path(temp_file.name)

        config["video"]["input"] = str(temp_video_path)
        config["video"]["display"] = False
        config["video"]["save_output"] = True
        config["output"]["save_video"] = True
        config["output"]["save_csv"] = True

        outputs_dir = PROJECT_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        output_video = (outputs_dir / f"{uploaded_file.name}").with_suffix(".mp4")
        config["video"]["output"] = str(output_video)

        csv_output = outputs_dir / "traffic_features.csv"
        config["video"]["features_csv"] = str(csv_output)

        pipeline = TrafficAccidentPipeline(
            config=config,
            project_root=PROJECT_ROOT,
        )

        progress_bar = st.progress(0)
        last_progress_percent = [-1]

        def update_progress(value: float) -> None:
            percent = int(value * 100)
            if percent != last_progress_percent[0]:
                last_progress_percent[0] = percent
                progress_bar.progress(value)

        try:
            with st.spinner("🔄 Đang xử lý video... Vui lòng đợi trong giây lát."):
                result = pipeline.run(progress_callback=update_progress)

            progress_bar.empty()

            # Lưu vào session_state để không bị mất kết quả khi bấm nút tải video
            st.session_state["analysis_result"] = result
            st.session_state["last_analyzed_file"] = uploaded_file.name

        except Exception as exc:
            st.error("❌ Quá trình phân tích gặp sự cố. Vui lòng thử lại với video khác.")
            st.exception(exc)

    # ========================================================
    # HIỂN THỊ KẾT QUẢ PHÂN TÍCH (LƯU TRỮ VĨNH VIỄN TRONG PHIÊN)
    # ========================================================
    if "analysis_result" in st.session_state and st.session_state.get("last_analyzed_file") == uploaded_file.name:
        result = st.session_state["analysis_result"]
        events = result.get("events", [])
        has_accident = len(events) > 0

        st.markdown("---")
        st.markdown("## 📊 Kết Quả Kiểm Tra")

        # Banner trạng thái lớn
        if has_accident:
            st.markdown(
                f"""
                <div class="alert-banner danger">
                    <span>🚨</span> <b>CẢNH BÁO: PHÁT HIỆN {len(events)} TÌNH HUỐNG TAI NẠN / VA CHẠM TRONG VIDEO!</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="alert-banner safe">
                    <span>✅</span> <b>TÌNH TRẠNG AN TOÀN: KHÔNG PHÁT HIỆN TAI NẠN TRONG VIDEO NÀY.</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3 Thẻ số liệu chính
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Tình trạng",
                value="🚨 Có va chạm" if has_accident else "✅ Bình thường",
            )
        with c2:
            st.metric(
                label="Số vụ va chạm",
                value=f"{len(events)} vụ",
            )
        with c3:
            st.metric(
                label="Tổng số khung hình",
                value=f"{result['frames_processed']:,} frames",
            )

        st.write("")

        # Video đầu ra (có đánh dấu xe và va chạm)
        output_video_path = Path(result["video_output"])
        if output_video_path.exists():
            st.markdown("### 🎬 Video Kết Quả (Đánh Dấu Vị Trí Sự Cố)")
            st.video(str(output_video_path))

            # Nút tải video về máy
            with open(output_video_path, "rb") as vf:
                video_data = vf.read()
            st.download_button(
                label="📥 Tải video kết quả về máy (.mp4)",
                data=video_data,
                file_name=output_video_path.name,
                mime="video/mp4",
                type="primary",
                key="download_result_video_btn",
            )

        # Bảng chi tiết sự việc nếu có va chạm
        if events:
            st.markdown("### 📋 Danh Sách Các Vụ Va Chạm Được Ghi Nhận")

            ACCIDENT_TYPE_MAP = {
                "rear_end_collision": "Va chạm phía sau (Tông đuôi)",
                "head_on_collision": "Va chạm đối đầu",
                "side_impact_collision": "Va chạm sườn xe",
                "multi_vehicle_pileup": "Va chạm liên hoàn",
                "single_vehicle_spin_or_rollover": "Mất lái quay vòng / Lật xe",
                "single_vehicle_loss_of_control": "Mất lái chệch làn",
                "single_vehicle_sudden_stop": "Phanh gấp bất thường",
                "single_vehicle_incident": "Sự cố phương tiện",
                "vehicle_collision": "Va chạm phương tiện",
                "unknown": "Va chạm giao thông",
            }

            table_rows = []
            for idx, e in enumerate(events, start=1):
                raw_type = e.get("accident_type", "unknown")
                type_display = ACCIDENT_TYPE_MAP.get(raw_type, raw_type.replace("_", " ").title())
                
                start_s = e.get("start_time_s", e.get("start_frame", 0) / 30.0)
                end_s = e.get("end_time_s", e.get("end_frame", 0) / 30.0)
                duration = e.get("duration_s", end_s - start_s)
                
                tids = e.get("track_ids", [])
                v_types = e.get("vehicle_types", {})
                vehicles_str = (
                    ", ".join([f"Xe #{tid} ({v_types.get(tid, 'xe')})" for tid in tids])
                    if tids
                    else "Không xác định"
                )

                table_rows.append({
                    "STT": idx,
                    "Loại va chạm": type_display,
                    "Phương tiện liên quan": vehicles_str,
                    "Thời điểm xuất hiện": f"{start_s:.1f}s ➔ {end_s:.1f}s (Kéo dài {duration:.1f}s)",
                    "Xác suất": f"{e.get('probability', 0.0) * 100:.1f}%",
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

else:
    # Trạng thái ban đầu khi chưa chọn video
    st.markdown(
        """
        <div class="clean-box" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📁</div>
            <h3 style="margin-bottom: 8px; color: #f1f5f9;">Vui lòng tải video lên</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 500px; margin: 0 auto;">
                Kéo và thả tệp video từ máy tính vào khung bên trên để bắt đầu kiểm tra giao thông.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
