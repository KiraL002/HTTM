"""
AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh
Giao diện Web Dashboard (Streamlit) — Tối Giản, Hiện Đại & Chuyên Nghiệp
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
    page_title="AcciVision — Giám Sát & Phát Hiện Tai Nạn",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GIAO DIỆN TÙY BIẾN (Modern Clean Custom CSS)
# ============================================================

st.markdown(
    """
    <style>
    /* Tổng thể typography & padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Header Card */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #1e293b15;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
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

    /* Alert Banner Clean Styling */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-badge.danger {
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .status-badge.safe {
        background-color: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }

    /* Clean Card Container */
    .clean-card {
        background: #1e293b08;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# NẠP CẤU HÌNH (Load Configuration)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

try:
    config = load_config(str(CONFIG_PATH))
except Exception as exc:
    st.error(f"❌ Không thể nạp tệp cấu hình config.yaml: {exc}")
    st.stop()

# ============================================================
# THANH BÊN (Sidebar — Gọn gàng & Tối giản)
# ============================================================

with st.sidebar:
    st.markdown("### 🚦 AcciVision")
    st.caption("Hệ thống phân tích & phát hiện va chạm tự động")

    st.markdown("---")

    # Tùy chỉnh tham số được gom gọn vào Expander để tránh chiếm diện tích
    with st.expander("⚙️ Tùy chỉnh nâng cao", expanded=False):
        sidebar_yolo_conf = st.slider(
            "Độ tin cậy YOLO",
            min_value=0.15,
            max_value=0.70,
            value=float(config["yolo"]["confidence"]),
            step=0.05,
            help="Ngưỡng phát hiện phương tiện tối thiểu.",
        )
        sidebar_thresh = st.slider(
            "Ngưỡng xác định tai nạn",
            min_value=0.50,
            max_value=0.95,
            value=float(config["classifier"].get("threshold", 0.75)),
            step=0.05,
            help="Tăng ngưỡng để triệt tiêu báo động giả trong điều kiện mật độ cao.",
        )
        sidebar_conf_frames = st.slider(
            "Số frame xác nhận",
            min_value=4,
            max_value=20,
            value=int(config["classifier"].get("confirmation_frames", 8)),
            step=1,
            help="Số frame liên tiếp duy trì để chốt sự kiện va chạm.",
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            <b>Trạng thái:</b> Sẵn sàng hoạt động<br>
            <b>Phiên bản:</b> v1.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER CHÍNH
# ============================================================

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">
            <span>🚦</span> AcciVision Dashboard
        </div>
        <p class="hero-subtitle">
            Hệ thống phát hiện và phân tích tai nạn giao thông thông minh từ camera giám sát.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# KHU VỰC TẢI & THIẾT LẬP VIDEO
# ============================================================

uploaded_file = st.file_uploader(
    "Tải lên video giao thông cần phân tích (MP4, AVI, MOV, MKV)",
    type=["mp4", "avi", "mov", "mkv"],
    help="Hỗ trợ các định dạng video chuẩn từ camera giám sát.",
)

if uploaded_file is not None:
    # Bố cục 2 cột: Xem trước video gốc & Tùy chọn xử lý
    col_preview, col_action = st.columns([1.1, 0.9])

    with col_preview:
        st.video(uploaded_file)

    with col_action:
        st.markdown("##### ⚙️ Tùy chọn xuất kết quả")
        save_output_video = st.checkbox(
            "🎬 Vẽ hộp phát hiện & quỹ đạo (Bounding Box)",
            value=True,
            help="Xuất video có đánh dấu phương tiện và sự kiện va chạm.",
        )
        save_csv = st.checkbox(
            "📊 Trích xuất dữ liệu đặc trưng (CSV)",
            value=True,
            help="Ghi lại các thông số tốc độ, gia tốc và góc di chuyển.",
        )

        st.write("")
        run_analysis = st.button(
            "▶ Bắt đầu phân tích video",
            type="primary",
            use_container_width=True,
        )

    # Tự động dọn session state nếu người dùng chọn video khác
    if "last_analyzed_file" in st.session_state and st.session_state["last_analyzed_file"] != uploaded_file.name:
        st.session_state.pop("analysis_result", None)

    # ----------------------------------------------------
    # QUÁ TRÌNH XỬ LÝ (KHI BẤM NÚT)
    # ----------------------------------------------------
    if run_analysis:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_video_path = Path(temp_file.name)

        # Cập nhật thông số vào config
        config["video"]["input"] = str(temp_video_path)
        config["video"]["display"] = False
        config["video"]["save_output"] = save_output_video
        config["yolo"]["confidence"] = sidebar_yolo_conf
        config["classifier"]["threshold"] = sidebar_thresh
        config["classifier"]["confirmation_frames"] = sidebar_conf_frames
        config["output"]["save_video"] = save_output_video
        config["output"]["save_csv"] = save_csv

        outputs_dir = PROJECT_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        output_video = (outputs_dir / f"{uploaded_file.name}").with_suffix(".mp4")
        config["video"]["output"] = str(output_video)

        csv_output = outputs_dir / "streamlit_features.csv"
        config["video"]["features_csv"] = str(csv_output)

        pipeline = TrafficAccidentPipeline(
            config=config,
            project_root=PROJECT_ROOT,
        )

        progress_bar = st.progress(0)
        status_text = st.empty()
        last_progress_percent = [-1]

        def update_progress(value: float) -> None:
            percent = int(value * 100)
            if percent != last_progress_percent[0]:
                last_progress_percent[0] = percent
                progress_bar.progress(value)

        try:
            with st.spinner("🔄 Đang phân tích luồng video — Vui lòng chờ..."):
                result = pipeline.run(progress_callback=update_progress)

            progress_bar.empty()
            status_text.empty()

            # LƯU KẾT QUẢ VÀO SESSION STATE ĐỂ GIỮ NGUYÊN GIAO DIỆN KHI DOWNLOAD
            st.session_state["analysis_result"] = result
            st.session_state["last_analyzed_file"] = uploaded_file.name
            st.session_state["save_output_video_opt"] = save_output_video

        except Exception as exc:
            st.error("❌ Quá trình phân tích gặp lỗi. Vui lòng kiểm tra video đầu vào.")
            st.exception(exc)

    # ========================================================
    # HIỂN THỊ KẾT QUẢ TẬP TRUNG (DUY TRÌ TRẠNG THÁI TRANG WEB)
    # ========================================================
    if "analysis_result" in st.session_state and st.session_state.get("last_analyzed_file") == uploaded_file.name:
        result = st.session_state["analysis_result"]
        save_output_video_saved = st.session_state.get("save_output_video_opt", True)
        events = result.get("events", [])
        has_accident = len(events) > 0

        st.markdown("### 📊 Kết Quả Phân Tích")

        # 4 Thẻ KPI Tóm Tắt (Gọn gàng, không trùng lặp)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Kết luận tổng quát",
                value="🚨 CÓ TAI NẠN" if has_accident else "✅ AN TOÀN",
            )
        with m2:
            st.metric(
                label="Số vụ va chạm",
                value=f"{len(events)} vụ",
            )
        with m3:
            st.metric(
                label="Tổng số khung hình",
                value=f"{result['frames_processed']:,}",
            )
        with m4:
            st.metric(
                label="Tốc độ xử lý",
                value=f"{result['average_fps']:.1f} FPS",
            )

        st.write("")

        # Phân tách nội dung bằng Tabs thay vì cuộn dài
        tab_video, tab_events, tab_export = st.tabs(
            ["🎬 Video Kết Quả", "🚨 Chi Tiết Sự Kiện", "📁 Xuất Dữ Liệu"]
        )

        # --- TAB 1: VIDEO KẾT QUẢ ---
        with tab_video:
            if save_output_video_saved and Path(result["video_output"]).exists():
                st.video(result["video_output"])
            else:
                if not save_output_video_saved:
                    st.info("ℹ️ Tùy chọn xuất video đã tắt để tối ưu tốc độ phân tích.")
                else:
                    st.warning("⚠️ Không tìm thấy tệp video đầu ra.")

        # --- TAB 2: CHI TIẾT SỰ KIỆN ---
        with tab_events:
            ACCIDENT_TYPE_MAP = {
                "rear_end_collision": "Va chạm phía sau (Rear-end)",
                "head_on_collision": "Va chạm đối đầu (Head-on)",
                "side_impact_collision": "Va chạm sườn/chữ T (Side Impact)",
                "multi_vehicle_pileup": "Va chạm liên hoàn (Pileup)",
                "single_vehicle_spin_or_rollover": "Mất lái quay vòng / lật xe",
                "single_vehicle_loss_of_control": "Mất lái chệch làn",
                "single_vehicle_sudden_stop": "Phanh gấp / va chạm vật cản",
                "single_vehicle_incident": "Sự cố đơn phương tiện",
                "vehicle_collision": "Va chạm phương tiện",
                "unknown": "Chưa xác định",
            }

            if events:
                table_rows = []
                for e in events:
                    raw_type = e.get("accident_type", "unknown")
                    type_display = ACCIDENT_TYPE_MAP.get(
                        raw_type,
                        raw_type.replace("_", " ").title(),
                    )
                    tids = e.get("track_ids", [])
                    v_types = e.get("vehicle_types", {})
                    vehicles_str = (
                        ", ".join([f"ID {tid} ({v_types.get(tid, 'xe')})" for tid in tids])
                        if tids
                        else "Không xác định"
                    )

                    start_s = e.get("start_time_s", e.get("start_frame", 0) / 30.0)
                    end_s = e.get("end_time_s", e.get("end_frame", 0) / 30.0)
                    duration = e.get("duration_s", end_s - start_s)

                    table_rows.append(
                        {
                            "Mã": f"#{e['event_id']}",
                            "Phân loại va chạm": type_display,
                            "Phương tiện liên quan": vehicles_str,
                            "Thời gian": f"{start_s:.1f}s ➔ {end_s:.1f}s ({duration:.1f}s)",
                            "Độ tin cậy": f"{e.get('probability', 0.0) * 100:.1f}%",
                            "Trạng thái": "Đang diễn ra" if e.get("active") else "Đã kết thúc",
                        }
                    )

                st.dataframe(
                    pd.DataFrame(table_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("✅ Toàn bộ video không ghi nhận bất kỳ sự cố va chạm nào.")

        # --- TAB 3: XUẤT DỮ LIỆU ---
        with tab_export:
            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                if save_output_video_saved and Path(result["video_output"]).exists():
                    with open(result["video_output"], "rb") as vf:
                        video_bytes = vf.read()
                    st.download_button(
                        label="📥 Tải xuống Video Kết Quả (.mp4)",
                        data=video_bytes,
                        file_name=Path(result["video_output"]).name,
                        mime="video/mp4",
                        use_container_width=True,
                        key="dl_btn_result_video",
                    )
                else:
                    st.write("Không có video xuất kèm.")

            with col_dl2:
                csv_path = Path(result["features_csv"])
                if csv_path.exists():
                    with csv_path.open("rb") as cf:
                        csv_bytes = cf.read()
                    st.download_button(
                        label="📥 Tải xuống Bảng Đặc Trưng (.csv)",
                        data=csv_bytes,
                        file_name=csv_path.name,
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_btn_features_csv",
                    )
                else:
                    st.write("Không có file CSV đặc trưng.")

            # Xem trước dữ liệu đặc trưng dưới dạng thu gọn tùy chọn
            if Path(result["features_csv"]).exists():
                with st.expander("🔍 Xem trước bảng số liệu đặc trưng (50 dòng đầu)", expanded=False):
                    df_preview = pd.read_csv(result["features_csv"]).head(50)
                    st.dataframe(df_preview, use_container_width=True)

else:
    # Khi chưa tải video: Hiển thị giao diện hướng dẫn tối giản, gọn gàng
    st.markdown(
        """
        <div class="clean-card" style="text-align: center; padding: 36px 20px;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">📹</div>
            <h4 style="margin-bottom: 8px;">Chưa có video được chọn</h4>
            <p style="color: #64748b; font-size: 0.95rem; max-width: 500px; margin: 0 auto;">
                Hãy kéo thả hoặc chọn tệp video giám sát giao thông ở trên để bắt đầu phân tích và phát hiện va chạm tự động.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
