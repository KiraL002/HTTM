"""
AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh
Giao diện Web Dashboard (Streamlit)

Module này cung cấp giao diện web trực quan cho phép người dùng:
- Tải lên video giao thông để phân tích
- Theo dõi tiến trình xử lý pipeline
- Xem kết quả phát hiện tai nạn chi tiết
- Tải xuống video kết quả và tệp CSV đặc trưng
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from main import (
    TrafficAccidentPipeline,
    load_config,
)


# ============================================================
# CẤU HÌNH TRANG (Page Configuration)
# ============================================================

st.set_page_config(
    page_title="AcciVision — Phát Hiện Tai Nạn Giao Thông",
    page_icon="🚦",
    layout="wide",
)


# ============================================================
# HEADER — TIÊU ĐỀ HỆ THỐNG
# ============================================================

st.title(
    "🚦 AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh"
)

st.caption(
    "Sử dụng Thị giác Máy tính (YOLOv8 + ByteTrack) kết hợp "
    "Chuyển đổi Phối cảnh & Phân loại Học máy (Random Forest) "
    "để phát hiện và phân loại tai nạn giao thông từ video."
)

st.divider()


# ============================================================
# NẠP CẤU HÌNH (Load Configuration)
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

CONFIG_PATH = (
    PROJECT_ROOT / "config.yaml"
)

try:
    config = load_config(
        str(CONFIG_PATH)
    )
except Exception as exc:
    st.error(
        f"❌ Không thể nạp tệp cấu hình config.yaml: {exc}"
    )
    st.stop()


# ============================================================
# THANH BÊN — CẤU HÌNH HỆ THỐNG (Sidebar Configuration)
# ============================================================

st.sidebar.image(
    "https://img.icons8.com/fluency/96/traffic-light.png",
    width=64,
)

st.sidebar.title("⚙️ Cấu Hình Hệ Thống")

st.sidebar.markdown("---")

st.sidebar.subheader("📷 Mô hình phát hiện")
st.sidebar.code(
    config["yolo"]["model"],
    language=None,
)

st.sidebar.subheader("🧠 Mô hình phân loại")
st.sidebar.code(
    config["classifier"]["model"],
    language=None,
)

st.sidebar.subheader("📊 Tinh chỉnh độ nhạy (Tùy chọn)")
sidebar_yolo_conf = st.sidebar.slider(
    "Ngưỡng tin cậy YOLO",
    min_value=0.15,
    max_value=0.70,
    value=float(config["yolo"]["confidence"]),
    step=0.05,
    help="Độ tin cậy tối thiểu để YOLO nhận diện phương tiện.",
)
sidebar_thresh = st.sidebar.slider(
    "Ngưỡng phát hiện tai nạn",
    min_value=0.50,
    max_value=0.95,
    value=float(config["classifier"].get("threshold", 0.75)),
    step=0.05,
    help="Tăng ngưỡng lên 0.75 - 0.85 để triệt tiêu báo động giả khi xe lưu thông bình thường.",
)
sidebar_conf_frames = st.sidebar.slider(
    "Số frame xác nhận",
    min_value=4,
    max_value=20,
    value=int(config["classifier"].get("confirmation_frames", 8)),
    step=1,
    help="Số frame liên tiếp duy trì trạng thái bất thường để chốt tai nạn. Tăng giá trị để loại bỏ rung lắc camera.",
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔧 Phối cảnh (Perspective)")
perspective_status = "✅ Bật" if config["perspective"]["enabled"] else "❌ Tắt"
st.sidebar.write(f"Trạng thái: {perspective_status}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "AcciVision v1.0 — Hệ thống phát hiện tai nạn giao thông thông minh"
)


# ============================================================
# KHU VỰC CHÍNH — TẢI VIDEO & PHÂN TÍCH
# ============================================================

st.subheader("📤 Bước 1: Tải lên video giao thông")

uploaded_file = st.file_uploader(
    "Chọn tệp video giao thông cần phân tích",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
    help="Hỗ trợ định dạng: MP4, AVI, MOV, MKV. Khuyến nghị video từ camera giám sát giao thông.",
)


if uploaded_file is not None:

    st.video(
        uploaded_file
    )

    st.subheader("⚙️ Bước 2: Tùy chọn xử lý")

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        save_output_video = st.checkbox(
            "🎬 Xuất video kết quả (vẽ BBox + Quỹ đạo)",
            value=True,
            help=(
                "Tạo video đầu ra có vẽ bounding box, quỹ đạo chuyển động "
                "và trạng thái phát hiện. Bỏ chọn để tăng tốc xử lý 2-3 lần."
            ),
        )

    with col_opt2:
        save_csv = st.checkbox(
            "📊 Xuất tệp CSV đặc trưng",
            value=True,
            help="Lưu toàn bộ đặc trưng động học trích xuất được vào tệp CSV.",
        )

    st.subheader("🚀 Bước 3: Bắt đầu phân tích")

    if st.button(
        "▶ Bắt Đầu Phát Hiện Tai Nạn",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # Lưu video tạm thời
        # ----------------------------------------------------

        suffix = Path(
            uploaded_file.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_video_path = Path(
                temp_file.name
            )

        # ----------------------------------------------------
        # Ghi đè cấu hình video đầu vào
        # ----------------------------------------------------

        config["video"]["input"] = (
            str(temp_video_path)
        )

        # Streamlit không sử dụng cửa sổ OpenCV
        config["video"]["display"] = False

        config["video"]["save_output"] = save_output_video

        config["yolo"]["confidence"] = sidebar_yolo_conf
        config["classifier"]["threshold"] = sidebar_thresh
        config["classifier"]["confirmation_frames"] = sidebar_conf_frames

        config["output"]["save_video"] = save_output_video
        config["output"]["save_csv"] = save_csv

        # ----------------------------------------------------
        # Cấu hình đường dẫn đầu ra
        # ----------------------------------------------------

        outputs_dir = (
            PROJECT_ROOT / "outputs"
        )

        outputs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_video = (
            outputs_dir
            / f"{uploaded_file.name}"
        )

        # Đảm bảo định dạng MP4
        output_video = (
            output_video.with_suffix(
                ".mp4"
            )
        )

        config["video"]["output"] = (
            str(output_video)
        )

        csv_output = (
            outputs_dir
            / "streamlit_features.csv"
        )

        config["video"][
            "features_csv"
        ] = str(csv_output)

        # ----------------------------------------------------
        # Khởi tạo và chạy Pipeline
        # ----------------------------------------------------

        pipeline = (
            TrafficAccidentPipeline(
                config=config,
                project_root=PROJECT_ROOT,
            )
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Chỉ cập nhật progress bar mỗi 1% để tránh làm chậm pipeline
        last_progress_percent = [-1]

        def update_progress(value: float) -> None:
            percent = int(value * 100)
            if percent != last_progress_percent[0]:
                last_progress_percent[0] = percent
                progress_bar.progress(value)

        try:

            with st.spinner(
                "🔄 Đang phân tích video — Vui lòng chờ..."
            ):

                result = pipeline.run(
                    progress_callback=update_progress
                )

            status_text.success(
                "✅ Phân tích hoàn tất!"
            )

            # ================================================
            # HIỂN THỊ KẾT QUẢ PHÂN TÍCH
            # ================================================

            st.divider()

            st.header(
                "📊 Kết Quả Phân Tích — AcciVision"
            )

            events = result.get("events", [])
            has_accident = len(events) > 0

            # ------------------------------------------------
            # 4 Thẻ Chỉ Số Tổng Quan (Metric Cards)
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "📌 KẾT LUẬN VIDEO",
                    "🚨 CÓ TAI NẠN" if has_accident else "✅ AN TOÀN",
                    delta="Phát hiện sự cố" if has_accident else "Không có tai nạn",
                    delta_color="inverse" if has_accident else "normal",
                )

            with col2:
                st.metric(
                    "🚨 Số vụ tai nạn",
                    f"{len(events)} vụ",
                )

            with col3:
                st.metric(
                    "🎞️ Tổng số Frame",
                    f"{result['frames_processed']:,}",
                )

            with col4:
                st.metric(
                    "⚡ Tốc độ xử lý",
                    f"{result['average_fps']:.1f} FPS",
                )

            # ------------------------------------------------
            # Banner Trạng Thái (Status Banner)
            # ------------------------------------------------

            if has_accident:
                st.error(
                    f"🚨 **KẾT LUẬN: VIDEO CÓ XẢY RA TAI NẠN GIAO THÔNG** — "
                    f"Hệ thống xác định chính xác {len(events)} vụ va chạm trong video."
                )
            else:
                st.success(
                    "✅ **KẾT LUẬN: VIDEO KHÔNG CÓ TAI NẠN** — "
                    "Toàn bộ quá trình lưu thông bình thường, an toàn."
                )

            # ------------------------------------------------
            # Bảng Phân Loại Tai Nạn
            # ------------------------------------------------

            ACCIDENT_TYPE_MAP = {
                "rear_end_collision": "🚗💥🚙 Va chạm phía sau (Rear-end)",
                "head_on_collision": "🚙💥🚗 Va chạm đối đầu (Head-on)",
                "side_impact_collision": "🚗💥🏎️ Va chạm sườn/chữ T (Side Impact)",
                "multi_vehicle_pileup": "🚗💥🚙💥🚛 Va chạm liên hoàn (Pileup)",
                "single_vehicle_spin_or_rollover": "🔄🚗 Mất lái quay vòng/lật (Spin/Rollover)",
                "single_vehicle_loss_of_control": "⚠️🚗 Mất lái chệch quỹ đạo (Loss of Control)",
                "single_vehicle_sudden_stop": "🛑🚗 Dừng đột ngột/đâm vật cản (Sudden Stop)",
                "single_vehicle_incident": "⚠️ Sự cố đơn phương tiện",
                "vehicle_collision": "💥 Va chạm phương tiện",
                "unknown": "❓ Chưa xác định",
            }

            st.subheader("🚨 Chi Tiết Các Sự Kiện Tai Nạn")

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
                        ", ".join(
                            [
                                f"ID:{tid} ({v_types.get(tid, 'phương tiện')})"
                                for tid in tids
                            ]
                        )
                        if tids
                        else "Không xác định"
                    )

                    start_s = e.get(
                        "start_time_s",
                        e.get("start_frame", 0) / 30.0,
                    )
                    end_s = e.get(
                        "end_time_s",
                        e.get("end_frame", 0) / 30.0,
                    )
                    duration = e.get(
                        "duration_s",
                        end_s - start_s,
                    )

                    table_rows.append(
                        {
                            "Mã sự kiện": f"#{e['event_id']}",
                            "Loại tai nạn": type_display,
                            "Phương tiện liên quan": vehicles_str,
                            "Thời điểm bắt đầu": f"{start_s:.2f}s (Frame {e['start_frame']})",
                            "Thời điểm kết thúc": f"{end_s:.2f}s (Frame {e['end_frame']})",
                            "Thời lượng": f"{duration:.2f}s",
                            "Xác suất": f"{e.get('probability', 0.0) * 100:.1f}%",
                            "Trạng thái": "🔴 Đang diễn ra" if e.get("active") else "⚪ Đã kết thúc",
                        }
                    )

                st.dataframe(
                    __import__("pandas").DataFrame(table_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "📌 **Ghi chú:** Track ID là mã định danh phiên theo dõi (ByteTrack session ID), "
                    "không phải biển số xe thực tế."
                )

                # Chi tiết từng sự kiện (Expandable Cards)
                for e in events:
                    raw_type = e.get("accident_type", "unknown")
                    type_display = ACCIDENT_TYPE_MAP.get(
                        raw_type,
                        raw_type.replace("_", " ").title(),
                    )
                    with st.expander(
                        f"🔍 Chi tiết Sự kiện #{e['event_id']} — {type_display}"
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**⏱️ Thông tin thời gian:**")
                            st.write(
                                f"- Bắt đầu: **{e.get('start_time_s', 0):.2f}s** (Frame {e['start_frame']})"
                            )
                            st.write(
                                f"- Kết thúc: **{e.get('end_time_s', 0):.2f}s** (Frame {e['end_frame']})"
                            )
                            st.write(
                                f"- Thời lượng: **{e.get('duration_s', 0):.2f}s** "
                                f"({e.get('end_frame', 0) - e['start_frame'] + 1} frames)"
                            )
                            st.write(
                                f"- Xác suất tai nạn: **{e.get('probability', 0.0) * 100:.1f}%**"
                            )
                        with c2:
                            tids = e.get("track_ids", [])
                            v_types = e.get("vehicle_types", {})
                            st.markdown("**🚗 Phương tiện liên quan:**")
                            st.write(f"- Số phương tiện: **{len(tids)}**")
                            for tid in tids:
                                vtype = v_types.get(tid, "phương tiện")
                                st.write(
                                    f"- Phương tiện **ID {tid}** ({vtype})"
                                )
                            if e.get("description"):
                                st.markdown("**📝 Mô tả:**")
                                st.write(e["description"])
            else:
                st.info(
                    "✅ Không phát hiện sự kiện tai nạn nào trong toàn bộ video. "
                    "Giao thông hoạt động bình thường."
                )

            # ------------------------------------------------
            # Video kết quả đầu ra
            # ------------------------------------------------

            if save_output_video and Path(
                result["video_output"]
            ).exists():

                st.subheader(
                    "🎬 Video Kết Quả"
                )

                st.video(
                    result["video_output"]
                )

                # Nút tải xuống video
                with open(result["video_output"], "rb") as video_file:
                    st.download_button(
                        label="📥 Tải xuống Video kết quả",
                        data=video_file,
                        file_name=Path(result["video_output"]).name,
                        mime="video/mp4",
                    )

            # ------------------------------------------------
            # Tệp CSV đặc trưng
            # ------------------------------------------------

            csv_path = Path(
                result["features_csv"]
            )

            if csv_path.exists():

                st.subheader(
                    "📋 Dữ Liệu Đặc Trưng Trích Xuất"
                )

                df_features = __import__(
                    "pandas"
                ).read_csv(
                    csv_path
                ).head(100)

                st.dataframe(
                    df_features,
                    use_container_width=True,
                )

                st.caption(
                    f"Hiển thị 100 dòng đầu tiên. Tổng số dòng: {len(df_features)}"
                )

                with csv_path.open(
                    "rb"
                ) as file:

                    st.download_button(
                        label="📥 Tải xuống tệp CSV đặc trưng",
                        data=file,
                        file_name=csv_path.name,
                        mime="text/csv",
                    )

        except Exception as exc:

            st.error(
                "❌ Pipeline xử lý gặp lỗi. Vui lòng kiểm tra video đầu vào và thử lại."
            )

            st.exception(
                exc
            )

else:
    # Hiển thị hướng dẫn khi chưa có video
    st.info(
        "👆 Vui lòng tải lên video giao thông để bắt đầu phân tích. "
        "Hệ thống hỗ trợ các định dạng: MP4, AVI, MOV, MKV."
    )

    # Giới thiệu quy trình hệ thống
    st.subheader("📖 Quy Trình Hoạt Động Của AcciVision")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            "### 1️⃣ Phát hiện & Theo dõi\n"
            "- YOLOv8 phát hiện phương tiện\n"
            "- ByteTrack theo dõi liên tục\n"
            "- Quản lý quỹ đạo chuyển động"
        )

    with col_b:
        st.markdown(
            "### 2️⃣ Trích xuất đặc trưng\n"
            "- Chuyển đổi phối cảnh Bird's Eye View\n"
            "- Tính toán tốc độ, gia tốc, hướng di chuyển\n"
            "- Phân tích tương tác giữa các phương tiện"
        )

    with col_c:
        st.markdown(
            "### 3️⃣ Phân loại & Cảnh báo\n"
            "- Random Forest phân loại tai nạn\n"
            "- Quản lý vòng đời sự kiện\n"
            "- Xuất báo cáo chi tiết"
        )
