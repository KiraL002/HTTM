from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from main import (
    TrafficAccidentPipeline,
    load_config,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="HTTM - Traffic Accident Detection",
    page_icon="🚗",
    layout="wide",
)


st.title(
    "🚗 HTTM - Traffic Accident Detection"
)

st.caption(
    "YOLO + ByteTrack + Perspective Transform "
    "+ Kinematic Features + Random Forest"
)


# ============================================================
# LOAD CONFIG
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
        f"Cannot load config.yaml: {exc}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Configuration"
)

st.sidebar.write(
    "YOLO model:"
)

st.sidebar.code(
    config["yolo"]["model"]
)

st.sidebar.write(
    "Classifier:"
)

st.sidebar.code(
    config["classifier"]["model"]
)

st.sidebar.write(
    "Accident threshold:"
)

st.sidebar.write(
    f"{config['classifier']['threshold']:.2f}"
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload traffic video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)


if uploaded_file is not None:

    st.video(
        uploaded_file
    )

    if st.button(
        "▶ Start Detection",
        type="primary",
    ):

        # ----------------------------------------------------
        # Save uploaded video temporarily
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
        # Override input path
        # ----------------------------------------------------

        config["video"]["input"] = (
            str(temp_video_path)
        )

        # Streamlit không cần cửa sổ OpenCV
        config["video"]["display"] = False

        config["video"]["save_output"] = True

        config["output"]["save_video"] = True

        # ----------------------------------------------------
        # Output path
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

        # Đảm bảo extension mp4
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
        # Pipeline
        # ----------------------------------------------------

        pipeline = (
            TrafficAccidentPipeline(
                config=config,
                project_root=PROJECT_ROOT,
            )
        )

        progress_bar = st.progress(
            0
        )

        status_text = st.empty()

        try:

            with st.spinner(
                "Running traffic accident detection..."
            ):

                result = pipeline.run(
                    progress_callback=(
                        lambda value:
                        progress_bar.progress(
                            value
                        )
                    )
                )

            status_text.success(
                "Processing completed."
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            st.subheader(
                "Detection Result"
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Frames",
                    result[
                        "frames_processed"
                    ],
                )

            with col2:

                st.metric(
                    "Average FPS",
                    f"{result['average_fps']:.2f}",
                )

            with col3:

                st.metric(
                    "Accident Probability",
                    f"{result['accident_probability'] * 100:.1f}%",
                )

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if (
                result["status"]
                == "POSSIBLE ACCIDENT"
            ):

                st.error(
                    "🚨 POSSIBLE ACCIDENT"
                )

            else:

                st.success(
                    "✅ NORMAL"
                )

            # ------------------------------------------------
            # Output video
            # ------------------------------------------------

            if Path(
                result["video_output"]
            ).exists():

                st.subheader(
                    "Processed Video"
                )

                st.video(
                    result["video_output"]
                )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_path = Path(
                result["features_csv"]
            )

            if csv_path.exists():

                st.subheader(
                    "Feature CSV"
                )

                st.dataframe(
                    # Chỉ đọc vài dòng đầu
                    # để dashboard nhẹ.
                    __import__(
                        "pandas"
                    ).read_csv(
                        csv_path
                    ).head(100),
                    use_container_width=True,
                )

                with csv_path.open(
                    "rb"
                ) as file:

                    st.download_button(
                        label=(
                            "Download CSV"
                        ),
                        data=file,
                        file_name=(
                            csv_path.name
                        ),
                        mime="text/csv",
                    )

        except Exception as exc:

            st.error(
                "Pipeline failed."
            )

            st.exception(
                exc
            )