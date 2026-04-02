"""
app.py — Campus Abnormal Behaviour Detection System
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import time
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Local modules
from model import BehaviorClassifier, CLASS_NAMES, SEQUENCE_LENGTH
from preprocessing import preprocess_frame, extract_frames_from_video, FRAME_SKIP
from utils import (
    DetectionLogger,
    FPSTracker,
    draw_overlay,
    format_confidence_bar,
    get_alert_html,
    BEHAVIOUR_ICONS,
    BEHAVIOUR_COLORS,
    ABNORMAL_BEHAVIOURS,
)


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Campus Behaviour Detection",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* ── Global dark theme tweaks ── */
.stApp {
    background: #0a0c10;
    color: #e2e8f0;
}

/* ── Title ── */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-bottom: 2px solid #1e40af44;
    padding-bottom: 10px;
    margin-bottom: 6px;
}
.hero-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 20px;
}

/* ── Metric cards ── */
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Status badge ── */
.status-normal   { color: #4ade80; font-weight: 700; }
.status-abnormal { color: #f87171; font-weight: 700; }

/* ── Video frame border ── */
.video-container img {
    border-radius: 12px;
    border: 2px solid #1e293b;
}

/* ── Log table ── */
.log-entry-normal   { color: #4ade80; }
.log-entry-abnormal { color: #f87171; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #070910;
    border-right: 1px solid #1e293b;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "classifier":   None,   # BehaviorClassifier instance
        "logger":       DetectionLogger(maxlen=100),
        "fps_tracker":  FPSTracker(),
        "running":      False,
        "last_label":   "—",
        "last_conf":    0.0,
        "last_abnormal": False,
        "total_frames": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Control Panel")
    st.divider()

    input_type = st.radio(
        "**Input Source**",
        ["📁 Upload Video", "📷 Webcam (Live)"],
        index=0,
    )

    st.divider()
    st.markdown("**Model Settings**")
    weights_path = st.text_input(
        "Weights file (.h5)",
        value="",
        placeholder="Leave blank for demo mode",
        help="Path to a pretrained .h5 weight file. Leave empty to run in demo mode with random weights.",
    )

    st.divider()
    st.markdown("**Performance**")
    st.caption(f"Frame skip: every **{FRAME_SKIP}th** frame processed")
    st.caption(f"Sequence length: **{SEQUENCE_LENGTH}** frames")

    st.divider()
    st.markdown("**Behaviour Legend**")
    for cls in CLASS_NAMES:
        icon = BEHAVIOUR_ICONS[cls]
        tag  = "⚡ abnormal" if cls in ABNORMAL_BEHAVIOURS else "✓ normal"
        st.markdown(f"{icon} **{cls}** — *{tag}*")

    st.divider()
    if st.button("🗑️ Clear Detection Log", use_container_width=True):
        st.session_state.logger.clear()
        st.session_state.last_label   = "—"
        st.session_state.last_conf    = 0.0
        st.session_state.last_abnormal = False
        st.session_state.total_frames  = 0
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🎓 Campus Abnormal Behaviour Detection System</div>'
    '<div class="hero-sub">Real-time AI-powered surveillance · CNN + LSTM · MobileNetV2</div>',
    unsafe_allow_html=True,
)

# ─── KPI row ──────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi_ph = {
    "label": kpi1.empty(),
    "conf":  kpi2.empty(),
    "abnormal": kpi3.empty(),
    "frames": kpi4.empty(),
}


def refresh_kpis():
    logger: DetectionLogger = st.session_state.logger
    label  = st.session_state.last_label
    conf   = st.session_state.last_conf
    is_ab  = st.session_state.last_abnormal

    color = "#f87171" if is_ab else "#4ade80"
    icon  = BEHAVIOUR_ICONS.get(label, "🔍")

    kpi_ph["label"].markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}; font-size:1.6rem;">{icon} {label}</div>
        <div class="metric-label">Current Behaviour</div>
    </div>""", unsafe_allow_html=True)

    kpi_ph["conf"].markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{conf*100:.0f}%</div>
        <div class="metric-label">Confidence</div>
    </div>""", unsafe_allow_html=True)

    kpi_ph["abnormal"].markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#f87171;">{logger.abnormal_count}</div>
        <div class="metric-label">Abnormal Events</div>
    </div>""", unsafe_allow_html=True)

    kpi_ph["frames"].markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.total_frames}</div>
        <div class="metric-label">Frames Processed</div>
    </div>""", unsafe_allow_html=True)


refresh_kpis()

st.divider()

# ─── Main layout: video | alerts + log ───────────────────────────────────────
col_video, col_info = st.columns([3, 2], gap="large")

with col_video:
    st.markdown("#### 📹 Live Feed")
    video_placeholder = st.empty()
    # Default placeholder
    video_placeholder.info("👆 Configure input on the sidebar and click **Start** below.")

with col_info:
    st.markdown("#### 🔔 Detection Alerts")
    alert_placeholder = st.empty()
    st.markdown("#### 📋 Recent Events")
    log_placeholder   = st.empty()

st.divider()


# ─── Lazy model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN-LSTM model…")
def get_classifier(weights: str) -> BehaviorClassifier:
    return BehaviorClassifier(weights_path=weights if weights else None)


# ─── Inference callback ───────────────────────────────────────────────────────
def run_inference(display_frame: np.ndarray, model_frame: np.ndarray):
    """Push one frame through the classifier, update state, refresh UI."""
    clf: BehaviorClassifier = st.session_state.classifier

    result = clf.update(model_frame)
    st.session_state.total_frames += 1

    if result is None:
        # Buffer still filling
        video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
        return

    label, conf, is_abnormal = result
    st.session_state.last_label    = label
    st.session_state.last_conf     = conf
    st.session_state.last_abnormal = is_abnormal
    st.session_state.logger.log(label, conf, is_abnormal)

    # Annotate frame
    annotated = draw_overlay(display_frame, label, conf, is_abnormal)
    video_placeholder.image(annotated, channels="RGB", use_container_width=True)

    # Alert box
    if is_abnormal:
        alert_placeholder.markdown(get_alert_html(label, conf), unsafe_allow_html=True)
    else:
        alert_placeholder.success(f"✅ **Normal behaviour** detected — {conf*100:.1f}% confidence")

    # Recent log
    events = st.session_state.logger.recent(8)
    if events:
        rows = []
        for ev in reversed(events):
            css  = "log-entry-abnormal" if ev.is_abnormal else "log-entry-normal"
            icon = BEHAVIOUR_ICONS.get(ev.label, "")
            rows.append(
                f'<div class="{css}" style="padding:4px 0; border-bottom:1px solid #1e293b22;">'
                f'{ev.time_str} &nbsp; {icon} <strong>{ev.label}</strong> &nbsp; '
                f'<span style="color:#64748b;">{ev.confidence*100:.0f}%</span>'
                f'</div>'
            )
        log_placeholder.markdown(
            '<div style="font-family:monospace; font-size:0.82rem;">'
            + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    refresh_kpis()


# ─── Upload-video mode ────────────────────────────────────────────────────────
def run_uploaded_video(uploaded_file):
    # Save to temp file so OpenCV can read it
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info(f"📂 Processing **{uploaded_file.name}** — frame skip every {FRAME_SKIP} frames …")

    try:
        clf: BehaviorClassifier = st.session_state.classifier
        clf.reset()

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not open video file.")
            return

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0, text="Analysing video…")
        frame_idx    = 0
        fps_tracker  = FPSTracker()

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Frame skip
            if frame_idx % FRAME_SKIP != 0:
                continue

            display_frame, model_frame = preprocess_frame(frame_bgr)
            fps_tracker.tick()

            run_inference(display_frame, model_frame)

            # Update progress
            progress = min(frame_idx / max(total_video_frames, 1), 1.0)
            progress_bar.progress(progress, text=f"Frame {frame_idx}/{total_video_frames} — {fps_tracker.fps:.1f} FPS")

        cap.release()
        progress_bar.progress(1.0, text="✅ Analysis complete!")

    finally:
        os.unlink(tmp_path)


# ─── Webcam mode ──────────────────────────────────────────────────────────────
def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error(
            "🚫 Could not open webcam. "
            "Make sure a camera is connected and browser permissions are granted."
        )
        return

    clf: BehaviorClassifier = st.session_state.classifier
    clf.reset()

    stop_btn   = st.button("⏹️ Stop Webcam", key="stop_webcam", type="primary")
    fps_tracker = FPSTracker()
    frame_idx   = 0

    st.session_state.running = True

    while st.session_state.running and not stop_btn:
        ret, frame_bgr = cap.read()
        if not ret:
            st.warning("Lost webcam feed.")
            break

        frame_idx += 1

        # Always display, only infer on frame_skip boundary
        display_frame, model_frame = preprocess_frame(frame_bgr)

        if frame_idx % FRAME_SKIP == 0:
            fps_tracker.tick()
            run_inference(display_frame, model_frame)
        else:
            video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

        time.sleep(0.01)  # Yield to Streamlit event loop

    cap.release()
    st.session_state.running = False
    st.info("Webcam stopped.")


# ─── Start button & dispatch ──────────────────────────────────────────────────
is_upload = "Upload" in input_type

if is_upload:
    uploaded_file = st.file_uploader(
        "Drop a video file here",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )
    start_disabled = uploaded_file is None
else:
    uploaded_file  = None
    start_disabled = False

btn_col, _ = st.columns([1, 4])
with btn_col:
    start = st.button(
        "▶ Start Analysis",
        disabled=start_disabled,
        use_container_width=True,
        type="primary",
    )

if start:
    # Load / retrieve cached model
    st.session_state.classifier = get_classifier(weights_path.strip())

    if is_upload and uploaded_file:
        run_uploaded_video(uploaded_file)
    elif not is_upload:
        run_webcam()
