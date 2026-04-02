"""
utils.py — Helper / utility functions for Campus Behaviour Detection System
Provides overlay rendering, alert formatting, confidence bar generation,
and logging helpers used by app.py.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# ─── Behaviour color map ──────────────────────────────────────────────────────
# BGR colours used for OpenCV overlays
BEHAVIOUR_COLORS: dict[str, tuple[int, int, int]] = {
    "Normal":    (50, 205, 50),    # Lime green
    "Running":   (0, 165, 255),    # Orange
    "Fighting":  (0, 0, 220),      # Red
    "Loitering": (0, 200, 255),    # Yellow
}

# Human-readable icons / prefixes for the Streamlit UI
BEHAVIOUR_ICONS: dict[str, str] = {
    "Normal":    "✅",
    "Running":   "🏃",
    "Fighting":  "🚨",
    "Loitering": "⚠️",
}

ABNORMAL_BEHAVIOURS = {"Running", "Fighting", "Loitering"}


# ─── Overlay renderer ─────────────────────────────────────────────────────────
def draw_overlay(
    frame_rgb: np.ndarray,
    label: str,
    confidence: float,
    is_abnormal: bool,
) -> np.ndarray:
    """
    Draw a semi-transparent label and confidence bar on a copy of the frame.

    Args:
        frame_rgb  : RGB uint8 numpy array (H, W, 3).
        label      : Predicted behaviour class name.
        confidence : Model confidence in [0, 1].
        is_abnormal: Whether to render a red alert border.

    Returns:
        Annotated RGB frame copy.
    """
    frame = frame_rgb.copy()
    h, w  = frame.shape[:2]
    color_bgr = BEHAVIOUR_COLORS.get(label, (200, 200, 200))
    # Convert BGR → RGB for display
    color_rgb = color_bgr[::-1]

    # ── Alert border for abnormal behaviour ──
    if is_abnormal:
        thickness = 6
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (220, 0, 0), thickness)

    # ── Label background pill ──
    text      = f"{BEHAVIOUR_ICONS.get(label, '')} {label}  {confidence * 100:.1f}%"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad = 8
    x1, y1 = 10, 10
    x2, y2 = x1 + tw + pad * 2, y1 + th + pad * 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Text
    cv2.putText(
        frame, text,
        (x1 + pad, y2 - pad),
        font, font_scale, color_rgb, thickness, cv2.LINE_AA,
    )

    # ── Confidence bar ──
    bar_x1  = 10
    bar_y1  = y2 + 6
    bar_w   = w - 20
    bar_h   = 8
    bar_x2  = bar_x1 + bar_w
    bar_y2  = bar_y1 + bar_h
    fill_x2 = bar_x1 + int(bar_w * confidence)

    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), color_rgb, -1)

    return frame


# ─── Detection log ────────────────────────────────────────────────────────────
@dataclass
class DetectionEvent:
    timestamp: float
    label: str
    confidence: float
    is_abnormal: bool

    @property
    def time_str(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class DetectionLogger:
    """
    Keeps a rolling window of the last N detection events for display.
    """

    def __init__(self, maxlen: int = 50):
        self._log: deque[DetectionEvent] = deque(maxlen=maxlen)

    def log(self, label: str, confidence: float, is_abnormal: bool):
        self._log.append(
            DetectionEvent(
                timestamp=time.time(),
                label=label,
                confidence=confidence,
                is_abnormal=is_abnormal,
            )
        )

    def recent(self, n: int = 10) -> list[DetectionEvent]:
        return list(self._log)[-n:]

    def clear(self):
        self._log.clear()

    @property
    def abnormal_count(self) -> int:
        return sum(1 for e in self._log if e.is_abnormal)

    @property
    def total_count(self) -> int:
        return len(self._log)


# ─── FPS tracker ──────────────────────────────────────────────────────────────
class FPSTracker:
    """Simple exponential moving average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha   = alpha
        self._fps     = 0.0
        self._last_ts = None

    def tick(self) -> float:
        now = time.time()
        if self._last_ts is not None:
            instant_fps  = 1.0 / max(now - self._last_ts, 1e-6)
            self._fps    = self._alpha * instant_fps + (1 - self._alpha) * self._fps
        self._last_ts = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ─── Misc helpers ─────────────────────────────────────────────────────────────
def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """Return a Unicode progress bar string for the sidebar."""
    filled = int(round(confidence * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"{bar} {confidence * 100:.1f}%"


def get_alert_html(label: str, confidence: float) -> str:
    """
    Return an HTML snippet for a Streamlit st.markdown alert box
    rendered with unsafe_allow_html=True.
    """
    icon = BEHAVIOUR_ICONS.get(label, "⚠️")
    return f"""
    <div style="
        background: linear-gradient(135deg, #ff1a1a22, #ff000011);
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        animation: pulse 1s ease-in-out infinite alternate;
    ">
        <div style="font-size: 1.4rem; font-weight: 700; color: #ff4444;">
            {icon} ABNORMAL BEHAVIOUR DETECTED
        </div>
        <div style="font-size: 1rem; color: #ffaaaa; margin-top: 4px;">
            <strong>{label}</strong> — confidence {confidence * 100:.1f}%
        </div>
    </div>
    <style>
        @keyframes pulse {{
            from {{ box-shadow: 0 0 8px #ff4444aa; }}
            to   {{ box-shadow: 0 0 20px #ff4444ff; }}
        }}
    </style>
    """
