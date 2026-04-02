"""
preprocessing.py — Frame extraction & normalization utilities
Handles resizing, normalization, and frame-skip logic for both
webcam and uploaded-video pipelines.
"""

import cv2
import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────────
FRAME_SIZE   = (224, 224)   # Target spatial size expected by MobileNetV2
FRAME_SKIP   = 5            # Process every N-th frame (performance trade-off)

# ImageNet mean/std used during MobileNetV2 pretraining
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Single-frame helpers ─────────────────────────────────────────────────────
def resize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize a BGR frame to FRAME_SIZE (H×W).
    Returns the resized frame still in BGR uint8.
    """
    return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR uint8 frame → RGB uint8."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def normalize_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize an RGB uint8 frame (H, W, 3) to float32 in [0, 1],
    then apply ImageNet mean/std standardization so values are
    compatible with MobileNetV2's pretrained weights.
    """
    frame_f32 = frame_rgb.astype(np.float32) / 255.0
    frame_f32 = (frame_f32 - _IMAGENET_MEAN) / _IMAGENET_STD
    return frame_f32  # (224, 224, 3), float32


def preprocess_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Full single-frame pipeline.

    Args:
        frame_bgr: Raw OpenCV frame (H, W, 3) in BGR uint8.

    Returns:
        display_frame : RGB uint8 resized to FRAME_SIZE  — for Streamlit display
        model_frame   : float32 normalized tensor         — for model input
    """
    resized       = resize_frame(frame_bgr)
    rgb           = bgr_to_rgb(resized)
    model_frame   = normalize_frame(rgb)
    display_frame = rgb                    # Already uint8 RGB
    return display_frame, model_frame


# ─── Video-file batch extractor ───────────────────────────────────────────────
def extract_frames_from_video(
    video_path: str,
    frame_skip: int = FRAME_SKIP,
    max_frames: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract frames from a video file, applying frame-skip.

    Args:
        video_path : Path to the video file.
        frame_skip : Keep 1 frame every `frame_skip` frames.
        max_frames : Hard cap on total frames extracted (memory guard).

    Returns:
        List of (display_frame, model_frame) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames  = []
    idx     = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame-skip: only process every N-th raw frame
        if idx % frame_skip == 0:
            display, model = preprocess_frame(frame)
            frames.append((display, model))

        idx += 1

    cap.release()
    return frames


# ─── Webcam frame generator ───────────────────────────────────────────────────
class WebcamStream:
    """
    Thin wrapper around cv2.VideoCapture for webcam input.
    Yields preprocessed frames on demand.
    """

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {camera_index}. "
                "Make sure a webcam is connected."
            )
        self._frame_count = 0

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """
        Read one frame from the webcam.

        Returns:
            (should_process, display_frame, model_frame)
            `should_process` is True only for every FRAME_SKIP-th call.
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        self._frame_count += 1
        should_process = (self._frame_count % FRAME_SKIP == 0)

        if should_process:
            display, model = preprocess_frame(frame)
            return True, display, model
        else:
            # Return the raw frame for display even if we skip inference
            rgb = bgr_to_rgb(resize_frame(frame))
            return False, rgb, None

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()
