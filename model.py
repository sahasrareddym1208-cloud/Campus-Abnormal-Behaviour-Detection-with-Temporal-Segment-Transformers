"""
model.py — CNN + LSTM model for Campus Abnormal Behaviour Detection
Combines a pretrained MobileNetV2 (feature extractor) with an LSTM (temporal modeling).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


# ─── Constants ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 16       # Number of frames in one temporal window
FRAME_SIZE      = (224, 224)
NUM_CLASSES     = 4
CLASS_NAMES     = ["Normal", "Running", "Fighting", "Loitering"]

# Behaviours considered abnormal (will trigger alert)
ABNORMAL_CLASSES = {"Running", "Fighting", "Loitering"}


# ─── Feature Extractor (CNN) ──────────────────────────────────────────────────
def build_cnn_encoder(trainable: bool = False) -> Model:
    """
    Load MobileNetV2 pretrained on ImageNet.
    Remove the top classification head; return spatial feature maps.
    """
    base = MobileNetV2(
        input_shape=(*FRAME_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",          # Global average pool → (batch, 1280)
    )
    base.trainable = trainable  # Freeze during initial training
    return base


# ─── Temporal Model (CNN + LSTM) ─────────────────────────────────────────────
def build_cnn_lstm_model(
    sequence_length: int = SEQUENCE_LENGTH,
    num_classes: int = NUM_CLASSES,
    lstm_units: int = 256,
    dropout_rate: float = 0.5,
) -> Model:
    """
    Build the full CNN-LSTM pipeline:
        Input  : (batch, sequence_length, 224, 224, 3)
        CNN    : MobileNetV2 applied to every frame via TimeDistributed
        LSTM   : 2-layer LSTM for temporal reasoning
        Output : softmax over NUM_CLASSES
    """
    cnn_encoder = build_cnn_encoder(trainable=False)

    # ── Input ──
    inp = layers.Input(shape=(sequence_length, *FRAME_SIZE, 3), name="video_input")

    # ── CNN feature extraction (one frame at a time) ──
    x = layers.TimeDistributed(cnn_encoder, name="frame_features")(inp)
    # x shape: (batch, sequence_length, 1280)

    # ── Temporal modeling with stacked LSTM ──
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(lstm_units // 2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Classification head ──
    x = layers.Dense(128, activation="relu", name="fc")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inp, outputs=out, name="CampusBehaviorCNN_LSTM")
    return model


# ─── Inference helper ────────────────────────────────────────────────────────
class BehaviorClassifier:
    """
    Wraps the CNN-LSTM model and manages the rolling frame buffer.
    Keeps a FIFO buffer of `sequence_length` preprocessed frames.
    When the buffer is full, it runs inference and returns a prediction.
    """

    def __init__(self, weights_path: str | None = None):
        self.model = build_cnn_lstm_model()
        self.buffer: list[np.ndarray] = []

        if weights_path:
            try:
                self.model.load_weights(weights_path)
                print(f"[model] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[model] Could not load weights ({e}). Using random init.")
        else:
            print("[model] No weights path provided — using random weights (demo mode).")

        # Warm up the model so the first real call is fast
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((1, SEQUENCE_LENGTH, *FRAME_SIZE, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def update(self, frame: np.ndarray) -> tuple[str, float, bool] | None:
        """
        Add one preprocessed frame (224×224, float32 [0,1]) to the buffer.
        Returns (class_name, confidence, is_abnormal) once the buffer is full,
        then slides the window by 1.
        Returns None while the buffer is still filling.
        """
        self.buffer.append(frame)

        if len(self.buffer) < SEQUENCE_LENGTH:
            return None  # Not enough frames yet

        # Keep only the most recent `SEQUENCE_LENGTH` frames
        self.buffer = self.buffer[-SEQUENCE_LENGTH:]

        # Build (1, T, H, W, C) tensor
        seq = np.stack(self.buffer, axis=0)[np.newaxis]   # (1, 16, 224, 224, 3)
        probs = self.model.predict(seq, verbose=0)[0]      # (4,)

        class_idx   = int(np.argmax(probs))
        class_name  = CLASS_NAMES[class_idx]
        confidence  = float(probs[class_idx])
        is_abnormal = class_name in ABNORMAL_CLASSES

        return class_name, confidence, is_abnormal

    def reset(self):
        """Clear the frame buffer (call when switching video sources)."""
        self.buffer = []
