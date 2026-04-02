# 🎓 Campus Abnormal Behaviour Detection System

Real-time AI surveillance using **CNN + LSTM** (MobileNetV2 + stacked LSTM) built with Streamlit and TensorFlow.

---

## 📁 Project Structure

```
campus_behavior/
├── app.py              # Streamlit UI — entry point
├── model.py            # CNN + LSTM architecture (MobileNetV2 + LSTM)
├── preprocessing.py    # Frame extraction, resize, normalization
├── utils.py            # Overlay drawing, FPS tracker, detection logger
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🚀 Usage

| Mode | Steps |
|------|-------|
| **Upload Video** | Select "📁 Upload Video" → drop a file (mp4/avi/mov/mkv) → click **▶ Start Analysis** |
| **Webcam (Live)** | Select "📷 Webcam (Live)" → click **▶ Start Analysis** → allow camera permissions |

---

## 🧠 Model Architecture

```
Input: (batch, 16, 224, 224, 3)   ← 16-frame temporal window
         │
TimeDistributed(MobileNetV2)       ← pretrained ImageNet weights, frozen
         │  (batch, 16, 1280)
LSTM(256, return_sequences=True)   ← temporal pattern learning
         │
LSTM(128, return_sequences=False)  ← sequence summary
         │
Dense(128, relu)
         │
Dense(4, softmax)                  ← Normal / Running / Fighting / Loitering
```

---

## 🏷️ Behaviour Classes

| Label | Type | Alert |
|-------|------|-------|
| ✅ Normal | Normal | — |
| 🏃 Running | **Abnormal** | ⚠️ |
| 🚨 Fighting | **Abnormal** | 🚨 |
| ⚠️ Loitering | **Abnormal** | ⚠️ |

---

## 🔧 Training Your Own Weights

```python
from model import build_cnn_lstm_model, SEQUENCE_LENGTH, FRAME_SIZE

model = build_cnn_lstm_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# X_train shape: (N, 16, 224, 224, 3) — float32 normalized frames
# y_train shape: (N,)                 — integer class labels 0-3
model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.2)
model.save_weights("campus_behavior_weights.h5")
```

Then set "Weights file" in the sidebar to `campus_behavior_weights.h5`.

---

## ⚡ Performance Notes

- **Frame skip**: Every 5th raw frame is sent to the model (configurable via `FRAME_SKIP` in `preprocessing.py`).
- **Sequence window**: 16 consecutive (post-skip) frames form one prediction window.
- **GPU**: TensorFlow will use CUDA automatically if a compatible GPU and `tensorflow-gpu` are present.
- **CPU-only**: Inference is still real-time for 224×224 input due to MobileNetV2's lightweight design.

---

## 📦 Key Libraries

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `tensorflow` | CNN + LSTM model |
| `opencv-python-headless` | Video capture & frame processing |
| `numpy` | Numerical ops |
