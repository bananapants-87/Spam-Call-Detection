# Spam-Call-Detection

A demo-ready **multi-modal early spam detection system** that combines:

1. **Behavioral call pattern modeling** (works before call acceptance).
2. **Early audio classification** using first 5–10 seconds.
3. **Ensemble scoring** to output final spam probability.

---

## Architecture

```text
Incoming Call
   ├── Behavioral model (telemetry features)
   ├── Audio model (MFCC on first 5-10 sec)
   └── Ensemble: final = 0.6 * behavior + 0.4 * audio
```

---

## Repository layout

```text
app.py                    # Streamlit UI
src/train_behavior.py     # synthetic telecom behavior model training
src/train_audio.py        # audio model training from dataset folders
src/features.py           # MFCC and audio preprocessing helpers
src/inference.py          # inference + ensemble logic
requirements.txt
```

---

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the behavioral model

```bash
python -m src.train_behavior
```

Outputs:
- `models/behavior_model.joblib`
- `models/behavior_synthetic_data.csv`

### 3) Prepare audio datasets

Create folders and place labeled clips:

```text
data/audio_spam/
data/audio_normal/
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

Then train:

```bash
python -m src.train_audio
```

Output:
- `models/audio_model.joblib`

### 4) Run the app

```bash
streamlit run app.py
```

---

## How to execute your 7-day plan with this repo

### Day 1
- Run `python -m src.train_behavior`
- Verify classification report + ROC-AUC in terminal.

### Day 2
- Populate `data/audio_spam/` and `data/audio_normal/`.

### Day 3
- Run `python -m src.train_audio`.

### Day 4
- Re-run both trainings after dataset balancing/cleanup.
- Track accuracy/precision/recall/ROC-AUC from script output.

### Day 5
- Launch Streamlit UI (`streamlit run app.py`) and test behavior + file upload flow.

### Day 6
- Integrate realtime audio via `streamlit-webrtc` frame buffering.

### Day 7
- Perform end-to-end dry-run and document limits (demo-level assumptions).

---

## Notes and realism

- This is a **demo architecture**, not telecom-grade.
- Production systems require large-scale signaling data, reputation graphs, and continual feedback loops.
- Start simple, collect evidence (metrics), then iterate weights and feature set.

