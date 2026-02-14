from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.inference import SpamDetector, ensemble_score
from src.train_behavior import FEATURE_COLUMNS

st.set_page_config(page_title="Spam Call Early Detection", page_icon="📞", layout="wide")
st.title("📞 Multi-Modal Spam Call Early Detection")
st.caption("Behavior + first 5-10 seconds of audio")

behavior_model_path = Path("models/behavior_model.joblib")
audio_model_path = Path("models/audio_model.joblib")

detector = SpamDetector(behavior_model_path, audio_model_path)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Behavioral Pattern Input")
    behavior_values = {
        "calls_per_day": st.number_input("Calls per day", min_value=0.0, value=25.0),
        "avg_duration": st.number_input("Average call duration (seconds)", min_value=0.0, value=90.0),
        "unique_numbers_called": st.number_input("Unique numbers called per day", min_value=0.0, value=15.0),
        "night_ratio": st.slider("Night call ratio", min_value=0.0, max_value=1.0, value=0.2),
        "short_call_ratio": st.slider("Short call ratio (< 20 sec)", min_value=0.0, max_value=1.0, value=0.3),
    }

with col2:
    st.subheader("2) Audio Input")
    uploaded_audio = st.file_uploader("Upload first 5-10 seconds audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

st.subheader("3) Inference")
if st.button("Predict Spam Probability"):
    b_score = detector.behavior_score(behavior_values) if detector.behavior_model else None
    a_score = None

    if uploaded_audio is not None and detector.audio_model:
        suffix = Path(uploaded_audio.name).suffix or ".wav"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_audio.read())
            tmp_path = Path(tmp.name)
        a_score = detector.audio_score(tmp_path)

    if b_score is None and a_score is None:
        st.error("No model available. Train behavior/audio model first.")
    else:
        final = ensemble_score(b_score, a_score)
        st.metric("Final spam probability", f"{100 * final:.2f}%")

        if b_score is not None:
            st.write(f"Behavior score: {b_score:.3f}")
        if a_score is not None:
            st.write(f"Audio score: {a_score:.3f}")

        if final >= 0.7:
            st.error("High risk: likely spam")
        elif final >= 0.4:
            st.warning("Medium risk: suspicious")
        else:
            st.success("Low risk")

st.divider()
st.markdown(
    """
### Optional realtime recording
To capture microphone audio in-app, integrate `streamlit-webrtc` and use a 5s buffer.
This scaffold keeps upload flow first for reliability, then you can add realtime capture.
"""
)

missing = [
    str(path) for path in [behavior_model_path, audio_model_path] if not path.exists()
]
if missing:
    st.info(
        "Missing trained model files: "
        + ", ".join(missing)
        + ". Run training scripts before full inference."
    )

st.code("python -m src.train_behavior\npython -m src.train_audio")
