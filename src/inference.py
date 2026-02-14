from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import extract_mfcc_summary
from src.train_behavior import FEATURE_COLUMNS


class SpamDetector:
    def __init__(self, behavior_model_path: Path, audio_model_path: Path | None = None) -> None:
        self.behavior_model = self._load_model(behavior_model_path)
        self.audio_model = self._load_model(audio_model_path) if audio_model_path else None

    @staticmethod
    def _load_model(model_path: Path | None) -> Any:
        if model_path is None or not model_path.exists():
            return None
        return joblib.load(model_path)

    def behavior_score(self, behavior_features: dict[str, float]) -> float | None:
        if self.behavior_model is None:
            return None

        row = pd.DataFrame([[behavior_features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        return float(self.behavior_model.predict_proba(row)[0, 1])

    def audio_score(self, audio_path: Path) -> float | None:
        if self.audio_model is None:
            return None

        feat = extract_mfcc_summary(audio_path)
        return float(self.audio_model.predict_proba(np.expand_dims(feat, axis=0))[0, 1])


def ensemble_score(behavior_score: float | None, audio_score: float | None) -> float:
    if behavior_score is not None and audio_score is not None:
        return 0.6 * behavior_score + 0.4 * audio_score
    if behavior_score is not None:
        return behavior_score
    if audio_score is not None:
        return audio_score
    raise ValueError("No score available. Provide at least one model output.")
