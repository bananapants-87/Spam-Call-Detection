from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import build_audio_feature_matrix

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def gather_audio_files(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def train_audio_model(
    spam_dir: Path = Path("data/audio_spam"),
    normal_dir: Path = Path("data/audio_normal"),
    output_dir: Path = Path("models"),
) -> None:
    if not spam_dir.exists() or not normal_dir.exists():
        raise FileNotFoundError(
            "Expected audio folders: data/audio_spam and data/audio_normal"
        )

    spam_files = gather_audio_files(spam_dir)
    normal_files = gather_audio_files(normal_dir)

    if not spam_files or not normal_files:
        raise ValueError("No audio files found in one or both class directories.")

    output_dir.mkdir(parents=True, exist_ok=True)

    X_spam = build_audio_feature_matrix(spam_files)
    X_normal = build_audio_feature_matrix(normal_files)

    X = np.vstack([X_normal, X_spam])
    y = np.concatenate([
        np.zeros(len(X_normal), dtype=np.int8),
        np.ones(len(X_spam), dtype=np.int8),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Audio model report:\n", classification_report(y_test, y_pred))
    print(f"Audio ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, output_dir / "audio_model.joblib")


if __name__ == "__main__":
    train_audio_model()
