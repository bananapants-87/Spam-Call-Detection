from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import build_audio_feature_matrix

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _sine_wave(freq_hz: float, sample_rate: int, duration_seconds: float, amplitude: float = 0.35) -> np.ndarray:
    t = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _write_demo_clip(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, samplerate=16_000)


def bootstrap_demo_audio_dataset(
    spam_dir: Path,
    normal_dir: Path,
    samples_per_class: int = 60,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    spam_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    duration_seconds = 6.0
    sample_rate = 16_000

    for idx in range(samples_per_class):
        base_normal = _sine_wave(
            freq_hz=float(rng.uniform(120.0, 320.0)),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            amplitude=float(rng.uniform(0.15, 0.3)),
        )
        normal_noise = rng.normal(0.0, 0.01, size=base_normal.shape[0])
        normal_signal = (base_normal + normal_noise).astype(np.float32)
        _write_demo_clip(normal_dir / f"normal_{idx:03d}.wav", normal_signal)

        tone_a = _sine_wave(
            freq_hz=float(rng.uniform(650.0, 1200.0)),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            amplitude=float(rng.uniform(0.18, 0.35)),
        )
        tone_b = _sine_wave(
            freq_hz=float(rng.uniform(1350.0, 2400.0)),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            amplitude=float(rng.uniform(0.08, 0.2)),
        )
        pulse = np.sign(_sine_wave(5.0, sample_rate, duration_seconds, amplitude=1.0))
        spam_noise = rng.normal(0.0, 0.02, size=tone_a.shape[0])
        spam_signal = (tone_a + tone_b * pulse + spam_noise).astype(np.float32)
        _write_demo_clip(spam_dir / f"spam_{idx:03d}.wav", spam_signal)


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
    bootstrap_demo: bool = False,
) -> None:
    if not spam_dir.exists() or not normal_dir.exists():
        if bootstrap_demo:
            bootstrap_demo_audio_dataset(spam_dir=spam_dir, normal_dir=normal_dir)
        else:
            raise FileNotFoundError(
                "Expected audio folders: data/audio_spam and data/audio_normal"
            )

    spam_files = gather_audio_files(spam_dir)
    normal_files = gather_audio_files(normal_dir)

    if not spam_files or not normal_files:
        if bootstrap_demo:
            bootstrap_demo_audio_dataset(spam_dir=spam_dir, normal_dir=normal_dir)
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
    parser = argparse.ArgumentParser(description="Train audio spam classifier")
    parser.add_argument(
        "--bootstrap-demo",
        action="store_true",
        help="Generate synthetic demo audio data if dataset is missing/empty",
    )
    args = parser.parse_args()
    train_audio_model(bootstrap_demo=args.bootstrap_demo)
