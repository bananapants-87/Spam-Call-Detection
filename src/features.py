from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16_000
    duration_seconds: int = 10
    n_mfcc: int = 20


DEFAULT_AUDIO_CONFIG = AudioFeatureConfig()


def load_audio_segment(audio_path: str | Path, config: AudioFeatureConfig = DEFAULT_AUDIO_CONFIG) -> np.ndarray:
    """Load and pad/crop to fixed duration."""
    max_samples = config.sample_rate * config.duration_seconds
    signal, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)

    if signal.shape[0] < max_samples:
        signal = np.pad(signal, (0, max_samples - signal.shape[0]))
    else:
        signal = signal[:max_samples]

    return signal


def extract_mfcc_summary(audio_path: str | Path, config: AudioFeatureConfig = DEFAULT_AUDIO_CONFIG) -> np.ndarray:
    """Return fixed-size summary MFCC features (mean + std per coefficient)."""
    signal = load_audio_segment(audio_path, config)

    mfcc = librosa.feature.mfcc(y=signal, sr=config.sample_rate, n_mfcc=config.n_mfcc)
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)

    return np.concatenate([means, stds]).astype(np.float32)


def build_audio_feature_matrix(file_paths: list[Path], config: AudioFeatureConfig = DEFAULT_AUDIO_CONFIG) -> np.ndarray:
    features = [extract_mfcc_summary(path, config) for path in file_paths]
    return np.vstack(features)
