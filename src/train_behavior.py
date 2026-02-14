from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "calls_per_day",
    "avg_duration",
    "unique_numbers_called",
    "night_ratio",
    "short_call_ratio",
]


def generate_behavior_data(size: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    normal = pd.DataFrame(
        {
            "calls_per_day": rng.normal(12, 5, size=size // 2).clip(1, 60),
            "avg_duration": rng.normal(180, 70, size=size // 2).clip(20, 900),
            "unique_numbers_called": rng.normal(7, 3, size=size // 2).clip(1, 35),
            "night_ratio": rng.beta(1.5, 6, size=size // 2),
            "short_call_ratio": rng.beta(1.5, 5, size=size // 2),
            "label": 0,
        }
    )

    spam = pd.DataFrame(
        {
            "calls_per_day": rng.normal(85, 20, size=size // 2).clip(8, 300),
            "avg_duration": rng.normal(40, 20, size=size // 2).clip(3, 220),
            "unique_numbers_called": rng.normal(55, 15, size=size // 2).clip(6, 250),
            "night_ratio": rng.beta(3.8, 2, size=size // 2),
            "short_call_ratio": rng.beta(5, 1.6, size=size // 2),
            "label": 1,
        }
    )

    return pd.concat([normal, spam], ignore_index=True).sample(frac=1.0, random_state=seed)


def train_behavior_model(output_dir: Path = Path("models")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_behavior_data()
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Behavior model report:\n", classification_report(y_test, y_pred))
    print(f"Behavior ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, output_dir / "behavior_model.joblib")
    df.to_csv(output_dir / "behavior_synthetic_data.csv", index=False)


if __name__ == "__main__":
    train_behavior_model()
