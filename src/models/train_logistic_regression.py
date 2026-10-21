
"""Train a Logistic Regression baseline for hourly sepsis prediction."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"patient_id", "hospital", "SepsisLabel"}


def load_split(name, feature_columns=None):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    if feature_columns is None:
        feature_columns = [
            col for col in df.columns if col not in EXCLUDE
        ]

    return (
        df[feature_columns],
        df["SepsisLabel"].astype(int),
        feature_columns,
    )


def evaluate(y, probabilities):
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def main():
    X_train, y_train, features = load_split("train")
    X_val, y_val, _ = load_split("val", features)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, val_probs)

    joblib.dump(model, MODEL_DIR / "logistic_regression.joblib")

    with (MODEL_DIR / "logistic_regression_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
