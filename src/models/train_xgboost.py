
"""Train XGBoost baseline for hourly sepsis prediction."""

from pathlib import Path
import json

import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"patient_id", "hospital", "SepsisLabel"}


def load_split(name, features=None):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    if features is None:
        features = [
            c for c in df.columns if c not in EXCLUDE
        ]

    return (
        df[features],
        df["SepsisLabel"].astype(int),
        features,
    )


def evaluate(y, probs):
    return {
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
    }


def main():
    X_train, y_train, features = load_split("train")
    X_val, y_val, _ = load_split("val", features)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    probabilities = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, probabilities)

    joblib.dump(model, MODEL_DIR / "xgboost.joblib")

    with (
        MODEL_DIR / "xgboost_metrics.json"
    ).open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
