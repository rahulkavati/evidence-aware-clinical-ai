"""Validation-only ablations for temporal and unit shortcut features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"patient_id", "hospital", "SepsisLabel"}
SHORTCUT_GROUPS = {
    "none": (),
    "time": ("ICULOS", "HospAdmTime"),
    "unit": ("Unit1", "Unit2"),
    "time_and_unit": ("ICULOS", "HospAdmTime", "Unit1", "Unit2"),
}


def feature_variants(columns: list[str]) -> dict[str, list[str]]:
    """Return feature lists with only the named shortcut group removed."""
    base = [column for column in columns if column not in EXCLUDE]
    return {
        name: [column for column in base if column not in removed]
        for name, removed in SHORTCUT_GROUPS.items()
    }


def fit_and_evaluate(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], n_estimators: int, n_jobs: int) -> dict[str, float | int]:
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=n_jobs,
    )
    y_train = train["SepsisLabel"].astype(int)
    y_validation = validation["SepsisLabel"].astype(int)
    model.fit(train[features], y_train, eval_set=[(validation[features], y_validation)], verbose=False)
    probabilities = model.predict_proba(validation[features])[:, 1]
    return {
        "feature_count": len(features),
        "validation_patients": int(validation["patient_id"].nunique()),
        "validation_rows": int(len(validation)),
        "positive_rate": float(y_validation.mean()),
        "auroc": float(roc_auc_score(y_validation, probabilities)),
        "auprc": float(average_precision_score(y_validation, probabilities)),
        "brier_score": float(brier_score_loss(y_validation, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.parquet")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/val.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/next_phase/shortcut_ablations.json")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    variants = feature_variants(train.columns.tolist())
    result = {
        "protocol": "validation-only patient split; no held-out test data loaded",
        "model": "XGBoost with fixed seed 42",
        "n_estimators": args.n_estimators,
        "ablations": {
            name: fit_and_evaluate(train, validation, features, args.n_estimators, args.n_jobs)
            for name, features in variants.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
