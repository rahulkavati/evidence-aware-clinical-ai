
"""Patient-disjoint probability calibration for the XGBoost baseline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("outputs/models")
RESULT_DIR = Path("experiments/results/calibration")

EXCLUDE = {"patient_id", "hospital", "SepsisLabel"}
RANDOM_STATE = 42


def expected_calibration_error(y, probabilities, bins=10):
    """Binary-event ECE with equal-width probability bins."""
    y = np.asarray(y)
    p = np.asarray(probabilities)

    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Labels and probabilities must have equal nonzero length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")

    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)

    ece = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def evaluate(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_bins": expected_calibration_error(y, p),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_DIR / "val.parquet")
    model = joblib.load(MODEL_DIR / "xgboost.joblib")

    # Use the original model's feature ordering.
    features = list(model.feature_names_in_)

    if EXCLUDE.intersection(features):
        raise ValueError("Identifiers or labels found in model features")

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    # Split by patient, not hourly observation.
    patient_labels = df.groupby("patient_id")["SepsisLabel"].max()

    calibration_ids, selection_ids = train_test_split(
        patient_labels.index.to_numpy(),
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=patient_labels.to_numpy(),
    )

    calibration_mask = df["patient_id"].isin(calibration_ids)
    selection_mask = df["patient_id"].isin(selection_ids)

    assert not set(calibration_ids).intersection(selection_ids)

    X_cal = df.loc[calibration_mask, features]
    y_cal = df.loc[calibration_mask, "SepsisLabel"].astype(int).to_numpy()

    X_sel = df.loc[selection_mask, features]
    y_sel = df.loc[selection_mask, "SepsisLabel"].astype(int).to_numpy()

    p_cal = model.predict_proba(X_cal)[:, 1]
    p_sel = model.predict_proba(X_sel)[:, 1]

    # Platt scaling: logistic mapping of raw model scores.
    eps = 1e-7
    logit_cal = np.log(
        np.clip(p_cal, eps, 1 - eps) /
        (1 - np.clip(p_cal, eps, 1 - eps))
    )
    logit_sel = np.log(
        np.clip(p_sel, eps, 1 - eps) /
        (1 - np.clip(p_sel, eps, 1 - eps))
    )

    sigmoid = LogisticRegression(max_iter=1000)
    sigmoid.fit(logit_cal.reshape(-1, 1), y_cal)
    p_sigmoid = sigmoid.predict_proba(logit_sel.reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    isotonic.fit(p_cal, y_cal)
    p_isotonic = isotonic.predict(p_sel)

    results = {
        "dataset": "PhysioNet 2019",
        "evaluation_split": "patient-disjoint validation selection subset",
        "calibration_patients": len(calibration_ids),
        "selection_patients": len(selection_ids),
        "calibration_hourly_rows": len(y_cal),
        "selection_hourly_rows": len(y_sel),
        "raw": evaluate(y_sel, p_sel),
        "sigmoid": evaluate(y_sel, p_sigmoid),
        "isotonic": evaluate(y_sel, p_isotonic),
    }

    joblib.dump(sigmoid, MODEL_DIR / "xgboost_sigmoid_calibrator.joblib")
    joblib.dump(isotonic, MODEL_DIR / "xgboost_isotonic_calibrator.joblib")

    output = RESULT_DIR / "calibration_comparison.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
