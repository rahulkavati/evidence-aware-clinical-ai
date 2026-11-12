"""Train and validate a clean all-26-laboratory mask-aware baseline."""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data.build_temporal_features import LAB_FEATURES
from src.data.g1_masking import apply_hidden_set
from src.evaluation.selective_prediction import evaluate_selective, select_threshold
from src.uncertainty.calibrate_models import split_patient_ids
from src.uncertainty.platt import apply_platt, fit_platt


TRAIN_PATH = Path("data/processed/train.parquet")
VAL_PATH = Path("data/processed/val.parquet")
TEST_PATH = Path("data/processed/test.parquet")
COHORT_PATH = Path("outputs/g1_cohorts/predictor_fit.csv")
CALIBRATION_PATH = Path("outputs/g1_cohorts/calibration.csv")
OUTPUT_DIR = Path(os.environ.get("NEXT_PHASE_OUTPUT_DIR", "outputs/next_phase_all_labs"))
EXCLUDE = {"patient_id", "hospital", "SepsisLabel"}
ALL_LABS = tuple(LAB_FEATURES)


def mask_patient_consistently(frame: pd.DataFrame, features: list[str], seed: int = 42) -> pd.DataFrame:
    """Mask all-lab triplets using one random mask per patient."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    ids = frame["patient_id"].astype(str)
    patients = ids.unique()
    for lab in ALL_LABS:
        columns = [f"{lab}_last", f"{lab}_observed", f"{lab}_age"]
        if not set(columns).issubset(features):
            raise ValueError(f"Missing lab triplet columns for {lab}")
        masked_patients = patients[rng.random(len(patients)) < 0.5]
        rows = ids.isin(masked_patients)
        out.loc[rows, f"{lab}_last"] = np.nan
        out.loc[rows, f"{lab}_observed"] = 0.0
        out.loc[rows, f"{lab}_age"] = np.nan
    return out


def patient_subset(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    ids = set(pd.read_csv(manifest_path)["patient_id"].astype(str))
    return frame.loc[frame["patient_id"].astype(str).isin(ids)].copy()


def train_model() -> tuple[object, list[str]]:
    data = patient_subset(pd.read_parquet(TRAIN_PATH), COHORT_PATH)
    features = [column for column in data.columns if column not in EXCLUDE]
    masked = mask_patient_consistently(data, features)
    X = pd.concat([data[features], masked[features]], ignore_index=True)
    y = np.tile(data["SepsisLabel"].astype(int).to_numpy(), 2)
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=4,
    )
    model.fit(X, y, verbose=False)
    return model, features


def calibrated_predictions(model, calibrator, frame: pd.DataFrame, hidden: bool) -> np.ndarray:
    state = apply_hidden_set(frame, ALL_LABS) if hidden else frame
    raw = model.predict_proba(state[list(model.feature_names_in_)])[:, 1]
    return apply_platt(calibrator, raw)


def evaluate_state(model, calibrator, frame: pd.DataFrame, threshold: float) -> dict:
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    hidden = calibrated_predictions(model, calibrator, frame, hidden=True)
    visible = calibrated_predictions(model, calibrator, frame, hidden=False)
    return {
        "patients": int(frame["patient_id"].nunique()),
        "rows": int(len(frame)),
        "all_labs_hidden": evaluate_selective(labels, hidden, threshold),
        "all_labs_visible": evaluate_selective(labels, visible, threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-held-out",
        action="store_true",
        help="Run the single locked held-out evaluation after validation selection.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, features = train_model()
    calibration = patient_subset(pd.read_parquet(TRAIN_PATH), CALIBRATION_PATH)
    calibration_masked = mask_patient_consistently(calibration, features, seed=43)
    states = pd.concat([calibration[features], calibration_masked[features]], ignore_index=True)
    labels = np.tile(calibration["SepsisLabel"].astype(int).to_numpy(), 2)
    calibrator = fit_platt(model.predict_proba(states)[:, 1], labels)

    validation = pd.read_parquet(VAL_PATH)
    selection_ids, evaluation_ids = split_patient_ids(
        validation.groupby("patient_id")["SepsisLabel"].max(), 42
    )
    selection = validation[validation["patient_id"].isin(selection_ids)]
    evaluation = validation[validation["patient_id"].isin(evaluation_ids)]
    selection_probabilities = calibrated_predictions(model, calibrator, selection, hidden=True)
    threshold_result = select_threshold(
        selection["SepsisLabel"].astype(int).to_numpy(), selection_probabilities,
        max_risk=0.01, min_coverage=0.0,
    )
    threshold = float(threshold_result["threshold"])
    result = {
        "config": {
            "all_labs": list(ALL_LABS),
            "candidate_lab_count": len(ALL_LABS),
            "model": "mask-aware XGBoost",
            "calibration": "Platt sigmoid",
            "threshold_risk_target": 0.01,
            "held_out_test_accessed": bool(args.include_held_out),
        },
        "threshold": threshold,
        "validation": {
            "selection": threshold_result,
            "evaluation": evaluate_state(model, calibrator, evaluation, threshold),
        },
    }
    if args.include_held_out:
        result["held_out_test"] = evaluate_state(
            model, calibrator, pd.read_parquet(TEST_PATH), threshold
        )
    joblib.dump(model, OUTPUT_DIR / "all_lab_mask_aware_xgboost.joblib")
    joblib.dump(calibrator, OUTPUT_DIR / "all_lab_sigmoid_calibrator.joblib")
    (OUTPUT_DIR / "all_lab_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
