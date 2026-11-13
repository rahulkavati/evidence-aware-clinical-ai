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
from src.data.g1_masking import apply_hidden_matrix, apply_hidden_set, sample_hidden_matrix
from src.evaluation.selective_prediction import evaluate_selective, select_threshold
from src.uncertainty.calibrate_models import expected_calibration_error, split_patient_ids
from src.uncertainty.platt import apply_platt, fit_platt


TRAIN_PATH = Path("data/processed/train.parquet")
VAL_PATH = Path("data/processed/val.parquet")
TEST_PATH = Path("data/processed/test.parquet")
COHORT_PATH = Path("outputs/g1_cohorts/predictor_fit.csv")
CALIBRATION_PATH = Path("outputs/g1_cohorts/calibration.csv")
OUTPUT_DIR = Path(os.environ.get("NEXT_PHASE_OUTPUT_DIR", "outputs/next_phase_all_labs"))
EXCLUDE = {"patient_id", "hospital", "SepsisLabel"}
ALL_LABS = tuple(LAB_FEATURES)
STATE_BANDS = {
    "all_visible": (0, 0),
    "lightly_hidden": (1, 8),
    "moderately_hidden": (9, 17),
    "mostly_hidden": (18, 25),
    "all_hidden": (26, 26),
}


def mask_patient_consistently(
    frame: pd.DataFrame,
    features: list[str],
    seed: int = 42,
    hidden_count: int | None = None,
) -> pd.DataFrame:
    """Mask all-lab triplets with one uniform-size mask per patient."""
    rng = np.random.default_rng(seed)
    ids = frame["patient_id"].astype(str)
    required = {
        f"{lab}_{suffix}"
        for lab in ALL_LABS
        for suffix in ("last", "observed", "age")
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Missing lab feature columns: {sorted(missing)}")
    patients = ids.drop_duplicates().to_numpy()
    if hidden_count is None:
        patient_masks = sample_hidden_matrix(len(patients), len(ALL_LABS), rng)
    else:
        if not 0 <= hidden_count <= len(ALL_LABS):
            raise ValueError("hidden_count must be between 0 and the number of labs")
        ranks = rng.random((len(patients), len(ALL_LABS))).argsort(axis=1)
        patient_masks = np.zeros((len(patients), len(ALL_LABS)), dtype=bool)
        patient_masks[np.arange(len(patients))[:, None], ranks[:, :hidden_count]] = True
    patient_index = {patient: index for index, patient in enumerate(patients)}
    row_masks = np.vstack([patient_masks[patient_index[patient]] for patient in ids])
    return apply_hidden_matrix(frame, ALL_LABS, row_masks)


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
    selected = calibrator["all_hidden" if hidden else "all_visible"] if isinstance(calibrator, dict) else calibrator
    return apply_platt(selected, raw)


def fit_state_calibrators(model, calibration: pd.DataFrame, features: list[str]) -> dict[str, object]:
    """Fit one Platt calibrator per hidden-lab state band."""
    labels = calibration["SepsisLabel"].astype(int).to_numpy()
    calibrators = {}
    for index, (name, (low, high)) in enumerate(STATE_BANDS.items()):
        hidden_count = low if low == high else (low + high) // 2
        state = mask_patient_consistently(
            calibration, features, seed=4300 + index, hidden_count=hidden_count
        )
        raw = model.predict_proba(state[features])[:, 1]
        calibrators[name] = fit_platt(raw, labels)
    return calibrators


def state_calibrated_predictions(model, calibrators, frame, hidden_count, seed=9001):
    """Predict a representative hidden-count state with its state calibrator."""
    state_name = next(
        name for name, (low, high) in STATE_BANDS.items() if low <= hidden_count <= high
    )
    features = list(model.feature_names_in_)
    state = mask_patient_consistently(frame, features, seed=seed, hidden_count=hidden_count)
    raw = model.predict_proba(state[features])[:, 1]
    return apply_platt(calibrators[state_name], raw)


def evaluate_state(model, calibrator, frame: pd.DataFrame, threshold: float) -> dict:
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    hidden = calibrated_predictions(model, calibrator, frame, hidden=True)
    visible = calibrated_predictions(model, calibrator, frame, hidden=False)
    result = {
        "patients": int(frame["patient_id"].nunique()),
        "rows": int(len(frame)),
        "all_labs_hidden": evaluate_selective(labels, hidden, threshold),
        "all_labs_visible": evaluate_selective(labels, visible, threshold),
    }
    if isinstance(calibrator, dict):
        state_metrics = {}
        for index, (name, (low, high)) in enumerate(STATE_BANDS.items()):
            hidden_count = low if low == high else (low + high) // 2
            probabilities = state_calibrated_predictions(
                model, calibrator, frame, hidden_count, seed=9100 + index
            )
            state_metrics[name] = {
                "representative_hidden_count": hidden_count,
                "brier_score": float(np.mean((probabilities - labels) ** 2)),
                "expected_calibration_error": float(
                    expected_calibration_error(labels, probabilities)
                ),
                "selective": evaluate_selective(labels, probabilities, threshold),
            }
        result["state_calibration"] = state_metrics
    return result


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
    calibrator = fit_state_calibrators(model, calibration, features)

    validation = pd.read_parquet(VAL_PATH)
    selection_ids, evaluation_ids = split_patient_ids(
        validation.groupby("patient_id")["SepsisLabel"].max(), 42
    )
    selection = validation[validation["patient_id"].isin(selection_ids)]
    evaluation = validation[validation["patient_id"].isin(evaluation_ids)]
    selection_probabilities = state_calibrated_predictions(
        model, calibrator, selection, hidden_count=26
    )
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
            "calibration": "state-specific Platt sigmoid",
            "training_masking": "patient-consistent uniform hidden-lab count (0..26)",
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
