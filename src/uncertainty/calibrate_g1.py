
"""G1.4: independent calibration for mask-aware XGBoost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.train_mask_aware_xgboost import (
    CANDIDATE_LABS,
    EXCLUDE,
    mask_training_matrix,
)
from src.uncertainty.platt import fit_platt, apply_platt

DATA_PATH = Path("data/processed/train.parquet")
COHORT_DIR = Path("outputs/g1_cohorts")
MODEL_DIR = Path("outputs/g1_models")


def load_calibration_data(
    data: pd.DataFrame,
    calibration: pd.DataFrame,
    predictor_fit: pd.DataFrame,
    policy_fit: pd.DataFrame,
) -> pd.DataFrame:
    """Enforce patient separation and exact calibration membership."""

    def ids(frame):
        if "patient_id" not in frame.columns:
            raise ValueError("Missing patient_id")
        if frame["patient_id"].isna().any():
            raise ValueError("Null patient IDs")
        return set(frame["patient_id"].astype(str))

    for name, frame in (
        ("calibration", calibration),
        ("predictor_fit", predictor_fit),
        ("policy_fit", policy_fit),
    ):
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {name}")

    cal_ids = ids(calibration)
    fit_ids = ids(predictor_fit)
    policy_ids = ids(policy_fit)

    if cal_ids & fit_ids or cal_ids & policy_ids or fit_ids & policy_ids:
        raise ValueError("G1 research cohorts overlap")

    selected = data.loc[
        data["patient_id"].astype(str).isin(cal_ids)
    ].copy()

    if ids(selected) != cal_ids:
        raise ValueError("Calibration patients missing from train data")

    if selected.duplicated(["patient_id", "ICULOS"]).any():
        raise ValueError("Duplicate patient-hour observations")

    return selected


def build_calibration_states(
    data: pd.DataFrame,
    features: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One original and one patient-consistently masked state per row."""

    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    X = data.loc[:, features].reset_index(drop=True)
    y = data["SepsisLabel"].astype(int).to_numpy()
    ids = data["patient_id"].reset_index(drop=True)

    masked = mask_training_matrix(
        X,
        ids,
        seed=seed,
        masked_probability=0.5,
    )

    combined = pd.concat([X, masked], ignore_index=True)
    labels = np.concatenate([y, y])

    return combined, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = joblib.load(
        MODEL_DIR / "mask_aware_xgboost.joblib"
    )
    features = list(model.feature_names_in_)

    if set(features) & EXCLUDE:
        raise ValueError("Forbidden predictor features")

    expected = {
        f"{lab}_{suffix}"
        for lab in CANDIDATE_LABS
        for suffix in ("last", "observed", "age")
    }
    if not expected.issubset(features):
        raise ValueError("Missing candidate laboratory triplets")

    data = pd.read_parquet(DATA_PATH)

    manifests = {
        name: pd.read_csv(COHORT_DIR / f"{name}.csv")
        for name in ("predictor_fit", "calibration", "policy_fit")
    }

    calibration_data = load_calibration_data(
        data,
        manifests["calibration"],
        manifests["predictor_fit"],
        manifests["policy_fit"],
    )

    X, y = build_calibration_states(
        calibration_data,
        features,
        seed=args.seed,
    )

    raw = model.predict_proba(X)[:, 1]

    calibrator = fit_platt(raw, y)
    calibrated = apply_platt(calibrator, raw)

    if not np.isfinite(calibrated).all():
        raise ValueError("Non-finite calibrated probabilities")

    if ((calibrated < 0) | (calibrated > 1)).any():
        raise ValueError("Calibrated probabilities outside [0, 1]")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        calibrator,
        MODEL_DIR / "mask_aware_sigmoid_calibrator.joblib",
    )

    metadata = {
        "seed": args.seed,
        "predictor_model": "mask_aware_xgboost.joblib",
        "calibration_patients": int(
            calibration_data["patient_id"].nunique()
        ),
        "original_hourly_rows": int(len(calibration_data)),
        "calibration_state_rows": int(len(X)),
        "calibration_state_mixture": (
            "Original state plus one patient-consistent "
            "Bernoulli-0.5 masked state"
        ),
        "calibration_method": "Platt sigmoid on raw probability logits",
        "positive_calibration_rows": int(y.sum()),
        "predictor_features": len(features),
        "validation_or_test_loaded": False,
        "evaluation_independent": False,
        "note": (
            "Calibration was fitted on these rows. "
            "Do not report in-sample calibration metrics "
            "as independent performance estimates."
        ),
    }

    (MODEL_DIR / "g1_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
