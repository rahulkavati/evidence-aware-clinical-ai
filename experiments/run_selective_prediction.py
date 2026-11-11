"""Evaluate G1 selective prediction on patient-disjoint validation subsets.

The threshold is selected on one half of ``data/processed/val.parquet`` and
evaluated on the other half. This script deliberately has no data-path option
so the held-out test split cannot be selected accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.selective_prediction import evaluate_selective, select_threshold
from src.uncertainty.calibrate_models import split_patient_ids
from src.evaluation.g1_headroom import predict_calibrated

VALIDATION_PATH = Path("data/processed/val.parquet")
MODEL_PATH = Path("outputs/g1_models/mask_aware_xgboost.joblib")
CALIBRATOR_PATH = Path("outputs/g1_models/mask_aware_sigmoid_calibrator.joblib")
RESULTS_DIR = Path("experiments/results/selective_prediction")
CHECKPOINTS = (6, 12, 24)


def split_validation_frame(
    frame: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return disjoint threshold-selection and evaluation patient cohorts."""
    required = {"patient_id", "SepsisLabel"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Validation frame is missing required columns: {sorted(missing)}")
    patient_labels = frame.groupby("patient_id")["SepsisLabel"].max()
    selection_ids, evaluation_ids = split_patient_ids(patient_labels, random_state)
    if set(selection_ids).intersection(evaluation_ids):
        raise AssertionError("Selection and evaluation patients must be disjoint")
    selection = frame[frame["patient_id"].isin(selection_ids)].copy()
    evaluation = frame[frame["patient_id"].isin(evaluation_ids)].copy()
    return selection, evaluation


def predict_frame(frame: pd.DataFrame, model: Any, calibrator: Any) -> np.ndarray:
    """Generate calibrated probabilities using the saved G1 model only."""
    features = list(getattr(model, "feature_names_in_", ()))
    if not features:
        raise ValueError("Saved G1 model does not expose feature names")
    forbidden = {"patient_id", "hospital", "SepsisLabel"}.intersection(features)
    if forbidden:
        raise ValueError(f"Saved G1 model contains forbidden predictors: {sorted(forbidden)}")
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Validation frame is missing model features: {missing}")
    return predict_calibrated(model, calibrator, frame[features])


def _scope_results(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    scopes: dict[str, np.ndarray] = {"pooled_all": np.ones(len(frame), dtype=bool)}
    if "hospital" in frame:
        for hospital in sorted(frame["hospital"].dropna().unique()):
            scopes[f"hospital_{hospital}"] = (frame["hospital"].to_numpy() == hospital)
    if "ICULOS" in frame:
        for checkpoint in CHECKPOINTS:
            scopes[f"hour_{checkpoint}"] = (frame["ICULOS"].to_numpy() == checkpoint)

    result: dict[str, Any] = {}
    for name, mask in scopes.items():
        if not mask.any():
            continue
        metrics = evaluate_selective(labels[mask], probabilities[mask], threshold)
        metrics.update(
            {
                "patients": int(frame.loc[mask, "patient_id"].nunique()),
                "positive_labels": int(labels[mask].sum()),
            }
        )
        result[name] = metrics
    return result


def run_selective_evaluation(
    frame: pd.DataFrame,
    model: Any,
    calibrator: Any,
    max_risk: float = 0.10,
    min_coverage: float = 0.50,
    random_state: int = 42,
) -> dict[str, Any]:
    """Select a threshold on validation patients and evaluate disjoint patients."""
    selection, evaluation = split_validation_frame(frame, random_state)
    selection_probabilities = predict_frame(selection, model, calibrator)
    evaluation_probabilities = predict_frame(evaluation, model, calibrator)
    threshold_result = select_threshold(
        selection["SepsisLabel"].astype(int).to_numpy(),
        selection_probabilities,
        max_risk=max_risk,
        min_coverage=min_coverage,
    )
    threshold = float(threshold_result["threshold"])
    return {
        "selection": {
            "patients": int(selection["patient_id"].nunique()),
            "rows": int(len(selection)),
            "positive_labels": int(selection["SepsisLabel"].sum()),
            "threshold_selection": threshold_result,
        },
        "evaluation": {
            "patients": int(evaluation["patient_id"].nunique()),
            "rows": int(len(evaluation)),
            "positive_labels": int(evaluation["SepsisLabel"].sum()),
            "scopes": _scope_results(evaluation, evaluation_probabilities, threshold),
        },
        "config": {
            "max_risk": max_risk,
            "min_coverage": min_coverage,
            "random_state": random_state,
            "validation_path": str(VALIDATION_PATH),
            "model_path": str(MODEL_PATH),
            "calibrator_path": str(CALIBRATOR_PATH),
            "held_out_test_accessed": False,
        },
    }


def render_report(result: dict[str, Any]) -> str:
    """Render a compact Markdown report from the JSON result."""
    selection = result["selection"]
    config = result["config"]
    lines = [
        "# Selective prediction evaluation",
        "",
        "Validation-only evaluation using patient-disjoint threshold-selection and evaluation cohorts.",
        "",
        f"- Maximum selection risk: `{config['max_risk']:.3f}`",
        f"- Minimum selection coverage: `{config['min_coverage']:.3f}`",
        f"- Selected confidence threshold: `{selection['threshold_selection']['threshold']:.6f}`",
        f"- Held-out test accessed: `{config['held_out_test_accessed']}`",
        "",
        "| Scope | Patients | Positive labels | Coverage | Abstention | Selective risk | Selective Brier | Retained AUROC | Retained AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, metrics in result["evaluation"]["scopes"].items():
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{value:.5f}"

        lines.append(
            f"| {scope} | {metrics['patients']} | {metrics['positive_labels']} | "
            f"{fmt(metrics['coverage'])} | {fmt(1 - metrics['coverage'])} | "
            f"{fmt(metrics['selective_risk'])} | {fmt(metrics['selective_brier_score'])} | "
            f"{fmt(metrics['retained_auroc'])} | {fmt(metrics['retained_auprc'])} |"
        )
    lines += [
        "",
        "This is an exploratory retrospective evaluation. The confidence threshold is not a clinically validated decision threshold.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-risk", type=float, default=0.10)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.max_risk <= 1.0:
        parser.error("--max-risk must be within [0, 1]")
    if not 0.0 <= args.min_coverage <= 1.0:
        parser.error("--min-coverage must be within [0, 1]")

    frame = pd.read_parquet(VALIDATION_PATH)
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    result = run_selective_evaluation(
        frame,
        model,
        calibrator,
        max_risk=args.max_risk,
        min_coverage=args.min_coverage,
        random_state=args.seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "selective_prediction.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "selective_prediction.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
