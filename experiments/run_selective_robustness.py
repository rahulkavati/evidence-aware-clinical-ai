"""Run validation-only selective-prediction sensitivity and robustness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from experiments.run_selective_prediction import (
    CALIBRATOR_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    VALIDATION_PATH,
    _scope_results,
    predict_frame,
    split_validation_frame,
)
from src.evaluation.selective_prediction import patient_cluster_bootstrap_ci, select_threshold

DEFAULT_MAX_RISKS = (0.01, 0.05, 0.10, 0.20)
CHECKPOINTS = (6, 12, 24)


def robustness_scope_results(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    bootstrap_resamples: int,
    random_state: int,
) -> dict[str, Any]:
    """Add patient-cluster intervals to the standard selective metrics."""
    result = _scope_results(frame, probabilities, threshold)
    for name, metrics in result.items():
        if name == "pooled_all":
            mask = np.ones(len(frame), dtype=bool)
        elif name.startswith("hospital_"):
            mask = frame["hospital"].to_numpy() == name.removeprefix("hospital_")
        else:
            mask = frame["ICULOS"].to_numpy() == int(name.removeprefix("hour_"))
        metrics["bootstrap_95_ci"] = patient_cluster_bootstrap_ci(
            frame.loc[mask, "SepsisLabel"].astype(int).to_numpy(),
            probabilities[mask],
            frame.loc[mask, "patient_id"].to_numpy(),
            threshold,
            n_resamples=bootstrap_resamples,
            random_state=random_state,
        )
    return result


def run_robustness(
    frame: pd.DataFrame,
    model: Any,
    calibrator: Any,
    max_risks: Sequence[float] = DEFAULT_MAX_RISKS,
    random_state: int = 42,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    """Select and evaluate one threshold for each pre-specified risk target."""
    selection, evaluation = split_validation_frame(frame, random_state)
    selection_probabilities = predict_frame(selection, model, calibrator)
    evaluation_probabilities = predict_frame(evaluation, model, calibrator)
    labels = selection["SepsisLabel"].astype(int).to_numpy()
    targets: dict[str, Any] = {}
    for max_risk in max_risks:
        if not 0.0 <= max_risk <= 1.0:
            raise ValueError("All max_risks must be within [0, 1]")
        threshold_result = select_threshold(
            labels, selection_probabilities, max_risk=max_risk, min_coverage=0.0
        )
        threshold = float(threshold_result["threshold"])
        targets[f"{max_risk:g}"] = {
            "max_risk": max_risk,
            "selection": threshold_result,
            "evaluation_scopes": robustness_scope_results(
                evaluation, evaluation_probabilities, threshold,
                bootstrap_resamples, random_state,
            ),
        }
    return {
        "config": {
            "max_risks": list(max_risks),
            "minimum_selection_coverage": 0.0,
            "bootstrap_resamples": bootstrap_resamples,
            "random_state": random_state,
            "validation_path": str(VALIDATION_PATH),
            "model_path": str(MODEL_PATH),
            "calibrator_path": str(CALIBRATOR_PATH),
            "held_out_test_accessed": False,
        },
        "selection": {
            "patients": int(selection["patient_id"].nunique()),
            "rows": int(len(selection)),
            "positive_labels": int(selection["SepsisLabel"].sum()),
        },
        "evaluation": {
            "patients": int(evaluation["patient_id"].nunique()),
            "rows": int(len(evaluation)),
            "positive_labels": int(evaluation["SepsisLabel"].sum()),
        },
        "targets": targets,
    }


def render_report(result: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.5f}"

    lines = [
        "# Selective prediction robustness gate", "",
        "Validation-only sensitivity analysis with patient-disjoint threshold selection and evaluation.", "",
        f"- Held-out test accessed: `{result['config']['held_out_test_accessed']}`",
        f"- Patient-cluster bootstrap resamples: `{result['config']['bootstrap_resamples']}`", "",
        "## Pooled evaluation", "",
        "| Max risk | Threshold | Coverage | Coverage 95% CI | Selective risk | Risk 95% CI | Abstention |",
        "|---:|---:|---:|---|---:|---|---:|",
    ]
    for key, target in result["targets"].items():
        metrics = target["evaluation_scopes"]["pooled_all"]
        ci = metrics["bootstrap_95_ci"]
        lines.append(
            f"| {float(key):.2f} | {target['selection']['threshold']:.6f} | {fmt(metrics['coverage'])} | "
            f"[{fmt(ci['coverage']['lower_95'])}, {fmt(ci['coverage']['upper_95'])}] | "
            f"{fmt(metrics['selective_risk'])} | [{fmt(ci['selective_risk']['lower_95'])}, {fmt(ci['selective_risk']['upper_95'])}] | "
            f"{fmt(1 - metrics['coverage'])} |"
        )
    lines += ["", "## Subgroup evaluation", ""]
    for key, target in result["targets"].items():
        lines += [f"### Maximum risk {float(key):.2f}", "", "| Scope | Coverage | Risk | Risk 95% CI |", "|---|---:|---:|---|"]
        for scope, metrics in target["evaluation_scopes"].items():
            ci = metrics["bootstrap_95_ci"]["selective_risk"]
            lines.append(f"| {scope} | {fmt(metrics['coverage'])} | {fmt(metrics['selective_risk'])} | [{fmt(ci['lower_95'])}, {fmt(ci['upper_95'])}] |")
        lines.append("")
    lines.append("Thresholds and risk targets are exploratory and are not clinically validated.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-risks", type=float, nargs="+", default=DEFAULT_MAX_RISKS)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    frame = pd.read_parquet(VALIDATION_PATH)
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    result = run_robustness(frame, model, calibrator, args.max_risks, args.seed, args.bootstrap)
    output_dir = RESULTS_DIR / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selective_robustness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "selective_robustness.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
