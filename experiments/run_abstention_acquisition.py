"""Evaluate one-step acquisition for validation cases that would abstain."""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from experiments.run_selective_prediction import (
    CALIBRATOR_PATH, MODEL_PATH, RESULTS_DIR, VALIDATION_PATH,
    predict_frame, split_validation_frame,
)
from src.data.g1_masking import CANDIDATE_LABS, apply_hidden_set
from src.evaluation.abstention_acquisition import decision_metrics, one_step_fixed_order_frame
from src.evaluation.selective_prediction import confidence, select_threshold

OUTPUT_DIR = RESULTS_DIR / "abstention_acquisition"


def _scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "pooled_all":
        return np.ones(len(frame), dtype=bool)
    if scope.startswith("hospital_"):
        return frame["hospital"].to_numpy() == scope.removeprefix("hospital_")
    return frame["ICULOS"].to_numpy() == int(scope.removeprefix("hour_"))


def run_abstention_acquisition(
    frame: pd.DataFrame,
    model: Any,
    calibrator: Any,
    max_risk: float = 0.01,
    random_state: int = 42,
) -> dict[str, Any]:
    """Select a strict threshold, then allow one fixed-order reveal for abstentions."""
    selection, evaluation = split_validation_frame(frame, random_state)
    selection_hidden = apply_hidden_set(selection, CANDIDATE_LABS)
    evaluation_hidden = apply_hidden_set(evaluation, CANDIDATE_LABS)
    selection_probabilities = predict_frame(selection_hidden, model, calibrator)
    baseline_probabilities = predict_frame(evaluation_hidden, model, calibrator)
    acquisition_frame, selected_indices = one_step_fixed_order_frame(evaluation)
    acquired_probabilities = predict_frame(acquisition_frame, model, calibrator)

    threshold_result = select_threshold(
        selection["SepsisLabel"].astype(int).to_numpy(), selection_probabilities,
        max_risk=max_risk, min_coverage=0.0,
    )
    threshold = float(threshold_result["threshold"])
    baseline_retained = confidence(baseline_probabilities) >= threshold
    eligible = selected_indices >= 0
    abstained = ~baseline_retained
    acquired_retained = abstained & eligible & (confidence(acquired_probabilities) >= threshold)
    final_retained = baseline_retained | acquired_retained
    final_probabilities = np.where(baseline_retained, baseline_probabilities, acquired_probabilities)
    labels = evaluation["SepsisLabel"].astype(int).to_numpy()

    scopes = {"pooled_all": np.ones(len(evaluation), dtype=bool)}
    for hospital in sorted(evaluation["hospital"].dropna().unique()):
        scopes[f"hospital_{hospital}"] = evaluation["hospital"].to_numpy() == hospital
    for checkpoint in (6, 12, 24):
        scopes[f"hour_{checkpoint}"] = evaluation["ICULOS"].to_numpy() == checkpoint

    scope_results: dict[str, Any] = {}
    for name, mask in scopes.items():
        if not mask.any():
            continue
        scope_results[name] = {
            "patients": int(evaluation.loc[mask, "patient_id"].nunique()),
            "baseline_abstention": decision_metrics(labels[mask], baseline_probabilities[mask], baseline_retained[mask]),
            "acquisition_assisted": decision_metrics(labels[mask], final_probabilities[mask], final_retained[mask]),
            "abstention_rows": int((abstained & mask).sum()),
            "eligible_abstention_rows": int((abstained & eligible & mask).sum()),
            "rescued_rows": int((acquired_retained & mask).sum()),
        }

    return {
        "config": {
            "max_risk": max_risk, "random_state": random_state,
            "policy": "one-step fixed-order acquisition among baseline abstentions",
            "priority": list(CANDIDATE_LABS), "validation_path": str(VALIDATION_PATH),
            "model_path": str(MODEL_PATH), "calibrator_path": str(CALIBRATOR_PATH),
            "held_out_test_accessed": False,
        },
        "selection": {"patients": int(selection["patient_id"].nunique()), "rows": int(len(selection)), "threshold_selection": threshold_result},
        "evaluation": {
            "patients": int(evaluation["patient_id"].nunique()), "rows": int(len(evaluation)),
            "baseline_abstention_rows": int(abstained.sum()),
            "eligible_abstention_rows": int((abstained & eligible).sum()),
            "rescued_rows": int(acquired_retained.sum()), "scopes": scope_results,
        },
    }


def render_report(result: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.5f}"
    lines = [
        "# Acquisition after abstention", "",
        "Validation-only one-step retrospective reveal for cases that fail the strict selective-prediction threshold.", "",
        f"- Selected threshold: `{result['selection']['threshold_selection']['threshold']:.6f}`",
        f"- Held-out test accessed: `{result['config']['held_out_test_accessed']}`",
        "- Policy: reveal the first historically eligible lab in fixed order, then re-apply the same threshold.", "",
        "| Scope | Baseline coverage | Assisted coverage | Baseline risk | Assisted risk | Eligible abstentions | Rescued |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, item in result["evaluation"]["scopes"].items():
        base, assisted = item["baseline_abstention"], item["acquisition_assisted"]
        lines.append(
            f"| {scope} | {fmt(base['coverage'])} | {fmt(assisted['coverage'])} | {fmt(base['risk'])} | {fmt(assisted['risk'])} | {item['eligible_abstention_rows']} | {item['rescued_rows']} |"
        )
    lines += ["", "This is a retrospective historical-information reveal, not a clinical test-ordering study. The fixed-order policy does not inspect hidden measurement values."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-risk", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_parquet(VALIDATION_PATH)
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    result = run_abstention_acquisition(frame, model, calibrator, args.max_risk, args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "abstention_acquisition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "abstention_acquisition.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
