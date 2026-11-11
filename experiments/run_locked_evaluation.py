"""Run the single locked held-out evaluation.

The test path is fixed and the threshold is loaded from the committed
validation artifact. This script intentionally has no retuning or data-path
arguments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.g1_masking import CANDIDATE_LABS, apply_hidden_set
from src.evaluation.abstention_acquisition import decision_metrics, one_step_fixed_order_frame
from src.evaluation.selective_prediction import confidence
from experiments.run_selective_prediction import predict_frame

TEST_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("outputs/g1_models/mask_aware_xgboost.joblib")
CALIBRATOR_PATH = Path("outputs/g1_models/mask_aware_sigmoid_calibrator.joblib")
THRESHOLD_ARTIFACT = Path("experiments/results/selective_prediction/abstention_acquisition/abstention_acquisition.json")
OUTPUT_DIR = Path("experiments/results/final")


def evaluate_locked_frame(
    frame: pd.DataFrame,
    model: Any,
    calibrator: Any,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate baseline abstention and fixed-order acquisition on one split."""
    hidden = apply_hidden_set(frame, CANDIDATE_LABS)
    baseline_probabilities = predict_frame(hidden, model, calibrator)
    acquired_frame, selected_indices = one_step_fixed_order_frame(frame)
    acquired_probabilities = predict_frame(acquired_frame, model, calibrator)
    baseline_retained = confidence(baseline_probabilities) >= threshold
    eligible = selected_indices >= 0
    abstained = ~baseline_retained
    rescued = abstained & eligible & (confidence(acquired_probabilities) >= threshold)
    retained = baseline_retained | rescued
    probabilities = np.where(baseline_retained, baseline_probabilities, acquired_probabilities)
    labels = frame["SepsisLabel"].astype(int).to_numpy()

    scopes = {"pooled_all": np.ones(len(frame), dtype=bool)}
    for hospital in sorted(frame["hospital"].dropna().unique()):
        scopes[f"hospital_{hospital}"] = frame["hospital"].to_numpy() == hospital
    for checkpoint in (6, 12, 24):
        scopes[f"hour_{checkpoint}"] = frame["ICULOS"].to_numpy() == checkpoint

    results: dict[str, Any] = {}
    for name, mask in scopes.items():
        if not mask.any():
            continue
        results[name] = {
            "patients": int(frame.loc[mask, "patient_id"].nunique()),
            "baseline_abstention": decision_metrics(labels[mask], baseline_probabilities[mask], baseline_retained[mask]),
            "acquisition_assisted": decision_metrics(labels[mask], probabilities[mask], retained[mask]),
            "abstention_rows": int((abstained & mask).sum()),
            "eligible_abstention_rows": int((abstained & eligible & mask).sum()),
            "rescued_rows": int((rescued & mask).sum()),
        }
    return {
        "patients": int(frame["patient_id"].nunique()),
        "rows": int(len(frame)),
        "threshold": threshold,
        "scopes": results,
    }


def render_report(result: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.5f}"
    lines = [
        "# Locked held-out evaluation", "",
        "Single final evaluation using the frozen validation-selected threshold and fixed one-step policy.", "",
        f"- Held-out test accessed: `{result['config']['held_out_test_accessed']}`",
        f"- Frozen threshold: `{result['evaluation']['threshold']:.10f}`",
        "- No threshold or policy retuning was performed on test data.", "",
        "| Scope | Baseline coverage | Assisted coverage | Baseline risk | Assisted risk | Rescued |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope, item in result["evaluation"]["scopes"].items():
        base, assisted = item["baseline_abstention"], item["acquisition_assisted"]
        lines.append(f"| {scope} | {fmt(base['coverage'])} | {fmt(assisted['coverage'])} | {fmt(base['risk'])} | {fmt(assisted['risk'])} | {item['rescued_rows']} |")
    lines += ["", "This retrospective evaluation does not establish prospective clinical utility."]
    return "\n".join(lines) + "\n"


def main() -> None:
    threshold_artifact = json.loads(THRESHOLD_ARTIFACT.read_text(encoding="utf-8"))
    threshold = float(threshold_artifact["selection"]["threshold_selection"]["threshold"])
    frame = pd.read_parquet(TEST_PATH)
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    result = {
        "config": {
            "test_path": str(TEST_PATH), "model_path": str(MODEL_PATH),
            "calibrator_path": str(CALIBRATOR_PATH),
            "threshold_artifact": str(THRESHOLD_ARTIFACT),
            "held_out_test_accessed": True,
            "test_evaluation_count": 1,
        },
        "evaluation": evaluate_locked_frame(frame, model, calibrator, threshold),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "locked_test_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "locked_test_evaluation.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
