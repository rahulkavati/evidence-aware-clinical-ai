"""Diagnose VOI estimates across disjoint training-policy cohort sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_voi_acquisition import (
    GLOBAL_SEED,
    RESULTS_DIR,
    run_diagnostics,
)


def _render(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def write_report(payload: dict) -> None:
    lines = [
        "# VOI estimator diagnostics and training-size evaluation",
        "",
        "- No validation or held-out test data were used to fit these diagnostics.",
        "- One independent predictor-fit cohort was held fixed and kept disjoint from every policy-fitting cohort.",
        "- The estimator, `min_examples=30`, shrinkage=100, and cost weight=0 were not changed.",
        "- Benefits are squared error before reveal minus squared error after reveal.",
        "- Values are shown with sufficient precision to distinguish small nonzero values from true zero.",
        "",
        "## Why the 100-patient development policy acquired zero measurements",
        "",
        "The existing development command samples 100 training patients and then assigns 20% "
        "to the policy cohort, leaving only 20 policy-training patients. Feature-level counts "
        "can therefore remain below `min_examples=30`; suppressed estimates are set exactly "
        "to zero before policy selection. A nonzero estimate must also be positive after the "
        "configured simulated cost. This diagnostic separates those mechanisms.",
        "",
        "### Actual 100-patient development reference",
        "",
        f"- Development sample: {payload['development_reference']['development_sample_patients']:,} patients",
        f"- Independent predictor-fit cohort: {payload['development_reference']['predictor_fit_patients']:,} patients",
        f"- Policy-training cohort: {payload['development_reference']['policy_training_patients']:,} patients",
        f"- Training outcome records: {payload['development_reference']['outcome_records']:,}",
        f"- Patient-disjoint cohorts: {'yes' if payload['development_reference']['patient_disjoint'] else 'no'}",
        "",
        "In this exact development reference, all feature-level empirical benefits are "
        "zero before rounding, so min-example suppression is not the only mechanism. "
        "The independent predictor produced no squared-error change between the masked "
        "and one-feature-revealed states for these sparse policy-fitting patients.",
        "",
        "## Training-size sensitivity",
        "",
        "| Policy patients | Predictor-fit patients | Outcome records | Patient-disjoint |",
        "|---:|---:|---:|---|",
    ]
    for analysis in payload["analyses"]:
        lines.append(
            f"| {analysis['policy_training_patients']:,} | "
            f"{analysis['predictor_fit_patients']:,} | "
            f"{analysis['outcome_records']:,} | "
            f"{'yes' if analysis['patient_disjoint'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Feature-level diagnostics",
            "",
            "| Policy patients | Feature | Examples | Mean unshrunk benefit | SD | Median | Positive fraction | Shrunk benefit | Min-example suppression | Unshrunk sign | Shrinkage factor | Positive after cost | Unavailable |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|",
        ]
    )
    for analysis in payload["analyses"]:
        for row in analysis["features"]:
            lines.append(
                f"| {analysis['policy_training_patients']:,} | {row['feature']} | "
                f"{row['training_policy_examples']:,} | "
                f"{_render(row['mean_unshrunk_brier_benefit'])} | "
                f"{_render(row['std_unshrunk_brier_benefit'])} | "
                f"{_render(row['median_unshrunk_brier_benefit'])} | "
                f"{_render(row['positive_benefit_fraction'])} | "
                f"{_render(row['shrunk_estimated_benefit'])} | "
                f"{_render(row['min_examples_suppressed'])} | "
                f"{row['unshrunk_sign']} | "
                f"{_render(row['shrinkage_factor'])} | "
                f"{_render(row['positive_net_benefit_after_cost'])} | "
                f"{_render(row['candidate_unavailable'])} |"
            )
    lines.extend(
        [
            "",
            "## Actual development-reference feature diagnostics",
            "",
            "| Feature | Examples | Mean unshrunk benefit | Shrunk benefit | Min-example suppression | Unshrunk sign | Positive after cost |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in payload["development_reference"]["features"]:
        lines.append(
            f"| {row['feature']} | {row['training_policy_examples']:,} | "
            f"{_render(row['mean_unshrunk_brier_benefit'])} | "
            f"{_render(row['shrunk_estimated_benefit'])} | "
            f"{_render(row['min_examples_suppressed'])} | "
            f"{row['unshrunk_sign']} | "
            f"{_render(row['positive_net_benefit_after_cost'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `min_examples_suppressed=yes` means the estimator deliberately returns exactly zero, regardless of the raw empirical mean.",
            "- `unshrunk_sign=negative` means the empirical mean benefit is negative before shrinkage and rounding.",
            "- `positive after cost=no` is evaluated using the configured simulated cost; this is not an observed clinical cost.",
            "- Larger policy cohorts can reduce suppression and shrinkage, but any change is a training-size sensitivity result, not a validated improvement.",
            "- Historical feature availability remains an oracle retrospective assumption, and the independent predictor remains a policy-fitting model rather than the deployed validation predictor.",
        ]
    )
    (RESULTS_DIR / "diagnostics_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use small deterministic policy cohorts for a quick smoke test.",
    )
    args = parser.parse_args()
    sizes = (10, 20) if args.smoke else (20, 100, 500, 1000, 3000)
    payload = run_diagnostics(
        policy_sizes=sizes,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "diagnostics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(payload)
    print(json.dumps(payload["configuration"], indent=2))


if __name__ == "__main__":
    main()
