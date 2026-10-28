"""Paired patient-cluster bootstrap evaluation of acquisition policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from experiments.run_batch_acquisition import (
        ACQUISITION_BUDGETS,
        CALIBRATOR_FILE,
        CHECKPOINTS,
        GLOBAL_SEED,
        MODEL_FILE,
        RANDOM_REPLICATES,
        SPLIT_FILE,
        acquisition_order,
        checkpoint_label,
        historical_frame,
        historically_available_features,
        initial_hidden_features,
        limit_acquisitions,
        run_batch,
        sample_development_patients,
    )
except ModuleNotFoundError as error:
    if error.name != "experiments":
        raise
    from run_batch_acquisition import (
        ACQUISITION_BUDGETS,
        CALIBRATOR_FILE,
        CHECKPOINTS,
        GLOBAL_SEED,
        MODEL_FILE,
        RANDOM_REPLICATES,
        SPLIT_FILE,
        acquisition_order,
        checkpoint_label,
        historical_frame,
        historically_available_features,
        initial_hidden_features,
        limit_acquisitions,
        run_batch,
        sample_development_patients,
    )
from src.evaluation.paired_bootstrap import paired_bootstrap


SIGNIFICANCE_OUTPUT_DIR = Path("outputs/acquisition_significance")
AGGREGATE_DIR = Path("experiments/results/acquisition")
BOOTSTRAP_RESAMPLES = 1_000


def _record(
    patient_id: str,
    hospital: str,
    checkpoint: int,
    policy: str,
    budget: int,
    replicate: int | None,
    label: int,
    probability: float,
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "hospital": hospital,
        "checkpoint": checkpoint,
        "policy": policy,
        "budget": budget,
        "replicate": replicate,
        "label": label,
        "probability": probability,
    }


def collect_patient_predictions(
    patients: pd.DataFrame,
    model: Any,
    calibrator: Any,
    *,
    random_replicates: int = RANDOM_REPLICATES,
    global_seed: int = GLOBAL_SEED,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run the existing cached batch evaluator and materialize aligned records."""
    if "file_path" not in patients or "patient_id" not in patients:
        raise ValueError("Validation manifest must include patient_id and file_path")
    if any("test" in str(path).lower() for path in patients["file_path"]):
        raise ValueError("Test-data paths are not permitted")
    _, _, diagnostics = run_batch(
        patients,
        model,
        calibrator,
        random_replicates=random_replicates,
        global_seed=global_seed,
        workers=workers,
        return_diagnostics=True,
    )
    probabilities = diagnostics["probability_by_state"]
    records: list[dict[str, Any]] = []
    initial_hidden = initial_hidden_features()
    decisions = {
        (patient_index, checkpoint, policy, replicate): tuple(order)
        for patient_index, checkpoint, policy, replicate, order in diagnostics["decisions"]
    }
    for patient_index, manifest in enumerate(patients.itertuples(index=False)):
        patient = pd.read_csv(manifest.file_path, sep="|")
        for checkpoint in CHECKPOINTS:
            if not (patient["ICULOS"] == checkpoint).any():
                continue
            label = checkpoint_label(patient, checkpoint)
            available = historically_available_features(
                historical_frame(patient, checkpoint)
            )
            patient_id = str(manifest.patient_id)
            hospital = str(manifest.hospital)

            def probability(hidden: set[str] | tuple[str, ...]) -> float:
                key = (patient_index, checkpoint, tuple(sorted(hidden)))
                if key not in probabilities:
                    raise ValueError(f"Missing cached prediction for state {key}")
                return float(probabilities[key])

            for scope in ("All", hospital):
                records.append(
                    _record(
                        patient_id, scope, checkpoint, "no_acquisition", 0, None,
                        label, probability(set(initial_hidden)),
                    )
                )
            fixed_order = acquisition_order(available, "fixed_order")
            for budget in ACQUISITION_BUDGETS:
                selected = limit_acquisitions(fixed_order, budget)
                for scope in ("All", hospital):
                    records.append(
                        _record(
                            patient_id, scope, checkpoint, "fixed_order", budget, None,
                            label, probability(set(initial_hidden) - set(selected)),
                        )
                    )
            for replicate in range(random_replicates):
                order = decisions[(patient_index, checkpoint, "random", replicate)]
                for budget in ACQUISITION_BUDGETS:
                    selected = limit_acquisitions(order, budget)
                    for scope in ("All", hospital):
                        records.append(
                            _record(
                                patient_id, scope, checkpoint, "random", budget,
                                replicate, label,
                                probability(set(initial_hidden) - set(selected)),
                            )
                        )
    return records


def _comparison(
    records: list[dict[str, Any]],
    checkpoint: int,
    hospital: str,
    policy: str,
    budget: int,
    replicate: int | None,
) -> dict[str, Any]:
    scoped = [
        row for row in records
        if row["checkpoint"] == checkpoint
        and row["hospital"] == hospital
        and row["policy"] in ("no_acquisition", policy)
        and row["budget"] == (0 if row["policy"] == "no_acquisition" else budget)
        and row["replicate"] == (None if row["policy"] == "no_acquisition" else replicate)
    ]
    no_rows = [row for row in scoped if row["policy"] == "no_acquisition"]
    acquisition_rows = [row for row in scoped if row["policy"] == policy]
    return {
        "checkpoint": checkpoint,
        "hospital": hospital,
        "policy": policy,
        "budget": budget,
        "replicate": replicate,
        **paired_bootstrap(
            no_rows,
            acquisition_rows,
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=GLOBAL_SEED,
        ),
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for hospital in ("All", "A", "B"):
            for policy in ("fixed_order", "random"):
                for budget in ACQUISITION_BUDGETS:
                    replicates = (None,) if policy == "fixed_order" else range(RANDOM_REPLICATES)
                    for replicate in replicates:
                        comparisons.append(
                            _comparison(
                                records, checkpoint, hospital, policy, budget, replicate
                            )
                        )
    random_summary = []
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        if row["policy"] == "random":
            grouped[(row["checkpoint"], row["hospital"], row["budget"])].append(row)
    for key, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "checkpoint": key[0], "hospital": key[1], "budget": key[2],
            "replicates": len(rows),
        }
        for metric in ("auroc", "auprc", "brier_improvement"):
            values = [row["observed"][metric] for row in rows if row["observed"][metric] is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        random_summary.append(summary)
    return {"comparisons": comparisons, "random_replicate_summary": random_summary}


def _write_report(result: dict[str, Any]) -> None:
    lines = [
        "# Paired statistical evaluation of acquisition policies",
        "",
        "- Dataset: validation patients only; no test data were accessed.",
        f"- Bootstrap: {BOOTSTRAP_RESAMPLES:,} patient-cluster resamples, seed {GLOBAL_SEED}.",
        "- Differences are acquisition minus no acquisition for AUROC/AUPRC, and no acquisition minus acquisition for Brier improvement.",
        "- Confidence intervals are 95% percentile intervals; undefined AUROC/AUPRC resamples are skipped and counted.",
        "- The five random-policy replicates use the same patients and are summarized as policy-sequence variation, not independent cohorts.",
        "",
        "## Paired differences",
        "",
        "| Hour | Hospital | Policy | Budget | Replicate | Patients | Positive labels | AUROC difference (95% CI) | AUPRC difference (95% CI) | Brier improvement (95% CI) | Valid AUROC/AUPRC/Brier |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in result["comparisons"]:
        def display(metric: str) -> str:
            observed = row["observed"][metric]
            interval = row["confidence_intervals_95"][metric]
            return "—" if observed is None else f"{observed:.5f} ({interval[0]:.5f}, {interval[1]:.5f})"
        valid = row["valid_bootstrap_replicates"]
        lines.append(
            f"| {row['checkpoint']} | {row['hospital']} | {row['policy']} | {row['budget']} | "
            f"{'—' if row['replicate'] is None else row['replicate']} | {row['patient_count']:,} | "
            f"{row['positive_label_count']:,} | {display('auroc')} | {display('auprc')} | "
            f"{display('brier_improvement')} | {valid['auroc']}/{valid['auprc']}/{valid['brier_improvement']} |"
        )
    lines.extend([
        "",
        "## Random-policy replicate variation",
        "",
        "Values are mean ± sample SD of observed paired differences across the five random sequences.",
        "",
        "| Hour | Hospital | Budget | AUROC difference | AUPRC difference | Brier improvement |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in result["random_replicate_summary"]:
        def mean_sd(metric: str) -> str:
            value = row[f"{metric}_mean"]
            return "—" if value is None else f"{value:.5f} ± {row[f'{metric}_std']:.5f}"
        lines.append(
            f"| {row['checkpoint']} | {row['hospital']} | {row['budget']} | "
            f"{mean_sd('auroc')} | {mean_sd('auprc')} | {mean_sd('brier_improvement')} |"
        )
    lines.extend([
        "",
        "## Methodological limitations",
        "",
        "- This is retrospective historical information reveal, not a prospective clinical ordering study.",
        "- Candidate-feature availability is an oracle assumption based on finite historical observations.",
        "- Artificial masking may produce out-of-distribution feature patterns.",
        "- Bootstrap intervals quantify patient-sampling uncertainty and do not establish clinical usefulness or causality.",
        "- Random-policy replicate variation reflects acquisition-sequence variation on the same patients.",
        "",
    ])
    (AGGREGATE_DIR / "significance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if "test" in str(SPLIT_FILE).lower():
        raise ValueError("Test split cannot be used")
    patients = pd.read_csv(SPLIT_FILE)
    if args.development:
        patients = sample_development_patients(patients, args.sample_size, args.seed)
    model = joblib.load(MODEL_FILE)
    calibrator = joblib.load(CALIBRATOR_FILE)
    records = collect_patient_predictions(
        patients, model, calibrator, global_seed=args.seed, workers=args.workers
    )
    SIGNIFICANCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (SIGNIFICANCE_OUTPUT_DIR / "patient_predictions.json").write_text(
        json.dumps(records, allow_nan=False), encoding="utf-8"
    )
    result = evaluate(records)
    result["configuration"] = {
        "dataset": "PhysioNet 2019 validation patients",
        "development_mode": bool(args.development),
        "patients_processed": int(len(patients)),
        "checkpoints": list(CHECKPOINTS),
        "random_replicates": RANDOM_REPLICATES,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": GLOBAL_SEED,
    }
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    (AGGREGATE_DIR / "significance_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_report(result)
    print(json.dumps(result["configuration"], indent=2))


if __name__ == "__main__":
    main()
