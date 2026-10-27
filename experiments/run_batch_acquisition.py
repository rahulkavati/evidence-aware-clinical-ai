"""Compare retrospective acquisition baselines on validation patients."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from src.acquisition.baseline_policies import fixed_order_policy, random_policy
from src.data.build_temporal_features import (
    build_checkpoint_feature_vector,
    build_temporal_features,
    mask_checkpoint_feature_vector,
)
from src.simulation.acquisition_simulator import AcquisitionSimulator

SPLIT_FILE = Path("outputs/splits/val.csv")
MODEL_FILE = Path("outputs/models/xgboost.joblib")
CALIBRATOR_FILE = Path("outputs/models/xgboost_sigmoid_calibrator.joblib")
RESULTS_DIR = Path("experiments/results/acquisition")
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS = (6, 12, 24)
CANDIDATE_LABS = (
    "Lactate",
    "WBC",
    "Creatinine",
    "Platelets",
    "BUN",
    "Glucose",
)
ACQUISITION_BUDGETS = (1, 2, 3)
RANDOM_REPLICATES = 5
GLOBAL_SEED = 42
DEFAULT_DEVELOPMENT_SAMPLE = 100
EXCLUDED_FEATURES = {"patient_id", "hospital", "SepsisLabel"}
FORBIDDEN_OUTPUT_KEYS = {
    "patient_id",
    "patient_ids",
    "file_path",
    "clinical_record",
    "clinical_records",
    "predictions",
    "patient_predictions",
    "raw_values",
    "measurements",
}
METRIC_NAMES = ("auroc", "auprc", "brier_score")


def initial_hidden_features() -> tuple[str, ...]:
    """Return the same candidate mask for every policy at a checkpoint."""
    return CANDIDATE_LABS


def historical_frame(patient: pd.DataFrame, checkpoint: int) -> pd.DataFrame:
    """Restrict a patient record to observations available at the checkpoint."""
    history = patient.loc[patient["ICULOS"] <= checkpoint].copy()
    if not (history["ICULOS"] == checkpoint).any():
        raise ValueError(f"Patient has no row exactly at ICU hour {checkpoint}")
    for feature in CANDIDATE_LABS:
        values = pd.to_numeric(history[feature], errors="coerce")
        history[feature] = values.replace([np.inf, -np.inf], np.nan)
    return history


def historically_available_features(history: pd.DataFrame) -> tuple[str, ...]:
    """Return candidate names with finite recorded values in bounded history."""
    return tuple(
        sorted(
            feature
            for feature in CANDIDATE_LABS
            if np.isfinite(
                pd.to_numeric(history[feature], errors="coerce").to_numpy(dtype=float)
            ).any()
        )
    )


def checkpoint_label(patient: pd.DataFrame, checkpoint: int) -> int:
    """Return the official SepsisLabel recorded at the exact checkpoint."""
    rows = patient.loc[patient["ICULOS"] == checkpoint, "SepsisLabel"]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row at ICU hour {checkpoint}")
    label = int(rows.iloc[0])
    if label not in (0, 1):
        raise ValueError(f"Invalid SepsisLabel at ICU hour {checkpoint}: {label}")
    return label


def stable_random_seed(
    global_seed: int,
    hospital: str,
    patient_identifier: str,
    checkpoint: int,
    replicate: int,
) -> int:
    """Create a process-independent seed for one random-policy trajectory."""
    seed_material = json.dumps(
        [
            int(global_seed),
            str(hospital),
            str(patient_identifier),
            int(checkpoint),
            int(replicate),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")


def acquisition_order(
    eligible_features: Iterable[str],
    policy: str,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    """Create an acquisition trajectory without inspecting hidden values."""
    eligible = sorted(set(eligible_features))
    if policy == "random":
        if not eligible:
            return ()
        generator = rng if rng is not None else random.Random(GLOBAL_SEED)
        order: list[str] = []
        remaining = eligible.copy()
        while remaining:
            selected = random_policy(remaining, rng=generator)
            if selected is None:
                break
            order.append(selected)
            remaining.remove(selected)
        return tuple(order)
    if policy == "fixed_order":
        order = []
        remaining = eligible.copy()
        while remaining:
            selected = fixed_order_policy(remaining)
            if selected is None:
                break
            order.append(selected)
            remaining.remove(selected)
        return tuple(order)
    raise ValueError(f"Unsupported acquisition policy: {policy}")


def limit_acquisitions(order: Iterable[str], budget: int) -> tuple[str, ...]:
    if budget < 0:
        raise ValueError("Acquisition budget cannot be negative")
    return tuple(order)[:budget]


def validate_aggregate_output(
    payload: dict[str, Any],
    forbidden_identifiers: Iterable[str] = (),
    forbidden_raw_values: Iterable[str] = (),
) -> None:
    """Reject patient-level fields and known sensitive values before publication."""
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    forbidden_keys = {key.lower() for key in FORBIDDEN_OUTPUT_KEYS}

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    raise ValueError(f"Patient-level output field is forbidden: {key}")
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(payload)
    for identifier in forbidden_identifiers:
        if identifier and identifier in serialized:
            raise ValueError("Patient identifier found in aggregate output")
    for raw_value in forbidden_raw_values:
        if raw_value and raw_value in serialized:
            raise ValueError("Raw clinical value found in aggregate output")


def sample_development_patients(
    patients: pd.DataFrame,
    sample_size: int,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("Development sample size must be at least 1")
    if sample_size >= len(patients):
        return patients.copy()
    sample = patients.sample(n=sample_size, random_state=seed)
    return sample.reset_index(drop=True)


def _calibrated_probability(model: Any, calibrator: Any, features: pd.DataFrame) -> float:
    model_features = list(model.feature_names_in_)
    missing = [name for name in model_features if name not in features.columns]
    if missing:
        raise ValueError(f"Model features missing from temporal features: {missing}")
    raw = float(model.predict_proba(features.loc[:, model_features])[:, 1][0])
    if not np.isfinite(raw) or raw < 0.0 or raw > 1.0:
        raise ValueError("XGBoost produced an invalid probability")
    clipped = np.clip(raw, 1e-7, 1.0 - 1e-7)
    logit = np.log(clipped / (1.0 - clipped))
    probability = float(calibrator.predict_proba(np.array([[logit]]))[0, 1])
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("Sigmoid calibrator produced an invalid probability")
    return probability


def _probability_from_visible_history(
    history: pd.DataFrame,
    checkpoint: int,
    model: Any,
    calibrator: Any,
    hidden_features: Iterable[str],
) -> float:
    visible = history.copy()
    for feature in hidden_features:
        visible[feature] = np.nan
    temporal = build_temporal_features(visible)
    current = temporal.loc[temporal["ICULOS"] == checkpoint]
    if len(current) != 1:
        raise ValueError("Expected exactly one engineered feature row at checkpoint")
    return _calibrated_probability(model, calibrator, current)


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Cannot compute metrics for an empty or misaligned cohort")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Evaluation probabilities must be finite and within [0, 1]")
    return {
        "eligible_patients": int(len(y)),
        "positive_labels_at_checkpoint": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "brier_score": float(brier_score_loss(y, p)),
    }


def _record_group(
    records: dict[tuple[Any, ...], dict[str, list[Any]]],
    key: tuple[Any, ...],
    label: int,
    probability: float,
    acquired_count: int,
    available_count: int,
) -> None:
    group = records[key]
    group["labels"].append(label)
    group["probabilities"].append(probability)
    group["acquired_counts"].append(acquired_count)
    group["available_counts"].append(available_count)
    group["acquisition_patients"].append(int(acquired_count > 0))


def _make_result_rows(
    records: dict[tuple[Any, ...], dict[str, list[Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for key, group in sorted(records.items(), key=lambda item: str(item[0])):
        checkpoint, hospital, policy, budget, replicate = key
        metrics = _metrics(group["labels"], group["probabilities"])
        acquired_counts = np.asarray(group["acquired_counts"], dtype=int)
        available_counts = np.asarray(group["available_counts"], dtype=int)
        rows.append(
            {
                "checkpoint": checkpoint,
                "hospital": hospital,
                "policy": policy,
                "budget": budget,
                "replicate": replicate,
                **metrics,
                "patients_with_available_candidate_features": int(
                    (available_counts > 0).sum()
                ),
                "mean_available_candidate_features": float(available_counts.mean()),
                "acquired_feature_count": int(acquired_counts.sum()),
                "mean_acquisition_count": float(acquired_counts.mean()),
                "acquisition_coverage": float(
                    np.mean(np.asarray(group["acquisition_patients"], dtype=float))
                ),
            }
        )
    policy_order = {
        "no_acquisition": 0,
        "no_masking_reference": 1,
        "fixed_order": 2,
        "random": 3,
    }
    hospital_order = {"All": 0, "A": 1, "B": 2}
    rows.sort(
        key=lambda row: (
            row["checkpoint"],
            hospital_order[row["hospital"]],
            policy_order[row["policy"]],
            row["budget"],
            -1 if row["replicate"] is None else row["replicate"],
        )
    )

    random_summary: list[dict[str, Any]] = []
    random_groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["policy"] == "random":
            random_groups[(row["checkpoint"], row["hospital"], row["budget"])].append(row)
    for (checkpoint, hospital, budget), replicates in sorted(random_groups.items()):
        summary: dict[str, Any] = {
            "checkpoint": checkpoint,
            "hospital": hospital,
            "budget": budget,
            "replicates": len(replicates),
        }
        for metric in (*METRIC_NAMES, "mean_acquisition_count", "acquisition_coverage"):
            values = np.asarray(
                [row[metric] for row in replicates if row[metric] is not None],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        random_summary.append(summary)
    return rows, random_summary


def _plot_metric(
    result_rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    figure, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(15, 4.5), sharey=True)
    for axis, checkpoint in zip(axes, CHECKPOINTS):
        scoped = [
            row
            for row in result_rows
            if row["checkpoint"] == checkpoint and row["hospital"] == "All"
        ]
        for policy, color, marker in (
            ("no_acquisition", "#666666", "o"),
            ("fixed_order", "#D55E00", "s"),
        ):
            values = sorted(
                (row["budget"], row[metric])
                for row in scoped
                if row["policy"] == policy and row[metric] is not None
            )
            if values:
                axis.plot(
                    [item[0] for item in values],
                    [item[1] for item in values],
                    marker=marker,
                    color=color,
                    label=policy.replace("_", " ").title(),
                )
        random_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            if row["policy"] == "random":
                random_groups[row["budget"]].append(row)
        if random_groups:
            x = sorted(random_groups)
            mean_values = []
            std_values = []
            for budget in x:
                values = np.asarray(
                    [
                        row[metric]
                        for row in random_groups[budget]
                        if row[metric] is not None
                    ],
                    dtype=float,
                )
                mean_values.append(float(values.mean()) if len(values) else np.nan)
                std_values.append(
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            axis.errorbar(
                x,
                mean_values,
                yerr=std_values,
                marker="^",
                capsize=3,
                color="#0072B2",
                label="Random (mean ± SD)",
            )
        reference = next(
            (
                row[metric]
                for row in scoped
                if row["policy"] == "no_masking_reference"
                and row[metric] is not None
            ),
            None,
        )
        if reference is not None:
            axis.axhline(
                reference,
                linestyle=":",
                color="#009E73",
                label="No-masking reference",
            )
        axis.set(
            title=f"ICU hour {checkpoint}",
            xlabel="Acquisition budget",
            xticks=(0, 1, 2, 3),
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.15, 1, 1))
    figure.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(figure)


def _write_markdown(result: dict[str, Any]) -> None:
    lines = [
        "# Batch retrospective acquisition evaluation",
        "",
        f"- Mode: {'development sample' if result['configuration']['development_mode'] else 'full validation'}",
        f"- Patients processed: {result['configuration']['patients_processed']:,}",
        f"- Random policy replicates: {result['configuration']['random_replicates']}",
        f"- Global seed: {result['configuration']['global_seed']}",
        "- Evaluation unit: one patient per checkpoint; official SepsisLabel at that hour.",
        "- Oracle availability: policies may use candidate feature names with finite observations through the checkpoint, but not hidden measurement values.",
        "- No-masking reference is descriptive only, not an acquisition policy.",
        "- AUROC is unavailable when a cohort has only one outcome class; AUPRC is unavailable when there are no positive checkpoint labels.",
        "",
        "## Metrics by checkpoint, hospital, budget, and replicate",
        "",
        "| Hour | Hospital | Policy | Budget | Replicate | Patients | Positive labels | AUROC | AUPRC | Brier | Acquired features | Mean acquired | Acquisition coverage |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        render = lambda value: "—" if value is None else f"{value:.5f}"
        lines.append(
            f"| {row['checkpoint']} | {row['hospital']} | {row['policy']} | "
            f"{row['budget']} | {'—' if row['replicate'] is None else row['replicate']} | "
            f"{row['eligible_patients']:,} | {row['positive_labels_at_checkpoint']:,} | "
            f"{render(row['auroc'])} | {render(row['auprc'])} | "
            f"{row['brier_score']:.5f} | {row['acquired_feature_count']:,} | "
            f"{row['mean_acquisition_count']:.3f} | "
            f"{row['acquisition_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Random-policy replicate variation",
            "",
            "Values show mean ± sample standard deviation across replicates.",
            "",
            "| Hour | Hospital | Budget | AUROC | AUPRC | Brier | Mean acquired | Acquisition coverage |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["random_replicate_summary"]:
        def mean_sd(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            return "—" if mean is None else f"{mean:.5f} ± {std:.5f}"

        lines.append(
            f"| {row['checkpoint']} | {row['hospital']} | {row['budget']} | "
            f"{mean_sd('auroc')} | {mean_sd('auprc')} | {mean_sd('brier_score')} | "
            f"{mean_sd('mean_acquisition_count')} | "
            f"{mean_sd('acquisition_coverage')} |"
        )
    lines.extend(
        [
            "",
            "## Scientific limitations",
            "",
            "- This is retrospective historical information reveal, not clinical test ordering.",
            "- Feature eligibility is based on historically recorded values and is an oracle availability assumption.",
            "- Artificial masking may create out-of-distribution feature patterns.",
            "- Risk changes or metric differences alone do not establish useful prediction improvement.",
            "- Development mode is a small validation sample, not a held-out final test result.",
            "- Patients without any finite candidate measurement remain in the checkpoint cohort and receive no acquisition.",
            "",
        ]
    )
    (RESULTS_DIR / "batch_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_batch_scalar_reference(
    patients: pd.DataFrame,
    model: Any,
    calibrator: Any,
    random_replicates: int = RANDOM_REPLICATES,
    global_seed: int = GLOBAL_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if random_replicates < 5:
        raise ValueError("At least five random-policy replicates are required")
    records: dict[tuple[Any, ...], dict[str, list[Any]]] = defaultdict(
        lambda: {
            "labels": [],
            "probabilities": [],
            "acquired_counts": [],
            "available_counts": [],
            "acquisition_patients": [],
        }
    )
    for patient_manifest in patients.itertuples(index=False):
        patient = pd.read_csv(patient_manifest.file_path, sep="|")
        for checkpoint in CHECKPOINTS:
            if not (patient["ICULOS"] == checkpoint).any():
                continue
            label = checkpoint_label(patient, checkpoint)
            history = historical_frame(patient, checkpoint)
            available = historically_available_features(history)
            initial_hidden = initial_hidden_features()
            no_acquisition_probability = _probability_from_visible_history(
                history,
                checkpoint,
                model,
                calibrator,
                initial_hidden,
            )
            no_mask_probability = _probability_from_visible_history(
                history,
                checkpoint,
                model,
                calibrator,
                (),
            )
            scope_names = ("All", str(patient_manifest.hospital))
            available_count = len(available)
            for scope in scope_names:
                common = (checkpoint, scope)
                _record_group(
                    records,
                    (*common, "no_acquisition", 0, None),
                    label,
                    no_acquisition_probability,
                    0,
                    available_count,
                )
                _record_group(
                    records,
                    (*common, "no_masking_reference", 0, None),
                    label,
                    no_mask_probability,
                    0,
                    available_count,
                )

            fixed_simulator = AcquisitionSimulator(
                patient=history,
                checkpoint=checkpoint,
                model=model,
                calibrator=calibrator,
                hidden_features=initial_hidden,
            )
            fixed_eligible = tuple(fixed_simulator.eligible_features())
            if fixed_eligible != available:
                raise AssertionError(
                    "Simulator availability differs from finite-history oracle"
                )
            fixed_order = acquisition_order(fixed_eligible, "fixed_order")
            for budget in ACQUISITION_BUDGETS:
                selected = limit_acquisitions(fixed_order, budget)
                for feature in selected[len(fixed_simulator.acquired) :]:
                    fixed_simulator.acquire(feature)
                probability = fixed_simulator.predict_risk()
                for scope in scope_names:
                    _record_group(
                        records,
                        (checkpoint, scope, "fixed_order", budget, None),
                        label,
                        probability,
                        len(fixed_simulator.acquired),
                        available_count,
                    )

            for replicate in range(random_replicates):
                rng = random.Random(
                    stable_random_seed(
                        global_seed,
                        patient_manifest.hospital,
                        patient_manifest.patient_id,
                        checkpoint,
                        replicate,
                    )
                )
                simulator = AcquisitionSimulator(
                    patient=history,
                    checkpoint=checkpoint,
                    model=model,
                    calibrator=calibrator,
                    hidden_features=initial_hidden,
                )
                eligible = tuple(simulator.eligible_features())
                if eligible != available:
                    raise AssertionError(
                        "Simulator availability differs from finite-history oracle"
                    )
                order = acquisition_order(eligible, "random", rng)
                for budget in ACQUISITION_BUDGETS:
                    selected = limit_acquisitions(order, budget)
                    for feature in selected[len(simulator.acquired) :]:
                        simulator.acquire(feature)
                    probability = simulator.predict_risk()
                    for scope in scope_names:
                        _record_group(
                            records,
                            (checkpoint, scope, "random", budget, replicate),
                            label,
                            probability,
                            len(simulator.acquired),
                            available_count,
                        )
    return _make_result_rows(records)


def _prepare_patient_states(
    patient_manifest: Any,
    patient_index: int,
    model_features: list[str],
    random_replicates: int,
    global_seed: int,
) -> dict[str, Any]:
    io_started = perf_counter()
    patient = pd.read_csv(patient_manifest.file_path, sep="|")
    io_seconds = perf_counter() - io_started
    feature_seconds = 0.0
    state_vectors: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}
    observations: list[tuple[Any, ...]] = []
    decisions: list[tuple[Any, ...]] = []

    for checkpoint in CHECKPOINTS:
        if not (patient["ICULOS"] == checkpoint).any():
            continue
        label = checkpoint_label(patient, checkpoint)
        history = historical_frame(patient, checkpoint)
        available = historically_available_features(history)

        feature_started = perf_counter()
        visible_vector = build_checkpoint_feature_vector(
            history,
            checkpoint,
            model_features,
        )
        feature_seconds += perf_counter() - feature_started

        def get_state_key(hidden: Iterable[str]) -> tuple[int, tuple[str, ...]]:
            nonlocal feature_seconds
            hidden_key = tuple(sorted(set(hidden)))
            local_key = (checkpoint, hidden_key)
            if local_key not in state_vectors:
                feature_started = perf_counter()
                state_vectors[local_key] = mask_checkpoint_feature_vector(
                    visible_vector,
                    model_features,
                    hidden_key,
                )
                feature_seconds += perf_counter() - feature_started
            return local_key

        initial_hidden = initial_hidden_features()
        no_acquisition_key = get_state_key(initial_hidden)
        reference_key = get_state_key(())
        scope_names = ("All", str(patient_manifest.hospital))
        for scope in scope_names:
            observations.append(
                (
                    (checkpoint, scope, "no_acquisition", 0, None),
                    label,
                    no_acquisition_key,
                    0,
                    len(available),
                )
            )
            observations.append(
                (
                    (checkpoint, scope, "no_masking_reference", 0, None),
                    label,
                    reference_key,
                    0,
                    len(available),
                )
            )

        fixed_order = acquisition_order(available, "fixed_order")
        decisions.append(
            (patient_index, checkpoint, "fixed_order", None, fixed_order)
        )
        for budget in ACQUISITION_BUDGETS:
            selected = limit_acquisitions(fixed_order, budget)
            hidden = set(initial_hidden) - set(selected)
            state_key = get_state_key(hidden)
            for scope in scope_names:
                observations.append(
                    (
                        (checkpoint, scope, "fixed_order", budget, None),
                        label,
                        state_key,
                        len(selected),
                        len(available),
                    )
                )

        for replicate in range(random_replicates):
            rng = random.Random(
                stable_random_seed(
                    global_seed,
                    patient_manifest.hospital,
                    patient_manifest.patient_id,
                    checkpoint,
                    replicate,
                )
            )
            order = acquisition_order(available, "random", rng)
            decisions.append(
                (patient_index, checkpoint, "random", replicate, order)
            )
            for budget in ACQUISITION_BUDGETS:
                selected = limit_acquisitions(order, budget)
                hidden = set(initial_hidden) - set(selected)
                state_key = get_state_key(hidden)
                for scope in scope_names:
                    observations.append(
                        (
                            (checkpoint, scope, "random", budget, replicate),
                            label,
                            state_key,
                            len(selected),
                            len(available),
                        )
                    )

    return {
        "patient_index": patient_index,
        "state_vectors": state_vectors,
        "observations": observations,
        "decisions": decisions,
        "io_seconds": io_seconds,
        "feature_seconds": feature_seconds,
    }


def run_batch(
    patients: pd.DataFrame,
    model: Any,
    calibrator: Any,
    random_replicates: int = RANDOM_REPLICATES,
    global_seed: int = GLOBAL_SEED,
    workers: int = 1,
    profile: dict[str, float] | None = None,
    return_diagnostics: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Prepare unique states, then run one deterministic batch prediction."""
    if random_replicates < 5:
        raise ValueError("At least five random-policy replicates are required")
    if workers not in (1, 2, 4):
        raise ValueError("workers must be one of 1, 2, or 4")
    started = perf_counter()
    model_features = list(model.feature_names_in_)
    if EXCLUDED_FEATURES.intersection(model_features):
        raise ValueError("Patient identifiers, hospital, or label found in model features")

    manifests = list(patients.itertuples(index=False))
    if workers == 1:
        prepared = [
            _prepare_patient_states(
                manifest,
                index,
                model_features,
                random_replicates,
                global_seed,
            )
            for index, manifest in enumerate(manifests)
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _prepare_patient_states,
                    manifest,
                    index,
                    model_features,
                    random_replicates,
                    global_seed,
                )
                for index, manifest in enumerate(manifests)
            ]
            prepared = [future.result() for future in futures]

    io_seconds = sum(item["io_seconds"] for item in prepared)
    feature_seconds = sum(item["feature_seconds"] for item in prepared)
    state_refs: list[tuple[int, int, tuple[str, ...]]] = []
    state_rows: list[np.ndarray] = []
    all_observations: list[tuple[Any, ...]] = []
    all_decisions: list[tuple[Any, ...]] = []
    for patient in prepared:
        patient_index = patient["patient_index"]
        for (checkpoint, hidden_key), vector in patient["state_vectors"].items():
            state_refs.append((patient_index, checkpoint, hidden_key))
            state_rows.append(vector)
        all_observations.extend(
            (patient_index, *observation)
            for observation in patient["observations"]
        )
        all_decisions.extend(patient["decisions"])

    feature_started = perf_counter()
    model_input = pd.DataFrame(np.vstack(state_rows), columns=model_features)
    feature_seconds += perf_counter() - feature_started

    model_inference_started = perf_counter()
    original_n_jobs = None
    if hasattr(model, "get_params") and "n_jobs" in model.get_params(deep=False):
        original_n_jobs = model.get_params(deep=False)["n_jobs"]
        model.set_params(n_jobs=1)
    try:
        with threadpool_limits(limits=1):
            raw_probabilities = np.asarray(
                model.predict_proba(model_input)[:, 1],
                dtype=float,
            )
    finally:
        if original_n_jobs is not None:
            model.set_params(n_jobs=original_n_jobs)
    if not np.isfinite(raw_probabilities).all() or (
        (raw_probabilities < 0.0) | (raw_probabilities > 1.0)
    ).any():
        raise ValueError("XGBoost produced invalid probabilities")
    clipped = np.clip(raw_probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    probabilities = np.asarray(
        calibrator.predict_proba(logits.reshape(-1, 1))[:, 1],
        dtype=float,
    )
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("Sigmoid calibrator produced invalid probabilities")
    inference_seconds = perf_counter() - model_inference_started

    probability_by_state = dict(zip(state_refs, probabilities.tolist()))
    records: dict[tuple[Any, ...], dict[str, list[Any]]] = defaultdict(
        lambda: {
            "labels": [],
            "probabilities": [],
            "acquired_counts": [],
            "available_counts": [],
            "acquisition_patients": [],
        }
    )
    evaluation_started = perf_counter()
    for (
        patient_index,
        group_key,
        label,
        local_state_key,
        acquired_count,
        available_count,
    ) in all_observations:
        state_key = (patient_index, local_state_key[0], local_state_key[1])
        _record_group(
            records,
            group_key,
            label,
            probability_by_state[state_key],
            acquired_count,
            available_count,
        )
    results, random_summary = _make_result_rows(records)
    evaluation_seconds = perf_counter() - evaluation_started
    elapsed = perf_counter() - started

    timing = {
        "file_io_seconds": io_seconds,
        "pandas_feature_engineering_seconds": feature_seconds,
        "model_inference_and_calibration_seconds": inference_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": elapsed,
    }
    if profile is not None:
        profile.update(timing)
        profile["unique_states"] = float(len(state_refs))
        profile["batched_model_calls"] = 1.0
        profile["patient_files_read"] = float(len(manifests))
    if return_diagnostics:
        return results, random_summary, {
            "probability_by_state": probability_by_state,
            "decisions": all_decisions,
            "timing": timing,
        }
    return results, random_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Run on a deterministic small sample from outputs/splits/val.csv.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_DEVELOPMENT_SAMPLE,
        help=f"Patient count in development mode (default: {DEFAULT_DEVELOPMENT_SAMPLE}).",
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Patient preprocessing workers; XGBoost uses one CPU thread.",
    )
    args = parser.parse_args()

    patients = pd.read_csv(SPLIT_FILE)
    required = {"patient_id", "hospital", "file_path"}
    if not required.issubset(patients.columns):
        raise ValueError(
            f"Validation split manifest missing columns: {sorted(required - set(patients.columns))}"
        )
    if patients["patient_id"].duplicated().any():
        raise ValueError("Validation split manifest contains duplicate patient IDs")
    if args.development:
        patients = sample_development_patients(
            patients,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    model = joblib.load(MODEL_FILE)
    features = list(model.feature_names_in_)
    if EXCLUDED_FEATURES.intersection(features):
        raise ValueError("Patient identifiers, hospital, or label found in model features")
    calibrator = joblib.load(CALIBRATOR_FILE)
    profile: dict[str, float] = {}
    result_rows, random_summary = run_batch(
        patients,
        model,
        calibrator,
        random_replicates=args.random_replicates,
        global_seed=args.seed,
        workers=args.workers,
        profile=profile,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "PhysioNet 2019 validation patients",
        "configuration": {
            "development_mode": bool(args.development),
            "patients_processed": int(len(patients)),
            "checkpoints": list(CHECKPOINTS),
            "candidate_features": list(CANDIDATE_LABS),
            "acquisition_budgets": list(ACQUISITION_BUDGETS),
            "random_replicates": args.random_replicates,
            "global_seed": args.seed,
            "availability_setting": "oracle eligibility by finite historically recorded values at or before checkpoint",
            "initial_mask": "all six candidate laboratory features, identical across policies",
            "outcome": "official SepsisLabel on the exact checkpoint row",
            "prediction": "XGBoost followed by the saved sigmoid calibrator",
            "test_data_loaded": False,
        },
        "results": result_rows,
        "random_replicate_summary": random_summary,
    }
    validate_aggregate_output(
        output,
        forbidden_identifiers=patients["patient_id"].astype(str).tolist(),
    )
    _plot_metric(
        result_rows,
        "brier_score",
        "Brier score (lower is better)",
        "brier_vs_budget.png",
    )
    _plot_metric(result_rows, "auroc", "AUROC", "auroc_vs_budget.png")
    _plot_metric(result_rows, "auprc", "AUPRC", "auprc_vs_budget.png")
    _plot_metric(
        result_rows,
        "mean_acquisition_count",
        "Mean acquired features per patient",
        "acquisitions_vs_budget.png",
    )
    (RESULTS_DIR / "batch_metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(output)
    print(
        json.dumps(
            {
                "configuration": output["configuration"],
                "result_rows": len(result_rows),
                "random_replicate_summary": random_summary,
                "performance_profile": profile,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()