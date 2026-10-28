"""Evaluate a training-derived feature-level value-of-information policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_batch_acquisition import (
    ACQUISITION_BUDGETS,
    CANDIDATE_LABS,
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
    sample_development_patients,
    stable_random_seed,
)
from src.acquisition.baseline_policies import fixed_order_policy
from src.acquisition.value_of_information import FeatureVOIPolicy, VOIConfig
from src.data.build_temporal_features import (
    build_checkpoint_feature_vector,
    mask_checkpoint_feature_vector,
)
from src.evaluation.paired_bootstrap import paired_bootstrap
from src.simulation.acquisition_simulator import AcquisitionSimulator


TRAIN_SPLIT_FILE = Path("outputs/splits/train.csv")
TRAIN_DATA_FILE = Path("data/processed/train.parquet")
VALIDATION_DATA_FILE = Path("data/processed/val.parquet")
VOI_MODEL_FILE = Path("outputs/models/feature_voi_policy.joblib")
RESULTS_DIR = Path("experiments/results/acquisition/voi")
FIGURES_DIR = RESULTS_DIR / "figures"
BOOTSTRAP_RESAMPLES = 1_000
TRAIN_POLICY_FRACTION = 0.2
VOI_COST_WEIGHTS = (0.0, 0.0001, 0.0005, 0.001)
POLICIES = ("voi", "fixed_order", "random")
EXCLUDED_FEATURES = {"patient_id", "hospital", "SepsisLabel"}


def _validate_validation_manifest(patients: pd.DataFrame) -> None:
    required = {"patient_id", "hospital", "file_path"}
    missing = required - set(patients.columns)
    if missing:
        raise ValueError(f"Validation manifest missing columns: {sorted(missing)}")
    if patients["patient_id"].duplicated().any():
        raise ValueError("Validation manifest contains duplicate patient IDs")
    if any("test" in str(path).lower() for path in patients["file_path"]):
        raise ValueError("Test data are not permitted")


def split_training_cohorts(
    train_manifest: pd.DataFrame,
    *,
    policy_fraction: float = TRAIN_POLICY_FRACTION,
    seed: int = GLOBAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create patient-disjoint model-fit and VOI-outcome training cohorts."""
    if not 0 < policy_fraction < 1:
        raise ValueError("policy_fraction must be between zero and one")
    if train_manifest["patient_id"].duplicated().any():
        raise ValueError("Training manifest contains duplicate patient IDs")
    fit_ids, policy_ids = train_test_split(
        train_manifest["patient_id"].to_numpy(),
        test_size=policy_fraction,
        random_state=seed,
    )
    if set(fit_ids).intersection(policy_ids):
        raise AssertionError("Training fit and policy cohorts overlap")
    fit = train_manifest[train_manifest["patient_id"].isin(fit_ids)].copy()
    policy = train_manifest[train_manifest["patient_id"].isin(policy_ids)].copy()
    return fit.reset_index(drop=True), policy.reset_index(drop=True)


def _fit_independent_predictor(
    fit_manifest: pd.DataFrame,
    *,
    n_estimators: int = 120,
    seed: int = GLOBAL_SEED,
) -> tuple[Any, list[str]]:
    """Fit a predictor only on the disjoint training fit cohort."""
    if not TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing training feature data: {TRAIN_DATA_FILE}")
    fit_ids = set(fit_manifest["patient_id"].astype(str))
    data = pd.read_parquet(TRAIN_DATA_FILE)
    data_ids = data["patient_id"].astype(str)
    fit_data = data.loc[data_ids.isin(fit_ids)].copy()
    if not set(fit_data["patient_id"].astype(str)).issubset(fit_ids):
        raise AssertionError("Independent predictor received patients outside fit cohort")
    features = [column for column in data.columns if column not in EXCLUDED_FEATURES]
    if not features:
        raise ValueError("No predictor features found")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        SGDClassifier(
            loss="log_loss",
            max_iter=max(10, n_estimators),
            random_state=seed,
            n_jobs=-1,
        ),
    )
    model.fit(
        fit_data.loc[:, features],
        fit_data["SepsisLabel"].astype(int),
    )
    return model, features


def _probability(
    model: Any,
    history: pd.DataFrame,
    checkpoint: int,
    hidden: Iterable[str],
) -> float:
    features = list(model.feature_names_in_)
    vector = build_checkpoint_feature_vector(
        history,
        checkpoint,
        features,
        hidden_features=set(hidden),
    )
    probability = float(
        model.predict_proba(pd.DataFrame([vector], columns=features))[0, 1]
    )
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Independent predictor produced an invalid probability")
    return probability


def _probability_from_vector(model: Any, vector: np.ndarray) -> float:
    features = list(model.feature_names_in_)
    probability = float(
        model.predict_proba(pd.DataFrame([vector], columns=features))[0, 1]
    )
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Independent predictor produced an invalid probability")
    return probability


def generate_training_outcomes(
    policy_patients: pd.DataFrame,
    predictor: Any,
    *,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
) -> list[dict[str, Any]]:
    """Generate feature outcomes using only the disjoint policy training cohort."""
    outcome_metadata: list[tuple[str, int, float, int]] = []
    vectors: list[np.ndarray] = []
    initial_hidden = initial_hidden_features()
    if not TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing training feature data: {TRAIN_DATA_FILE}")
    policy_ids = set(policy_patients["patient_id"].astype(str))
    data = pd.read_parquet(TRAIN_DATA_FILE)
    data = data[data["patient_id"].astype(str).isin(policy_ids)]
    data = data[data["ICULOS"].isin(checkpoints)]
    model_features = list(predictor.feature_names_in_)
    for patient_id, patient_rows in data.groupby("patient_id", sort=False):
        for checkpoint in checkpoints:
            checkpoint_rows = patient_rows[patient_rows["ICULOS"] == checkpoint]
            if len(checkpoint_rows) == 0:
                continue
            if len(checkpoint_rows) != 1:
                raise ValueError(
                    f"Expected one processed row at checkpoint {checkpoint}"
                )
            row = checkpoint_rows.iloc[0]
            label = int(row["SepsisLabel"])
            if label not in (0, 1):
                raise ValueError("Training checkpoint labels must be binary")
            available = tuple(
                sorted(
                    feature
                    for feature in CANDIDATE_LABS
                    if float(row[f"{feature}_observed"]) > 0
                )
            )
            visible_vector = row.loc[model_features].to_numpy(dtype=float)
            before_vector = mask_checkpoint_feature_vector(
                visible_vector,
                model_features,
                set(initial_hidden),
            )
            before_index = len(vectors)
            vectors.append(before_vector)
            for feature in available:
                after_vector = mask_checkpoint_feature_vector(
                    visible_vector,
                    model_features,
                    set(initial_hidden) - {feature},
                )
                after_index = len(vectors)
                vectors.append(after_vector)
                outcome_metadata.append((feature, label, before_index, after_index))
    if not vectors:
        return []
    probabilities = np.asarray(
        predictor.predict_proba(
            pd.DataFrame(np.vstack(vectors), columns=model_features)
        )[:, 1],
        dtype=float,
    )
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Independent predictor produced invalid probabilities")
    outcomes = []
    for feature, label, before_index, after_index in outcome_metadata:
        outcomes.append(
            {
                "feature": feature,
                "loss_before": (probabilities[before_index] - label) ** 2,
                "loss_after": (probabilities[after_index] - label) ** 2,
            }
        )
    return outcomes


def fit_voi_policy(
    outcomes: list[dict[str, Any]],
    *,
    cost_weight: float = 0.0,
) -> FeatureVOIPolicy:
    policy = FeatureVOIPolicy(
        VOIConfig(cost_weight=cost_weight, shrinkage=100.0, min_examples=30)
    )
    policy.fit(outcomes)
    return policy


def _policy_order(
    policy: str,
    eligible: Iterable[str],
    *,
    voi_policy: FeatureVOIPolicy | None = None,
    costs: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    remaining = set(eligible)
    order: list[str] = []
    while remaining:
        if policy == "voi":
            selected = voi_policy.select(remaining, costs=costs) if voi_policy else None
        elif policy == "fixed_order":
            selected = fixed_order_policy(sorted(remaining))
        elif policy == "random":
            selected = acquisition_order(remaining, "random", rng)[0]
        else:
            raise ValueError(f"Unsupported policy: {policy}")
        if selected is None:
            break
        order.append(selected)
        remaining.remove(selected)
    return tuple(order)


def evaluate_patient_records(
    patients: pd.DataFrame,
    model: Any,
    calibrator: Any,
    voi_policy: FeatureVOIPolicy,
    *,
    random_replicates: int = RANDOM_REPLICATES,
    global_seed: int = GLOBAL_SEED,
    policies: tuple[str, ...] = ("voi", "fixed_order", "random"),
) -> list[dict[str, Any]]:
    """Evaluate every policy from identical initial masked states."""
    _validate_validation_manifest(patients)
    pending: list[tuple[dict[str, Any], np.ndarray]] = []
    initial_hidden = initial_hidden_features()
    model_features = list(model.feature_names_in_)
    if not VALIDATION_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing validation feature data: {VALIDATION_DATA_FILE}")
    validation_data = pd.read_parquet(VALIDATION_DATA_FILE)
    hospital_by_patient = dict(
        zip(
            patients["patient_id"].astype(str),
            patients["hospital"].astype(str),
        )
    )
    validation_data = validation_data[
        validation_data["patient_id"].astype(str).isin(hospital_by_patient)
    ]
    validation_data = validation_data[
        validation_data["ICULOS"].isin(CHECKPOINTS)
    ]
    for row in validation_data.itertuples(index=False):
        patient_id = str(row.patient_id)
        hospital = hospital_by_patient[patient_id]
        checkpoint = int(row.ICULOS)
        label = int(row.SepsisLabel)
        if label not in (0, 1):
            raise ValueError("Validation checkpoint labels must be binary")
        available = tuple(
            sorted(
                feature
                for feature in CANDIDATE_LABS
                if float(getattr(row, f"{feature}_observed")) > 0
            )
        )
        visible_vector = np.asarray(
            [getattr(row, feature) for feature in model_features],
            dtype=float,
        )
        for checkpoint in (checkpoint,):
            scope_names = ("All", hospital)
            no_vector = mask_checkpoint_feature_vector(
                visible_vector,
                model_features,
                set(initial_hidden),
            )
            for scope in scope_names:
                pending.append(
                    (
                        {
                            "patient_id": patient_id,
                            "hospital": scope,
                            "checkpoint": checkpoint,
                            "policy": "no_acquisition",
                            "budget": 0,
                            "replicate": None,
                            "label": label,
                            "acquired_count": 0,
                            "selected_features": [],
                        },
                        no_vector,
                    )
                )

            policy_specs = []
            if "voi" in policies:
                policy_specs.append(("voi", None))
            if "fixed_order" in policies:
                policy_specs.append(("fixed_order", None))
            if "random" in policies:
                policy_specs.extend(
                    ("random", replicate) for replicate in range(random_replicates)
                )
            for policy, replicate in policy_specs:
                rng = (
                    random.Random(
                        stable_random_seed(
                            global_seed,
                            hospital,
                            patient_id,
                            checkpoint,
                            replicate or 0,
                        )
                    )
                    if policy == "random"
                    else None
                )
                costs = {feature: 1.0 for feature in available}
                order = _policy_order(
                    policy,
                    available,
                    voi_policy=voi_policy,
                    costs=costs,
                    rng=rng,
                )
                for budget in (0, *ACQUISITION_BUDGETS):
                    selected = list(order[:budget])
                    hidden = set(initial_hidden) - set(selected)
                    vector = mask_checkpoint_feature_vector(
                        visible_vector,
                        model_features,
                        hidden,
                    )
                    for scope in scope_names:
                        pending.append(
                            (
                                {
                                    "patient_id": patient_id,
                                    "hospital": scope,
                                    "checkpoint": checkpoint,
                                    "policy": policy,
                                    "budget": budget,
                                    "replicate": replicate,
                                    "label": label,
                                    "acquired_count": len(selected),
                                    "selected_features": selected,
                                },
                                vector,
                            )
                        )
    if not pending:
        return []
    vectors = pd.DataFrame(
        np.vstack([vector for _, vector in pending]),
        columns=model_features,
    )
    raw = np.asarray(model.predict_proba(vectors)[:, 1], dtype=float)
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Validation predictor produced invalid probabilities")
    clipped = np.clip(raw, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    probabilities = np.asarray(
        calibrator.predict_proba(logits.reshape(-1, 1))[:, 1],
        dtype=float,
    )
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Validation calibrator produced invalid probabilities")
    records = []
    for (record, _), probability in zip(pending, probabilities):
        record["probability"] = float(probability)
        records.append(record)
    return records


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    return {
        "eligible_patients": int(len(y)),
        "positive_labels_at_checkpoint": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier_score": float(brier_score_loss(y, p)),
    }


def aggregate_results(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["checkpoint"],
                record["hospital"],
                record["policy"],
                record["budget"],
                record["replicate"],
            )
        ].append(record)
    results = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        metrics = _metrics(
            [row["label"] for row in group],
            [row["probability"] for row in group],
        )
        results.append(
            {
                "checkpoint": key[0],
                "hospital": key[1],
                "policy": key[2],
                "budget": key[3],
                "replicate": key[4],
                **metrics,
                "mean_acquisition_count": float(
                    np.mean([row["acquired_count"] for row in group])
                ),
                "acquisition_coverage": float(
                    np.mean([row["acquired_count"] > 0 for row in group])
                ),
            }
        )
    return results


def paired_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    keys = sorted(
        {
            (r["checkpoint"], r["hospital"], r["policy"], r["budget"], r["replicate"])
            for r in records
            if r["policy"] in ("voi", "fixed_order", "random")
        }
    )
    for checkpoint, hospital, policy, budget, replicate in keys:
        no_rows = [
            row
            for row in records
            if row["checkpoint"] == checkpoint
            and row["hospital"] == hospital
            and row["policy"] == "no_acquisition"
        ]
        acquisition_rows = [
            row
            for row in records
            if row["checkpoint"] == checkpoint
            and row["hospital"] == hospital
            and row["policy"] == policy
            and row["budget"] == budget
            and row["replicate"] == replicate
        ]
        comparisons.append(
            {
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
        )
    return comparisons


def cost_sensitivity(
    patients: pd.DataFrame,
    policy: FeatureVOIPolicy,
) -> list[dict[str, Any]]:
    """Assess simulated dimensionless acquisition costs, not clinical costs."""
    if not VALIDATION_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing validation feature data: {VALIDATION_DATA_FILE}")
    validation_data = pd.read_parquet(VALIDATION_DATA_FILE)
    patient_ids = set(patients["patient_id"].astype(str))
    validation_data = validation_data[
        validation_data["patient_id"].astype(str).isin(patient_ids)
    ]
    validation_data = validation_data[
        validation_data["ICULOS"].isin(CHECKPOINTS)
    ]
    rows = []
    for weight in VOI_COST_WEIGHTS:
        acquisition_counts = []
        for row in validation_data.itertuples(index=False):
            available = tuple(
                sorted(
                    feature
                    for feature in CANDIDATE_LABS
                    if float(getattr(row, f"{feature}_observed")) > 0
                )
            )
            selected = 0
            remaining = set(available)
            for _ in ACQUISITION_BUDGETS:
                candidates = [
                    (policy.values.get(feature, 0.0) - weight, feature)
                    for feature in remaining
                ]
                if not candidates:
                    break
                score, feature = max(candidates, key=lambda item: (item[0], item[1]))
                if score <= 0:
                    break
                remaining.remove(feature)
                selected += 1
            acquisition_counts.append(selected)
        rows.append(
            {
                "cost_weight": weight,
                "simulated_cost_units_per_acquisition": 1.0,
                "mean_acquisition_count": float(np.mean(acquisition_counts)),
                "acquisition_coverage": float(np.mean(np.asarray(acquisition_counts) > 0)),
            }
        )
    return rows


def _write_figures(results: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy in ("voi", "fixed_order", "random"):
        rows = [
            row
            for row in results
            if row["policy"] == policy
            and row["hospital"] == "All"
            and row["replicate"] in (None, 0)
        ]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["budget"]].append(row["brier_score"])
        if grouped:
            x = sorted(grouped)
            axis.plot(x, [np.mean(grouped[value]) for value in x], marker="o", label=policy)
    axis.set(xlabel="Acquisition budget", ylabel="Brier score")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "brier_by_policy.png", dpi=160)
    plt.close(figure)


def _write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# VOI acquisition evaluation",
        "",
        "- Dataset: validation patients only for evaluation; the held-out test split was not accessed.",
        "- VOI fitting: a patient-disjoint training fit cohort trained an independent predictor, and a separate training policy cohort supplied out-of-sample acquisition outcomes.",
        "- Existing validation XGBoost and sigmoid calibrator were used for all reported validation predictions.",
        "- Budgets 0-3 are evaluated from identical initial masked states; VOI stops early when no positive estimated net benefit remains.",
        "- Simulated costs are dimensionless policy-analysis costs, not observed clinical costs.",
        "- Bootstrap intervals use 1,000 paired patient-cluster resamples with seed 42.",
        "",
        "## Fitted feature-level VOI estimates",
        "",
        "| Feature | Training examples | Shrunk estimated Brier benefit |",
        "|---|---:|---:|",
    ]
    for feature in CANDIDATE_LABS:
        lines.append(
            f"| {feature} | {payload['voi_policy']['training_counts'].get(feature, 0):,} | "
            f"{payload['voi_policy']['values'].get(feature, 0.0):.8f} |"
        )
    lines.extend(
        [
            "",
            "## VOI policy selection frequencies",
            "",
            "Counts are validation patient-checkpoint selections across the evaluated VOI trajectories.",
            "",
            "| Feature | Selections |",
            "|---|---:|",
        ]
    )
    for feature, count in payload["voi_policy"]["selection_counts"].items():
        lines.append(f"| {feature} | {count:,} |")
    lines.extend(
        [
            "",
            "## Simulated cost sensitivity",
            "",
            "Costs are dimensionless simulated policy-analysis units, not observed clinical costs.",
            "",
            "| Cost weight | Simulated cost units/acquisition | Mean acquisitions | Coverage |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cost_sensitivity"]:
        lines.append(
            f"| {row['cost_weight']:.4f} | "
            f"{row['simulated_cost_units_per_acquisition']:.1f} | "
            f"{row['mean_acquisition_count']:.3f} | "
            f"{row['acquisition_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Validation metrics",
            "",
            "| Hour | Hospital | Policy | Budget | Replicate | Patients | Positive labels | AUROC | AUPRC | Brier | Coverage | Mean acquisitions |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["results"]:
        if row["replicate"] not in (None, 0):
            continue
        render = lambda value: "—" if value is None else f"{value:.5f}"
        lines.append(
            f"| {row['checkpoint']} | {row['hospital']} | {row['policy']} | {row['budget']} | "
            f"{'—' if row['replicate'] is None else row['replicate']} | {row['eligible_patients']:,} | "
            f"{row['positive_labels_at_checkpoint']:,} | {render(row['auroc'])} | "
            f"{render(row['auprc'])} | {row['brier_score']:.5f} | "
            f"{row['acquisition_coverage']:.3f} | {row['mean_acquisition_count']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences versus no acquisition",
            "",
            "AUROC/AUPRC differences are acquisition minus no acquisition. Brier improvement is no acquisition minus acquisition.",
            "",
            "| Hour | Hospital | Policy | Budget | Replicate | AUROC difference (95% CI) | AUPRC difference (95% CI) | Brier improvement (95% CI) | Valid AUROC/AUPRC/Brier |",
            "|---:|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in payload["comparisons"]:
        def display(metric: str) -> str:
            observed = row["observed"][metric]
            interval = row["confidence_intervals_95"][metric]
            return "—" if observed is None else f"{observed:.5f} ({interval[0]:.5f}, {interval[1]:.5f})"
        valid = row["valid_bootstrap_replicates"]
        lines.append(
            f"| {row['checkpoint']} | {row['hospital']} | {row['policy']} | {row['budget']} | "
            f"{'—' if row['replicate'] is None else row['replicate']} | {display('auroc')} | "
            f"{display('auprc')} | {display('brier_improvement')} | "
            f"{valid['auroc']}/{valid['auprc']}/{valid['brier_improvement']} |"
        )
    lines.extend(
        [
            "",
            "## Methodological limitations",
            "",
            "- The independent VOI predictor is a policy-fitting model, not the deployed validation predictor.",
            "- The policy cohort is disjoint from the cohort used to fit that predictor, but the VOI estimator remains subject to finite-sample and model-specification error.",
            "- Historical feature availability is an oracle retrospective assumption.",
            "- Artificial masking may produce out-of-distribution feature patterns.",
            "- Confidence intervals quantify patient-sampling variation and do not establish clinical usefulness or causality.",
        ]
    )
    (RESULTS_DIR / "voi_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    development: bool = False,
    sample_size: int = 100,
    seed: int = GLOBAL_SEED,
    n_estimators: int = 120,
) -> tuple[dict[str, Any], Any, FeatureVOIPolicy]:
    train_manifest = pd.read_csv(TRAIN_SPLIT_FILE)
    validation = pd.read_csv(SPLIT_FILE)
    _validate_validation_manifest(validation)
    if development:
        train_manifest = sample_development_patients(train_manifest, sample_size, seed)
        validation = sample_development_patients(validation, sample_size, seed)
    fit_manifest, policy_manifest = split_training_cohorts(
        train_manifest,
        policy_fraction=TRAIN_POLICY_FRACTION,
        seed=seed,
    )
    predictor, _ = _fit_independent_predictor(
        fit_manifest,
        n_estimators=n_estimators,
        seed=seed,
    )
    outcomes = generate_training_outcomes(policy_manifest, predictor)
    policy = fit_voi_policy(outcomes)
    model = joblib.load(MODEL_FILE)
    calibrator = joblib.load(CALIBRATOR_FILE)
    result_records = evaluate_patient_records(
        validation,
        model,
        calibrator,
        policy,
        global_seed=seed,
    )
    results = aggregate_results(result_records)
    comparisons = paired_comparisons(result_records)
    selection_counts = Counter(
        feature
        for row in result_records
        if row["policy"] == "voi"
        for feature in row["selected_features"]
    )
    payload = {
        "configuration": {
            "development_mode": development,
            "training_patients": len(train_manifest),
            "training_fit_patients": len(fit_manifest),
            "training_policy_patients": len(policy_manifest),
            "validation_patients": len(validation),
            "checkpoints": list(CHECKPOINTS),
            "candidate_features": list(CANDIDATE_LABS),
            "budgets": [0, *ACQUISITION_BUDGETS],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": GLOBAL_SEED,
            "random_replicates": RANDOM_REPLICATES,
        },
        "voi_policy": {
            "values": {
                feature: float(policy.values.get(feature, 0.0))
                for feature in CANDIDATE_LABS
            },
            "training_counts": {
                feature: int(policy.counts.get(feature, 0))
                for feature in CANDIDATE_LABS
            },
            "selection_counts": dict(sorted(selection_counts.items())),
            "outcome_records": len(outcomes),
            "cost_weight": policy.config.cost_weight,
        },
        "results": results,
        "comparisons": comparisons,
        "cost_sensitivity": cost_sensitivity(
            validation,
            policy,
        ),
    }
    return payload, predictor, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--n-estimators", type=int, default=120)
    args = parser.parse_args()
    payload, _, policy = run_experiment(
        development=args.development,
        sample_size=args.sample_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    VOI_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, VOI_MODEL_FILE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "voi_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_report(payload)
    _write_figures(payload["results"])
    print(json.dumps(payload["configuration"], indent=2))


if __name__ == "__main__":
    main()
