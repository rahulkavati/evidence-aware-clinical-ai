"""Paired patient-cluster bootstrap utilities for acquisition comparisons."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


METRICS = ("auroc", "auprc", "brier_improvement")


def _arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not records:
        raise ValueError("Prediction records cannot be empty")
    patient_ids = [str(record["patient_id"]) for record in records]
    labels = np.asarray([record["label"] for record in records], dtype=int)
    probabilities = np.asarray(
        [record["probability"] for record in records], dtype=float
    )
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Prediction records contain duplicate patient IDs")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return labels, probabilities, patient_ids


def validate_alignment(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> None:
    """Require identical patient IDs and labels in the same order."""
    left_labels, _, left_ids = _arrays(left)
    right_labels, _, right_ids = _arrays(right)
    if left_ids != right_ids:
        raise ValueError("Compared policies do not contain identical patient IDs")
    if not np.array_equal(left_labels, right_labels):
        raise ValueError("Compared policies do not contain identical labels")


def metric_values(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    """Compute metrics, returning None for undefined single-class metrics."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be binary")
    return {
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(brier_score_loss(y, p)),
    }


def paired_bootstrap(
    no_acquisition: Sequence[Mapping[str, Any]],
    acquisition: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two aligned policies using a patient-cluster percentile bootstrap."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    validate_alignment(no_acquisition, acquisition)
    labels, no_probabilities, patient_ids = _arrays(no_acquisition)
    _, acquisition_probabilities, _ = _arrays(acquisition)
    observed_no = metric_values(labels, no_probabilities)
    observed_acquisition = metric_values(labels, acquisition_probabilities)
    observed = {
        "auroc": (
            None
            if observed_no["auroc"] is None or observed_acquisition["auroc"] is None
            else observed_acquisition["auroc"] - observed_no["auroc"]
        ),
        "auprc": (
            None
            if observed_no["auprc"] is None or observed_acquisition["auprc"] is None
            else observed_acquisition["auprc"] - observed_no["auprc"]
        ),
        "brier_improvement": observed_no["brier"] - observed_acquisition["brier"],
    }
    bootstrap_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = rng.integers(0, len(patient_ids), size=len(patient_ids))
        sampled_labels = labels[indices]
        sampled_no = no_probabilities[indices]
        sampled_acquisition = acquisition_probabilities[indices]
        no_metrics = metric_values(sampled_labels, sampled_no)
        acquisition_metrics = metric_values(sampled_labels, sampled_acquisition)
        if no_metrics["auroc"] is not None and acquisition_metrics["auroc"] is not None:
            bootstrap_values["auroc"].append(
                acquisition_metrics["auroc"] - no_metrics["auroc"]
            )
        if no_metrics["auprc"] is not None and acquisition_metrics["auprc"] is not None:
            bootstrap_values["auprc"].append(
                acquisition_metrics["auprc"] - no_metrics["auprc"]
            )
        bootstrap_values["brier_improvement"].append(
            no_metrics["brier"] - acquisition_metrics["brier"]
        )
    confidence_intervals = {}
    valid_bootstrap_replicates = {}
    for metric, values in bootstrap_values.items():
        valid_bootstrap_replicates[metric] = len(values)
        confidence_intervals[metric] = (
            None
            if not values
            else [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
        )
    return {
        "observed": observed,
        "confidence_intervals_95": confidence_intervals,
        "valid_bootstrap_replicates": valid_bootstrap_replicates,
        "n_resamples": n_resamples,
        "seed": seed,
        "patient_count": len(patient_ids),
        "positive_label_count": int(labels.sum()),
    }
