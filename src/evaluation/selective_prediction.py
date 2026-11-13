"""Utilities for validation-selected selective prediction.

Selective prediction retains predictions whose confidence exceeds a threshold
and abstains on the remainder. Threshold selection must be performed on a
patient-disjoint development cohort; evaluation is reported separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_binary_inputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional binary labels and probabilities."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be equal, nonzero one-dimensional arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return y, p


def confidence(probabilities: np.ndarray) -> np.ndarray:
    """Return the probability of the model's predicted class."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be one-dimensional and within [0, 1]")
    return np.maximum(p, 1.0 - p)


def retained_mask(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return the rows retained at a confidence threshold."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be finite and within [0, 1]")
    return confidence(probabilities) >= threshold


def evaluate_selective(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate coverage, selective risk, and retained-set quality."""
    y, p = validate_binary_inputs(labels, probabilities)
    mask = retained_mask(p, threshold)
    retained_y = y[mask]
    retained_p = p[mask]
    retained_count = int(mask.sum())
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "total_count": int(len(y)),
        "retained_count": retained_count,
        "abstained_count": int(len(y) - retained_count),
        "coverage": float(mask.mean()),
        "selective_risk": None,
        "selective_brier_score": None,
        "retained_auroc": None,
        "retained_auprc": None,
        "positive_count": int((y == 1).sum()),
        "retained_positive_count": int((mask & (y == 1)).sum()),
        "abstained_positive_count": int(((~mask) & (y == 1)).sum()),
        "abstained_positive_capture": (
            float(((~mask) & (y == 1)).sum() / (y == 1).sum())
            if (y == 1).sum() else None
        ),
        "retained_sensitivity": None,
        "retained_specificity": None,
        "retained_ppv": None,
        "retained_npv": None,
    }
    if retained_count:
        predicted = retained_p >= 0.5
        result["selective_risk"] = float(np.mean(predicted != retained_y.astype(bool)))
        result["selective_brier_score"] = float(np.mean((retained_p - retained_y) ** 2))
        tp = int((predicted & (retained_y == 1)).sum())
        tn = int((~predicted & (retained_y == 0)).sum())
        fp = int((predicted & (retained_y == 0)).sum())
        fn = int((~predicted & (retained_y == 1)).sum())
        result["retained_sensitivity"] = float(tp / (tp + fn)) if tp + fn else None
        result["retained_specificity"] = float(tn / (tn + fp)) if tn + fp else None
        result["retained_ppv"] = float(tp / (tp + fp)) if tp + fp else None
        result["retained_npv"] = float(tn / (tn + fn)) if tn + fn else None
        if len(np.unique(retained_y)) == 2:
            result["retained_auroc"] = float(roc_auc_score(retained_y, retained_p))
            result["retained_auprc"] = float(average_precision_score(retained_y, retained_p))
    return result


def patient_cluster_bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Return percentile intervals after resampling patients as clusters."""
    y, p = validate_binary_inputs(labels, probabilities)
    ids = np.asarray(patient_ids)
    if ids.ndim != 1 or len(ids) != len(y):
        raise ValueError("Patient IDs must be one-dimensional and align with labels")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    _, codes = np.unique(ids, return_inverse=True)
    retained = retained_mask(p, threshold)
    errors = ((p >= 0.5) != y.astype(bool)).astype(float)
    rng = np.random.default_rng(random_state)
    patient_count = int(codes.max()) + 1
    coverage_values = np.empty(n_resamples, dtype=float)
    risk_values = np.empty(n_resamples, dtype=float)
    brier_values = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        sampled = rng.integers(0, patient_count, size=patient_count)
        weights = np.bincount(sampled, minlength=patient_count)[codes]
        retained_weight = weights[retained].sum()
        coverage_values[iteration] = retained_weight / weights.sum()
        if retained_weight:
            risk_values[iteration] = np.average(errors[retained], weights=weights[retained])
            brier_values[iteration] = np.average(
                (p[retained] - y[retained]) ** 2,
                weights=weights[retained],
            )
        else:
            risk_values[iteration] = np.nan
            brier_values[iteration] = np.nan

    def interval(values: np.ndarray) -> dict[str, float]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return {"lower_95": float("nan"), "upper_95": float("nan")}
        return {
            "lower_95": float(np.percentile(finite, 2.5)),
            "upper_95": float(np.percentile(finite, 97.5)),
        }

    return {
        "coverage": interval(coverage_values),
        "selective_risk": interval(risk_values),
        "selective_brier_score": interval(brier_values),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    max_risk: float,
    min_coverage: float = 0.0,
) -> dict[str, Any]:
    """Choose the lowest confidence threshold meeting a validation risk target.

    The returned threshold maximizes retained coverage subject to empirical
    selective risk being at most ``max_risk`` and coverage being at least
    ``min_coverage``. Ties are retained as a group, so the rule is stable and
    does not partially retain equal-confidence predictions.
    """
    y, p = validate_binary_inputs(labels, probabilities)
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError("max_risk must be within [0, 1]")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be within [0, 1]")

    scores = confidence(p)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_errors = ((p[order] >= 0.5) != y[order].astype(bool)).astype(int)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    group_counts = np.diff(np.r_[-1, group_ends])
    cumulative_counts = np.cumsum(group_counts)
    cumulative_errors = np.cumsum(
        np.add.reduceat(sorted_errors, np.r_[0, group_ends[:-1]])
    )
    coverage = cumulative_counts / len(y)
    risks = cumulative_errors / cumulative_counts
    feasible = np.flatnonzero((coverage >= min_coverage) & (risks <= max_risk))
    if len(feasible) == 0:
        raise ValueError("No confidence threshold satisfies the requested risk and coverage")
    # Groups are added from highest to lowest confidence, so the last
    # feasible point has the greatest coverage. Ties remain grouped.
    selected_index = int(feasible[-1])
    threshold = float(sorted_scores[group_ends[selected_index]])
    return evaluate_selective(y, p, threshold)
