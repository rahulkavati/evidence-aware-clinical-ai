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
    }
    if retained_count:
        predicted = retained_p >= 0.5
        result["selective_risk"] = float(np.mean(predicted != retained_y.astype(bool)))
        result["selective_brier_score"] = float(np.mean((retained_p - retained_y) ** 2))
        if len(np.unique(retained_y)) == 2:
            result["retained_auroc"] = float(roc_auc_score(retained_y, retained_p))
            result["retained_auprc"] = float(average_precision_score(retained_y, retained_p))
    return result


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
    # The first feasible point has the greatest coverage because groups are
    # added from highest to lowest confidence. Ties remain grouped.
    selected_index = int(feasible[0])
    threshold = float(sorted_scores[group_ends[selected_index]])
    return evaluate_selective(y, p, threshold)
