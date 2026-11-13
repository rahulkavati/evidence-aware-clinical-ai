"""One-step acquisition evaluation for predictions that would abstain."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data.g1_masking import CANDIDATE_LABS, apply_hidden_matrix, triplet_columns
from src.evaluation.selective_prediction import confidence, validate_binary_inputs

DEFAULT_PRIORITY = CANDIDATE_LABS


def one_step_fixed_order_frame(
    frame: pd.DataFrame,
    labs: Sequence[str] = CANDIDATE_LABS,
    priority: Sequence[str] = DEFAULT_PRIORITY,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Hide all labs, then reveal the first historically eligible lab per row."""
    labs = list(labs)
    priority = list(priority)
    if set(priority) != set(labs) or len(priority) != len(labs):
        raise ValueError("Priority must contain each candidate lab exactly once")
    for lab in labs:
        _, observed, _ = triplet_columns(lab)
        if observed not in frame.columns:
            raise ValueError(f"Missing observed-mask column for {lab}: {observed}")

    observed = np.column_stack([
        frame[triplet_columns(lab)[1]].fillna(0).to_numpy(dtype=float) == 1
        for lab in labs
    ])
    lab_index = {lab: index for index, lab in enumerate(labs)}
    selected_index = np.full(len(frame), -1, dtype=int)
    for lab in priority:
        index = lab_index[lab]
        available = (selected_index < 0) & observed[:, index]
        selected_index[available] = index

    hidden = np.ones((len(frame), len(labs)), dtype=bool)
    rows = np.flatnonzero(selected_index >= 0)
    hidden[rows, selected_index[rows]] = False
    return apply_hidden_matrix(frame, labs, hidden), selected_index


def decision_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    retained: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure prediction quality only among rows on which a decision is made."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    mask = np.asarray(retained, dtype=bool)
    if len(y) == 0 or len(y) != len(p) or len(y) != len(mask):
        raise ValueError("Labels, probabilities, and retention mask must align")
    count = int(mask.sum())
    predicted = p >= 0.5
    positives = y == 1
    retained_positives = mask & positives
    abstained_positives = (~mask) & positives
    retained_predicted = predicted[mask]
    retained_labels = y[mask]

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    result: dict[str, float | int | None] = {
        "total_count": int(len(y)),
        "retained_count": count,
        "abstained_count": int(len(y) - count),
        "coverage": float(mask.mean()),
        "risk": None,
        "brier_score": None,
        "positive_count": int(positives.sum()),
        "retained_positive_count": int(retained_positives.sum()),
        "abstained_positive_count": int(abstained_positives.sum()),
        "abstained_positive_capture": safe_ratio(
            int(abstained_positives.sum()), int(positives.sum())
        ),
        "retained_sensitivity": None,
        "retained_specificity": None,
        "retained_ppv": None,
        "retained_npv": None,
    }
    if count:
        errors = retained_predicted != retained_labels.astype(bool)
        result["risk"] = float(np.mean(errors))
        result["brier_score"] = float(np.mean((p[mask] - y[mask]) ** 2))
        tp = int((retained_predicted & (retained_labels == 1)).sum())
        tn = int((~retained_predicted & (retained_labels == 0)).sum())
        fp = int((retained_predicted & (retained_labels == 0)).sum())
        fn = int((~retained_predicted & (retained_labels == 1)).sum())
        result["retained_sensitivity"] = safe_ratio(tp, tp + fn)
        result["retained_specificity"] = safe_ratio(tn, tn + fp)
        result["retained_ppv"] = safe_ratio(tp, tp + fp)
        result["retained_npv"] = safe_ratio(tn, tn + fn)
    return result


def risk_coverage_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    """Return tied-confidence risk/coverage operating points.

    Rows are added from highest confidence to lowest confidence. This makes
    every point compatible with the selective rule used by ``retained_mask``
    and avoids optimistic partial retention of tied rows.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must align and be non-empty")
    validate_binary_inputs(y, p)
    scores = confidence(p)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_errors = ((p[order] >= 0.5) != y[order].astype(bool)).astype(int)
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    counts = ends - starts + 1
    cumulative_count = np.cumsum(counts)
    cumulative_errors = np.cumsum(
        np.add.reduceat(sorted_errors, starts)
    )
    return [
        {
            "threshold": float(sorted_scores[index]),
            "retained_count": int(cumulative_count[index]),
            "coverage": float(cumulative_count[index] / len(y)),
            "risk": float(cumulative_errors[index] / cumulative_count[index]),
        }
        for index in range(len(ends))
    ]


def evaluate_trivial_baselines(
    labels: np.ndarray,
    probabilities: np.ndarray,
    abstention_fraction: float = 0.08,
) -> dict[str, dict[str, float | int | None]]:
    """Evaluate simple reference systems for selective rule-out claims."""
    y, p = validate_binary_inputs(labels, probabilities)
    if not 0.0 <= abstention_fraction < 1.0:
        raise ValueError("abstention_fraction must be in [0, 1)")

    always_negative = np.zeros(len(y), dtype=bool)
    top_count = int(round(len(y) * abstention_fraction))
    order = np.argsort(-p, kind="stable")
    abstain = np.zeros(len(y), dtype=bool)
    if top_count:
        abstain[order[:top_count]] = True
    retained = ~abstain
    return {
        "predict_negative_everywhere": decision_metrics(y, np.zeros_like(p), np.ones(len(y), dtype=bool)),
        "abstain_highest_risk_fraction": decision_metrics(y, p, retained),
    }
