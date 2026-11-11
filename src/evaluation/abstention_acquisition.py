"""One-step acquisition evaluation for predictions that would abstain."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data.g1_masking import CANDIDATE_LABS, apply_hidden_matrix, triplet_columns

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
    result: dict[str, float | int | None] = {
        "total_count": int(len(y)),
        "retained_count": count,
        "abstained_count": int(len(y) - count),
        "coverage": float(mask.mean()),
        "risk": None,
        "brier_score": None,
    }
    if count:
        result["risk"] = float(np.mean((p[mask] >= 0.5) != y[mask].astype(bool)))
        result["brier_score"] = float(np.mean((p[mask] - y[mask]) ** 2))
    return result
