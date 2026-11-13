"""Official PhysioNet/CinC 2019 sepsis utility scoring.

This follows the reference implementation published by the PhysioNet
Challenges evaluation repository, including its timing window and cohort
normalization. Predictions are binary and must contain one value per hourly
label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_sequence(labels: Sequence[int], predictions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(predictions, dtype=int)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and predictions must be non-empty aligned vectors")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1")
    if not np.isin(p, (0, 1)).all():
        raise ValueError("Predictions must contain only 0 and 1")
    return y, p


def compute_prediction_utility(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    dt_early: int = -12,
    dt_optimal: int = -6,
    dt_late: int = 3,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
) -> float:
    """Return the unnormalized utility for one patient sequence."""
    y, p = _validate_sequence(labels, predictions)
    if not dt_early < dt_optimal < dt_late:
        raise ValueError("Require dt_early < dt_optimal < dt_late")
    septic = bool(y.any())
    t_sepsis = int(np.argmax(y) - dt_optimal) if septic else float("inf")
    m1 = max_u_tp / (dt_optimal - dt_early)
    b1 = -m1 * dt_early
    m2 = -max_u_tp / (dt_late - dt_optimal)
    b2 = -m2 * dt_late
    m3 = min_u_fn / (dt_late - dt_optimal)
    b3 = -m3 * dt_optimal
    utility = np.zeros(len(y), dtype=float)
    for t in range(len(y)):
        if t <= t_sepsis + dt_late:
            if septic and p[t]:
                if t <= t_sepsis + dt_optimal:
                    utility[t] = max(m1 * (t - t_sepsis) + b1, u_fp)
                else:
                    utility[t] = m2 * (t - t_sepsis) + b2
            elif not septic and p[t]:
                utility[t] = u_fp
            elif septic and not p[t] and t > t_sepsis + dt_optimal:
                utility[t] = m3 * (t - t_sepsis) + b3
            elif not septic and not p[t]:
                utility[t] = u_tn
    return float(utility.sum())


def normalized_utility(
    patient_sequences: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> dict[str, float | int]:
    """Return observed and normalized utility for a cohort of patients."""
    if not patient_sequences:
        raise ValueError("patient_sequences cannot be empty")
    observed = 0.0
    best = 0.0
    worst = 0.0
    inaction = 0.0
    for labels, predictions in patient_sequences:
        y, p = _validate_sequence(labels, predictions)
        best_predictions = np.zeros(len(y), dtype=int)
        if y.any():
            t_sepsis = int(np.argmax(y) - (-6))
            best_predictions[max(0, t_sepsis - 12): min(t_sepsis + 3 + 1, len(y))] = 1
        observed += compute_prediction_utility(y, p)
        best += compute_prediction_utility(y, best_predictions)
        worst += compute_prediction_utility(y, 1 - best_predictions)
        inaction += compute_prediction_utility(y, np.zeros(len(y), dtype=int))
    denominator = best - inaction
    if np.isclose(denominator, 0.0):
        raise ValueError("Best and inaction utilities are indistinguishable")
    return {
        "observed_utility": float(observed),
        "best_utility": float(best),
        "worst_utility": float(worst),
        "inaction_utility": float(inaction),
        "normalized_utility": float((observed - inaction) / denominator),
        "patient_count": int(len(patient_sequences)),
    }


def score_frame(
    frame: pd.DataFrame,
    predictions: Sequence[int],
    *,
    patient_column: str = "patient_id",
    time_column: str = "ICULOS",
    label_column: str = "SepsisLabel",
) -> dict[str, float | int]:
    """Score aligned hourly frame rows using the official cohort protocol."""
    required = {patient_column, time_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    p = np.asarray(predictions, dtype=int)
    if len(p) != len(frame):
        raise ValueError("Predictions must align with frame rows")
    ordered = frame.assign(_prediction=p).sort_values(
        [patient_column, time_column], kind="stable"
    )
    sequences = [
        (
            patient[label_column].astype(int).to_numpy(),
            patient["_prediction"].astype(int).to_numpy(),
        )
        for _, patient in ordered.groupby(patient_column, sort=False)
    ]
    return normalized_utility(sequences)
