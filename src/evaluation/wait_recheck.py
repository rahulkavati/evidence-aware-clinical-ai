"""Leakage-safe wait/re-check action helpers."""

from __future__ import annotations

import pandas as pd
import numpy as np


def next_observation_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return positional current/next-row pairs within each patient.

    The returned arrays are aligned and contain only rows with a strictly
    future observation. Labels are not read or returned by this helper.
    """
    required = {"patient_id", "ICULOS"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ordered = frame.reset_index(drop=False).rename(columns={"index": "_original_position"}).sort_values(
        ["patient_id", "ICULOS"], kind="stable"
    )
    current: list[int] = []
    future: list[int] = []
    for _, patient in ordered.groupby("patient_id", sort=False):
        positions = patient["_original_position"].to_numpy()
        if len(positions) > 1:
            current.extend(positions[:-1])
            future.extend(positions[1:])
    return np.asarray(current, dtype=int), np.asarray(future, dtype=int)


def wait_recheck_candidates(
    frame: pd.DataFrame,
    abstained: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next-row pairs only for rows selected to wait."""
    abstained = np.asarray(abstained, dtype=bool)
    if len(abstained) != len(frame):
        raise ValueError("abstained mask must align with frame")
    current, future = next_observation_pairs(frame)
    selected = abstained[current]
    return current[selected], future[selected]


def next_observation(
    frame: pd.DataFrame,
    current_hour: int | float,
    *,
    patient_id: object | None = None,
) -> pd.Series | None:
    """Return the next strictly future row without exposing its label."""
    required = {"ICULOS"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows = frame.loc[frame["ICULOS"] > current_hour].sort_values("ICULOS")
    if patient_id is not None and "patient_id" in rows:
        rows = rows.loc[rows["patient_id"] == patient_id]
    if rows.empty:
        return None
    result = rows.iloc[0].copy()
    if "SepsisLabel" in result.index:
        result.pop("SepsisLabel")
    return result


def wait_recheck_action(
    frame: pd.DataFrame,
    current_hour: int | float,
    *,
    patient_id: object | None = None,
) -> dict[str, object]:
    """Describe a wait action using only the next available observation time."""
    row = next_observation(frame, current_hour, patient_id=patient_id)
    if row is None:
        return {"action": "abstain", "reason": "no_future_observation"}
    return {
        "action": "wait_recheck",
        "current_hour": current_hour,
        "next_hour": row["ICULOS"],
        "patient_id": row.get("patient_id", patient_id),
    }
