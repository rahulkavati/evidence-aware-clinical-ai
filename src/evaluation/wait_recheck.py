"""Leakage-safe wait/re-check action helpers."""

from __future__ import annotations

import pandas as pd


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
