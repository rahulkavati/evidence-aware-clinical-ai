"""Future sepsis targets for multi-horizon urgency experiments."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_horizon_targets(
    frame: pd.DataFrame,
    horizons: Sequence[int] = (6, 12, 24),
) -> pd.DataFrame:
    """Add targets for a positive label strictly after the current hour.

    The target is a future outcome for supervised training/evaluation. It is
    intentionally not suitable as a model input or policy-state feature.
    """
    required = {"patient_id", "ICULOS", "SepsisLabel"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if any(int(h) <= 0 for h in horizons):
        raise ValueError("Horizons must be positive")

    result = frame.copy()
    result["SepsisLabel"] = result["SepsisLabel"].astype(int)
    grouped = result.groupby("patient_id", sort=False)
    for horizon in horizons:
        name = f"SepsisWithin{int(horizon)}h"
        values = pd.Series(0, index=result.index, dtype="int8")
        for _, patient in grouped:
            hours = patient["ICULOS"].to_numpy()
            labels = patient["SepsisLabel"].to_numpy(dtype=int)
            for index, hour in zip(patient.index, hours):
                future = (hours > hour) & (hours <= hour + int(horizon))
                values.loc[index] = int(labels[future].any())
        result[name] = values
    return result


def future_onset_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows eligible for future-onset evaluation.

    Rows already carrying ``SepsisLabel == 1`` cannot represent a future
    onset prediction, so they are excluded from horizon analyses. The input
    is copied and its ordering is preserved.
    """
    required = {"SepsisLabel"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    labels = frame["SepsisLabel"].astype(int)
    if not labels.isin((0, 1)).all():
        raise ValueError("SepsisLabel must contain only 0 and 1")
    return frame.loc[labels == 0].copy()
