"""Future sepsis targets for multi-horizon urgency experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
    patient_codes = pd.factorize(result["patient_id"])[0]
    hours_all = result["ICULOS"].to_numpy(dtype=float)
    labels_all = result["SepsisLabel"].to_numpy(dtype=int)
    order = np.lexsort((hours_all, patient_codes))
    codes = patient_codes[order]
    hours = hours_all[order]
    labels = labels_all[order]
    # Prefix sums of positive labels within each patient (in hour order).
    cumulative = np.cumsum(labels)
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    patient_start = np.repeat(starts, np.diff(np.r_[starts, len(codes)]))
    before_start = np.where(patient_start > 0, cumulative[patient_start - 1], 0)
    for horizon in horizons:
        name = f"SepsisWithin{int(horizon)}h"
        # Composite sort key keeps the binary search inside each patient.
        key = codes * 1e7 + hours
        upper = np.searchsorted(key, codes * 1e7 + hours + int(horizon), side="right")
        upper_sum = np.where(upper > 0, cumulative[upper - 1], 0)
        future_positive = upper_sum - cumulative  # strictly after current row
        values = np.zeros(len(result), dtype="int8")
        values[order] = (future_positive > 0).astype("int8")
        result[name] = values
    del before_start
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
