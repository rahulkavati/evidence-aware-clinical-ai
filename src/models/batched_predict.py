"""Bounded-batch prediction helpers for native estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def batched_predict_proba(
    model,
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | None = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict class-1 probabilities without one oversized native call."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    names = list(feature_names or getattr(model, "feature_names_in_", frame.columns))
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    outputs = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size][names]
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None:
            margins = np.asarray(booster.inplace_predict(batch, predict_type="margin"))
            outputs.append(1.0 / (1.0 + np.exp(-margins)))
        else:
            probabilities = np.asarray(model.predict_proba(batch))
            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                raise ValueError("model.predict_proba must return two class columns")
            outputs.append(probabilities[:, 1])
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=float)
