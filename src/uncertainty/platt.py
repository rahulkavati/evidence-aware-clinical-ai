
"""Platt (sigmoid) calibration on log-odds."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(
    raw_probabilities: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    y = np.asarray(y, dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError("Calibration labels need both classes")

    model = LogisticRegression(max_iter=1000)
    model.fit(logit(raw_probabilities).reshape(-1, 1), y)
    return model


def apply_platt(
    calibrator,
    raw_probabilities: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(
        logit(raw_probabilities).reshape(-1, 1)
    )[:, 1]
