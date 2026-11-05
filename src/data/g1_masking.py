"""Vectorised laboratory masking for the mask-aware prediction model.

Semantics are identical to ``mask_checkpoint_feature_vector`` in
``src.data.build_temporal_features``: hiding a laboratory feature sets its
``_last`` and ``_age`` columns to NaN and its ``_observed`` column to 0.
Hiding a feature that was never observed is therefore a no-op.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.build_temporal_features import LAB_FEATURES

CANDIDATE_LABS: tuple[str, ...] = (
    "Lactate",
    "WBC",
    "Creatinine",
    "Platelets",
    "BUN",
    "Glucose",
)


def triplet_columns(feature: str) -> tuple[str, str, str]:
    """Return the (last, observed, age) model columns for one lab."""
    return f"{feature}_last", f"{feature}_observed", f"{feature}_age"


def _validate_labs(X: pd.DataFrame, labs: Iterable[str]) -> list[str]:
    labs = list(labs)
    invalid = set(labs) - set(LAB_FEATURES)
    if invalid:
        raise ValueError(f"Not laboratory features: {sorted(invalid)}")
    if len(set(labs)) != len(labs):
        raise ValueError("Duplicate laboratory names")
    for lab in labs:
        missing = [c for c in triplet_columns(lab) if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for {lab}: {missing}")
    return labs


def apply_hidden_set(X: pd.DataFrame, hidden: Iterable[str]) -> pd.DataFrame:
    """Hide the same set of labs in every row. Returns a copy."""
    labs = _validate_labs(X, hidden)
    out = X.copy()
    for lab in labs:
        last, observed, age = triplet_columns(lab)
        out[last] = np.nan
        out[observed] = 0
        out[age] = np.nan
    return out


def sample_hidden_matrix(
    n_rows: int,
    n_labs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a boolean (n_rows, n_labs) hide matrix.

    The number of hidden labs per row is uniform on {0..n_labs}, then the
    hidden subset is uniformly random given its size. This guarantees that
    the evaluation states (all hidden, one revealed, none hidden) each get
    real coverage during training; independent Bernoulli(0.5) hiding would
    show the all-hidden state in only 1/64 of rows.
    """
    sizes = rng.integers(0, n_labs + 1, size=n_rows)
    ranks = rng.random((n_rows, n_labs)).argsort(axis=1).argsort(axis=1)
    return ranks < sizes[:, None]


def apply_hidden_matrix(
    X: pd.DataFrame,
    labs: Sequence[str],
    hidden: np.ndarray,
) -> pd.DataFrame:
    """Hide lab ``labs[j]`` in the rows where ``hidden[:, j]`` is True."""
    labs = _validate_labs(X, labs)
    hidden = np.asarray(hidden, dtype=bool)
    if hidden.shape != (len(X), len(labs)):
        raise ValueError("Hide matrix shape must be (n_rows, n_labs)")
    out = X.copy()
    for j, lab in enumerate(labs):
        mask = hidden[:, j]
        if not mask.any():
            continue
        last, observed, age = triplet_columns(lab)
        out.loc[mask, last] = np.nan
        out.loc[mask, observed] = 0
        out.loc[mask, age] = np.nan
    return out


def random_hide(
    X: pd.DataFrame,
    labs: Sequence[str] = CANDIDATE_LABS,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly hide candidate labs per row (see ``sample_hidden_matrix``)."""
    rng = np.random.default_rng(seed)
    matrix = sample_hidden_matrix(len(X), len(labs), rng)
    return apply_hidden_matrix(X, labs, matrix), matrix
