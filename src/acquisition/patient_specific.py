"""Patient-specific acquisition recommendations and policy comparators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def recommendation_score(
    current_probability: float,
    possible_probabilities: Sequence[float],
    likelihoods: Sequence[float],
    decision_threshold: float = 0.5,
) -> float:
    """Expected probability movement across a decision threshold.

    ``possible_probabilities`` represent model probabilities under possible
    results of one test, and ``likelihoods`` represent the predicted result
    probabilities. This score is a recommendation signal, not a causal
    estimate of clinical utility.
    """
    values = np.asarray(possible_probabilities, dtype=float)
    weights = np.asarray(likelihoods, dtype=float)
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("Possible probabilities and likelihoods must align")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Likelihoods must sum to one")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("Decision threshold must be within [0, 1]")
    current_side = current_probability >= decision_threshold
    crosses = (values >= decision_threshold) != current_side
    movement = np.abs(values - current_probability)
    return float(np.sum(weights[crosses] * movement[crosses]))


def rank_recommendations(
    current_probability: float,
    candidates: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    decision_threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """Return candidates ranked by patient-specific expected crossing score."""
    scores = [
        (
            name,
            recommendation_score(current_probability, values, likelihoods, decision_threshold),
        )
        for name, (values, likelihoods) in candidates.items()
    ]
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def choose_fixed_order(candidates: Sequence[str], priority: Sequence[str]) -> str | None:
    """Choose the first available candidate in a fixed priority list."""
    available = set(candidates)
    return next((name for name in priority if name in available), None)


def choose_random(candidates: Sequence[str], seed: int = 42) -> str | None:
    """Choose an available candidate reproducibly at random."""
    if not candidates:
        return None
    rng = np.random.default_rng(seed)
    return str(rng.choice(np.asarray(list(candidates), dtype=str)))


def choose_oracle(
    candidates: Mapping[str, float],
) -> str | None:
    """Choose the candidate with the largest observed post-acquisition gain."""
    if not candidates:
        return None
    return max(candidates, key=lambda name: (candidates[name], name))


def selector_features(
    frame: pd.DataFrame,
    all_labs: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Build features available before acquiring any laboratory group."""
    lab_columns = {
        f"{lab}_{suffix}"
        for lab in all_labs
        for suffix in ("last", "observed", "age")
    }
    excluded = lab_columns | {"patient_id", "hospital", "SepsisLabel"}
    columns = [column for column in frame.columns if column not in excluded]
    result = frame[columns].copy()
    for name, labs in groups.items():
        observed_columns = [f"{lab}_observed" for lab in labs]
        missing = set(observed_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing availability columns for {name}: {sorted(missing)}")
        result[f"available_{name}"] = np.column_stack([
            frame[column].fillna(0).to_numpy() == 1 for column in observed_columns
        ]).any(axis=1).astype(float)
    return result.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def choose_patient_specific(
    scores: np.ndarray,
    eligible: np.ndarray,
    abstained: np.ndarray,
    group_names: Sequence[str],
) -> np.ndarray:
    """Choose the highest predicted crossing-probability eligible group."""
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    abstained = np.asarray(abstained, dtype=bool)
    if scores.ndim != 2 or eligible.shape != scores.shape:
        raise ValueError("Scores and eligibility must have shape (rows, groups)")
    if len(abstained) != len(scores) or len(group_names) != scores.shape[1]:
        raise ValueError("Patient-specific policy inputs are misaligned")
    selected = np.full(len(scores), -1, dtype=int)
    for row in range(len(scores)):
        choices = np.flatnonzero(eligible[row] & abstained[row])
        if len(choices):
            selected[row] = choices[int(np.argmax(scores[row, choices]))]
    return selected


class PatientSpecificPolicy:
    """Training-derived pre-acquisition group recommendation policy.

    Training targets are generated from post-acquisition model behavior, but
    inference uses only non-laboratory features and historical availability.
    """

    def __init__(self, group_names, models, priors, feature_columns):
        self.group_names = tuple(group_names)
        self.models = models
        self.priors = np.asarray(priors, dtype=float)
        self.feature_columns = list(feature_columns)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        base_probabilities: np.ndarray,
        group_probabilities: Mapping[str, np.ndarray],
        groups: Mapping[str, Sequence[str]],
        all_labs: Sequence[str],
        decision_threshold: float = 0.5,
    ) -> "PatientSpecificPolicy":
        features = selector_features(frame, all_labs, groups)
        base = np.asarray(base_probabilities, dtype=float)
        if len(base) != len(frame):
            raise ValueError("Base probabilities must align with frame")
        models = []
        priors = []
        for name in groups:
            candidate = np.asarray(group_probabilities[name], dtype=float)
            if len(candidate) != len(base):
                raise ValueError(f"Group probabilities do not align for {name}")
            targets = ((candidate >= decision_threshold) != (base >= decision_threshold)).astype(int)
            priors.append(float(targets.mean()))
            if len(np.unique(targets)) < 2:
                models.append(None)
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(features, targets)
                models.append(model)
        return cls(tuple(groups), models, priors, list(features.columns))

    def score(self, frame: pd.DataFrame, groups: Mapping[str, Sequence[str]], all_labs: Sequence[str]) -> np.ndarray:
        features = selector_features(frame, all_labs, groups)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)
        scores = np.zeros((len(frame), len(self.group_names)), dtype=float)
        for index, (model, prior) in enumerate(zip(self.models, self.priors)):
            scores[:, index] = prior if model is None else model.predict_proba(features)[:, 1]
        return scores
