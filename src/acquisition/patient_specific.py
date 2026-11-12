"""Patient-specific acquisition recommendations and policy comparators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


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
