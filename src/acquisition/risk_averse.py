"""Risk-sensitive acquisition selection primitives.

This module implements the decision core needed for a CVaR-style active-
sensing comparator.  It is deliberately model-agnostic: callers provide the
candidate-specific loss scenarios and acquisition costs.  The implementation
is therefore useful for a reproducible protocol comparison without claiming
to reproduce the full recurrent policy from the published paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def cvar(losses: Sequence[float], alpha: float = 0.9) -> float:
    """Return the upper-tail conditional value at risk of ``losses``.

    ``alpha`` is the confidence level.  With finite retrospective scenarios,
    the tail is linearly weighted at the boundary so the result remains
    deterministic when the number of scenarios is small.
    """
    values = np.sort(np.asarray(losses, dtype=float))
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("losses must be a non-empty finite one-dimensional sequence")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if alpha == 0.0:
        return float(values.mean())

    tail_mass = (1.0 - alpha) * len(values)
    start = int(np.floor(alpha * len(values)))
    # Each empirical scenario has unit mass here.  The first retained
    # scenario contributes only the fraction remaining after alpha*n.
    boundary_weight = (start + 1) - alpha * len(values)
    tail = values[start:].copy()
    tail[0] *= boundary_weight
    return float(tail.sum() / tail_mass)


def choose_risk_averse(
    candidate_losses: Mapping[str, Sequence[float]],
    costs: Mapping[str, float] | None = None,
    *,
    alpha: float = 0.9,
    cost_weight: float = 0.0,
) -> str | None:
    """Choose the eligible candidate with the lowest risk-adjusted loss.

    The objective is ``CVaR_alpha(loss) + cost_weight * cost``.  Ties are
    resolved lexicographically for reproducibility.  This selector must only
    receive scenarios available to the policy at decision time; it does not
    inspect labels or post-acquisition held-out outcomes.
    """
    if not candidate_losses:
        return None
    costs = costs or {}
    ranked = []
    for name, losses in candidate_losses.items():
        cost = float(costs.get(name, 1.0))
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("candidate costs must be finite and non-negative")
        objective = cvar(losses, alpha=alpha) + cost_weight * cost
        ranked.append((objective, str(name)))
    return min(ranked, key=lambda item: (item[0], item[1]))[1]
