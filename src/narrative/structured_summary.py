"""Deterministic narrative rendering for structured model outputs.

This layer is deliberately downstream of prediction and policy decisions. It
does not choose an action, change a probability, or infer an outcome.
"""

from __future__ import annotations

from collections.abc import Mapping


def render_decision_summary(decision: Mapping[str, object]) -> str:
    """Render a concise, auditable narrative from an existing decision."""
    required = {"action", "probability", "confidence"}
    missing = required - set(decision)
    if missing:
        raise ValueError(f"Missing structured decision fields: {sorted(missing)}")
    action = str(decision["action"])
    probability = float(decision["probability"])
    confidence = float(decision["confidence"])
    if not 0.0 <= probability <= 1.0 or not 0.0 <= confidence <= 1.0:
        raise ValueError("Probability and confidence must be within [0, 1]")
    detail = str(decision.get("detail", ""))
    suffix = f" {detail}" if detail else ""
    return (
        f"Action: {action}. Estimated sepsis probability: {probability:.3f}; "
        f"confidence: {confidence:.3f}.{suffix}"
    )
