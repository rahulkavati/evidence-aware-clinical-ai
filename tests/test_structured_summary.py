import pytest

from src.narrative.structured_summary import render_decision_summary


def test_language_layer_only_renders_structured_decision():
    text = render_decision_summary({
        "action": "abstain",
        "probability": 0.23,
        "confidence": 0.77,
        "detail": "Additional evidence may be considered.",
    })
    assert text == (
        "Action: abstain. Estimated sepsis probability: 0.230; "
        "confidence: 0.770. Additional evidence may be considered."
    )


def test_language_layer_validates_structured_input():
    with pytest.raises(ValueError):
        render_decision_summary({"action": "predict"})
