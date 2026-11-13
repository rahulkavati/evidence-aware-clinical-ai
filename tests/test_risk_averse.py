import pytest

from src.acquisition.risk_averse import cvar, choose_risk_averse


def test_cvar_is_mean_at_zero_and_upper_tail_at_high_alpha():
    assert cvar([0.0, 1.0, 2.0, 3.0], alpha=0.0) == pytest.approx(1.5)
    assert cvar([0.0, 1.0, 2.0, 3.0], alpha=0.75) == pytest.approx(3.0)


def test_risk_averse_selection_includes_cost_and_is_deterministic():
    losses = {"cheap": [0.0, 1.0], "safer": [0.2, 0.2]}
    assert choose_risk_averse(losses, costs={"cheap": 0.0, "safer": 0.1}, alpha=0.9) == "safer"
    assert choose_risk_averse({"B": [0.2], "A": [0.2]}) == "A"


def test_risk_averse_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        cvar([], alpha=0.9)
    with pytest.raises(ValueError):
        cvar([1.0], alpha=1.0)
    with pytest.raises(ValueError):
        choose_risk_averse({"x": [0.1]}, costs={"x": -1})
