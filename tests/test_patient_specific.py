import pytest

from src.acquisition.patient_specific import (
    choose_fixed_order,
    choose_oracle,
    choose_random,
    rank_recommendations,
    recommendation_score,
)


def test_recommendation_score_rewards_expected_threshold_crossing():
    score = recommendation_score(0.40, [0.60, 0.42], [0.5, 0.5])
    assert score == pytest.approx(0.10)


def test_recommendations_rank_patient_specific_candidates():
    ranked = rank_recommendations(
        0.40,
        {"lab_a": ([0.60], [1.0]), "lab_b": ([0.45], [1.0])},
    )
    assert ranked[0] == ("lab_a", pytest.approx(0.20))


def test_fixed_random_and_oracle_policies():
    assert choose_fixed_order(["B", "A"], ["A", "B"]) == "A"
    assert choose_random(["A", "B"], seed=42) in {"A", "B"}
    assert choose_oracle({"A": 0.1, "B": 0.3}) == "B"
