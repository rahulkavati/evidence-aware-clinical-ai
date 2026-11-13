import pytest
import numpy as np
import pandas as pd

from src.acquisition.patient_specific import (
    choose_fixed_order,
    choose_oracle,
    choose_random,
    rank_recommendations,
    recommendation_score,
    choose_patient_specific,
    selector_features,
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


def test_patient_specific_selection_is_abstention_and_eligibility_limited():
    selected = choose_patient_specific(
        np.array([[0.2, 0.9], [0.8, 0.1]]),
        np.array([[True, True], [True, True]]),
        np.array([True, False]),
        ["A", "B"],
    )
    np.testing.assert_array_equal(selected, [1, -1])


def test_selector_features_exclude_laboratory_values():
    frame = pd.DataFrame({
        "HR_last": [80], "Lactate_last": [4.0], "Lactate_observed": [1],
        "Lactate_age": [0], "ICULOS": [3], "SepsisLabel": [0],
    })
    result = selector_features(frame, ["Lactate"], {"perfusion": ["Lactate"]})
    assert "Lactate_last" not in result.columns
    assert "available_perfusion" in result.columns
