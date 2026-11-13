import numpy as np
import pandas as pd
import pytest

from src.evaluation.physionet_utility import (
    compute_prediction_utility,
    normalized_utility,
    score_frame,
)


def test_reference_patient_utility_matches_official_example():
    assert compute_prediction_utility(
        [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1]
    ) == pytest.approx(3.388888888888889)


def test_no_predictions_normalize_to_zero():
    result = normalized_utility([
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 1, 1], [0, 0, 0, 0]),
    ])
    assert result["normalized_utility"] == pytest.approx(0.0)


def test_score_frame_sorts_rows_within_each_patient():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "ICULOS": [4, 1, 1],
        "SepsisLabel": [1, 0, 0],
    })
    result = score_frame(frame, np.array([1, 0, 0]))
    assert result["patient_count"] == 2
    assert result["normalized_utility"] >= 0.0


def test_invalid_prediction_values_are_rejected():
    with pytest.raises(ValueError):
        compute_prediction_utility([0, 1], [0, 2])
