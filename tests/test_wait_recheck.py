import pandas as pd

import numpy as np

from src.evaluation.wait_recheck import (
    next_observation,
    next_observation_pairs,
    wait_recheck_action,
    wait_recheck_candidates,
)


def test_wait_recheck_uses_next_future_row_without_label():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1"],
        "ICULOS": [1, 3, 6],
        "HR": [80, 82, 84],
        "SepsisLabel": [0, 0, 1],
    })
    row = next_observation(frame, 1, patient_id="p1")
    assert row["ICULOS"] == 3
    assert "SepsisLabel" not in row.index
    assert wait_recheck_action(frame, 1, patient_id="p1")["action"] == "wait_recheck"


def test_wait_recheck_abstains_when_no_future_row_exists():
    frame = pd.DataFrame({"ICULOS": [6], "SepsisLabel": [0]})
    assert wait_recheck_action(frame, 6)["action"] == "abstain"


def test_wait_candidates_only_include_abstained_rows():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "ICULOS": [1, 3, 2],
        "SepsisLabel": [0, 0, 0],
    })
    current, future = wait_recheck_candidates(frame, np.array([True, False, True]))
    np.testing.assert_array_equal(current, [0])
    np.testing.assert_array_equal(future, [1])


def test_next_observation_pairs_are_strictly_future_and_patient_local():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p2", "p1", "p1"],
        "ICULOS": [3, 1, 1, 6],
    })
    current, future = next_observation_pairs(frame)
    assert list(zip(current, future)) == [(2, 0), (0, 3)]
