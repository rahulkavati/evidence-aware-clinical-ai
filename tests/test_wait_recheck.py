import pandas as pd

from src.evaluation.wait_recheck import next_observation, wait_recheck_action


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
