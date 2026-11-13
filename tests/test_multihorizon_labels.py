import pandas as pd

from src.data.multihorizon_labels import add_horizon_targets, future_onset_rows


def test_horizon_targets_use_only_strictly_future_labels():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1", "p1"],
        "ICULOS": [1, 4, 8, 20],
        "SepsisLabel": [0, 1, 0, 1],
    })
    result = add_horizon_targets(frame, horizons=(3, 6, 24))
    assert result["SepsisWithin3h"].tolist() == [1, 0, 0, 0]
    assert result["SepsisWithin6h"].tolist() == [1, 0, 0, 0]
    assert result["SepsisWithin24h"].tolist() == [1, 1, 1, 0]


def test_horizon_targets_require_patient_time_and_label_columns():
    try:
        add_horizon_targets(pd.DataFrame({"ICULOS": [1]}))
    except ValueError as error:
        assert "patient_id" in str(error)
    else:
        raise AssertionError("Expected missing-column validation")


def test_future_onset_rows_exclude_currently_positive_rows():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "ICULOS": [1, 2, 1],
        "SepsisLabel": [0, 1, 0],
    })
    result = future_onset_rows(frame)
    assert result["SepsisLabel"].tolist() == [0, 0]
    assert result["ICULOS"].tolist() == [1, 1]


def test_future_onset_rows_validate_binary_labels():
    frame = pd.DataFrame({"SepsisLabel": [0, 2]})
    try:
        future_onset_rows(frame)
    except ValueError as error:
        assert "only 0 and 1" in str(error)
    else:
        raise AssertionError("Expected binary-label validation")
