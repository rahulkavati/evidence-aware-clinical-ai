import pandas as pd

from src.data.multihorizon_labels import add_horizon_targets


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
