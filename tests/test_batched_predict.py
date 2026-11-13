import numpy as np
import pandas as pd
import pytest

from src.models.batched_predict import batched_predict_proba


class CountingModel:
    feature_names_in_ = np.array(["x"])

    def __init__(self):
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(len(frame))
        values = frame["x"].to_numpy(dtype=float)
        return np.column_stack([1.0 - values, values])


def test_predictions_are_batched_and_reassembled_in_order():
    model = CountingModel()
    result = batched_predict_proba(model, pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5]}), batch_size=2)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert model.calls == [2, 2, 1]


def test_batched_prediction_validates_inputs():
    with pytest.raises(ValueError):
        batched_predict_proba(CountingModel(), pd.DataFrame({"x": [0.1]}), batch_size=0)
