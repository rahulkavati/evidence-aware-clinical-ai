import numpy as np
import pytest

from src.models.sequence_baselines import append_missingness_channels


def test_missingness_channels_are_appended_and_values_are_filled():
    sequence = np.array([[[1.0, np.nan], [np.inf, 4.0]]])
    result = append_missingness_channels(sequence)
    assert result.shape == (1, 2, 4)
    np.testing.assert_array_equal(result[0, :, :2], [[1.0, 0.0], [0.0, 4.0]])
    np.testing.assert_array_equal(result[0, :, 2:], [[1.0, 0.0], [0.0, 1.0]])


def test_sequence_models_report_optional_dependency_when_torch_is_unavailable():
    try:
        import torch  # noqa: F401
    except ImportError:
        from src.models.sequence_baselines import GRUSequenceClassifier
        with pytest.raises(ImportError, match="optional dependency 'torch'"):
            GRUSequenceClassifier(4)
