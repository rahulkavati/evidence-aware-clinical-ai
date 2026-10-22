import unittest

import numpy as np
import pandas as pd

from src.uncertainty.evaluate_calibration import (
    align_features,
    expected_calibration_error,
    selective_risk_curve,
)


class TestCalibrationEvaluation(unittest.TestCase):

    def test_ece_uses_equal_width_bins_and_weights_by_frequency(self):
        ece, bins = expected_calibration_error(
            np.array([0, 1, 1]),
            np.array([0.05, 0.15, 0.95]),
            n_bins=2,
        )

        self.assertAlmostEqual(ece, (2 / 3) * 0.4 + (1 / 3) * 0.05)
        self.assertEqual(
            [(item["lower"], item["upper"]) for item in bins],
            [(0.0, 0.5), (0.5, 1.0)],
        )
        self.assertEqual([item["count"] for item in bins], [2, 1])

    def test_ece_bin_boundaries_include_zero_and_one(self):
        _, bins = expected_calibration_error(
            np.array([0, 0, 1, 1]),
            np.array([0.0, 0.1, 0.9, 1.0]),
            n_bins=10,
        )

        self.assertEqual([item["count"] for item in bins], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(bins[2]["mean_predicted_probability"], None)

    def test_selective_risk_uses_error_and_tied_uncertainty_groups(self):
        coverage, risk = selective_risk_curve(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.5, 0.5]),
        )

        np.testing.assert_allclose(coverage, [0.5, 1.0])
        np.testing.assert_allclose(risk, [0.0, 0.25])

    def test_feature_alignment_uses_saved_model_order_and_excludes_metadata(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["id-1"],
                "second": [2.0],
                "hospital": [1],
                "SepsisLabel": [0],
                "first": [1.0],
            }
        )

        aligned = align_features(frame, ["first", "second"])

        self.assertEqual(aligned.columns.tolist(), ["first", "second"])
        self.assertEqual(aligned.iloc[0].tolist(), [1.0, 2.0])
        for excluded in ("patient_id", "hospital", "SepsisLabel"):
            with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                align_features(frame, ["first", excluded])


if __name__ == "__main__":
    unittest.main()
