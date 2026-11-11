import unittest

import numpy as np

from src.evaluation.selective_prediction import (
    confidence,
    evaluate_selective,
    retained_mask,
    select_threshold,
)


class TestSelectivePrediction(unittest.TestCase):
    def test_confidence_is_probability_of_predicted_class(self):
        np.testing.assert_allclose(confidence(np.array([0.1, 0.5, 0.9])), [0.9, 0.5, 0.9])

    def test_retention_keeps_tied_confidence_values_together(self):
        mask = retained_mask(np.array([0.1, 0.9, 0.3, 0.7, 0.4]), 0.7)
        np.testing.assert_array_equal(mask, [True, True, True, True, False])

    def test_evaluation_reports_abstention_and_retained_quality(self):
        result = evaluate_selective(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.6]),
            0.8,
        )
        self.assertEqual(result["retained_count"], 2)
        self.assertEqual(result["abstained_count"], 2)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_risk"], 0.0)

    def test_threshold_selection_maximizes_coverage_under_risk_target(self):
        result = select_threshold(
            np.array([0, 1, 1, 1, 0]),
            np.array([0.1, 0.9, 0.4, 0.6, 0.55]),
            max_risk=0.0,
        )
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["coverage"], 0.4)

    def test_threshold_selection_uses_maximum_feasible_coverage(self):
        result = select_threshold(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.6]),
            max_risk=0.10,
        )
        self.assertEqual(result["threshold"], 0.6)
        self.assertEqual(result["coverage"], 1.0)

    def test_threshold_selection_rejects_impossible_minimum_coverage(self):
        with self.assertRaises(ValueError):
            select_threshold(
                np.array([0, 0]),
                np.array([0.1, 0.9]),
                max_risk=0.0,
                min_coverage=1.0,
            )


if __name__ == "__main__":
    unittest.main()
