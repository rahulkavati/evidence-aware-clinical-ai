import unittest

import numpy as np

from src.evaluation.selective_prediction import (
    confidence,
    evaluate_selective,
    retained_mask,
    select_threshold,
)
from src.evaluation.abstention_acquisition import (
    evaluate_trivial_baselines,
    risk_coverage_curve,
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

    def test_evaluation_reports_positive_capture_and_clinical_metrics(self):
        result = evaluate_selective(
            np.array([0, 1, 1, 0]),
            np.array([0.1, 0.9, 0.6, 0.4]),
            0.8,
        )
        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["retained_positive_count"], 1)
        self.assertEqual(result["abstained_positive_count"], 1)
        self.assertEqual(result["abstained_positive_capture"], 0.5)
        self.assertEqual(result["retained_sensitivity"], 1.0)
        self.assertEqual(result["retained_specificity"], 1.0)

    def test_risk_coverage_curve_keeps_ties_together(self):
        curve = risk_coverage_curve(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.6]),
        )
        self.assertEqual([point["retained_count"] for point in curve], [2, 4])
        self.assertEqual(curve[0]["coverage"], 0.5)

    def test_trivial_baselines_include_negative_and_top_risk_reference(self):
        result = evaluate_trivial_baselines(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.6]),
            abstention_fraction=0.25,
        )
        self.assertEqual(set(result), {
            "predict_negative_everywhere",
            "abstain_highest_risk_fraction",
        })
        self.assertEqual(result["abstain_highest_risk_fraction"]["abstained_count"], 1)

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
