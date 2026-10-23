import unittest

import numpy as np
import pandas as pd

from src.uncertainty.calibrate_models import split_patient_ids
from src.uncertainty.evaluate_calibration_robustness import (
    align_features,
    apply_calibrators,
    bootstrap_row_weights,
    hospital_row_indices,
    patient_cluster_bootstrap_ci,
    reliability_bins,
)


class FakeSigmoidCalibrator:
    def __init__(self):
        self.seen_logits = None

    def predict_proba(self, logits):
        self.seen_logits = logits.copy()
        probabilities = np.array([0.25, 0.5, 0.75])
        return np.column_stack((1.0 - probabilities, probabilities))


class FakeIsotonicCalibrator:
    def __init__(self):
        self.seen_probabilities = None

    def predict(self, probabilities):
        self.seen_probabilities = probabilities.copy()
        return np.array([0.2, 0.4, 0.6])


class TestCalibrationRobustness(unittest.TestCase):

    def test_patient_partition_is_deterministic_stratified_and_disjoint(self):
        labels = pd.Series(
            [0] * 20 + [1] * 10,
            index=[f"patient-{index}" for index in range(30)],
        )

        calibration_a, selection_a = split_patient_ids(labels, random_state=42)
        calibration_b, selection_b = split_patient_ids(labels, random_state=42)

        self.assertEqual(set(calibration_a), set(calibration_b))
        self.assertEqual(set(selection_a), set(selection_b))
        self.assertFalse(set(calibration_a).intersection(selection_a))
        self.assertEqual(len(calibration_a), 15)
        self.assertEqual(len(selection_a), 15)
        self.assertEqual(labels.loc[calibration_a].sum(), 5)
        self.assertEqual(labels.loc[selection_a].sum(), 5)

    def test_feature_alignment_preserves_model_order_and_excludes_metadata(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["id-1"],
                "second": [2.0],
                "hospital": ["A"],
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

    def test_sigmoid_receives_clipped_raw_log_odds_and_isotonic_raw_scores(self):
        sigmoid = FakeSigmoidCalibrator()
        isotonic = FakeIsotonicCalibrator()

        result = apply_calibrators(np.array([0.0, 0.5, 1.0]), sigmoid, isotonic)

        expected = np.log(np.array([1e-7, 0.5, 1 - 1e-7]) / np.array([1 - 1e-7, 0.5, 1e-7]))
        np.testing.assert_allclose(sigmoid.seen_logits[:, 0], expected)
        np.testing.assert_allclose(isotonic.seen_probabilities, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(result["sigmoid"], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(result["isotonic"], [0.2, 0.4, 0.6])

    def test_ece_bins_assign_zero_and_one_to_endpoint_bins(self):
        bins = reliability_bins(
            np.array([0, 0, 0, 1]),
            np.array([0.0, 0.1, 0.2, 1.0]),
            n_bins=10,
        )

        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[2]["count"], 1)
        self.assertEqual(bins[-1]["count"], 1)
        self.assertEqual(sum(item["count"] for item in bins), 4)

    def test_hospital_grouping_returns_only_rows_for_each_site(self):
        frame = pd.DataFrame({"hospital": ["A", "B", "A", "B"]})

        groups = hospital_row_indices(frame)

        np.testing.assert_array_equal(groups["A"], [0, 2])
        np.testing.assert_array_equal(groups["B"], [1, 3])

    def test_bootstrap_weights_retain_patient_clusters(self):
        patient_codes = np.array([0, 0, 1, 2, 2, 2])
        sampled_patients = np.array([0, 0, 2])

        weights = bootstrap_row_weights(patient_codes, sampled_patients, 3)

        np.testing.assert_array_equal(weights, [2, 2, 0, 1, 1, 1])
        intervals = patient_cluster_bootstrap_ci(
            np.array([0, 1, 0, 1, 0, 0]),
            np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.4]),
            patient_codes,
            patient_count=3,
            n_resamples=20,
            random_state=19,
        )
        self.assertLessEqual(
            intervals["brier_score"]["lower_95"],
            intervals["brier_score"]["upper_95"],
        )
        self.assertLessEqual(
            intervals["ece_10_bins"]["lower_95"],
            intervals["ece_10_bins"]["upper_95"],
        )


if __name__ == "__main__":
    unittest.main()
