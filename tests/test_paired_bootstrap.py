import unittest

import numpy as np
import pandas as pd

from experiments.evaluate_acquisition_significance import collect_patient_predictions
from src.evaluation.paired_bootstrap import (
    metric_values,
    paired_bootstrap,
    paired_cluster_bootstrap_rows,
)


def records(ids, labels, probabilities):
    return [
        {"patient_id": patient_id, "label": label, "probability": probability}
        for patient_id, label, probability in zip(ids, labels, probabilities)
    ]


class TestPairedBootstrap(unittest.TestCase):
    def setUp(self):
        self.ids = ["p1", "p2", "p3", "p4"]
        self.labels = [0, 0, 1, 1]
        self.no = records(self.ids, self.labels, [0.1, 0.3, 0.6, 0.8])
        self.acquisition = records(self.ids, self.labels, [0.2, 0.2, 0.7, 0.9])

    def test_identical_patient_alignment_is_required(self):
        result = paired_bootstrap(self.no, self.acquisition, n_resamples=10)
        self.assertEqual(result["patient_count"], 4)
        with self.assertRaises(ValueError):
            paired_bootstrap(self.no, records(["p1", "p2", "p4", "p5"], self.labels, [0.2] * 4))

    def test_paired_resampling_and_duplicate_multiplicity(self):
        result = paired_bootstrap(self.no, self.acquisition, n_resamples=100)
        self.assertEqual(result["valid_bootstrap_replicates"]["brier_improvement"], 100)
        self.assertAlmostEqual(
            result["observed"]["brier_improvement"],
            metric_values(self.labels, [0.1, 0.3, 0.6, 0.8])["brier"]
            - metric_values(self.labels, [0.2, 0.2, 0.7, 0.9])["brier"],
        )
        self.assertNotEqual(
            result["confidence_intervals_95"]["brier_improvement"][0],
            result["confidence_intervals_95"]["brier_improvement"][1],
        )

        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(self.ids), size=len(self.ids))
        self.assertGreater(np.unique(indices).size, 1)
        self.assertTrue((np.bincount(indices, minlength=len(self.ids)) > 1).any())
        sampled_labels = np.asarray(self.labels)[indices]
        sampled_no = np.asarray([row["probability"] for row in self.no])[indices]
        sampled_acquisition = np.asarray(
            [row["probability"] for row in self.acquisition]
        )[indices]
        manual = (
            metric_values(sampled_labels, sampled_no)["brier"]
            - metric_values(sampled_labels, sampled_acquisition)["brier"]
        )
        self.assertAlmostEqual(
            manual,
            metric_values(sampled_labels, sampled_no)["brier"]
            - metric_values(sampled_labels, sampled_acquisition)["brier"],
            places=12,
        )

    def test_missing_or_mismatched_predictions_fail(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.no, self.acquisition[:-1])
        with self.assertRaises(ValueError):
            paired_bootstrap(self.no, records(self.ids, [0, 1, 1, 1], [0.2] * 4))

    def test_single_class_metrics_are_undefined(self):
        single = records(["p1", "p2"], [0, 0], [0.1, 0.2])
        result = paired_bootstrap(single, records(["p1", "p2"], [0, 0], [0.2, 0.3]))
        self.assertIsNone(result["observed"]["auroc"])
        self.assertIsNone(result["observed"]["auprc"])
        self.assertEqual(result["valid_bootstrap_replicates"]["auroc"], 0)
        self.assertEqual(result["valid_bootstrap_replicates"]["auprc"], 0)

    def test_fixed_seed_is_reproducible(self):
        self.assertEqual(
            paired_bootstrap(self.no, self.acquisition, n_resamples=50, seed=42),
            paired_bootstrap(self.no, self.acquisition, n_resamples=50, seed=42),
        )

    def test_collect_patient_predictions_rejects_test_data(self):
        patients = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "hospital": ["A"],
                "file_path": ["/tmp/test_patient.csv"],
            }
        )
        with self.assertRaises(ValueError):
            collect_patient_predictions(patients, model=None, calibrator=None)

    def test_probabilities_are_validated(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.no, records(self.ids, self.labels, [0.2, np.nan, 0.7, 0.9]))

    def test_row_bootstrap_resamples_patient_clusters(self):
        no = records(
            ["p1", "p1", "p2", "p2", "p3", "p3"],
            [0, 0, 0, 0, 1, 1],
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7],
        )
        acquisition = records(
            ["p1", "p1", "p2", "p2", "p3", "p3"],
            [0, 0, 0, 0, 1, 1],
            [0.2, 0.2, 0.2, 0.3, 0.8, 0.9],
        )
        result = paired_cluster_bootstrap_rows(no, acquisition, n_resamples=50)
        self.assertEqual(result["row_count"], 6)
        self.assertEqual(result["patient_count"], 3)
        self.assertEqual(result["valid_bootstrap_replicates"]["brier_improvement"], 50)

    def test_row_bootstrap_requires_row_alignment(self):
        no = records(["p1", "p1"], [0, 1], [0.1, 0.8])
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap_rows(
                no, records(["p1", "p2"], [0, 1], [0.2, 0.7]), n_resamples=10
            )


if __name__ == "__main__":
    unittest.main()
