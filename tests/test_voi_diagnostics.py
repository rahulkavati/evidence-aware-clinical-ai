import unittest

import pandas as pd

from experiments.run_voi_acquisition import (
    diagnostic_training_cohorts,
    diagnose_policy_records,
    fit_voi_policy,
)
from src.acquisition.value_of_information import VOIConfig


class TestVOIDiagnostics(unittest.TestCase):
    def test_feature_diagnostics_report_raw_and_shrunk_values(self):
        records = [
            {
                "patient_id": f"p{i}",
                "feature": "WBC",
                "loss_before": 0.5,
                "loss_after": 0.4 if i < 2 else 0.6,
            }
            for i in range(4)
        ]
        policy = fit_voi_policy(records)
        policy.config = VOIConfig(shrinkage=2, min_examples=2)
        policy.fit(records)
        row = next(
            item
            for item in diagnose_policy_records(records, policy)
            if item["feature"] == "WBC"
        )
        self.assertEqual(row["training_policy_examples"], 4)
        self.assertAlmostEqual(row["mean_unshrunk_brier_benefit"], 0.0)
        self.assertAlmostEqual(row["std_unshrunk_brier_benefit"], 0.1154700538)
        self.assertAlmostEqual(row["median_unshrunk_brier_benefit"], 0.0)
        self.assertAlmostEqual(row["positive_benefit_fraction"], 0.5)
        self.assertAlmostEqual(row["shrunk_estimated_benefit"], 0.0)
        self.assertFalse(row["min_examples_suppressed"])
        self.assertEqual(row["unshrunk_sign"], "zero")

    def test_min_examples_suppresses_nonzero_mean(self):
        records = [
            {
                "patient_id": f"p{i}",
                "feature": "WBC",
                "loss_before": 0.5,
                "loss_after": 0.4,
            }
            for i in range(2)
        ]
        policy = fit_voi_policy(records)
        row = next(
            item
            for item in diagnose_policy_records(records, policy)
            if item["feature"] == "WBC"
        )
        self.assertTrue(row["min_examples_suppressed"])
        self.assertEqual(row["unshrunk_sign"], "positive")
        self.assertEqual(row["shrunk_estimated_benefit"], 0.0)

    def test_training_size_cohorts_are_disjoint_and_sized(self):
        manifest = pd.DataFrame(
            {
                "patient_id": [f"p{i}" for i in range(20)],
                "hospital": ["A", "B"] * 10,
                "file_path": [f"p{i}.psv" for i in range(20)],
            }
        )
        fit, cohorts = diagnostic_training_cohorts(
            manifest,
            policy_sizes=(2, 4),
            seed=42,
        )
        self.assertEqual(len(fit), 16)
        self.assertEqual(len(cohorts[2]), 2)
        self.assertEqual(len(cohorts[4]), 4)
        for cohort in cohorts.values():
            self.assertTrue(
                set(fit["patient_id"]).isdisjoint(cohort["patient_id"])
            )


if __name__ == "__main__":
    unittest.main()
