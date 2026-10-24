import json
import random
import unittest
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.run_batch_acquisition import (
    acquisition_order,
    checkpoint_label,
    historical_frame,
    historically_available_features,
    initial_hidden_features,
    limit_acquisitions,
    stable_random_seed,
    _make_result_rows,
    _record_group,
    validate_aggregate_output,
)


class TestBatchAcquisition(unittest.TestCase):

    def make_patient(self):
        return pd.DataFrame(
            {
                "ICULOS": [1, 6, 7],
                "SepsisLabel": [0, 0, 1],
                "Lactate": [1.0, np.nan, 99.0],
                "WBC": [np.nan, 7.0, 8.0],
                "Creatinine": [np.nan, np.nan, 2.0],
                "Platelets": [np.nan, np.nan, 100.0],
                "BUN": [np.inf, np.nan, 20.0],
                "Glucose": [np.nan, np.nan, 150.0],
            }
        )

    def test_initial_mask_is_identical_candidate_set(self):
        first_policy_mask = initial_hidden_features()
        second_policy_mask = initial_hidden_features()

        self.assertEqual(first_policy_mask, second_policy_mask)
        self.assertEqual(
            first_policy_mask,
            ("Lactate", "WBC", "Creatinine", "Platelets", "BUN", "Glucose"),
        )

    def test_historical_cutoff_excludes_future_measurements(self):
        patient = self.make_patient()

        history = historical_frame(patient, 6)

        self.assertEqual(history["ICULOS"].max(), 6)
        self.assertEqual(history["Lactate"].iloc[0], 1.0)
        self.assertTrue(history["Lactate"].iloc[1] != history["Lactate"].iloc[1])
        self.assertTrue(history["Creatinine"].isna().all())
        self.assertEqual(
            historically_available_features(history),
            ("Lactate", "WBC"),
        )

    def test_random_selection_is_stable_for_key_and_varies_by_replicate(self):
        features = ["Lactate", "WBC", "Creatinine", "BUN"]
        seed_a = stable_random_seed(42, "A", "patient-1", 12, 0)
        seed_b = stable_random_seed(42, "A", "patient-1", 12, 0)
        seed_c = stable_random_seed(42, "A", "patient-1", 12, 1)

        first = acquisition_order(features, "random", random.Random(seed_a))
        repeated = acquisition_order(features, "random", random.Random(seed_b))
        different = acquisition_order(features, "random", random.Random(seed_c))

        self.assertEqual(first, repeated)
        self.assertEqual(set(first), set(features))
        self.assertNotEqual(first, different)

    def test_budget_limits_acquisition_count(self):
        order = ("Lactate", "WBC", "BUN")

        self.assertEqual(limit_acquisitions(order, 0), ())
        self.assertEqual(limit_acquisitions(order, 2), ("Lactate", "WBC"))
        self.assertEqual(len(limit_acquisitions(order, 10)), len(order))
        with self.assertRaises(ValueError):
            limit_acquisitions(order, -1)

    def test_fixed_order_uses_declared_priority(self):
        order = acquisition_order(
            ["WBC", "BUN", "Lactate", "Creatinine"],
            "fixed_order",
        )

        self.assertEqual(order, ("Lactate", "WBC", "Creatinine", "BUN"))

    def test_empty_eligible_feature_set_produces_no_acquisitions(self):
        order = acquisition_order([], "random")

        self.assertEqual(order, ())
        self.assertEqual(limit_acquisitions(order, 3), ())
        records = defaultdict(
            lambda: {
                "labels": [],
                "probabilities": [],
                "acquired_counts": [],
                "available_counts": [],
                "acquisition_patients": [],
            }
        )
        _record_group(
            records,
            (6, "All", "random", 3, 0),
            label=0,
            probability=0.1,
            acquired_count=0,
            available_count=0,
        )
        rows, _ = _make_result_rows(records)
        self.assertEqual(rows[0]["eligible_patients"], 1)
        self.assertEqual(rows[0]["patients_with_available_candidate_features"], 0)
        self.assertEqual(rows[0]["mean_acquisition_count"], 0.0)
        self.assertEqual(rows[0]["acquisition_coverage"], 0.0)

    def test_checkpoint_label_is_not_replaced_by_later_outcome(self):
        patient = self.make_patient()

        self.assertEqual(checkpoint_label(patient, 6), 0)
        self.assertEqual(checkpoint_label(patient, 7), 1)
        with self.assertRaises(ValueError):
            checkpoint_label(patient, 5)

    def test_aggregate_output_rejects_patient_ids_and_clinical_values(self):
        aggregate = {
            "results": [
                {
                    "checkpoint": 6,
                    "hospital": "A",
                    "eligible_patients": 1,
                    "brier_score": 0.02,
                }
            ]
        }
        encoded = json.dumps(aggregate)
        self.assertNotIn("patient-1", encoded)
        self.assertNotIn("123.456", encoded)
        validate_aggregate_output(
            aggregate,
            forbidden_identifiers=("patient-1",),
            forbidden_raw_values=("123.456",),
        )
        with self.assertRaises(ValueError):
            validate_aggregate_output({"patient_id": "patient-1"})
        with self.assertRaises(ValueError):
            validate_aggregate_output(
                aggregate | {"debug_value": "123.456"},
                forbidden_raw_values=("123.456",),
            )


if __name__ == "__main__":
    unittest.main()
