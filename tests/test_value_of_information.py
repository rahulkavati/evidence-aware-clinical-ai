
import unittest

import pandas as pd

from experiments.run_voi_acquisition import (
    _policy_order,
    _validate_validation_manifest,
    split_training_cohorts,
)
from src.acquisition.value_of_information import (
    FeatureVOIPolicy,
    VOIConfig,
)


class TestVOI(unittest.TestCase):

    def test_selects_highest_benefit(self):
        records = (
            [{"feature": "Lactate",
              "loss_before": 0.5,
              "loss_after": 0.2}] * 40
            +
            [{"feature": "WBC",
              "loss_before": 0.5,
              "loss_after": 0.4}] * 40
        )

        policy = FeatureVOIPolicy(
            VOIConfig(shrinkage=0, min_examples=1)
        ).fit(records)

        self.assertEqual(
            policy.select(["WBC", "Lactate"]),
            "Lactate"
        )

    def test_stops_when_benefit_negative(self):
        records = [
            {
                "feature": "WBC",
                "loss_before": 0.1,
                "loss_after": 0.2,
            }
        ]

        policy = FeatureVOIPolicy(
            VOIConfig(shrinkage=0, min_examples=1)
        ).fit(records)

        self.assertIsNone(policy.select(["WBC"]))

    def test_cost_changes_selection(self):
        records = [
            {
                "feature": "Lactate",
                "loss_before": 0.5,
                "loss_after": 0.2,
            },
            {
                "feature": "WBC",
                "loss_before": 0.5,
                "loss_after": 0.3,
            },
        ]

        policy = FeatureVOIPolicy(
            VOIConfig(
                cost_weight=0.2,
                shrinkage=0,
                min_examples=1,
            )
        ).fit(records)

        selected = policy.select(
            ["Lactate", "WBC"],
            costs={"Lactate": 2.0, "WBC": 0.5},
        )

        self.assertEqual(selected, "WBC")

    def test_no_eligible_features(self):
        policy = FeatureVOIPolicy()
        self.assertIsNone(policy.select([]))

    def test_training_policy_cohorts_are_patient_disjoint_and_reproducible(self):
        manifest = pd.DataFrame(
            {
                "patient_id": [f"p{i}" for i in range(10)],
                "hospital": ["A", "B"] * 5,
                "file_path": [f"p{i}.psv" for i in range(10)],
            }
        )
        fit_a, policy_a = split_training_cohorts(manifest, seed=42)
        fit_b, policy_b = split_training_cohorts(manifest, seed=42)
        self.assertEqual(fit_a["patient_id"].tolist(), fit_b["patient_id"].tolist())
        self.assertEqual(policy_a["patient_id"].tolist(), policy_b["patient_id"].tolist())
        self.assertTrue(
            set(fit_a["patient_id"]).isdisjoint(policy_a["patient_id"])
        )

    def test_validation_manifest_rejects_test_data(self):
        manifest = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "hospital": ["A"],
                "file_path": ["src/data/raw/test/p1.psv"],
            }
        )
        with self.assertRaises(ValueError):
            _validate_validation_manifest(manifest)

    def test_policy_order_is_deterministic_and_budget_is_prefix(self):
        policy = FeatureVOIPolicy(
            VOIConfig(shrinkage=0, min_examples=1)
        ).fit(
            [
                {"feature": "WBC", "loss_before": 0.5, "loss_after": 0.4},
                {"feature": "Lactate", "loss_before": 0.5, "loss_after": 0.1},
            ]
        )
        first = _policy_order(
            "voi",
            ["WBC", "Lactate"],
            voi_policy=policy,
            costs={"WBC": 1.0, "Lactate": 1.0},
        )
        second = _policy_order(
            "voi",
            ["Lactate", "WBC"],
            voi_policy=policy,
            costs={"Lactate": 1.0, "WBC": 1.0},
        )
        self.assertEqual(first, second)
        self.assertEqual(first[:1], ("Lactate",))

    def test_policy_order_stops_when_net_benefit_is_nonpositive(self):
        policy = FeatureVOIPolicy(
            VOIConfig(shrinkage=0, min_examples=1)
        ).fit(
            [{"feature": "WBC", "loss_before": 0.1, "loss_after": 0.2}]
        )
        self.assertEqual(
            _policy_order(
                "voi",
                ["WBC"],
                voi_policy=policy,
                costs={"WBC": 1.0},
            ),
            (),
        )

    def test_candidate_order_is_sorted_before_selection(self):
        policy = FeatureVOIPolicy(
            VOIConfig(shrinkage=0, min_examples=1)
        ).fit(
            [{"feature": feature, "loss_before": 0.5, "loss_after": 0.4}
             for feature in ("BUN", "Creatinine", "WBC")]
        )
        self.assertEqual(
            _policy_order(
                "fixed_order",
                ["WBC", "BUN", "Creatinine"],
            ),
            ("WBC", "Creatinine", "BUN"),
        )


if __name__ == "__main__":
    unittest.main()
