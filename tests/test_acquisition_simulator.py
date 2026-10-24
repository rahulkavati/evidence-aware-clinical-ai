
import unittest
import numpy as np
import pandas as pd

from src.data.build_temporal_features import (
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
)
from src.simulation.acquisition_simulator import AcquisitionSimulator
from src.acquisition.baseline_policies import (
    random_policy,
    fixed_order_policy,
)


class FakeModel:
    feature_names_in_ = np.array(["Lactate_last", "ICULOS"])

    def predict_proba(self, X):
        value = X["Lactate_last"].iloc[0]
        risk = 0.1 if pd.isna(value) else min(0.9, value / 10)
        return np.array([[1 - risk, risk]])


class TestAcquisitionSimulator(unittest.TestCase):

    def make_patient(self):
        data = {
            feature: [np.nan] * 3
            for feature in DYNAMIC_FEATURES + STATIC_FEATURES
        }

        data["ICULOS"] = [1, 2, 3]
        data["SepsisLabel"] = [0, 0, 0]
        data["Lactate"] = [np.nan, 2.5, np.nan]
        data["WBC"] = [8.0, np.nan, np.nan]

        return pd.DataFrame(data)

    def test_masked_measurement(self):
        simulator = AcquisitionSimulator(
            self.make_patient(),
            checkpoint=3,
            model=FakeModel(),
            hidden_features=["Lactate"],
        )

        visible = simulator.visible_history()

        self.assertTrue(visible["Lactate"].isna().all())

    def test_acquisition_changes_prediction(self):
        simulator = AcquisitionSimulator(
            self.make_patient(),
            checkpoint=3,
            model=FakeModel(),
            hidden_features=["Lactate"],
        )

        result = simulator.acquire("Lactate")

        self.assertAlmostEqual(result.risk_before, 0.1)
        self.assertAlmostEqual(result.risk_after, 0.25)

    def test_unobserved_feature_not_eligible(self):
        simulator = AcquisitionSimulator(
            self.make_patient(),
            checkpoint=3,
            model=FakeModel(),
            hidden_features=["Fibrinogen"],
        )

        self.assertEqual(simulator.eligible_features(), [])

    def test_random_policy(self):
        result = random_policy(["WBC", "Lactate"])
        self.assertIn(result, ["WBC", "Lactate"])

    def test_fixed_order_policy(self):
        result = fixed_order_policy(["WBC", "Lactate"])
        self.assertEqual(result, "Lactate")


if __name__ == "__main__":
    unittest.main()
