
import unittest

import numpy as np
import pandas as pd

from src.data.build_temporal_features import (
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    build_temporal_features,
)


class TestTemporalFeatures(unittest.TestCase):

    def make_patient(self):
        data = {
            feature: [np.nan] * 4
            for feature in DYNAMIC_FEATURES + STATIC_FEATURES
        }

        data["ICULOS"] = [1, 2, 3, 4]
        data["SepsisLabel"] = [0, 0, 0, 1]
        data["HR"] = [80, np.nan, 100, np.nan]
        data["Lactate"] = [np.nan, 2.1, np.nan, np.nan]
        data["Age"] = [65] * 4

        return pd.DataFrame(data)

    def test_last_observation(self):
        result = build_temporal_features(
            self.make_patient()
        )

        self.assertEqual(
            result["HR_last"].tolist(),
            [80, 80, 100, 100],
        )

    def test_measurement_age(self):
        result = build_temporal_features(
            self.make_patient()
        )

        self.assertEqual(
            result["HR_age"].tolist(),
            [0, 1, 0, 1],
        )

    def test_missing_measurement(self):
        result = build_temporal_features(
            self.make_patient()
        )

        self.assertEqual(
            result["Lactate_observed"].tolist(),
            [0, 1, 1, 1],
        )

        self.assertTrue(
            np.isnan(result["Lactate_last"].iloc[0])
        )

    def test_future_leakage(self):
        patient = self.make_patient()

        original = build_temporal_features(patient)

        patient.loc[3, "HR"] = 200

        modified = build_temporal_features(patient)

        pd.testing.assert_frame_equal(
            original.iloc[:3],
            modified.iloc[:3],
        )


if __name__ == "__main__":
    unittest.main()
