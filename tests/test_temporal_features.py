
import unittest

import numpy as np
import pandas as pd

from src.data.build_temporal_features import (
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    build_checkpoint_feature_vector,
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

    def test_direct_checkpoint_vector_matches_temporal_builder(self):
        patient = self.make_patient()
        model_features = [
            column
            for column in build_temporal_features(patient).columns
            if column != "SepsisLabel"
        ]
        checkpoint = 3

        vector = build_checkpoint_feature_vector(
            patient,
            checkpoint,
            model_features,
        )
        expected = (
            build_temporal_features(patient)
            .loc[lambda frame: frame["ICULOS"] == checkpoint, model_features]
            .iloc[0]
            .to_numpy(dtype=float)
        )

        np.testing.assert_allclose(vector, expected, equal_nan=True)

    def test_direct_checkpoint_vector_matches_hidden_observations(self):
        patient = self.make_patient()
        patient["HR"] = [80, np.nan, 100, np.nan]
        model_features = [
            column
            for column in build_temporal_features(patient).columns
            if column != "SepsisLabel"
        ]
        patient["Lactate"] = [1.5, np.nan, 2.5, np.nan]
        hidden_history = patient.copy()
        hidden_history["Lactate"] = np.nan
        expected = (
            build_temporal_features(hidden_history)
            .loc[lambda frame: frame["ICULOS"] == 3, model_features]
            .iloc[0]
            .to_numpy(dtype=float)
        )

        vector = build_checkpoint_feature_vector(
            patient,
            3,
            model_features,
            hidden_features={"Lactate"},
        )

        np.testing.assert_allclose(vector, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
