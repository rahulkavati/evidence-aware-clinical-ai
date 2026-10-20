import unittest

import numpy as np
import pandas as pd

from src.data.audit_acquisition_feasibility import (
    LAB_FEATURES,
    VITAL_FEATURES,
    checkpoint_patient_metrics,
)


def patient_frame(rows):
    columns = [*LAB_FEATURES, *VITAL_FEATURES, "ICULOS", "SepsisLabel"]
    return pd.DataFrame(rows, columns=columns)


class CheckpointPatientMetricsTests(unittest.TestCase):
    def test_later_row_cannot_make_checkpoint_eligible_or_add_labs(self):
        row_width = len(LAB_FEATURES) + len(VITAL_FEATURES)
        frame = patient_frame(
            [
                [np.nan] * row_width + [5, 0],
                [np.nan] * row_width + [6, 0],
                [10.0] + [np.nan] * (row_width - 1) + [7, 1],
            ]
        )

        at_hour_6 = checkpoint_patient_metrics(frame, 6)
        at_hour_7 = checkpoint_patient_metrics(frame, 7)

        self.assertTrue(at_hour_6["eligible"])
        self.assertEqual(at_hour_6["observed_lab_features"], 0)
        self.assertTrue(at_hour_7["eligible"])
        self.assertEqual(at_hour_7["observed_lab_features"], 1)

    def test_missing_and_non_finite_values_are_not_observed(self):
        row_width = len(LAB_FEATURES) + len(VITAL_FEATURES)
        row = [np.nan] * row_width + [6, 0]
        row[0] = np.inf
        row[1] = 0.0
        metrics = checkpoint_patient_metrics(patient_frame([row]), 6)

        self.assertTrue(metrics["eligible"])
        self.assertEqual(metrics["observed_lab_features"], 1)
        self.assertEqual(metrics["lab_maskable_cells"], 1)

    def test_patient_without_checkpoint_observation_is_ineligible(self):
        row_width = len(LAB_FEATURES) + len(VITAL_FEATURES)
        rows = [
            [np.nan] * row_width + [hour, 0]
            for hour in (1, 2, 4, 5, 7)
        ]

        self.assertFalse(checkpoint_patient_metrics(patient_frame(rows), 6)["eligible"])

    def test_counts_distinct_laboratory_features_not_observation_rows(self):
        row_width = len(LAB_FEATURES) + len(VITAL_FEATURES)
        first = [np.nan] * row_width + [5, 0]
        second = [np.nan] * row_width + [6, 0]
        first[0] = 1.2
        second[0] = 1.2
        second[1] = 3.4
        metrics = checkpoint_patient_metrics(patient_frame([first, second]), 6)

        self.assertEqual(metrics["observed_lab_features"], 2)
        self.assertEqual(metrics["lab_maskable_cells"], 3)


if __name__ == "__main__":
    unittest.main()