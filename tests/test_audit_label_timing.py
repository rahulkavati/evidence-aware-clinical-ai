import unittest

import pandas as pd

from src.data.audit_acquisition_feasibility import CHECKPOINTS
from src.data.audit_label_timing import (
    checkpoint_label_metrics,
    first_positive_label_hour,
)


class FirstPositiveLabelTests(unittest.TestCase):
    def test_returns_earliest_positive_hour_independent_of_row_order(self):
        frame = pd.DataFrame(
            {"ICULOS": [8, 3, 6, 2], "SepsisLabel": [1, 0, 1, 0]}
        )

        self.assertEqual(first_positive_label_hour(frame), 6)

    def test_returns_none_for_never_positive_patient(self):
        frame = pd.DataFrame(
            {"ICULOS": [1, 2, 3], "SepsisLabel": [0, 0, 0]}
        )

        self.assertIsNone(first_positive_label_hour(frame))

    def test_positive_at_checkpoint_is_in_at_or_before_group(self):
        for checkpoint in CHECKPOINTS:
            with self.subTest(checkpoint=checkpoint):
                frame = pd.DataFrame(
                    {"ICULOS": [checkpoint - 1, checkpoint], "SepsisLabel": [0, 1]}
                )

                metrics = checkpoint_label_metrics(frame, checkpoint)

                self.assertTrue(metrics["eligible"])
                self.assertTrue(metrics["first_positive_label_at_or_before_checkpoint"])
                self.assertFalse(metrics["first_positive_label_after_checkpoint"])
                self.assertFalse(metrics["no_positive_label"])

    def test_first_positive_after_checkpoint_is_pre_positive_eligible(self):
        for checkpoint in CHECKPOINTS:
            with self.subTest(checkpoint=checkpoint):
                frame = pd.DataFrame(
                    {"ICULOS": [checkpoint, checkpoint + 1], "SepsisLabel": [0, 1]}
                )

                metrics = checkpoint_label_metrics(frame, checkpoint)

                self.assertTrue(metrics["eligible"])
                self.assertFalse(metrics["first_positive_label_at_or_before_checkpoint"])
                self.assertTrue(metrics["first_positive_label_after_checkpoint"])
                self.assertFalse(metrics["no_positive_label"])

    def test_missing_checkpoint_row_is_ineligible(self):
        frame = pd.DataFrame({"ICULOS": [5, 7], "SepsisLabel": [0, 1]})

        metrics = checkpoint_label_metrics(frame, 6)

        self.assertFalse(metrics["eligible"])
        self.assertFalse(metrics["first_positive_label_after_checkpoint"])

    def test_never_positive_eligible_patient_is_in_no_positive_group(self):
        frame = pd.DataFrame({"ICULOS": [1, 6, 9], "SepsisLabel": [0, 0, 0]})

        metrics = checkpoint_label_metrics(frame, 6)

        self.assertTrue(metrics["eligible"])
        self.assertTrue(metrics["no_positive_label"])
        self.assertFalse(metrics["first_positive_label_at_or_before_checkpoint"])
        self.assertFalse(metrics["first_positive_label_after_checkpoint"])


if __name__ == "__main__":
    unittest.main()