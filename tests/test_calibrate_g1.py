
import unittest

import numpy as np
import pandas as pd

from src.uncertainty.platt import fit_platt, apply_platt
from src.uncertainty.calibrate_g1 import (
    load_calibration_data,
    build_calibration_states,
)


class TestG1Calibration(unittest.TestCase):

    def setUp(self):
        self.fit = pd.DataFrame({"patient_id": ["p1"]})
        self.cal = pd.DataFrame({"patient_id": ["p2", "p3"]})
        self.policy = pd.DataFrame({"patient_id": ["p4"]})

        self.features = [
            f"{lab}_{suffix}"
            for lab in (
                "Lactate", "WBC", "Creatinine",
                "Platelets", "BUN", "Glucose"
            )
            for suffix in ("last", "observed", "age")
        ] + ["ICULOS"]

        rows = []
        for pid, label in [("p2", 0), ("p3", 1)]:
            row = {
                "patient_id": pid,
                "ICULOS": 6,
                "SepsisLabel": label,
            }
            for lab in (
                "Lactate", "WBC", "Creatinine",
                "Platelets", "BUN", "Glucose"
            ):
                row[f"{lab}_last"] = 1.0
                row[f"{lab}_observed"] = 1
                row[f"{lab}_age"] = 0.0
            rows.append(row)

        self.data = pd.DataFrame(rows)

    def test_disjoint_calibration_cohort(self):
        selected = load_calibration_data(
            self.data, self.cal, self.fit, self.policy
        )
        self.assertEqual(len(selected), 2)

    def test_rejects_overlap(self):
        bad_fit = pd.DataFrame({"patient_id": ["p2"]})
        with self.assertRaises(ValueError):
            load_calibration_data(
                self.data, self.cal, bad_fit, self.policy
            )

    def test_rejects_missing_patients(self):
        bad_cal = pd.DataFrame({"patient_id": ["p2", "missing"]})
        with self.assertRaises(ValueError):
            load_calibration_data(
                self.data, bad_cal, self.fit, self.policy
            )

    def test_calibration_states_are_reproducible(self):
        a, ya = build_calibration_states(
            self.data, self.features, seed=42
        )
        b, yb = build_calibration_states(
            self.data, self.features, seed=42
        )

        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(ya, yb)
        self.assertEqual(len(a), 4)
        self.assertEqual(ya.tolist(), [0, 1, 0, 1])

    def test_platt_probability_bounds(self):
        raw = np.array([0.02, 0.08, 0.4, 0.9])
        labels = np.array([0, 0, 1, 1])

        calibrator = fit_platt(raw, labels)
        probabilities = apply_platt(calibrator, raw)

        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) &
                         (probabilities <= 1)).all())


if __name__ == "__main__":
    unittest.main()
