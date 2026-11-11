import unittest

import numpy as np
import pandas as pd

from experiments.run_selective_prediction import (
    render_report,
    run_selective_evaluation,
    split_validation_frame,
)


class FakeModel:
    feature_names_in_ = np.array(["signal"])

    def predict_proba(self, frame):
        probabilities = frame["signal"].to_numpy()
        return np.column_stack((1.0 - probabilities, probabilities))


class FakeCalibrator:
    def predict_proba(self, logits):
        probabilities = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return np.column_stack((1.0 - probabilities, probabilities))


class TestSelectivePredictionRunner(unittest.TestCase):
    def make_frame(self):
        rows = []
        for patient in range(20):
            label = int(patient % 2)
            rows.append(
                {
                    "patient_id": f"p{patient}",
                    "hospital": "A" if patient < 10 else "B",
                    "ICULOS": 6,
                    "SepsisLabel": label,
                    "signal": 0.9 if label else 0.1,
                }
            )
        return pd.DataFrame(rows)

    def test_split_is_patient_disjoint(self):
        selection, evaluation = split_validation_frame(self.make_frame(), random_state=42)
        self.assertTrue(set(selection.patient_id).isdisjoint(evaluation.patient_id))
        self.assertEqual(len(selection) + len(evaluation), 20)

    def test_runner_reports_disjoint_evaluation_scopes(self):
        result = run_selective_evaluation(
            self.make_frame(), FakeModel(), FakeCalibrator(), max_risk=0.0, min_coverage=0.5
        )
        self.assertFalse(result["config"]["held_out_test_accessed"])
        self.assertIn("pooled_all", result["evaluation"]["scopes"])
        self.assertIn("hospital_A", result["evaluation"]["scopes"])
        self.assertIn("hour_6", result["evaluation"]["scopes"])
        self.assertIn("Selective prediction evaluation", render_report(result))


if __name__ == "__main__":
    unittest.main()
