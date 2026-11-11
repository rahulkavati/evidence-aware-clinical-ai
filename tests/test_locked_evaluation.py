import unittest

import numpy as np
import pandas as pd

from experiments.run_locked_evaluation import evaluate_locked_frame


class FakeModel:
    feature_names_in_ = np.array(["Lactate_last", "Lactate_observed", "Lactate_age", "WBC_last", "WBC_observed", "WBC_age"])

    def predict_proba(self, frame):
        p = np.where(frame["Lactate_observed"].to_numpy() == 1, 0.9, 0.1)
        return np.column_stack((1 - p, p))


class FakeCalibrator:
    def predict_proba(self, logits):
        return np.column_stack((np.full(len(logits), 0.1), np.full(len(logits), 0.9)))


class TestLockedEvaluation(unittest.TestCase):
    def test_locked_frame_reports_both_policies(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["a", "b"], "hospital": ["A", "B"], "ICULOS": [6, 6], "SepsisLabel": [1, 0],
                "Lactate_last": [1.0, 2.0], "Lactate_observed": [1, 1], "Lactate_age": [1.0, 1.0],
                "WBC_last": [1.0, 2.0], "WBC_observed": [1, 1], "WBC_age": [1.0, 1.0],
            }
        )
        for lab in ("Creatinine", "Platelets", "BUN", "Glucose"):
            frame[f"{lab}_last"] = np.nan
            frame[f"{lab}_observed"] = 0
            frame[f"{lab}_age"] = np.nan
        result = evaluate_locked_frame(frame, FakeModel(), FakeCalibrator(), threshold=0.8)
        self.assertIn("pooled_all", result["scopes"])
        self.assertIn("baseline_abstention", result["scopes"]["pooled_all"])
        self.assertIn("acquisition_assisted", result["scopes"]["pooled_all"])


if __name__ == "__main__":
    unittest.main()
