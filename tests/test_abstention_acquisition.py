import unittest

import numpy as np
import pandas as pd

from src.evaluation.abstention_acquisition import decision_metrics, one_step_fixed_order_frame


class TestAbstentionAcquisition(unittest.TestCase):
    def test_fixed_order_reveals_only_first_eligible_lab(self):
        frame = pd.DataFrame(
            {
                "Lactate_last": [1.0, np.nan], "Lactate_observed": [1, 0], "Lactate_age": [1.0, np.nan],
                "WBC_last": [2.0, 3.0], "WBC_observed": [1, 1], "WBC_age": [1.0, 2.0],
            }
        )
        masked, selected = one_step_fixed_order_frame(frame, labs=("Lactate", "WBC"), priority=("Lactate", "WBC"))
        np.testing.assert_array_equal(selected, [0, 1])
        self.assertEqual(masked.loc[0, "Lactate_observed"], 1)
        self.assertEqual(masked.loc[0, "WBC_observed"], 0)
        self.assertEqual(masked.loc[1, "Lactate_observed"], 0)
        self.assertEqual(masked.loc[1, "WBC_observed"], 1)

    def test_decision_metrics_excludes_abstentions(self):
        result = decision_metrics(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.6]), np.array([True, False, True]))
        self.assertEqual(result["retained_count"], 2)
        self.assertEqual(result["abstained_count"], 1)
        self.assertEqual(result["risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
