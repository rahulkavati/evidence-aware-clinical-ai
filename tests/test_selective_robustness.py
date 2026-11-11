import unittest

import numpy as np

from src.evaluation.selective_prediction import patient_cluster_bootstrap_ci


class TestSelectiveRobustness(unittest.TestCase):
    def test_patient_cluster_bootstrap_returns_ordered_intervals(self):
        result = patient_cluster_bootstrap_ci(
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.4, 0.6, 0.8, 0.2]),
            np.array(["a", "a", "b", "b", "c", "c"]),
            threshold=0.6,
            n_resamples=50,
            random_state=42,
        )
        for interval in result.values():
            self.assertLessEqual(interval["lower_95"], interval["upper_95"])


if __name__ == "__main__":
    unittest.main()
