
import unittest
import numpy as np

from src.uncertainty.calibrate_models import expected_calibration_error


class TestCalibration(unittest.TestCase):

    def test_perfect_calibration(self):
        y = np.array([0, 1])
        p = np.array([0.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0)

    def test_boundary_probabilities(self):
        y = np.array([0, 1, 1])
        p = np.array([0.0, 0.5, 1.0])
        result = expected_calibration_error(y, p)
        self.assertTrue(np.isfinite(result))

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([0, 1], [-0.1, 1.1])


if __name__ == "__main__":
    unittest.main()
