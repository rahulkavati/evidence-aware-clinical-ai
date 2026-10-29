import unittest
from experiments.audit_voi_numerics import summarize, patient_concentration

class TestVOINumerics(unittest.TestCase):
    def test_zero_vs_tiny(self):
        result = summarize([0., 1e-100, 1e-5])
        self.assertEqual(result['exact_zero'], 1)
        self.assertEqual(result['abs_le_1e_12'], 2)
        self.assertEqual(result['abs_gt_1e_6'], 1)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            summarize([float('nan')])

    def test_patient_concentration(self):
        records = [{'patient_id':'a','benefit':-3.},
                   {'patient_id':'a','benefit':1.},
                   {'patient_id':'b','benefit':2.}]
        self.assertAlmostEqual(patient_concentration(records)['total_absolute_benefit'], 6.)

    def test_zero_influence(self):
        result = patient_concentration([{'patient_id':'a','benefit':0.}])
        self.assertIsNone(result['top_1pct_share'])

if __name__ == '__main__':
    unittest.main()