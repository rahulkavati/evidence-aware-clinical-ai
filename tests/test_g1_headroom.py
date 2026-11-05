import unittest
import numpy as np
import pandas as pd

from src.evaluation.g1_headroom import (
    cluster_bootstrap_differences, checkpoint_rows, state_definitions,
)
from src.data.g1_masking import apply_hidden_set
from experiments.run_g1_gate import select_development_patients


class TestG1Headroom(unittest.TestCase):
    def test_states(self):
        states = state_definitions()
        self.assertEqual(len(states), 8)
        self.assertEqual(len(states['all_hidden']), 6)
        self.assertEqual(states['none_hidden'], ())
        self.assertEqual(len(states['reveal_WBC']), 5)

    def test_mask_triplet_and_preserve_original(self):
        frame = pd.DataFrame({'WBC_last': [8.0], 'WBC_observed': [1],
                              'WBC_age': [2.0], 'ICULOS': [12]})
        masked = apply_hidden_set(frame, ['WBC'])
        self.assertTrue(np.isnan(masked.loc[0, 'WBC_last']))
        self.assertEqual(masked.loc[0, 'WBC_observed'], 0)
        self.assertTrue(np.isnan(masked.loc[0, 'WBC_age']))
        self.assertEqual(frame.loc[0, 'WBC_last'], 8.0)

    def test_checkpoint_unique(self):
        frame = pd.DataFrame({'patient_id': ['p1', 'p1'], 'ICULOS': [6, 6]})
        with self.assertRaises(ValueError):
            checkpoint_rows(frame)

    def test_paired_bootstrap_zero_difference(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.array([.1, .8, .2, .7, .3, .9])
        ids = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        result = cluster_bootstrap_differences(y, p, {'same': p.copy()}, ids, 30, 42)
        self.assertAlmostEqual(result['differences']['same']['neg_brier']['observed'], 0)
        self.assertEqual(result['differences']['same']['neg_brier']['ci95'], [0, 0])

    def test_sample_keeps_patient_rows_together(self):
        frame = pd.DataFrame({'patient_id': ['a', 'a', 'b', 'b', 'c'],
                              'ICULOS': [6, 12, 6, 12, 6]})
        sampled = select_development_patients(frame, 2, 42)
        self.assertEqual(sampled['patient_id'].nunique(), 2)
        for pid in sampled['patient_id'].unique():
            self.assertEqual((sampled['patient_id'] == pid).sum(),
                             (frame['patient_id'] == pid).sum())


if __name__ == '__main__':
    unittest.main()
