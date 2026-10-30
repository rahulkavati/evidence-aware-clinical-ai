"""Tests for G1.3 patient isolation and mask-aware augmentation."""
import unittest

import numpy as np
import pandas as pd

from src.models.train_mask_aware_xgboost import (
    CANDIDATE_LABS,
    build_training_examples,
    mask_training_matrix,
    select_predictor_rows,
)


class TestMaskAwareXGBoost(unittest.TestCase):
    def setUp(self):
        self.features = {'ICULOS': [6, 12, 6, 12]}
        for lab in CANDIDATE_LABS:
            self.features[f'{lab}_last'] = [1., 2., 3., 4.]
            self.features[f'{lab}_observed'] = [1., 1., 1., 1.]
            self.features[f'{lab}_age'] = [0., 1., 0., 1.]
        self.X = pd.DataFrame(self.features)
        self.ids = pd.Series(['a', 'a', 'b', 'b'])

    def test_mask_triplets_and_preserve_unrelated_features(self):
        result = mask_training_matrix(self.X, self.ids, masked_probability=1.0)
        for lab in CANDIDATE_LABS:
            self.assertTrue(result[f'{lab}_last'].isna().all())
            self.assertTrue(result[f'{lab}_age'].isna().all())
            self.assertTrue((result[f'{lab}_observed'] == 0).all())
        pd.testing.assert_series_equal(result['ICULOS'], self.X['ICULOS'])

    def test_reproducibility_and_original_unchanged(self):
        original = self.X.copy(deep=True)
        a = mask_training_matrix(self.X, self.ids, seed=42)
        b = mask_training_matrix(self.X, self.ids, seed=42)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(self.X, original)

    def test_predictor_cohort_isolation(self):
        data = self.X.copy()
        data['patient_id'] = self.ids
        data['SepsisLabel'] = [0, 0, 1, 1]
        manifest = pd.DataFrame({'patient_id': ['a']})
        selected = select_predictor_rows(data, manifest)
        self.assertEqual(set(selected['patient_id']), {'a'})
        X, y, features = build_training_examples(selected)
        self.assertEqual(len(X), 4)
        self.assertEqual(len(y), 4)
        self.assertNotIn('patient_id', features)
        self.assertNotIn('SepsisLabel', features)

    def test_invalid_mask_probability(self):
        with self.assertRaises(ValueError):
            mask_training_matrix(self.X, self.ids, masked_probability=1.2)


if __name__ == '__main__':
    unittest.main()
