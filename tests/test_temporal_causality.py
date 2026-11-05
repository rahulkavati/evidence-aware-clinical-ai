import unittest
import numpy as np
import pandas as pd
from src.data.build_temporal_features import (
    DYNAMIC_FEATURES, STATIC_FEATURES, build_checkpoint_feature_vector,
)


class TestTemporalCausality(unittest.TestCase):
    def test_future_measurements_do_not_change_checkpoint(self):
        rows = []
        for hour in (1, 6, 12, 24):
            row = {name: np.nan for name in DYNAMIC_FEATURES}
            row.update({name: 0.0 for name in STATIC_FEATURES})
            row.update({'ICULOS': hour, 'SepsisLabel': 0})
            row['HR'] = 70 + hour
            row['WBC'] = 7 + hour if hour in (6, 24) else np.nan
            rows.append(row)
        original = pd.DataFrame(rows)
        altered = original.copy()
        altered.loc[altered['ICULOS'] > 12, ['HR', 'WBC', 'SepsisLabel']] = [999, 999, 1]
        features = ['HR_last', 'HR_observed', 'HR_age',
                    'WBC_last', 'WBC_observed', 'WBC_age', 'ICULOS']
        a = build_checkpoint_feature_vector(original, 12, features)
        b = build_checkpoint_feature_vector(altered, 12, features)
        np.testing.assert_allclose(a, b, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
