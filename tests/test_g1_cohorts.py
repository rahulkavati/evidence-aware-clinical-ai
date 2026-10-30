import unittest
import pandas as pd
from src.data.create_g1_cohorts import make_cohorts, fingerprint


class TestG1Cohorts(unittest.TestCase):
    def setUp(self):
        rows = [{"patient_id": f"P{i}", "hospital": "A" if i % 2 else "B",
                 "sepsis_positive": (i // 2) % 2} for i in range(200)]
        self.train = pd.DataFrame(rows[:160])
        self.val = pd.DataFrame(rows[160:180])
        self.test = pd.DataFrame(rows[180:])

    def test_disjoint_and_complete(self):
        result = make_cohorts(self.train, self.val, self.test)
        ids = [set(f.patient_id) for f in result.values()]
        self.assertEqual(set.union(*ids), set(self.train.patient_id))
        self.assertTrue(all(not ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))))
        self.assertTrue(all(not s & set(self.val.patient_id) for s in ids))
        self.assertTrue(all(not s & set(self.test.patient_id) for s in ids))

    def test_reproducible(self):
        a = make_cohorts(self.train, self.val, self.test)
        b = make_cohorts(self.train, self.val, self.test)
        self.assertEqual({k: fingerprint(v.patient_id) for k, v in a.items()},
                         {k: fingerprint(v.patient_id) for k, v in b.items()})

    def test_original_overlap_rejected(self):
        val = self.val.copy()
        val.loc[val.index[0], "patient_id"] = self.train.patient_id.iloc[0]
        with self.assertRaises(ValueError):
            make_cohorts(self.train, val, self.test)

    def test_duplicate_rejected(self):
        train = self.train.copy()
        train.loc[train.index[1], "patient_id"] = train.patient_id.iloc[0]
        with self.assertRaises(ValueError):
            make_cohorts(train, self.val, self.test)


if __name__ == "__main__":
    unittest.main()
