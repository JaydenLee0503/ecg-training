import copy
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c
import balanced
import numpy as np
from attention_ecg.data import FitStandardizer


class BalancingTests(unittest.TestCase):
    def fixture(self):
        return dict(X_fit=np.arange(16).reshape(8, 2), y_fit=np.array([0]*6+[1]*2),
            fit_record_ids=np.array([f'r{i}' for i in range(8)]), fit_patients=np.array([f'p{i}' for i in range(8)]),
            X_validation=np.array([[100, 101], [102, 103]]), y_validation=np.array([0, 1]),
            validation_record_ids=np.array(['v0', 'v1']), validation_patients=np.array(['q0', 'q1']))

    def test_exact_balance_retains_originals_and_shared_order(self):
        data = self.fixture()
        original = copy.deepcopy(data)
        idx = c.oversample_indices(data['y_fit'], 20261003)
        np.testing.assert_array_equal(idx[:8], np.arange(8))
        self.assertEqual(np.bincount(data['y_fit'][idx]).tolist(), [6, 6])
        self.assertTrue(np.all(data['y_fit'][idx[8:]] == 1))
        np.testing.assert_array_equal(idx, c.oversample_indices(data['y_fit'], 20261003))
        alternate = copy.deepcopy(data)
        alternate['X_fit'] += 50
        a, b = balanced.resample(data, idx), balanced.resample(alternate, idx)
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])
            if 'validation' in key:
                np.testing.assert_array_equal(a[key], data[key])
                np.testing.assert_array_equal(b[key], data[key])
        np.testing.assert_array_equal(a['fit_record_ids'], b['fit_record_ids'])
        np.testing.assert_array_equal(b['X_fit']-a['X_fit'], 50)

    def test_seeds_and_already_balanced(self):
        y = np.array([0]*40+[1]*8)
        self.assertFalse(np.array_equal(c.oversample_indices(y, 1), c.oversample_indices(y, 2)))
        np.testing.assert_array_equal(c.oversample_indices([0, 1, 0, 1], 0), np.arange(4))

    def test_invalid_labels_and_patient_leakage(self):
        for y in ([], [0], [0, 2], [0, .3, 1], [[0, 1]], [0, float('nan')]):
            with self.assertRaises((ValueError, TypeError)):
                c.oversample_indices(y, 1)
        data = self.fixture()
        data['validation_patients'][0] = data['fit_patients'][0]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            c.validate_inputs(data)
        data = self.fixture()
        data['fit_record_ids'][1] = data['fit_record_ids'][0]
        with self.assertRaisesRegex(ValueError, 'identities'):
            c.validate_inputs(data)

    def test_normalization_cannot_use_repeated_or_validation_records(self):
        rows = {'a': {'partition': 'fit'}, 'b': {'partition': 'fit'}, 'v': {'partition': 'validation'}}
        a, b = np.zeros((3, 1, 2)), np.full((3, 1, 2), 2.)
        norm = FitStandardizer().fit([('a', a), ('b', b)], rows)
        np.testing.assert_array_equal(norm.mean_, np.ones((1, 2)))
        np.testing.assert_array_equal(norm.scale_, np.ones((1, 2)))
        for samples in ([('a', a), ('a', a)], [('a', a), ('v', b)]):
            with self.assertRaisesRegex(ValueError, 'unique fit'):
                FitStandardizer().fit(samples, rows)

    def test_artifact_corruption_and_context_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            c.write(folder / 'audit.json', {'ok': True})
            c.complete(folder, ['audit.json'], 'a')
            self.assertTrue(c.verified(folder, 'a'))
            with self.assertRaisesRegex(ValueError, 'context'):
                c.verified(folder, 'b')
            c.write(folder / 'audit.json', {'ok': False})
            with self.assertRaisesRegex(ValueError, 'artifact'):
                c.verified(folder, 'a')


if __name__ == '__main__':
    unittest.main()
