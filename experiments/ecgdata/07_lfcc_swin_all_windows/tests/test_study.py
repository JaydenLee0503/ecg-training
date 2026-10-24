"""Synthetic checks for ECG-07; no real data, no writes outside a temporary directory."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study  # noqa: E402
from ecgvmd.segment import fixed_windows, standardise  # noqa: E402


class GridTests(unittest.TestCase):
    def test_sampled_v1_windows_are_exact_grid_windows(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(6, 65536))
        grid, rows, starts = study.grid_windows(data, 500)
        self.assertEqual(grid.shape, (6*131, 500))
        rows_label = np.arange(6)
        sampled, _, row_ids = fixed_windows(data, rows_label, rows_label, seg_len=500, n_per_record=10, seed=0)
        sampled = standardise(sampled)
        # Recover starts exactly as the v1 loader does.
        r = np.random.default_rng(0)
        v1_starts = np.concatenate([np.sort(r.choice(131, 10, replace=False))*500 for _ in range(6)])
        pos = study.subset_positions(row_ids.astype(int), v1_starts, 500, 131)
        np.testing.assert_array_equal(grid[pos], sampled)
        np.testing.assert_array_equal(rows[pos], row_ids)
        np.testing.assert_array_equal(starts[pos], v1_starts)

    def test_off_grid_start_rejected(self):
        with self.assertRaises(ValueError):
            study.subset_positions(np.array([0]), np.array([250]), 500, 131)

    def test_patient_crossing_folds_rejected(self):
        self.assertEqual(study.patient_folds(np.array(['a', 'a', 'b']), np.array([1, 1, 0])), {'a': 1, 'b': 0})
        with self.assertRaises(ValueError):
            study.patient_folds(np.array(['a', 'a']), np.array([1, 2]))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.patients = np.repeat(np.arange(30).astype(str), 5)
        self.y = np.repeat(np.arange(30) % 3, 5)
        self.pred = np.where(rng.random(150) < .8, self.y, (self.y+1) % 3)

    def test_identical_arms_have_zero_difference(self):
        out = study.paired_bootstrap(self.y, self.patients, {'a': [self.pred], 'b': [self.pred.copy()]},
                                     [('a', 'b')], 200, 3)
        for d in out['paired_differences']['a_minus_b'].values():
            self.assertEqual(d['interval_95'], [0.0, 0.0])

    def test_better_arm_is_positive(self):
        out = study.paired_bootstrap(self.y, self.patients, {'perfect': [self.y.copy()], 'noisy': [self.pred]},
                                     [('perfect', 'noisy')], 200, 3)
        self.assertGreater(out['paired_differences']['perfect_minus_noisy']['window_accuracy']['interval_95'][0], 0)


class FitTests(unittest.TestCase):
    def test_tiny_swin_and_control_fit_without_patient_leakage(self):
        from attention_ecg.ecgdata_training import fit_swin, fit_control, verify_trial
        p = study.protocol()
        p['training'] = dict(p['training'], epochs=2)
        rng = np.random.default_rng(0)
        patients = np.repeat(np.arange(15).astype(str), 4)
        y = np.repeat(np.arange(15) % 3, 4)
        fold = np.repeat(np.arange(15) % 5, 4)
        X = (rng.normal(size=(60, 13, 1, 16)) + y[:, None, None, None]).astype(np.float32)
        data = dict(X=X, y=y, patients=patients, fold=fold)
        with tempfile.TemporaryDirectory() as tmp:
            fit_swin(Path(tmp)/'swin', data, 0, 0, p, 'cpu')
            fit_control(Path(tmp)/'log', data, 0, p)
            self.assertTrue(verify_trial(Path(tmp)/'swin') and verify_trial(Path(tmp)/'log'))
            fit = study.read_json(Path(tmp)/'swin/fit.json')
            self.assertFalse(set(patients[fit['train_indices']]) & set(patients[fit['test_indices']]))
            self.assertEqual(len(study.read_json(Path(tmp)/'swin/history.json')), 2)


if __name__ == '__main__':
    unittest.main()
