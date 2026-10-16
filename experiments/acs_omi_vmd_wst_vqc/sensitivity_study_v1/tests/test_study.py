import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1 import common as c
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.metrics import Curve, at_threshold, analyze
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.prepare import partition, oversample
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.fit import run_trial


class ThresholdTests(unittest.TestCase):
    def test_ties_constant_and_exact_percentage_boundary(self):
        y = np.r_[np.ones(20), np.zeros(20)].astype(int)
        score = np.r_[np.linspace(1, .1, 20), np.linspace(.9, 0, 20)]
        p = Curve(y, score).operating_points([80, 90, 95])
        self.assertEqual([r['tp'] for r in p], [16, 18, 19])
        for point, target in zip(p, [80, 90, 95]):
            self.assertEqual(point['sensitivity'], target/100)
            higher = score[score > point['threshold']]
            self.assertLess(at_threshold(y, score, higher.min())['sensitivity'], target/100)
        p = Curve([0, 1, 0, 1], [.3]*4).operating_points([80, 90, 95])
        for row in p:
            self.assertEqual((row['threshold'], row['sensitivity'], row['specificity'], row['precision']),
                             (.3, 1., 0., .5))

    def test_weighted_curve_matches_literal_patient_duplicates(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        s = np.array([.9, .8, .8, .5, .1, .1])
        weights = np.array([2, 2, 0, 0, 3, 3])
        a = Curve(y, s).operating_points([80, 90, 95], weights)
        b = Curve(np.repeat(y, weights), np.repeat(s, weights)).operating_points([80, 90, 95])
        self.assertEqual(a, b)

    def test_cluster_bootstrap_seed_means_and_paired_zero(self):
        y = np.tile([0, 1], 10)
        patients = np.repeat(np.arange(10), 2)
        score = np.tile([.2, .8], 10)
        scores = {'new/lfcc/logistic': score, **{f'new/vmd/vqc_seed{s}': score for s in (0, 1, 2)}}
        result, draws = analyze(y, patients, scores, [80, 90, 95], 30, 42)
        for row in result['paired_differences']['new/vmd/vqc minus new/lfcc/logistic']['intervals']:
            self.assertTrue(all(v == [0, 0] for v in row.values()))
        self.assertEqual(result['bootstrap']['valid'], 30)
        result2, draws2 = analyze(y, patients, scores, [80, 90, 95], 30, 42)
        self.assertEqual(result, result2)
        for name in draws:
            np.testing.assert_array_equal(draws[name], draws2[name])

    def test_fixed_threshold_is_not_reselected_to_hit_target(self):
        y, s = np.array([0, 1, 0, 1]), np.array([.1, .2, .3, .4])
        r, _ = analyze(y, np.arange(4), {'new/lfcc/logistic': s}, [80, 90, 95], 30, 5,
                       {'new/lfcc/logistic': [.35, .35, .35]})
        self.assertEqual([p['sensitivity'] for p in r['per_model']['new/lfcc/logistic']], [.5]*3)
        for ci in r['per_model_intervals']['new/lfcc/logistic']:
            self.assertEqual(ci['threshold'], [.35, .35])

    def test_invalid_inputs_rejected(self):
        for y, s in [([0, 0], [.1, .2]), ([0, 1], [.1, np.nan]), ([0, 1], [-.1, .2])]:
            with self.assertRaises(ValueError):
                Curve(y, s)
        with self.assertRaises(ValueError):
            Curve([0, 1], [.1, .9]).operating_points([90.5])

    def test_patient_split_and_sampling_are_separate(self):
        patients = np.repeat(np.arange(100), 2)
        y = np.tile([0, 1], 100)  # Mixed-label patients remain intact.
        old = np.array(['fit']*160 + ['validation']*40)
        spec = dict(calibration_split=dict(n_splits=5, fold=0, seed=20261015))
        parts = partition(y, patients, old, spec)
        np.testing.assert_array_equal(parts[160:], old[160:])
        for p in np.unique(patients):
            self.assertEqual(len(set(parts[patients == p])), 1)
        subfit = np.flatnonzero(parts == 'subfit')
        chosen = oversample(y[subfit], 5)
        self.assertEqual(set(chosen), set(range(len(subfit))))
        self.assertTrue(np.all(parts[subfit[chosen]] == 'subfit'))
        self.assertEqual(np.bincount(y[subfit][chosen])[0], np.bincount(y[subfit][chosen])[1])

    def test_end_to_end_logistic_validation_labels_cannot_change_cutoffs(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(120, 4))
        y = np.tile([0, 1], 60)
        X[:, 0] += y
        parts = np.array(['subfit']*80 + ['calibration']*20 + ['validation']*20)
        spec = c.read(c.BASE / 'protocol.json')
        manifest = dict(protocol=spec, original_settings=dict(vqc=dict(class_weight=None, seeds=[0, 1, 2]),
            controls=dict(logistic=dict(C=1., class_weight=None, solver='lbfgs', max_iter=2000),
                          weighted_knn=dict(n_neighbors=10))))
        saved = []
        with tempfile.TemporaryDirectory() as tmp:
            for attempt in range(2):
                run = Path(tmp) / str(attempt)
                prepared = run / 'prepared'
                labels = y.copy()
                if attempt:
                    labels[100:] = 1-labels[100:]
                c.npz(prepared / 'cohort.npz', y=labels, record_ids=np.arange(120).astype(str),
                    patients=np.arange(120).astype(str), partitions=parts, sampled_indices=np.arange(80))
                c.npz(prepared / 'vmd.npz', X=X)
                c.complete(prepared, 'test')
                with patch.object(c, 'RUN', run), patch.object(c, 'manifest', return_value=(manifest, 'test')):
                    run_trial('vmd', 'logistic')
                    folder = run / 'models/vmd/logistic'
                    saved.append(c.read(folder / 'thresholds.json'))
                    self.assertTrue(c.verified(folder, 'test'))
                    before = c.sha(folder / 'model.joblib')
                    run_trial('vmd', 'logistic')  # Completed fit must be reused.
                    self.assertEqual(c.sha(folder / 'model.joblib'), before)
            self.assertEqual(saved[0], saved[1])

    def test_swin_adapter_keeps_calibration_and_validation_separate(self):
        spec = c.read(c.ROOT / 'correct balance training set OMI/protocol.json')
        spec['model'] = dict(n_leads=1, n_ceps=2, num_classes=2, embed_dim=4,
                             depths=[2, 2], heads=[1, 2], window_size=4)
        spec['training'].update(epochs=2, batch_size=4)
        manifest = dict(protocol=c.read(c.BASE / 'protocol.json'), balanced_settings=spec)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            prepared = run / 'prepared'
            c.npz(prepared / 'cohort.npz', y=np.tile([0, 1], 7),
                  record_ids=np.arange(14).astype(str), patients=np.arange(14).astype(str),
                  partitions=np.array(['subfit']*10 + ['calibration']*2 + ['validation']*2),
                  sampled_indices=np.arange(10))
            np.save(prepared / 'lfcc_normalized.npy', np.random.default_rng(9).normal(size=(14, 9, 1, 2)).astype(np.float32))
            c.complete(prepared, 'test')
            with patch.object(c, 'RUN', run), patch.object(c, 'manifest', return_value=(manifest, 'test')):
                run_trial('lfcc', 'swin_seed0', 'cpu')
                out = run / 'models/lfcc/swin_seed0'
                self.assertTrue(c.verified(out, 'test'))
                np.testing.assert_array_equal(c.arrays(out / 'calibration_predictions.npz')['record_ids'], ['10', '11'])
                np.testing.assert_array_equal(c.arrays(out / 'validation_predictions.npz')['record_ids'], ['12', '13'])
                before = c.sha(out / 'completed.json')
                run_trial('lfcc', 'swin_seed0', 'cpu')
                self.assertEqual(c.sha(out / 'completed.json'), before)


if __name__ == '__main__':
    unittest.main()
