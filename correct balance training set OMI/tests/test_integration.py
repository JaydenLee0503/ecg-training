"""Small fixture integration: production artifacts, recovery, and report identities."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c
import balanced
import lfcc
import report
import run
import numpy as np


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        spec = copy.deepcopy(c.read(c.SPEC))
        baseline_protocol = c.read(c.ROOT / 'experiments/acs_omi_vmd_wst_vqc/protocols/acs_omi_protocol_v1_retry256k.json')
        spec['sampling'].update(original_counts=[6, 2], balanced_counts=[6, 6], fit_records=8, validation_records=2)
        spec['extraction'].update(batch_records=3, shape=[3, 1, 2])
        spec['evaluation']['bootstrap_resamples'] = 20
        self.spec, self.ctx, self.source, self.root = spec, 'fixture-context', root / 'baseline', root
        self.data = dict(X_fit=np.arange(16).reshape(8, 2), y_fit=np.array([0]*6+[1]*2),
            fit_record_ids=np.array([f'r{i}' for i in range(8)]), fit_patients=np.array([f'p{i}' for i in range(8)]),
            X_validation=np.array([[20, 21], [22, 23]]), y_validation=np.array([0, 1]),
            validation_record_ids=np.array(['r8', 'r9']), validation_patients=np.array(['p8', 'p9']),
            selected_indices=np.array([0, 1]), selected_names=np.array(['f0', 'f1']))
        m = dict(protocol=baseline_protocol)
        c.write(self.source / 'manifest.json', m)
        self.baseline_sha = c.sha(self.source / 'manifest.json')
        for arm in ('vmd', 'wst'):
            folder = self.source / arm
            c.npz(folder / 'inputs.npz', **self.data)
            c.write_bytes(folder / 'preprocessor.joblib', b'opaque original preprocessor fixture')
            c.write(folder / 'preprocessing.json', dict(status='complete', context_sha=self.baseline_sha,
                files={n: c.sha(folder / n) for n in ('inputs.npz', 'preprocessor.joblib')}))
        self.write_trials(self.source, self.baseline_sha)
        rows = [[f'r{i}', f'p{i}', 'train', 'fit' if i < 8 else 'validation',
                 int(i in (6, 7, 9)), f'waveform-{i}'] for i in range(10)]
        c.write_bytes(root / 'splits.csv', c.csv_bytes(
            ['record_id', 'patient_id', 'split', 'partition', 'label', 'waveform_sha256'], rows))
        spec.update(split='splits.csv', split_sha256=c.sha(root / 'splits.csv'))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [('ROOT', root), ('RUN', root / 'newrun'), ('BASE', root / 'experiment')]:
            self.stack.enter_context(patch.object(c, name, value))
        c.RUN.mkdir()
        self.stack.enter_context(patch.object(c, 'context', return_value=(spec, self.ctx, self.source, m, self.baseline_sha)))
        self.stack.enter_context(patch.object(balanced.original, 'read_run',
            side_effect=lambda p: (c.read(p / 'manifest.json'), c.sha(p / 'manifest.json'))))

    def predictions(self):
        score = np.array([.2, .8])
        return dict(score=score, prediction=(score >= .5).astype(int), y=self.data['y_validation'],
                    record_ids=self.data['validation_record_ids'], patients=self.data['validation_patients'])

    def write_trials(self, folder, digest):
        for arm, name in balanced.original.TRIALS:
            target = folder / arm / name
            c.npz(target / 'predictions.npz', **self.predictions())
            c.complete(target, ['predictions.npz'], digest)
        c.write(folder / 'completed.json', dict(status='complete', context_sha=digest, fits=10))

    def test_prepare_reuse_and_tamper_guard(self):
        balanced.prepare()
        target = c.RUN / 'balanced'
        protected = [target / arm / name for arm in ('vmd', 'wst')
                     for name in ('inputs.npz', 'preprocessor.joblib', 'preprocessing.json')]
        hashes = {str(p): c.sha(p) for p in protected}
        balanced.prepare()
        self.assertEqual(hashes, {str(p): c.sha(p) for p in protected})
        m = c.read(target / 'manifest.json')
        self.assertIsNone(m['protocol']['vqc']['class_weight'])
        self.assertIsNone(m['protocol']['controls']['logistic']['class_weight'])
        self.assertEqual(c.read(self.source / 'manifest.json')['protocol']['vqc']['class_weight'], 'balanced')
        for arm in ('vmd', 'wst'):
            new = c.arrays(target / arm / 'inputs.npz')
            self.assertEqual(np.bincount(new['y_fit']).tolist(), [6, 6])
            for key in new:
                if 'validation' in key:
                    np.testing.assert_array_equal(new[key], self.data[key])
        c.write_bytes(target / 'vmd/inputs.npz', b'changed')
        with self.assertRaisesRegex(ValueError, 'Changed saved artifact'):
            balanced.prepare()

    def test_lfcc_batch_resume_and_unique_fit_normalization(self):
        balanced.prepare()
        seen = []
        fail = [True]
        class FakeDataset:
            source_manifest = {'fixture': True}
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def inspect(self, rid): return rid
        def extract(rid, rows, config):
            seen.append(rid)
            if fail[0] and rid == 'r3':
                raise OSError('simulated interruption')
            value = int(rid[1:])
            return np.full((3, 1, 2), value, dtype=np.float32), dict(record_id=rid)
        with patch.object(lfcc, 'ACSDataset', FakeDataset), patch.object(lfcc, 'extract_acs_record', extract), \
                patch.object(lfcc, 'engineering_check'):
            with self.assertRaisesRegex(OSError, 'interruption'):
                lfcc.extract()
            done = c.RUN / 'lfcc/batches/00000/completed.json'
            before = c.sha(done)
            fail[0] = False
            seen.clear()
            lfcc.extract()
            self.assertEqual(c.sha(done), before)
            self.assertFalse(set(seen) & {'r0', 'r1', 'r2'})
            lfcc.extract()  # Entire completed cache verifies and reuses.
        data, norm, tr, val = lfcc.load()
        np.testing.assert_array_equal(norm['mean'], np.full((1, 2), 3.5))
        self.assertEqual(np.bincount(data['y'][tr]).tolist(), [6, 6])
        np.testing.assert_array_equal(data['record_ids'][val], self.data['validation_record_ids'])
        corrupt = c.arrays(c.RUN / 'lfcc/features.npz')
        corrupt['patients'][0] = corrupt['patients'][-1]
        c.npz(c.RUN / 'lfcc/features.npz', **corrupt)
        with self.assertRaisesRegex(ValueError, 'artifact'):
            lfcc.load()

    def test_report_uses_natural_prior_and_rejects_identity_mismatch(self):
        balanced.prepare()
        out = c.RUN / 'balanced'
        self.write_trials(out, c.sha(out / 'manifest.json'))
        models = c.RUN / 'lfcc_models'
        c.write(models / 'manifest.json', dict(fixture=True))
        digest = c.sha(models / 'manifest.json')
        names = ['logistic', 'swin_seed0', 'swin_seed1', 'swin_seed2']
        for name in names:
            c.npz(models / name / 'predictions.npz', **self.predictions())
            c.complete(models / name, ['predictions.npz'], digest)
        c.complete(models, ['manifest.json'] + [f'{n}/completed.json' for n in names], self.ctx)
        _, _, reference, scores, groups, pairs, prior = report.collect()
        self.assertEqual(prior, .25)
        self.assertEqual(len(scores), 25)
        self.assertEqual(len(groups['balanced/lfcc/swin']), 3)
        np.testing.assert_array_equal(scores['constant_natural_prior'], [.25, .25])
        report.report()
        before = c.sha(c.RUN / 'report/completed.json')
        report.report()
        self.assertEqual(before, c.sha(c.RUN / 'report/completed.json'))
        c.write_bytes(c.BASE / 'reports/v1/report.md', b'incomplete export')
        report.report()
        self.assertEqual(c.sha(c.BASE / 'reports/v1/report.md'), c.sha(c.RUN / 'report/report.md'))
        # Even a correctly checksummed prediction file with different row order must fail.
        prediction = self.predictions()
        prediction['record_ids'] = prediction['record_ids'][::-1]
        c.npz(models / 'swin_seed1/predictions.npz', **prediction)
        c.complete(models / 'swin_seed1', ['predictions.npz'], digest)
        c.complete(models, ['manifest.json'] + [f'{n}/completed.json' for n in names], self.ctx)
        with self.assertRaises(AssertionError):
            report.collect()

    def test_stop_targets_owned_process_group_only(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        with patch.object(run.os, 'killpg') as kill:
            run.stop_child(child)
        kill.assert_called_once_with(4321, signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=10)
        child.poll.return_value = 0
        with patch.object(run.os, 'killpg') as kill:
            run.stop_child(child)
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
