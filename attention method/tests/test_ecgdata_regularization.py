import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from attention_ecg.ecgdata import read_json, fit_normalizer, file_hash
from attention_ecg.ecgdata_training import fit_swin, load_checkpoint
from attention_ecg.model import CepstralSwin
from attention_ecg.regularized_swin import make_model
from attention_ecg.ecgdata_regularization import (
    inner_splits, fit, choose, setup, load_state, reuse_v1)


def fixture():
    rng = np.random.default_rng(12)
    y = np.tile(np.repeat(np.arange(3), 4), 5)
    return dict(y=y, patients=np.repeat([f'p{i:03d}' for i in range(30)], 2),
                fold=np.repeat(np.arange(5), 12), X=rng.normal(size=(60, 13, 1, 16)).astype(np.float32))


def base_protocol():
    base = read_json(ROOT / 'ecgdata_protocol.json')
    base['training']['epochs'] = 4
    return base


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        setup('cpu')

    def test_zero_dropout_exact_forward_gradients_and_initialization(self):
        config = base_protocol()['model']
        torch.manual_seed(19)
        old = CepstralSwin(**config)
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        new = make_model(config, 0)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        x = torch.randn(3, 13, 1, 16)
        for t in (1, 4, 5, 13):
            a, b = old(x[:, :t]), new(x[:, :t])
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        old(x).square().sum().backward()
        new(x).square().sum().backward()
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        dropout = make_model(config, .1)
        dropout.load_state_dict(old.state_dict())
        dropout.eval()
        torch.testing.assert_close(old(x), dropout(x), rtol=0, atol=0)
        dropout.train()
        self.assertFalse(torch.equal(dropout(x), dropout(x)))

    def test_inner_patients_isolated_stratified_complete_reproducible(self):
        data = fixture()
        for outer in range(5):
            first = inner_splits(data, outer, 3, 20261015)
            again = inner_splits(data, outer, 3, 20261015)
            seen = []
            for (train, val), (tr2, va2) in zip(first, again):
                np.testing.assert_array_equal(train, tr2)
                np.testing.assert_array_equal(val, va2)
                self.assertFalse(set(data['patients'][train]) & set(data['patients'][val]))
                self.assertFalse(np.any(data['fold'][train] == outer))
                self.assertFalse(np.any(data['fold'][val] == outer))
                self.assertEqual(set(data['y'][train]), {0, 1, 2})
                self.assertEqual(set(data['y'][val]), {0, 1, 2})
                seen.extend(val)
            np.testing.assert_array_equal(np.sort(seen), np.flatnonzero(data['fold'] != outer))

    def test_zero_dropout_training_matches_frozen_v1(self):
        data, base = fixture(), base_protocol()
        base['training']['epochs'] = 2
        train, evaluation = np.flatnonzero(data['fold'] != 0), np.flatnonzero(data['fold'] == 0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fit_swin(folder / 'v1', data, 0, 2, base, 'cpu')
            fit(folder / 'v2', data, train, evaluation, 2, 0, 2, [2], base, 'cpu', 'test', 'outer')
            a, b = load_checkpoint(folder / 'v1'), load_state(folder / 'v2')
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            np.testing.assert_array_equal([r['loss'] for r in a['history']], [r['loss'] for r in b['history']])
            # Production's R0 epoch-40 selection reuses its matching frozen fit.
            (folder / 'reference').mkdir()
            (folder / 'v1').rename(folder / 'reference/swin_seed2_fold0')
            reused = reuse_v1(folder / 'reuse', data, 0, 2, base, 'test', 'cpu', folder / 'reference')
            self.assertTrue(reused['reused_v1'])
            with np.load(folder / 'reuse/epoch_002.npz') as x, np.load(folder / 'reference/swin_seed2_fold0/predictions.npz') as y:
                np.testing.assert_array_equal(x['probabilities'], y['probabilities'])

    def _resume_check(self, device):
        setup(device)
        data, base = fixture(), base_protocol()
        train, evaluation = inner_splits(data, 0, 3, 11)[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = (data, train, evaluation, 1, .1, 3, [1, 3], base, device, 'test', 'inner')
            fit(folder / 'full', *args)
            fit(folder / 'resumed', *args, stop_after=1)
            first = load_state(folder / 'resumed')
            # An uncommitted generation must not displace the committed pointer.
            (folder / 'resumed/rolling/epoch_002.pt').write_bytes(b'uncommitted')
            self.assertEqual(load_state(folder / 'resumed')['epoch'], 1)
            fit(folder / 'resumed', *args)
            a, b = load_state(folder / 'full'), load_state(folder / 'resumed')
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            self.assertEqual(a['history'], b['history'])
            self.assertEqual(b['scheduler']['T_max'], 4)
            self.assertEqual(b['scheduler']['last_epoch'], 3)
            self.assertEqual(first['history'][0]['epoch'], 1)
            mean, scale = fit_normalizer(data['X'], train)
            with np.load(folder / 'resumed/preprocessing.npz') as z:
                np.testing.assert_array_equal(z['mean'], mean)
                np.testing.assert_array_equal(z['scale'], scale)
            for epoch in (1, 3):
                with np.load(folder / f'full/epoch_{epoch:03d}.npz') as x, np.load(folder / f'resumed/epoch_{epoch:03d}.npz') as y:
                    np.testing.assert_array_equal(x['probabilities'], y['probabilities'])
            before = file_hash(folder / 'resumed/completed.json')
            fit(folder / 'resumed', *args)
            self.assertEqual(before, file_hash(folder / 'resumed/completed.json'))
            changed = list(args)
            changed[4] = .2
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                fit(folder / 'resumed', *changed)
            (folder / 'resumed/epoch_001.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                fit(folder / 'resumed', *args)

    def test_dropout_resume_retention_normalization_reuse_and_corruption(self):
        self._resume_check('cpu')

    def test_evaluation_values_do_not_affect_training_or_normalization(self):
        data, base = fixture(), base_protocol()
        train, evaluation = inner_splits(data, 0, 3, 11)[0]
        changed = copy.deepcopy(data)
        changed['X'][evaluation] *= 100
        changed['y'][evaluation] = (changed['y'][evaluation] + 1) % 3
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, sample in [('original', data), ('changed', changed)]:
                fit(folder / name, sample, train, evaluation, 1, .1, 2, [2], base, 'cpu', 'test', 'inner')
            a, b = load_state(folder / 'original'), load_state(folder / 'changed')
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)

    def test_selection_ties_prefer_epoch_then_candidate(self):
        rows = [dict(candidate=c, epoch=e, mean_macro_f1=.8) for c in ('R1','R0') for e in (40,10,20)]
        selected = choose(rows)
        self.assertEqual((selected['candidate'], selected['epoch']), ('R0', 10))
        rows[0]['mean_macro_f1'] = .81
        self.assertEqual(choose(rows), rows[0])

    @unittest.skipUnless(os.environ.get('ECG_REG_TEST_CUDA') == '1', 'Explicit GPU verification')
    def test_gpu_forward_gradient_and_resume(self):
        setup('cuda')
        torch.manual_seed(4)
        cpu = make_model(base_protocol()['model'], .1).eval()
        gpu = copy.deepcopy(cpu).cuda()
        x = torch.randn(4, 13, 1, 16)
        a, b = cpu(x), gpu(x.cuda())
        torch.testing.assert_close(a, b.cpu(), rtol=1e-4, atol=2e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        for left, right in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(left.grad, right.grad.cpu(), rtol=2e-3, atol=2e-5)
        self._resume_check('cuda')


if __name__ == '__main__':
    unittest.main()
