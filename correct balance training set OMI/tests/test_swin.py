import copy
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c
import swin
import numpy as np


class SwinTests(unittest.TestCase):
    def fixture(self):
        spec = copy.deepcopy(c.read(c.SPEC))
        spec['model'] = dict(n_leads=1, n_ceps=2, num_classes=2, embed_dim=4,
                             depths=[2, 2], heads=[1, 2], window_size=4)
        spec['training'].update(epochs=2, batch_size=4)
        rng = np.random.default_rng(5)
        data = dict(X=rng.normal(size=(10, 9, 1, 2)).astype(np.float32),
                    y=np.array([0]*5+[1]*3+[0, 1]),
                    record_ids=np.array([f'r{i}' for i in range(10)]),
                    patients=np.array([f'p{i}' for i in range(10)]))
        return data, c.oversample_indices(data['y'][:8], 7), np.arange(8, 10), spec

    def check_resume(self, device):
        swin.setup(device)
        data, tr, val, spec = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            full, resumed = Path(td)/'full', Path(td)/'resumed'
            swin.fit_swin(full, data, data['X'], tr, val, 1, spec, device, 'test')
            swin.fit_swin(resumed, data, data['X'], tr, val, 1, spec, device, 'test', stop_after=1)
            self.assertFalse((resumed / 'completed.json').exists())
            first = swin.load_checkpoint(resumed, 'test')
            self.assertEqual(first['epoch'], 1)
            # An interrupted second save must leave the first committed slot usable.
            broken = dict(first, epoch=2)
            with patch.object(c, 'write', side_effect=OSError('simulated interrupted commit')):
                with self.assertRaises(OSError):
                    swin.save_checkpoint(resumed, broken)
            self.assertEqual(swin.load_checkpoint(resumed, 'test')['epoch'], 1)
            swin.fit_swin(resumed, data, data['X'], tr, val, 1, spec, device, 'test')
            np.testing.assert_array_equal(c.arrays(full / 'predictions.npz')['score'],
                                          c.arrays(resumed / 'predictions.npz')['score'])
            h1, h2 = c.read(full / 'history.json'), c.read(resumed / 'history.json')
            self.assertEqual([h['loss'] for h in h1], [h['loss'] for h in h2])
            previous = c.sha(resumed / 'completed.json')
            swin.fit_swin(resumed, data, data['X'], tr, val, 1, spec, device, 'test')
            self.assertEqual(previous, c.sha(resumed / 'completed.json'))
            with self.assertRaisesRegex(ValueError, 'context'):
                swin.load_checkpoint(resumed, 'changed')
            marker = c.read(resumed / 'checkpoint.json')
            c.write_bytes(resumed / marker['file'], b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                swin.load_checkpoint(resumed, 'test')

    def test_cpu_checkpoint_resume(self):
        self.check_resume('cpu')

    @unittest.skipUnless(os.environ.get('OMI_TEST_CUDA') == '1', 'explicit CUDA test only')
    def test_cuda_checkpoint_resume(self):
        self.check_resume('cuda')

    @unittest.skipUnless(os.environ.get('OMI_TEST_CUDA') == '1', 'explicit CUDA test only')
    def test_full_acs_architecture_on_cuda(self):
        import torch
        from attention_ecg.model import CepstralSwin
        swin.setup('cuda')
        torch.manual_seed(0)
        model = CepstralSwin(**c.read(c.SPEC)['model']).cuda()
        inputs = torch.randn(128, 37, 12, 20, device='cuda')
        target = torch.arange(128, device='cuda') % 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(model(inputs), target)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertEqual(model(inputs).shape, (128, 2))

    def test_lfcc_control_uses_original_scaler_and_reloads(self):
        import joblib
        data, tr, val, spec = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            swin.fit_control(folder, data, tr, val, spec, 'test')
            model = joblib.load(folder / 'model.joblib')
            np.testing.assert_allclose(model['scaler'].mean_, swin.pooled(data['X'])[np.unique(tr)].mean(0),
                                       atol=1e-7, rtol=0)
            self.assertIsNone(model['classifier'].class_weight)
            np.testing.assert_array_equal(c.arrays(folder / 'predictions.npz')['record_ids'], data['record_ids'][val])


if __name__ == '__main__':
    unittest.main()
