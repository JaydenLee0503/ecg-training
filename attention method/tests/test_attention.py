import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from attention_ecg.cepstral import CepstralConfig, cepstral_features, filterbank
from attention_ecg.data import FitStandardizer, extract_acs_record, load_splits, validate_splits

try:
    import torch
    from attention_ecg.model import CepstralSwin, WindowAttention
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None


def fixture_rows():
    return [dict(record_id=str(i), patient_id=f'p{i}', split='train',
                 partition='fit' if i < 2 else 'validation', label=str(i % 2),
                 waveform_sha256='a' * 64) for i in range(4)]


class FeatureTests(unittest.TestCase):
    def test_reference_periodogram_and_explicit_cosine_matrix(self):
        c = CepstralConfig()
        x = np.random.default_rng(3).normal(size=(5000, 2))
        actual, metadata = cepstral_features(x, c)
        k = np.arange(c.n_ceps)[:, None]
        n = np.arange(c.n_filters)[None, :]
        cosine = np.sqrt(2 / c.n_filters) * np.cos(np.pi * k * (n + .5) / c.n_filters)
        cosine[0] /= np.sqrt(2)
        reference = []
        for start in range(0, 4501, 125):
            _, psd = periodogram(x[start:start + 500].T, fs=500, window='hann',
                                 nfft=512, detrend=False, scaling='density', axis=-1)
            reference.append(np.log(np.maximum(psd @ filterbank(c).T, c.power_floor)) @ cosine.T)
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=3e-6)
        self.assertEqual(actual.shape, (37, 2, 20))
        self.assertEqual(metadata['right_pad_samples'], 0)

    def test_gain_changes_c0_only_above_floor(self):
        x = np.random.default_rng(4).normal(size=(5000, 1))
        a, _ = cepstral_features(x)
        b, _ = cepstral_features(2 * x)
        np.testing.assert_allclose(b[..., 0] - a[..., 0], np.sqrt(40) * np.log(4), atol=5e-6)
        np.testing.assert_allclose(b[..., 1:], a[..., 1:], atol=5e-6)

    def test_padding_zero_mel_and_invalid_inputs(self):
        for spacing in ('linear', 'mel'):
            y, meta = cepstral_features(np.zeros((501, 12)), CepstralConfig(spacing=spacing))
            self.assertEqual(y.shape, (2, 12, 20))
            self.assertEqual(meta['right_pad_samples'], 124)
            self.assertTrue(np.isfinite(y).all())
        for x in (np.zeros(5000), np.zeros((100, 12)), np.full((500, 1), np.nan)):
            with self.assertRaises(ValueError):
                cepstral_features(x)
        with self.assertRaises(ValueError):
            CepstralConfig(fmax=300)
        with self.assertRaises(ValueError):
            filterbank(CepstralConfig(n_filters=500, n_ceps=20))


class DataTests(unittest.TestCase):
    def test_patient_boundaries_test_rejection_and_duplicates(self):
        rows = fixture_rows()
        validate_splits(rows)
        bad = copy.deepcopy(rows)
        bad[2]['patient_id'] = bad[0]['patient_id']
        with self.assertRaises(ValueError):
            validate_splits(bad)
        bad = copy.deepcopy(rows)
        bad[2]['split'] = 'test'
        with self.assertRaises(ValueError):
            validate_splits(bad)
        with self.assertRaises(ValueError):
            validate_splits(rows + rows[:1])

    def test_streaming_fit_statistics_and_validation_guard(self):
        splits = validate_splits(fixture_rows())
        a = np.arange(24).reshape(4, 2, 3)
        b = np.arange(30, 66).reshape(6, 2, 3)
        normalizer = FitStandardizer().fit([('0', a), ('1', b)], splits)
        combined = np.concatenate([a, b])
        np.testing.assert_allclose(normalizer.mean_, combined.mean(0))
        np.testing.assert_allclose(normalizer.scale_, combined.std(0))
        before = normalizer.state_dict()
        normalizer.transform(np.full((3, 2, 3), 1e5))
        self.assertEqual(before, normalizer.state_dict())
        with self.assertRaises(ValueError):
            FitStandardizer().fit([('2', a)], splits)
        with self.assertRaises(ValueError):
            FitStandardizer().fit([('0', a), ('0', a)], splits)

    def test_split_checksum_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'splits.csv'
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_splits(path, expected_sha256=hashlib.sha256(b'original').hexdigest())

    def test_acs_adapter_identity_and_official_test_guard(self):
        from experiments.acs_omi_vmd_wst_vqc.loader import ACSRecord, ACSInfo, ACSDecision, LEADS
        from dataclasses import replace
        splits = validate_splits(fixture_rows())
        decision = ACSDecision('0', 'p0', 'train', 0, True, 5000, 5000,
                               'fixture', (), (), 0, (), 'a' * 64)
        record = ACSRecord(ACSInfo('0', 'p0', 'train', 0), np.ones((5000, 12)), LEADS, decision)
        features, metadata = extract_acs_record(record, splits)
        self.assertEqual(features.shape, (37, 12, 20))
        self.assertEqual(metadata['partition'], 'fit')
        with self.assertRaises(ValueError):
            extract_acs_record(replace(record, info=ACSInfo('0', 'p0', 'test', None)), splits)
        with self.assertRaises(ValueError):
            extract_acs_record(replace(record, decision=replace(decision, waveform_sha256='b'*64)), splits)


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is absent')
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_shift_connects_neighbors_without_circular_wrap(self):
        x = torch.randn(1, 16, 8)
        unshifted = WindowAttention(8, 2, 8, 0).eval()
        shifted = WindowAttention(8, 2, 8, 4).eval()
        with torch.inference_mode():
            base_u, base_s = unshifted(x), shifted(x)
            neighbor = x.clone()
            neighbor[:, 8] += 10
            torch.testing.assert_close(unshifted(neighbor)[:, 7], base_u[:, 7], rtol=0, atol=0)
            self.assertGreater((shifted(neighbor)[:, 7] - base_s[:, 7]).abs().max().item(), 1e-5)
            edge = x.clone()
            edge[:, -1] += 100
            torch.testing.assert_close(shifted(edge)[:, 0], base_s[:, 0], rtol=0, atol=0)

    def test_padding_keys_are_excluded(self):
        # One real token: all attention weight must be on that token, regardless
        # of learned q/k, relative biases, or seven synthetic padded keys.
        attention = WindowAttention(8, 2, 8, 4).eval()
        x = torch.randn(2, 1, 8)
        with torch.inference_mode():
            expected = attention.proj(attention.qkv(x)[..., 16:])
            torch.testing.assert_close(attention(x), expected)

    def test_shapes_batch_independence_no_weight_updates_and_roundtrip(self):
        model = CepstralSwin().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            for length in (1, 7, 8, 9, 37, 40):
                x = torch.randn(2, length, 12, 20)
                out = model(x)
                self.assertEqual(out.shape, (2, 2))
                self.assertTrue(torch.isfinite(out).all())
                torch.testing.assert_close(out, torch.cat([model(x[:1]), model(x[1:])]), atol=2e-6, rtol=1e-5)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = CepstralSwin().eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored(x), out, rtol=0, atol=0)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_reject_wrong_axes_and_nonfinite(self):
        model = CepstralSwin()
        for x in (torch.zeros(1, 37, 20, 12), torch.full((1, 37, 12, 20), float('nan'))):
            with self.assertRaises(ValueError):
                model(x)


if __name__ == '__main__':
    unittest.main()
