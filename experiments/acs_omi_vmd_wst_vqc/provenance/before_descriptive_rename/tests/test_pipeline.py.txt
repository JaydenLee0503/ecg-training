"""Patient boundaries, shared inputs, convergence failures and binary model interface."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.acs.loader import ACSDecision, ACSInfo, ACSRecord, LEADS
from experiments.acs.pipeline import (engineering_ids, extract_record, fit_preprocessor,
                                   make_classifiers, patient_split)
from experiments.acs.scripts import experiment as runner

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] /
                       'protocols/acs_omi_protocol_v1.json').read_text())


def rows():
    return [dict(record_id=f'{i:05d}', patient_id=f'P{i // 2:05d}',
                 label=(i // 2) % 2, split='train') for i in range(120)]


def record():
    t = np.arange(5000) / 500
    signal = np.stack([(i + 1) * .02 * np.sin(2 * np.pi * (i + 1) * t)
                       + .005 * np.cos(2 * np.pi * 3.17 * t) for i in range(12)], axis=1)
    decision = ACSDecision('00001', 'P00001', 'train', 1, True, 5000, 5000,
                           'wfdb_standard', (), (), 0, (), 'fixture')
    return ACSRecord(ACSInfo('00001', 'P00001', 'train', 1), signal, LEADS, decision)


class ACSExperimentTests(unittest.TestCase):
    def test_layout_migration_keeps_strict_source_and_manifest_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, current, shared = [root / name for name in ('old.py.txt', 'new.py', 'shared.py')]
            for path, content in ((snapshot, 'old import'), (current, 'new import'), (shared, 'unchanged math')):
                path.write_text(content)
            recorded = {'old.py': runner.file_hash(snapshot), 'shared.py': runner.file_hash(shared)}
            layout = dict(legacy_manifest_sha256='known-run', legacy_code_sha256=recorded,
                          code_moves={'old.py': dict(snapshot='old.py.txt', current='new.py',
                                                    current_sha256=runner.file_hash(current))})
            layout_path = root / 'layout.json'
            layout_path.write_text(json.dumps(layout))
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'LAYOUT', layout_path):
                runner.check_implementation({'code_sha256': recorded}, 'known-run')
                with self.assertRaisesRegex(ValueError, 'No verified layout'):
                    runner.check_implementation({'code_sha256': recorded}, 'another-run')
                current.write_text('changed transform')
                with self.assertRaisesRegex(ValueError, 'Changed migrated'):
                    runner.check_implementation({'code_sha256': recorded}, 'known-run')
                current.write_text('new import')
                snapshot.write_text('changed evidence')
                with self.assertRaisesRegex(ValueError, 'Changed migrated'):
                    runner.check_implementation({'code_sha256': recorded}, 'known-run')
                snapshot.write_text('old import')
                shared.write_text('changed math')
                with self.assertRaisesRegex(ValueError, 'Changed shared'):
                    runner.check_implementation({'code_sha256': recorded}, 'known-run')

    def test_new_layout_manifests_still_reject_code_changes(self):
        recorded = {name: 'expected' for name in runner.CODE}
        with patch.object(runner, 'file_hash', return_value='expected'):
            runner.check_implementation({'code_sha256': recorded}, 'new-run')
        with patch.object(runner, 'file_hash', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Changed implementation'):
                runner.check_implementation({'code_sha256': recorded}, 'new-run')

    def test_checkpoint_resume_verifies_identity_and_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = dict(record_id='00001', patient_id='P00001', label='1', waveform_sha256='wave')
            path = folder / '00001.npz'
            np.savez_compressed(path, vmd=np.arange(3.), wst=np.arange(4.),
                                vmd_names=np.array(['a', 'b', 'c']), wst_names=np.array(['d', 'e', 'f', 'g']))
            meta = dict(status='complete', manifest_sha256='run', **row,
                        features_sha256=runner.file_hash(path), vmd_features=3, wst_features=4)
            meta['label'] = 1
            (folder / '00001.json').write_text(json.dumps(meta))
            _, loaded = runner.checked_feature(folder, row, 'run')
            self.assertEqual(loaded['record_id'], row['record_id'])
            with self.assertRaises(ValueError):
                runner.checked_feature(folder, {**row, 'patient_id': 'P00002'}, 'run')
            with self.assertRaises(ValueError):
                runner.checked_feature(folder, row, 'different_run')
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                runner.checked_feature(folder, row, 'run')

    def test_completed_smoke_restart_does_not_reopen_waveforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            folder = out / 'smoke'
            folder.mkdir()
            (folder / 'status.json').write_text(json.dumps({'status': 'complete'}))
            (out / 'manifest.json').write_text('{}')
            data = patient_split(rows(), PROTOCOL)
            for rid in engineering_ids(data, PROTOCOL):
                (folder / f'{rid}.json').write_text('{}')
            args = SimpleNamespace(out=out, command='smoke', data_dir=out / 'unavailable')
            with patch.object(runner, 'read_run', return_value=({'protocol': PROTOCOL}, data)), \
                    patch.object(runner, 'checked_feature', return_value=({}, {})) as verify, \
                    patch.object(runner, 'ACSDataset', side_effect=AssertionError('Unexpected ECG reread')):
                runner.extract(args)
            self.assertEqual(verify.call_count, 2)

    def test_split_is_deterministic_and_all_patient_ecgs_stay_together(self):
        result = patient_split(rows(), PROTOCOL)
        self.assertEqual(result, patient_split(list(reversed(rows())), PROTOCOL))
        self.assertEqual(len(result), 120)
        for patient in {r['patient_id'] for r in result}:
            self.assertEqual(len({r['fold'] for r in result if r['patient_id'] == patient}), 1)
        self.assertEqual({r['label'] for r in result if r['partition'] == 'validation'}, {0, 1})

    def test_split_rejects_test_records_duplicates_and_missing_labels(self):
        for bad in ({'split': 'test'}, {'label': None}, {'record_id': '00001'}):
            data = rows()
            data[0].update(bad)
            with self.assertRaises(ValueError):
                patient_split(data, PROTOCOL)

    def test_engineering_subset_is_fit_only_and_label_independent(self):
        data = patient_split(rows(), PROTOCOL)
        chosen = engineering_ids(data, PROTOCOL)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(all(r['partition'] == 'fit' for r in data if r['record_id'] in chosen))
        for r in data:
            r['label'] = 1 - r['label']
        self.assertEqual(chosen, engineering_ids(data, PROTOCOL))

    def test_preprocessing_cannot_learn_from_validation(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(60, 24))
        y = np.arange(60) % 2
        X[:, 0] += y * 2
        patients = np.array([f'P{i}' for i in range(60)])
        partitions = np.array(['fit'] * 40 + ['validation'] * 20)
        a, Z, fit, val = fit_preprocessor(X, y, patients, partitions, PROTOCOL)
        changed = X.copy()
        changed[val] = 1e9
        labels = y.copy()
        labels[val] = 1 - labels[val]
        b, Zb, _, _ = fit_preprocessor(changed, labels, patients, partitions, PROTOCOL)
        np.testing.assert_array_equal(a[0].idx_, b[0].idx_)
        np.testing.assert_array_equal(a[1].scaler_.mean_, b[1].scaler_.mean_)
        np.testing.assert_allclose(a[1].scaler_.mean_, X[fit][:, a[0].idx_].mean(0))
        np.testing.assert_array_equal(Z[fit], Zb[fit])
        self.assertEqual(Z.shape, (60, 12))
        self.assertTrue(np.isfinite(Zb).all())
        patients[-1] = patients[0]
        with self.assertRaisesRegex(ValueError, 'patient'):
            fit_preprocessor(X, y, patients, partitions, PROTOCOL)
        partitions[-1] = 'test'
        with self.assertRaises(ValueError):
            fit_preprocessor(X, y, np.arange(60), partitions, PROTOCOL)

    def test_classifier_budget_and_binary_quantum_interface(self):
        models = make_classifiers(PROTOCOL)
        self.assertEqual(set(models), {'vqc_seed0', 'vqc_seed1', 'vqc_seed2', 'logistic', 'weighted_knn'})
        for seed in range(3):
            model = models[f'vqc_seed{seed}']
            self.assertEqual(model.seed, seed)
            self.assertEqual(model.n_qubits, 6)
            self.assertFalse(model.reupload)
        model = models['vqc_seed0'].set_params(epochs=1)
        rng = np.random.default_rng(15)
        X = rng.normal(size=(16, 12))
        model.fit(X, np.arange(16) % 2)
        probabilities = model.predict_proba(X[:3])
        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_allclose(probabilities.sum(1), 1)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_extractor_rejects_official_test_or_changed_inputs_before_computing(self):
        r = record()
        invalid = (replace(r, fs=128), replace(r, leads=tuple(reversed(LEADS))),
                   replace(r, info=ACSInfo('00001', 'P00001', 'test', None)),
                   replace(r, signal_mV=None), replace(r, signal_mV=r.signal_mV[:3500]))
        for bad in invalid:
            with self.subTest(bad=bad.info):
                with self.assertRaises(ValueError):
                    extract_record(bad, PROTOCOL)

    def test_capped_vmd_is_logged_and_stops_instead_of_becoming_an_exclusion(self):
        attempts = []
        def capped(X, **kw):
            return SimpleNamespace(modes=np.zeros((len(X), 8, X.shape[1])),
                                   omega_hz=np.zeros((len(X), 8)),
                                   iters=np.full(len(X), kw['max_iter']), capped=np.ones(len(X), bool))
        with patch('experiments.acs.pipeline.vmd_batch', side_effect=capped):
            with self.assertRaisesRegex(RuntimeError, 'unconverged'):
                extract_record(record(), PROTOCOL, on_attempt=attempts.append)
        self.assertEqual({r['limit'] for r in attempts}, set(PROTOCOL['vmd']['iteration_limits']))
        self.assertTrue(all(r['capped'] for r in attempts))

    def test_both_frontends_receive_identical_physical_leads_and_order(self):
        r = record()
        seen = []
        def converged(X, **kw):
            seen.extend(X.copy())
            self.assertEqual(kw['fs'], 500)
            return SimpleNamespace(modes=X[:, None, :] * (np.arange(1, 9) / 36)[None, :, None],
                                   omega_hz=np.tile(np.arange(8), (len(X), 1)),
                                   iters=np.full(len(X), 10), capped=np.zeros(len(X), bool))
        with patch('experiments.acs.pipeline.vmd_batch', side_effect=converged):
            vectors, meta = extract_record(r, PROTOCOL, reference=True)
        np.testing.assert_array_equal(np.stack(seen), r.signal_mV.T)
        self.assertEqual(len(vectors['vmd']), 12 * 8 * 28)
        self.assertEqual(meta['label'], r.info.label)
        self.assertLess(meta['reference_max_abs_error'], 1e-12)
        for arm in ('vmd', 'wst'):
            self.assertTrue(np.isfinite(vectors[arm]).all())
            width = len(vectors[arm]) // 12
            self.assertTrue(vectors[arm + '_names'][width].startswith('II:'))
            self.assertEqual(len(set(vectors[arm + '_names'])), len(vectors[arm]))


if __name__ == '__main__':
    unittest.main()
