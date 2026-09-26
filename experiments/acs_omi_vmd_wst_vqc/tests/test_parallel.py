"""Actual spawned processes, bounded failure handling, locking and exact references."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.acs_omi_vmd_wst_vqc.parallel import (bounded_execute, check_loaded, compare_serial,
                                      pilot_ids, run_lock, save_feature)
from experiments.acs_omi_vmd_wst_vqc.scripts import experiment as serial


def tiny_worker(record, protocol, attempt_path, reference):
    if protocol.get('fail') and record.record_id == '00000':
        return {'ok': False, 'record_id': record.record_id, 'error': 'intentional fixture failure'}
    time.sleep(.05)
    Path(attempt_path).write_text('fixture convergence log\n')
    return {'ok': True, 'record_id': record.record_id, 'reference': reference}


class ParallelTests(unittest.TestCase):
    def test_spawned_workers_finish_every_row_once_with_bounded_loading(self):
        rows = [{'record_id': f'{i:05d}'} for i in range(5)]
        loaded, accepted = [], []
        def load(row):
            loaded.append(row['record_id'])
            self.assertLessEqual(len(loaded) - len(accepted), 2)
            return SimpleNamespace(**row)
        def accept(row, result):
            self.assertEqual(row['record_id'], result['record_id'])
            accepted.append(result['record_id'])
        with tempfile.TemporaryDirectory() as tmp:
            failures = bounded_execute(rows, load, accept, Path(tmp), {}, 2, {'00000'}, worker=tiny_worker)
        self.assertEqual(failures, [])
        self.assertEqual(sorted(accepted), [r['record_id'] for r in rows])

    def test_first_worker_failure_stops_new_submissions_and_retains_running_success(self):
        rows = [{'record_id': f'{i:05d}'} for i in range(10)]
        loaded, accepted = [], []
        def load(row):
            loaded.append(row['record_id'])
            return SimpleNamespace(**row)
        with tempfile.TemporaryDirectory() as tmp:
            failures = bounded_execute(rows, load, lambda row, result: accepted.append(row['record_id']),
                                       Path(tmp), {'fail': True}, 2, set(), worker=tiny_worker)
        self.assertEqual(loaded, ['00000', '00001'])
        self.assertEqual(accepted, ['00001'])
        self.assertEqual(failures[0]['record_id'], '00000')

    def test_parent_loader_failure_is_recorded_without_submitting_further_rows(self):
        def load(row):
            raise ValueError('Changed waveform identity')
        with tempfile.TemporaryDirectory() as tmp:
            failures = bounded_execute([{'record_id': '00000'}], load, lambda *_: None,
                                       Path(tmp), {}, 2, set(), worker=tiny_worker)
        self.assertEqual(failures[0]['error_type'], 'ValueError')
        self.assertIn('waveform identity', failures[0]['error'])

    def test_run_lock_excludes_second_writer_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'fixture'):
                with run_lock(folder):
                    with self.assertRaisesRegex(RuntimeError, 'Another process'):
                        with run_lock(folder):
                            pass
                    raise ValueError('fixture')
            with run_lock(folder):
                pass

    def test_pilot_selection_is_fit_only_deterministic_and_label_independent(self):
        rows = [{'record_id': str(i), 'partition': 'fit' if i < 20 else 'validation', 'label': i % 2}
                for i in range(30)]
        chosen = pilot_ids(rows, {'id': 'fixture'}, {'pilot': {'records': 16}})
        self.assertEqual(len(chosen), 16)
        self.assertTrue(all(int(rid) < 20 for rid in chosen))
        for row in rows:
            row['label'] = 1 - row['label']
        self.assertEqual(chosen, pilot_ids(list(reversed(rows)), {'id': 'fixture'}, {'pilot': {'records': 16}}))

    def test_reference_requires_identical_features_names_inputs_and_convergence(self):
        vectors = dict(vmd=np.arange(3.), wst=np.arange(4.), vmd_names=np.array(['a', 'b', 'c']),
                       wst_names=np.array(['d', 'e', 'f', 'g']))
        details = dict(record_id='00001', physical_input_sha256='wave', vmd_iterations=[4],
                       vmd_limits=[20], vmd_attempts=[{'capped': False}])
        self.assertEqual(compare_serial(vectors, details, vectors, details)['status'], 'passed')
        for key in vectors:
            changed = {k: v.copy() for k, v in vectors.items()}
            changed[key][0] = 'x' if key.endswith('names') else -1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                compare_serial(changed, details, vectors, details)
        with self.assertRaisesRegex(ValueError, 'diagnostic mismatch'):
            compare_serial(vectors, {**details, 'vmd_iterations': [5]}, vectors, details)

    def test_checkpoint_identity_and_corruption_checks_apply_to_parallel_outputs(self):
        row = dict(record_id='00001', patient_id='P00001', label='1', waveform_sha256='wave')
        vectors = dict(vmd=np.arange(3.), wst=np.arange(4.), vmd_names=np.array(['a', 'b', 'c']),
                       wst_names=np.array(['d', 'e', 'f', 'g']))
        details = dict(**row, vmd_features=3, wst_features=4)
        details['label'] = 1
        result = {'vectors': vectors, 'details': details}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            saved = save_feature(folder, row, result, 'parallel', 'parent')
            self.assertEqual(saved['parent_manifest_sha256'], 'parent')
            serial.checked_feature(folder, row, 'parallel')
            path = folder / '00001.npz'
            path.write_bytes(path.read_bytes() + b'corruption')
            with self.assertRaises(ValueError):
                serial.checked_feature(folder, row, 'parallel')

    def test_changed_test_or_patient_identity_never_reaches_workers(self):
        row = dict(record_id='00001', patient_id='P00001', label='1', waveform_sha256='wave')
        record = SimpleNamespace(info=SimpleNamespace(record_id='00001', patient_id='P00001', label=1, split='test'),
                                 decision=SimpleNamespace(waveform_sha256='wave'))
        with self.assertRaises(ValueError):
            check_loaded(record, row)
        record.info.split = 'train'
        check_loaded(record, row)
        record.info.patient_id = 'P00002'
        with self.assertRaises(ValueError):
            check_loaded(record, row)


if __name__ == '__main__':
    unittest.main()
