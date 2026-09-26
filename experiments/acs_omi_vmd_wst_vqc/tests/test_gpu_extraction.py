"""Failure visibility, identity boundaries, durable resume and rename provenance."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.acs_omi_vmd_wst_vqc import gpu_extraction as gpu, layout
from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as runner
from experiments.acs_omi_vmd_wst_vqc.tests.test_pipeline import record, PROTOCOL
from ecgvmd.vmd import VMDResult


class GPUExtractionTests(unittest.TestCase):
    def test_final_nonconvergence_logs_every_attempt_before_stopping(self):
        attempts = []
        def capped(X, **kw):
            return VMDResult(np.zeros((len(X), kw['K'], X.shape[1])),
                             np.zeros((len(X), kw['K'])), np.full(len(X), kw['max_iter']),
                             kw['max_iter'], fs=kw['fs'])
        with patch.object(gpu, 'vmd_gpu_batch', side_effect=capped), \
                patch.object(gpu, 'scatter_batch', side_effect=AssertionError('WST must not run after VMD failure')):
            with self.assertRaisesRegex(RuntimeError, 'Unconverged'):
                list(gpu.extract_records([record()], PROTOCOL,
                                         on_attempt=lambda rid, a: attempts.append((rid, a))))
        self.assertEqual(len(attempts), 12*len(PROTOCOL['vmd']['iteration_limits']))
        self.assertTrue(all(rid == '00001' and a['capped'] for rid, a in attempts))
        self.assertEqual(attempts[-1][1]['limit'], 32000)

    def test_reject_test_excluded_duplicate_and_nonfinite_before_cuda(self):
        r = record()
        nonfinite = r.signal_mV.copy()
        nonfinite[0, 0] = np.nan
        invalid = ([replace(r, info=replace(r.info, split='test', label=None))],
                   [replace(r, decision=replace(r.decision, eligible=False))],
                   [replace(r, signal_mV=nonfinite)], [r, r], [], [replace(r, fs=128)])
        with patch.object(gpu, 'vmd_gpu_batch', side_effect=AssertionError('Unexpected GPU work')):
            for records in invalid:
                with self.assertRaises(ValueError):
                    list(gpu.extract_records(records, PROTOCOL, on_attempt=lambda *a: None))

    def test_checkpoint_orphan_recovery_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = dict(record_id='00001', patient_id='P00001', label='1', waveform_sha256='wave')
            vectors = dict(vmd=np.zeros(2688, np.float32), wst=np.ones(2808, np.float32),
                           vmd_names=np.asarray([str(i) for i in range(2688)]),
                           wst_names=np.asarray([str(i) for i in range(2808)]))
            (folder/'00001.npz').write_bytes(b'incomplete orphan')
            pending, complete = runner.scan_checkpoints(folder, [row], 'run')
            self.assertEqual(pending, [row])
            self.assertFalse(complete)
            (folder/'00001_attempts.jsonl').write_text('{}\n')
            details = dict(row, label=1, vmd_features=2688, wst_features=2808)
            runner.save_feature(folder, row, vectors, details, 'run')
            pending, complete = runner.scan_checkpoints(folder, [row], 'run')
            self.assertFalse(pending)
            self.assertEqual(len(complete), 1)
            with self.assertRaises(ValueError):
                runner.scan_checkpoints(folder, [dict(row, patient_id='other')], 'run')
            (folder/'00001_attempts.jsonl').write_text('changed evidence')
            with self.assertRaisesRegex(ValueError, 'attempt log'):
                runner.scan_checkpoints(folder, [row], 'run')

    def test_rename_accepts_only_known_manifest_and_exact_snapshots_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root/'old.txt', root/'new.py'
            old.write_text('old import')
            new.write_text('new import')
            recorded = {'old.py': layout.file_hash(old)}
            migration = dict(support_sha256={}, manifests={'known': dict(original_code_sha256=recorded,
                files={'old.py': dict(snapshot='old.txt', current='new.py', current_sha256=layout.file_hash(new))})})
            path = root/'migration.json'
            path.write_text(json.dumps(migration))
            with patch.object(layout, 'ROOT', root), patch.object(layout, 'MIGRATION', path):
                self.assertFalse(layout.verify_known_migration({'code_sha256': recorded}, 'unknown'))
                self.assertTrue(layout.verify_known_migration({'code_sha256': recorded}, 'known'))
                with self.assertRaises(ValueError):
                    layout.verify_known_migration({'code_sha256': {}}, 'known')
                new.write_text('changed math')
                with self.assertRaises(ValueError):
                    layout.verify_known_migration({'code_sha256': recorded}, 'known')
                new.write_text('new import')
                old.write_text('changed source evidence')
                with self.assertRaises(ValueError):
                    layout.verify_known_migration({'code_sha256': recorded}, 'known')

    def test_sigterm_requests_batch_boundary_stop_and_restores_handler(self):
        before = signal.getsignal(signal.SIGTERM)
        with runner.graceful_stop() as state:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(state['requested'])
            self.assertEqual(state['signal'], signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_completed_extraction_restart_does_not_open_data_or_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'features').mkdir()
            (folder/'extraction_status.json').write_text('{"status":"complete"}')
            args = SimpleNamespace(out=folder, command='extract', max_batches=None, data_dir=folder/'absent')
            with patch.object(runner, 'read_run', return_value=({}, [{'record_id':'one'}], 'sha')), \
                    patch.object(runner, 'validate_pilot'), \
                    patch.object(runner, 'scan_checkpoints', return_value=([], [{}])), \
                    patch.object(runner, 'hardware', side_effect=AssertionError('Unexpected CUDA access')), \
                    patch.object(runner, 'ACSDataset', side_effect=AssertionError('Unexpected source reread')):
                runner.run(args)


if __name__ == '__main__':
    unittest.main()
