"""Retry amendment must preserve scientific settings and reject capped reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.acs_omi_vmd_wst_vqc.gpu_retry import (
    assert_compatible_extension, assert_converged, copy_checkpoint, inventory_entry)
from experiments.acs_omi_vmd_wst_vqc.loader import LEADS
from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as old

BASE = Path(__file__).resolve().parents[1]
PARENT = json.loads((BASE/'protocols/acs_omi_protocol_v1.json').read_text())
CANDIDATE = json.loads((BASE/'protocols/acs_omi_protocol_v1_retry128k.json').read_text())


def details():
    return dict(vmd_attempts=[dict(lead=lead, limit=2000, iterations=50, capped=False) for lead in LEADS],
                vmd_iterations=[50]*12, vmd_limits=[2000]*12)


class RetryReuseTests(unittest.TestCase):
    def test_only_append_policy_is_compatible(self):
        assert_compatible_extension(PARENT, CANDIDATE)
        for section, key, value in [('vmd','tol',1e-6), ('vmd','alpha',1000),
                                    ('vmd','iteration_limits',[64000,128000]),
                                    ('preprocessing','k',6), ('split','seed',123),
                                    ('vqc','n_qubits',8)]:
            changed = copy.deepcopy(CANDIDATE)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                assert_compatible_extension(PARENT, changed)

    def test_capped_or_inconsistent_results_cannot_be_reused(self):
        assert_converged(details(), PARENT)
        changes = [lambda m: m['vmd_attempts'].__setitem__(0, dict(lead='I',limit=2000,iterations=2000,capped=True)),
                   lambda m: m['vmd_iterations'].__setitem__(0,51),
                   lambda m: m['vmd_attempts'].pop(),
                   lambda m: m['vmd_attempts'][0].update(limit=4000)]
        for change in changes:
            meta = details()
            change(meta)
            with self.assertRaises(ValueError):
                assert_converged(meta, PARENT)

    def test_import_keeps_bytes_origin_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root/'source', root/'new'
            source.mkdir(); destination.mkdir()
            row = dict(record_id='00001', patient_id='P1', label='1', waveform_sha256='wave')
            vectors = dict(vmd=np.zeros(2688,np.float32),wst=np.zeros(2808,np.float32),
                           vmd_names=np.array([str(i) for i in range(2688)]),
                           wst_names=np.array([str(i) for i in range(2808)]))
            (source/'00001_attempts.jsonl').write_text('{}\n')
            meta = dict(**row, vmd_features=2688, wst_features=2808, execution_origin='gpu_fp64_vmd_cpu_wst', **details())
            meta['label'] = 1
            old.save_feature(source,row,vectors,meta,'old')
            entry = inventory_entry(source,row,'old',PARENT)
            imported = copy_checkpoint(source,destination,row,entry,'new','parent_run')
            self.assertEqual((source/'00001.npz').read_bytes(),(destination/'00001.npz').read_bytes())
            self.assertEqual(imported['execution_origin'],meta['execution_origin'])
            self.assertEqual(imported['manifest_sha256'],'new')
            self.assertEqual(json.loads((source/'00001.json').read_text())['manifest_sha256'],'old')
            copy_checkpoint(source,destination,row,entry,'new','parent_run')
            with self.assertRaises(ValueError):
                copy_checkpoint(source,destination,row,entry,'new','different_source')
            (destination/'00001_attempts.jsonl').write_text('corrupt')
            with self.assertRaises(ValueError):
                copy_checkpoint(source,destination,row,entry,'new','parent_run')


if __name__ == '__main__':
    unittest.main()
