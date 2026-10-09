"""Verify numerical failures and completed outputs survive resumptions safely."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot_assessment import compatible, task


class AssessmentTests(unittest.TestCase):
    def test_only_unused_ceiling_may_change_for_import(self):
        old = C.read(C.SPEC); new = copy.deepcopy(old)
        new['emd']['parameters']['MAX_ITERATION'] = 64000
        compatible(old, new)
        for key, value in [('FIXE_H', 5), ('std_thr', .3), ('MAX_ITERATION', 100)]:
            changed = copy.deepcopy(new); changed['emd']['parameters'][key] = value
            with self.assertRaises(ValueError): compatible(old, changed)
        changed = copy.deepcopy(new); changed['analytic']['trim_seconds'] = .1
        with self.assertRaises(ValueError): compatible(old, changed)

    def test_failed_task_is_recorded_not_recomputed_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = C.Path(tmp); x = np.zeros(5000); spec = C.read(C.SPEC)
            result = task(folder, x, spec, 'context', {'record_id':'fixture'})
            self.assertEqual(result['outcome'], 'numerical_failure')
            self.assertTrue(result['files'])
            with patch('experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot_assessment.one', side_effect=AssertionError('reran')):
                self.assertEqual(task(folder, x, spec, 'context', {'record_id':'fixture'}), result)
            with self.assertRaises(ValueError): task(folder, x+1, spec, 'context', {'record_id':'fixture'})
            failure = folder / next(iter(result['files']))
            failure.write_text('{}')
            with self.assertRaises(ValueError): task(folder, x, spec, 'context', {'record_id':'fixture'})

    def test_success_reused_and_corruption_rejected(self):
        t = np.arange(5000)/500
        x = sum(a*np.cos(2*np.pi*f*t) for a,f in [(.1,85),(.2,31),(.4,9),(.8,2)])
        with tempfile.TemporaryDirectory() as tmp:
            folder = C.Path(tmp); spec = C.read(C.SPEC)
            result = task(folder, x, spec, 'context', {'record_id':'fixture'}, True)
            self.assertEqual(result['outcome'], 'success')
            with patch('experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot_assessment.one', side_effect=AssertionError('reran')):
                self.assertEqual(task(folder, x, spec, 'context', {'record_id':'fixture'}, True), result)
            (folder/'features.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError): task(folder, x, spec, 'context', {'record_id':'fixture'}, True)


if __name__ == '__main__':
    unittest.main()
