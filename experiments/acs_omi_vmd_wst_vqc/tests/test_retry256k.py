import copy
import json
import unittest
from experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu_retry256k import assert_extension,BASE


class ExtensionTests(unittest.TestCase):
    def test_only_declared_cap_can_change(self):
        parent=json.loads((BASE/'protocols/acs_omi_protocol_v1_retry128k.json').read_text())
        candidate=json.loads((BASE/'protocols/acs_omi_protocol_v1_retry256k.json').read_text())
        assert_extension(parent,candidate)
        for field,value in [('tol',1e-6),('alpha',1000),('iteration_limits',[256000])]:
            changed=copy.deepcopy(candidate);changed['vmd'][field]=value
            with self.assertRaises(ValueError):
                assert_extension(parent,changed)
        changed=copy.deepcopy(candidate);changed['input']['normalization']='zscore'
        with self.assertRaises(ValueError):
            assert_extension(parent,changed)


if __name__=='__main__':
    unittest.main()
