import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.acs_omi_vmd_wst_vqc.scripts import bundle_gpu as bundler
from experiments.acs_omi_vmd_wst_vqc.loader import LEADS
import numpy as np


class BundleTests(unittest.TestCase):
    def test_complete_alignment_reuse_and_corruption_guard(self):
        protocol=json.loads((bundler.retry.BASE/'protocols/acs_omi_protocol_v1_retry128k.json').read_text())
        rows=[dict(record_id='00001',patient_id='p1',partition='fit',label='0'),
              dict(record_id='00002',patient_id='p2',partition='validation',label='1')]
        vectors={key:value for arm,n in [('vmd',2688),('wst',2808)] for key,value in
                 [(arm,np.arange(n,dtype=np.float32)),(arm+'_names',np.array([str(i) for i in range(n)]))]}
        meta=dict(vmd_iterations=[50]*12,vmd_limits=[2000]*12,
            vmd_attempts=[dict(lead=lead,limit=2000,iterations=50,capped=False) for lead in LEADS])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with (patch.object(bundler,'read_extraction_run',return_value=({'scientific_protocol':protocol},rows,'sha')),
                  patch.object(bundler.retry.old,'checked',return_value=(vectors,meta))):
                bundler.retry.old.atomic_json(out/'extraction_status.json',dict(status='interrupted',records_complete=1,manifest_sha256='sha'))
                with self.assertRaises(ValueError):
                    bundler.bundle(out)
                bundler.retry.old.atomic_json(out/'extraction_status.json',dict(status='complete',records_complete=2,manifest_sha256='sha'))
                bundler.bundle(out)
                with np.load(out/'features.npz') as z:
                    np.testing.assert_array_equal(z['record_ids'],['00001','00002'])
                    np.testing.assert_array_equal(z['vmd'][1],vectors['vmd'])
                bundler.bundle(out)
                (out/'features.npz').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    bundler.bundle(out)


if __name__=='__main__':
    unittest.main()
