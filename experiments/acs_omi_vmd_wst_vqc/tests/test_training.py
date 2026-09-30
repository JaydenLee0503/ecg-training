"""Numerical parity and exact recovery of the ACS-local VQC fit loop."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from ecgvmd.reupload import ReuploadVQC
from experiments.acs_omi_vmd_wst_vqc.training import CheckpointVQC


class TrainingTests(unittest.TestCase):
    def test_original_parity_resume_and_changed_input_rejection(self):
        rng=np.random.default_rng(52)
        X=rng.normal(size=(12,4))
        y=np.array([0,0,1]*4)
        settings=dict(n_qubits=2,n_layers=2,reupload=False,epochs=3,schedule_epochs=3,
                      batch_size=5,seed=2)
        reference=ReuploadVQC(**settings).fit(X,y)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            CheckpointVQC(**settings).fit(X,y,folder,'test',stop_after=1)
            resumed=CheckpointVQC(**settings).fit(X,y,folder,'test')
            np.testing.assert_array_equal(resumed.predict_proba(X),reference.predict_proba(X))
            np.testing.assert_array_equal(resumed.loss_,reference.loss_)
            self.assertEqual(resumed.n_steps_,reference.n_steps_)
            with self.assertRaises(ValueError):
                CheckpointVQC(**settings).fit(X+0.1,y,folder,'test')
            with self.assertRaises(ValueError):
                CheckpointVQC(**settings).fit(X,y,folder,'changed')
            (folder/'epoch_003.joblib').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                CheckpointVQC(**settings).fit(X,y,folder,'test')


if __name__=='__main__':
    unittest.main()
