"""Exercise durable classical fits and the refusal to reuse corrupted outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as runner
import numpy as np
from sklearn.preprocessing import StandardScaler


class RunnerTests(unittest.TestCase):
    def test_classical_artifacts_prediction_identity_and_tamper_rejection(self):
        rng=np.random.default_rng(61)
        X=rng.normal(size=(40,12)); y=np.array([0,1]*20)
        protocol=json.loads((runner.BASE/'protocols/acs_omi_protocol_v1_retry128k.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); arm=out/'vmd';arm.mkdir()
            scaler=StandardScaler().fit(X[:30])
            runner.atomic_joblib(arm/'preprocessor.joblib',scaler)
            runner.atomic_npz(arm/'inputs.npz',X_fit=scaler.transform(X[:30]),X_validation=scaler.transform(X[30:]),
                y_fit=y[:30],y_validation=y[30:],validation_record_ids=np.array([f'r{i}' for i in range(10)]),
                validation_patients=np.array([f'p{i}' for i in range(10)]))
            runner.old.atomic_json(arm/'preprocessing.json',dict(status='complete',context_sha='context',
                files={n:runner.old.serial.file_hash(arm/n) for n in ('inputs.npz','preprocessor.joblib')}))
            with patch.object(runner,'read_run',return_value=({'protocol':protocol},'context')):
                for name in ('logistic','weighted_knn'):
                    runner.trial(out,'vmd',name)
                    self.assertIn('verified complete',runner.trial(out,'vmd',name))
                    with np.load(arm/name/'predictions.npz') as z:
                        np.testing.assert_array_equal(z['record_ids'],[f'r{i}' for i in range(10)])
                        np.testing.assert_array_equal(z['y'],y[30:])
                (arm/'logistic/predictions.npz').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    runner.trial(out,'vmd','logistic')


if __name__=='__main__':
    unittest.main()
