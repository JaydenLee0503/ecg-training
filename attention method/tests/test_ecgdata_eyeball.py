import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attention_ecg.ecgdata_eyeball import _feature_spec, _load_inputs
from attention_ecg.ecgdata import read_json


class ECGDataEyeballTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(ROOT/'ecgdata_eyeball_protocol_v1.json')

    def test_feature_spec_matches_frozen_numerics(self):
        spec = _feature_spec(self.protocol)
        self.assertEqual(spec['minimum_seconds'], 30)
        self.assertEqual(spec['emd']['version'], '1.6.4')
        self.assertEqual(spec['emd']['components'], 4)
        self.assertEqual(spec['emd']['parameters']['MAX_ITERATION'], 64000)
        self.assertEqual(spec['analytic']['trim_seconds'], .25)

    def test_real_data_windows_keep_frozen_patient_folds(self):
        _, _, data = _load_inputs(ROOT.parent/'ECGData.mat')
        self.assertEqual(data['signals'].shape, (1620, 3840))
        self.assertEqual(len(set(data['patients'])), 80)
        self.assertEqual([int((data['fold']==f).sum()) for f in range(5)], [340,320,320,320,320])
        for patient in set(data['patients']):
            idx = data['patients'] == patient
            self.assertEqual(len(set(data['fold'][idx])), 1)
            self.assertEqual(len(set(data['y'][idx])), 1)
        self.assertTrue((data['starts'] % 3840 == 0).all())
        self.assertTrue((abs(data['signals'].mean(axis=1)) < 1e-12).all())
        self.assertTrue(((data['signals'].std(axis=1) - 1) < 1e-12).all())

    def test_protocol_is_valid_json(self):
        json.loads((ROOT/'ecgdata_eyeball_protocol_v1.json').read_text())


if __name__ == '__main__':
    unittest.main()
