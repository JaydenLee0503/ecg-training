import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
from attention_ecg.ecgdata import (CepstralConfig, cepstral_features, read_json, PROTOCOL,
    fit_normalizer, normalize, safe_output, validate_fold_arrays, atomic_json, writer_lock)
from attention_ecg.ecgdata_training import (partition,batch_indices,run_training,
    pooled_features,save_checkpoint,load_checkpoint,complete_trial,verify_trial)
from attention_ecg.ecgdata_report import confusion,scores,patient_confusions,bootstrap_comparison

try:
    import torch
except ModuleNotFoundError:
    torch=None


def fixture():
    y=np.tile(np.repeat(np.arange(3),2),5)
    return dict(y=y,patients=np.repeat([f'p{i}' for i in range(15)],2),
                fold=np.repeat(np.arange(5),6), X=np.arange(30*13*16).reshape(30,13,1,16).astype(np.float32))


class ECGDataTests(unittest.TestCase):
    def test_128hz_features_cover_tail_and_retain_single_lead(self):
        p=read_json(PROTOCOL)
        features,meta=cepstral_features(np.random.default_rng(0).normal(size=(500,1)),CepstralConfig(**p['cepstral']))
        self.assertEqual(features.shape,(13,1,16))
        self.assertEqual(meta['right_pad_samples'],12)
        self.assertEqual(meta['frame_start_samples'][-1],384)
        self.assertTrue(np.isfinite(features).all())

    def test_patient_partitions_and_leakage_rejection(self):
        data=fixture()
        validate_fold_arrays(data['y'],data['patients'],data['fold'])
        for fold in range(5):
            tr,te=partition(data,fold)
            self.assertFalse(set(data['patients'][tr])&set(data['patients'][te]))
        data['fold'][0]=1
        with self.assertRaises(ValueError):
            validate_fold_arrays(data['y'],data['patients'],data['fold'])

    def test_normalization_never_uses_held_out_windows(self):
        data=fixture()
        tr,te=partition(data,0)
        mean,scale=fit_normalizer(data['X'],tr)
        data['X'][te]=1e12
        again=fit_normalizer(data['X'],tr)
        np.testing.assert_array_equal(mean,again[0])
        np.testing.assert_array_equal(scale,again[1])
        transformed=normalize(data['X'][tr],mean,scale)
        np.testing.assert_allclose(transformed.mean((0,1)),0,atol=1e-6)
        np.testing.assert_allclose(transformed.std((0,1)),1,atol=1e-6)

    def test_training_flag_rejected_before_cache_or_model_access(self):
        with patch('attention_ecg.ecgdata_training.verify_cache',side_effect=AssertionError('Should not read cache')):
            with self.assertRaisesRegex(ValueError,'disabled'):
                run_training(execute=False)

    def test_writes_cannot_target_original_data_or_experiments(self):
        for path in (ROOT.parent/'experiments/acs_omi_vmd_wst_vqc/data',ROOT.parent/'results/patient_vmd_wst_vqc',ROOT):
            with self.assertRaises(ValueError):
                safe_output(path,'features')
        self.assertEqual(safe_output(ROOT/'features/test','features'),(ROOT/'features/test').resolve())

    def test_restart_shuffle_and_control_pooling(self):
        batches=batch_indices(31,8,2,17)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(31))
        np.testing.assert_array_equal(np.concatenate(batches),np.concatenate(batch_indices(31,8,2,17)))
        self.assertFalse(np.array_equal(np.concatenate(batches),np.concatenate(batch_indices(31,8,2,18))))
        self.assertEqual(pooled_features(fixture()['X']).shape,(30,32))

    def test_completed_trial_corruption_and_writer_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            atomic_json(path/'result.json',{'a':1})
            complete_trial(path,['result.json'])
            self.assertTrue(verify_trial(path))
            atomic_json(path/'result.json',{'a':2})
            with self.assertRaises(ValueError):
                verify_trial(path)
            with writer_lock(path):
                with self.assertRaises(RuntimeError):
                    with writer_lock(path):
                        pass

    def test_metrics_vote_ties_and_paired_cluster_intervals(self):
        data=fixture()
        y=data['y']
        self.assertEqual(scores(confusion(y,y))['macro_f1'],1)
        pred=y.copy()
        pred[:2]=[0,1]
        _,vote=patient_confusions(y,pred,data['patients'])
        self.assertEqual(vote[0,0,0],1)  # deterministic lowest-index tie
        report=bootstrap_comparison(y,data['patients'],{'swin':[pred,pred,pred],'control':[pred]},30,9)
        for interval in report['paired_difference_intervals_95']['swin_minus_control'].values():
            np.testing.assert_allclose(interval,[0,0],atol=1e-15)

    @unittest.skipIf(torch is None,'Optional PyTorch absent')
    def test_checkpoint_roundtrip_and_corruption_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            state={'epoch':3,'model':{'dummy':torch.arange(4)},'history':[{'loss':1.0}]}
            save_checkpoint(path,state)
            restored=load_checkpoint(path)
            self.assertEqual(restored['epoch'],3)
            torch.testing.assert_close(restored['model']['dummy'],state['model']['dummy'])
            (path/'checkpoint.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    @unittest.skipIf(torch is None,'Optional PyTorch absent')
    def test_small_model_forward_only_and_protocol_defaults(self):
        from attention_ecg.model import CepstralSwin
        torch.set_num_threads(1)
        torch.manual_seed(0)
        p=read_json(PROTOCOL)
        model=CepstralSwin(**p['model']).eval()
        original={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            out=model(torch.zeros(2,13,1,16))
        self.assertEqual(out.shape,(2,3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(sum(v.numel() for v in model.parameters()),100000)
        self.assertTrue(all(torch.equal(original[k],v) for k,v in model.state_dict().items()))


if __name__=='__main__':
    unittest.main()
