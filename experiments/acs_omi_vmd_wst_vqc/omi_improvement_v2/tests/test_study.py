import copy
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2 import common as c
from experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2.data import inner_splits,normalization,oversample
from experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2 import engine as e
from experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2.workflow import choose
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.metrics import Curve
import torch
from attention_ecg.model import CepstralSwin


class StudyTests(unittest.TestCase):
    def setUp(self):
        e.setup('cpu')
        self.spec=c.read(c.BASE/'protocol.json')

    def test_inner_patient_isolation_and_coverage(self):
        data=dict(y=np.tile([0,0,1,1],15),patients=np.repeat(np.arange(30),2),
                  partitions=np.array(['subfit']*48+['calibration']*6+['validation']*6))
        splits=inner_splits(data,self.spec)
        for train,held in splits:
            self.assertFalse(set(data['patients'][train]) & set(data['patients'][held]))
            self.assertTrue(np.all(data['partitions'][np.r_[train,held]]=='subfit'))

    def test_normalization_ignores_held_out(self):
        rng=np.random.default_rng(0)
        for raw,shape in [(False,(10,5,12,20)),(True,(10,12,51))]:
            X=rng.normal(size=shape).astype('float32')
            before=normalization(X,np.arange(6),raw)
            X[6:]=1e9
            after=normalization(X,np.arange(6),raw)
            for a,b in zip(before,after): np.testing.assert_array_equal(a,b)
            axis=(0,2) if raw else (0,1)
            expected=X[:6].astype(float).mean(axis)
            np.testing.assert_allclose(before[0].squeeze() if raw else before[0],expected,atol=1e-12)

    def test_sampling_preserves_originals(self):
        y=np.array([0]*13+[1]*3)
        ix=oversample(y,12)
        np.testing.assert_array_equal(ix[:len(y)],np.arange(len(y)))
        np.testing.assert_array_equal(np.bincount(y[ix]),[13,13])

    def test_zero_dropout_equivalent(self):
        spec=copy.deepcopy(self.spec)
        spec['swin_model']=dict(n_leads=12,n_ceps=20,num_classes=2,embed_dim=8,depths=[2],heads=[2],window_size=4)
        torch.manual_seed(0); base=CepstralSwin(**spec['swin_model'])
        torch.manual_seed(0); adapted=e.model('swin0',spec)
        x=torch.randn(3,9,12,20)
        torch.testing.assert_close(base(x),adapted(x),rtol=0,atol=0)
        base(x).sum().backward(); adapted(x).sum().backward()
        for a,b in zip(base.parameters(),adapted.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_dropout_recovery_and_reload(self):
        self.recovery('cpu')

    def recovery(self,device):
        e.setup(device)
        spec=copy.deepcopy(self.spec)
        spec['epochs']=[1,2]
        spec['swin_model']=dict(n_leads=12,n_ceps=20,num_classes=2,embed_dim=8,depths=[2],heads=[2],window_size=4)
        spec['training'].update(epochs=2,batch_size=4)
        rng=np.random.default_rng(13)
        X=rng.normal(size=(16,9,12,20)).astype('float32')
        data=dict(y=np.tile([0,1],8),patients=np.arange(16),record_ids=np.arange(16).astype(str))
        tr,held=np.arange(10),np.arange(10,16)
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'full',Path(temp)/'resume'
            e.fit_neural(a,data,X,tr,held,'swin1',0,2,spec,'test',device,inner=True)
            e.fit_neural(b,data,X,tr,held,'swin1',0,2,spec,'test',device,inner=True,stop_after=1)
            e.fit_neural(b,data,X,tr,held,'swin1',0,2,spec,'test',device,inner=True)
            np.testing.assert_array_equal(c.arrays(a/'epoch2/held.npz')['score'],c.arrays(b/'epoch2/held.npz')['score'])
            digest=c.sha(b/'completed.json')
            e.fit_neural(b,data,X,tr,held,'swin1',0,2,spec,'test',device,inner=True)
            self.assertEqual(digest,c.sha(b/'completed.json'))

    @unittest.skipUnless(os.environ.get('OMI_IMPROVEMENT_CUDA')=='1','GPU preflight explicitly enabled')
    def test_gpu_recovery_and_numerics(self):
        self.recovery('cuda')
        for arm in ('swin0','swin1','raw'):
            torch.manual_seed(17)
            cpu=e.model(arm,self.spec).eval()
            gpu=copy.deepcopy(cpu).cuda().eval()
            shape=(2,12,5000) if arm=='raw' else (2,37,12,20)
            x=torch.randn(shape)
            a,b=cpu(x),gpu(x.cuda())
            torch.testing.assert_close(a,b.cpu(),rtol=2e-4,atol=2e-5)
            a.square().sum().backward(); b.square().sum().backward()
            for p,q in zip(cpu.parameters(),gpu.parameters()):
                torch.testing.assert_close(p.grad,q.grad.cpu(),rtol=5e-3,atol=2e-4)

    def test_raw_shape_and_finite_gradients(self):
        net=e.model('raw',self.spec)
        out=net(torch.randn(2,12,5000))
        self.assertEqual(tuple(out.shape),(2,2))
        out.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_selection_ties_and_threshold_ties(self):
        rows=[dict(epoch=10,AP=.5),dict(epoch=40,AP=.5)]
        self.assertEqual(choose(rows)['epoch'],10)
        points=Curve(np.array([0,1,1,0]),np.array([.8,.5,.5,.2])).operating_points([90])
        self.assertEqual(points[0]['threshold'],.5)
        self.assertEqual(points[0]['sensitivity'],1.)

    def test_logistic_train_only_scaler(self):
        rng=np.random.default_rng(9)
        X=rng.normal(size=(20,6)); X[14:]+=100
        data=dict(y=np.tile([0,1],10),patients=np.arange(20),record_ids=np.arange(20).astype(str))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            e.fit_logistic(folder,data,X,np.arange(14),np.arange(14,20),.1,'test')
            import joblib
            scaler,_=joblib.load(folder/'model.joblib')
            np.testing.assert_allclose(scaler.mean_,X[:14].mean(0),atol=1e-12)


if __name__=='__main__':
    unittest.main()
