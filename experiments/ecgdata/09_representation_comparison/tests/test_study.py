import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.fft import dct
p=Path(__file__).resolve().parents[1]/'study.py';spec=importlib.util.spec_from_file_location('ecg09',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from attention_ecg.cepstral import cepstral_features,CepstralConfig

class RepresentationChecks(unittest.TestCase):
 def test_filterbank_matches_existing_lfcc_and_batch(self):
  cfg=CepstralConfig(fs=128,frame_samples=128,hop_samples=32,n_fft=256,n_filters=32,n_ceps=16,fmin=0.,fmax=64.)
  x=np.random.default_rng(10).normal(size=(3,500));bands=m.extract_bands(x,cfg,batch=2)
  np.testing.assert_allclose(bands,m.extract_bands(x,cfg,batch=3),rtol=0,atol=0)
  for i in range(3): np.testing.assert_allclose(dct(bands[i],type=2,norm='ortho',axis=-1)[...,:16],cepstral_features(x[i,:,None],cfg)[0],rtol=1e-5,atol=2e-5)
 def test_raw_cnn_shape_and_finite_gradient(self):
  model=m.RawCNN();x=m.torch.randn(4,1,500);output=model(x);self.assertEqual(tuple(output.shape),(4,3));output.sum().backward();self.assertTrue(all(m.torch.isfinite(v.grad).all() for v in model.parameters()))

 def test_raw_fit_patient_isolation_and_reuse(self):
  n=30;fold=np.repeat(np.arange(5),6);labels=np.tile(np.repeat(np.arange(3),2),5)
  data={'raw':np.random.default_rng(9).normal(size=(n,500)).astype(np.float32),'y':labels.astype(np.int64),'fold':fold,'patients':np.array([f'p{i}' for i in range(n)])}
  p=m.old.protocol();p['training']['epochs']=1;p['training']['batch_size']=8
  with tempfile.TemporaryDirectory() as td:
   path=Path(td);m.setup('cpu');m.raw_fit(path,data,0,0,p,'cpu');self.assertTrue(m.verify_trial(path))
   before=m.file_hash(path/'checkpoint.pt');m.raw_fit(path,data,0,0,p,'cpu');self.assertEqual(before,m.file_hash(path/'checkpoint.pt'))
   with np.load(path/'predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['indices'],np.arange(6))

 def test_32_band_swin_and_control_fit(self):
  n=30;fold=np.repeat(np.arange(5),6);labels=np.tile(np.repeat(np.arange(3),2),5)
  data={'X':np.random.default_rng(11).normal(size=(n,13,1,32)).astype(np.float32),'y':labels.astype(np.int64),'fold':fold,'patients':np.array([f'p{i}' for i in range(n)])}
  p=m.old.protocol();p['training']['epochs']=1;p['training']['batch_size']=8;p['model']['n_ceps']=32
  with tempfile.TemporaryDirectory() as td:
   m.setup('cpu');root=Path(td);m.fit_control(root/'logistic',data,0,p);m.fit_swin(root/'swin',data,0,0,p,'cpu')
   self.assertTrue(m.verify_trial(root/'logistic'));self.assertTrue(m.verify_trial(root/'swin'))
   for folder in ('logistic','swin'):
    with np.load(root/folder/'predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['indices'],np.arange(6))

if __name__=='__main__':unittest.main()
