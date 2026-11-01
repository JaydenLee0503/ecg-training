"""A real interrupted training checkpoint must reproduce an uninterrupted fit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
p=Path(__file__).resolve().parents[1]/'study.py'
spec=importlib.util.spec_from_file_location('ecg09_recovery',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RawRecoveryTest(unittest.TestCase):
 def test_interrupted_resume_matches_uninterrupted(self):
  n=30;labels=np.tile(np.repeat(np.arange(3),2),5)
  data={'raw':np.random.default_rng(19).normal(size=(n,500)).astype(np.float32),'y':labels.astype(np.int64),'fold':np.repeat(np.arange(5),6),'patients':np.array([f'p{i}' for i in range(n)])}
  protocol=m.old.protocol();protocol['training']['epochs']=2;protocol['training']['batch_size']=8
  m.setup('cpu')
  with tempfile.TemporaryDirectory() as td:
   baseline=Path(td)/'baseline';resumed=Path(td)/'resumed'
   m.raw_fit(baseline,data,0,1,protocol,'cpu')
   original=m.save_checkpoint
   def interrupt(folder,state):
    original(folder,state)
    if state['epoch']==1: raise KeyboardInterrupt('synthetic interruption after checkpoint')
   try:
    m.save_checkpoint=interrupt
    with self.assertRaises(KeyboardInterrupt):m.raw_fit(resumed,data,0,1,protocol,'cpu')
   finally:m.save_checkpoint=original
   m.raw_fit(resumed,data,0,1,protocol,'cpu')
   a=m.torch.load(baseline/'checkpoint.pt',map_location='cpu',weights_only=True)
   b=m.torch.load(resumed/'checkpoint.pt',map_location='cpu',weights_only=True)
   for key in a['model']:self.assertTrue(m.torch.equal(a['model'][key],b['model'][key]),key)
   with np.load(baseline/'predictions.npz',allow_pickle=False) as x,np.load(resumed/'predictions.npz',allow_pickle=False) as y:
    np.testing.assert_array_equal(x['probabilities'],y['probabilities'])

if __name__=='__main__':unittest.main()
