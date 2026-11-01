"""Numerical CPU/GPU check on the new filterbank input and raw control."""
import importlib.util,unittest,os
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'study.py'
spec=importlib.util.spec_from_file_location('ecg09_cuda',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from attention_ecg.model import CepstralSwin

@unittest.skipUnless(os.environ.get('ECG09_TEST_CUDA')=='1','set ECG09_TEST_CUDA=1 for host GPU check')
class CudaAgreement(unittest.TestCase):
 def test_forward_and_gradients(self):
  self.assertTrue(m.torch.cuda.is_available());m.setup('cuda');m.torch.manual_seed(123)
  config=dict(m.old.protocol()['model']);config['n_ceps']=32
  for ctor,shape in ((lambda:CepstralSwin(**config),(3,13,1,32)),(m.RawCNN,(3,1,500))):
   cpu=ctor();gpu=ctor().cuda();gpu.load_state_dict(cpu.state_dict());x=m.torch.randn(*shape)
   a=cpu(x);b=gpu(x.cuda());m.torch.testing.assert_close(a,b.cpu(),rtol=5e-4,atol=1e-5)
   a.square().mean().backward();b.square().mean().backward()
   for c,g in zip(cpu.parameters(),gpu.parameters()):m.torch.testing.assert_close(c.grad,g.grad.cpu(),rtol=1e-3,atol=2e-5)

if __name__=='__main__':unittest.main()
