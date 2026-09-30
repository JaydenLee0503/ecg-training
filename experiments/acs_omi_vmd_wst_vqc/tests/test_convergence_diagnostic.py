"""The diagnostic observer must not change the frozen batched solver."""
import unittest
import numpy as np
from ecgvmd.vmd import vmd_batch
from experiments.acs_omi_vmd_wst_vqc.vmd_convergence_diagnostic import traced_vmd_batch


class DiagnosticTests(unittest.TestCase):
    def test_observer_preserves_modes_iterations_and_centres(self):
        rng=np.random.default_rng(27)
        x=np.vstack([np.zeros(128),rng.normal(size=128),np.sin(np.arange(128)*0.4)])
        for limit,tol in [(3,1e-20),(1000,1e-7)]:
            traces=[]
            settings=dict(K=4,alpha=2000,dc=True,max_iter=limit,tol=tol)
            reference=vmd_batch(x,**settings)
            observed=traced_vmd_batch(x,**settings,on_iteration=lambda n,d,o:traces.append(n))
            for field in ('modes','iters','omega','capped'):
                np.testing.assert_array_equal(getattr(reference,field),getattr(observed,field))
            self.assertEqual(traces,list(range(1,max(observed.iters)+1)))


if __name__=='__main__':
    unittest.main()
