"""Independent numerical fixtures and provenance checks for rotational features."""
import copy
import tempfile
import unittest

from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from ecgvmd.rotational import analytic_descriptors, arc_centroid, decompose, extract, RotationalError
import numpy as np


class RotationalTests(unittest.TestCase):
    def setUp(self):
        self.spec = C.read(C.SPEC)

    def test_known_sinusoid_envelopes_frequencies_global_energy(self):
        fs=500; t=np.arange(5000)/fs
        amplitudes=np.array([.2,.4,.6,.8]); frequencies=np.array([3.,7.,19.,43.])
        modes=amplitudes[:,None]*np.cos(2*np.pi*frequencies[:,None]*t)
        values, details, _=analytic_descriptors(modes,fs,self.spec['analytic'])
        np.testing.assert_allclose(values[:4],frequencies,atol=1e-10,rtol=0)
        np.testing.assert_allclose(values[4:8],amplitudes,atol=1e-10,rtol=0)
        self.assertAlmostEqual(values[8],np.sum(amplitudes**2*frequencies)/np.sum(amplitudes**2),10)
        self.assertAlmostEqual(values[9],np.linalg.norm(amplitudes),10)
        self.assertEqual(details['retained_samples'],4750)

    def test_amplitude_modulation(self):
        t=np.arange(5000)/500
        env=1+.2*np.cos(2*np.pi*t)
        frequencies=np.array([10,20,30,40])
        values,_,_=analytic_descriptors(env[None,:]*np.cos(2*np.pi*frequencies[:,None]*t),500,self.spec['analytic'])
        np.testing.assert_allclose(values[:4],frequencies,atol=1e-9,rtol=0)
        np.testing.assert_allclose(values[4:8],env[125:-125].mean(),atol=1e-10,rtol=0)

    def test_centroid_density_invariance_circle_and_scale(self):
        # Same open piecewise-linear path with very different vertex density.
        original=np.array([0+0j,2+0j,2+2j])
        dense=np.r_[np.linspace(0,2,200)+0j,2+2j]
        self.assertAlmostEqual(abs(arc_centroid(original)[0]-(1.5+.5j)),0,12)
        self.assertAlmostEqual(abs(arc_centroid(original)[0]-arc_centroid(dense)[0]),0,12)
        circle=np.exp(1j*np.linspace(0,2*np.pi,10001))
        self.assertLess(abs(arc_centroid(circle)[0]),1e-12)
        self.assertAlmostEqual(abs(arc_centroid(2*original+3j)[0]-(3+4j)),0,12)
        with self.assertRaises(RotationalError): arc_centroid(np.zeros(10))

    def test_phase_floor_and_input_rejection(self):
        with self.assertRaises(RotationalError):
            analytic_descriptors(np.zeros((4,5000)),500,self.spec['analytic'])
        for data in (np.zeros(5000), np.full(5000,np.nan), np.arange(100)):
            with self.assertRaises(RotationalError): extract(data,500,self.spec)

    def test_emd_instrumentation_matches_unmodified_solver(self):
        from PyEMD import EMD
        t=np.arange(5000)/500
        signal=sum(a*np.cos(2*np.pi*f*t) for a,f in [(0.1,85),(0.2,31),(0.4,9),(0.8,2)])
        modes,residual,details=decompose(signal,self.spec['emd'])
        reference=EMD(**self.spec['emd']['parameters'])
        reference.emd(signal,max_imf=4)
        expected,remaining=reference.get_imfs_and_residue()
        np.testing.assert_array_equal(modes,expected)
        np.testing.assert_array_equal(residual,remaining)
        np.testing.assert_allclose(modes.sum(0)+residual,signal,atol=1e-14,rtol=0)
        self.assertEqual(len(details['sifting']),4)
        np.testing.assert_array_equal(extract(signal,500,self.spec)[0],extract(signal,500,self.spec)[0])

    def test_cap_and_insufficient_components_are_failures(self):
        t=np.arange(5000)/500
        with self.assertRaisesRegex(RotationalError,'Fewer than four'):
            decompose(np.cos(2*np.pi*7*t),self.spec['emd'])
        spec=copy.deepcopy(self.spec['emd'])
        spec['parameters']['MAX_ITERATION']=2
        with self.assertRaisesRegex(RotationalError,'iteration cap') as caught:
            decompose(np.sin(2*np.pi*7*t)+.2*np.sin(2*np.pi*31*t),spec)
        self.assertTrue(any(x['capped'] for x in caught.exception.diagnostics['sifting']))

    def test_selection_ignores_labels_order_and_excludes_validation(self):
        rows=[dict(record_id=str(i),patient_id=str(i//2),partition='validation' if i>15 else 'fit',label=str(i%2)) for i in range(20)]
        a=C.selection(rows,4,'test')
        modified=[dict(r,label=str(1-int(r['label']))) for r in reversed(rows)]
        b=C.selection(modified,4,'test')
        self.assertEqual([r['record_id'] for r in a],[r['record_id'] for r in b])
        self.assertEqual(len({r['patient_id'] for r in a}),4)
        self.assertTrue(all(r['partition']=='fit' for r in a))

    def test_completed_artifact_rejects_changed_bytes_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=C.Path(tmp)
            C.write(p/'value.json',{'x':1})
            C.complete(p,['value.json'],'a')
            self.assertTrue(C.verified(p,'a'))
            with self.assertRaises(ValueError): C.verified(p,'b')
            C.write(p/'value.json',{'x':2})
            with self.assertRaises(ValueError): C.verified(p,'a')


if __name__ == '__main__':
    unittest.main()
