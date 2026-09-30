import unittest
import numpy as np
from experiments.acs_omi_vmd_wst_vqc.reporting import binary_metrics,patient_weights,bootstrap


class ReportingTests(unittest.TestCase):
    def test_weighted_metrics_equal_duplicated_ecgs(self):
        y=np.array([0,1,0,1]); score=np.array([0.1,0.5,0.7,0.9]); weights=np.array([2,2,1,1])
        a=binary_metrics(y,score,weights)
        b=binary_metrics(np.repeat(y,weights),np.repeat(score,weights))
        for k in a:
            self.assertAlmostEqual(a[k],b[k])
        self.assertEqual(a['tp'],3)

    def test_patient_clusters_and_identical_model_difference(self):
        patients=np.array(['a','a','b','b','c','c'])
        weights=patient_weights(patients,np.random.default_rng(3))
        np.testing.assert_array_equal(weights[::2],weights[1::2])
        y=np.array([0,1,0,1,0,1]); s=np.linspace(0,1,6)
        result=bootstrap(y,patients,{'a':s,'b':s},{'a':['a'],'b':['b']},
                         {'a-b':('a','b')},20,42)
        for interval in result['paired_differences']['a-b']['intervals'].values():
            self.assertEqual(interval,[0,0])
        self.assertEqual(result['bootstrap']['valid'],20)

    def test_invalid_scores_and_single_class_rejected(self):
        for y,s in [([0,1],[0.1,np.nan]),([0,0],[0.1,0.9]),([0,1],[-0.1,0.5])]:
            with self.assertRaises(ValueError):
                binary_metrics(y,s)


if __name__=='__main__':
    unittest.main()
