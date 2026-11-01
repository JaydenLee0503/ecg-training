import importlib.util
import unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('study.py')
spec=importlib.util.spec_from_file_location('learning_study',p); s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class PatientSubsetTests(unittest.TestCase):
    def test_nested_deterministic_and_disjoint(self):
        patients=np.repeat(np.arange(60),2)
        data={'patients':patients,'y':np.repeat(np.repeat([0,1,2],20),2),'fold':np.repeat(np.tile(np.arange(5),12),2)}
        for fold in range(5):
            for rep in range(3):
                previous=set()
                for f in (.25,.5,.75):
                    tr,te=s.subset(data,fold,rep,f)
                    again,_=s.subset(data,fold,rep,f)
                    np.testing.assert_array_equal(tr,again)
                    self.assertTrue(previous<=set(tr)); previous=set(tr)
                    self.assertFalse(set(patients[tr])&set(patients[te]))
                    self.assertEqual(set(data['y'][tr]),{0,1,2})
                    for patient in set(patients[tr]):
                        self.assertEqual(sum(patients[tr]==patient),2)

if __name__=='__main__': unittest.main()
