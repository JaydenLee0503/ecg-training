import importlib.util
import unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('study.py')
spec=importlib.util.spec_from_file_location('undersample_study',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class SamplingTests(unittest.TestCase):
    def test_mixed_patient_retained_whole_and_calibration_excluded(self):
        data={'patients':np.array(['a','a','b','c','d','e','f','z']),
              'y':np.array([1,0,1,0,0,0,0,0]),
              'partitions':np.array(['subfit']*7+['validation'])}
        ix=s.select(data,42)
        self.assertEqual(np.bincount(data['y'][ix]).tolist(),[2,2])
        self.assertTrue({0,1,2}<=set(ix)); self.assertNotIn(7,ix)
        np.testing.assert_array_equal(ix,s.select(data,42))
        for patient in set(data['patients'][ix]):
            self.assertEqual(set(ix[data['patients'][ix]==patient]),set(np.flatnonzero((data['patients']==patient)&(data['partitions']=='subfit'))))

if __name__=='__main__': unittest.main()
