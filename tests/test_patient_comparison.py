"""Regression checks for source grouping and the matched VQC evaluation boundary."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

import ecgvmd as E
from ecgvmd.data import ECGDataset, load_subject_map
from scripts.matched_vqc import checked_splits
from scripts.verify_ecg_sources import decode_212
from scripts.summarize_patient_vqc import accuracy, macro_f1


class PatientComparisonTests(unittest.TestCase):
    def test_nondyadic_scattering_metadata_and_reference(self):
        from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D
        signal = np.random.default_rng(4).normal(size=(2, 500))
        for scale in (50, 64):
            reference = ScatteringNumPy1D(J=6, shape=500, Q=(8, 1), T=scale)
            result = E.scatter_batch(signal, T=scale)
            np.testing.assert_allclose(result.coeffs, reference(signal), atol=1e-12, rtol=0)
            self.assertEqual(result.bin_seconds, 2**reference.log2_T / result.fs)
            self.assertEqual(result.invariance_seconds, scale / result.fs)

    def test_bootstrap_metric_functions_match_sklearn(self):
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        truth = np.array(['ARR']*90+['CHF']*30+['NSR']*40)
        predicted = np.array(['ARR']*160)
        cm = confusion_matrix(truth,predicted,labels=E.CLASS_ORDER)
        self.assertAlmostEqual(float(accuracy(cm)),accuracy_score(truth,predicted))
        self.assertAlmostEqual(float(macro_f1(cm)),f1_score(truth,predicted,average='macro',labels=E.CLASS_ORDER))

    def test_wfdb_signed_decoding(self):
        # Format 212: first/last byte are low bytes, middle nibbles are high bits.
        raw = bytes([0xff, 0x7f, 0xff, 0x00, 0xf8, 0xff])
        np.testing.assert_array_equal(decode_212(raw), [[-1, 2047], [-2048, -1]])

    def test_segment_defaults_to_patient_and_preserves_windows(self):
        ds = ECGDataset(np.arange(64).reshape(4,16), np.array(['ARR']*4),
                        np.arange(4), 128, 'synthetic',
                        np.array(['a','a','b','b']), np.array(['person']*4))
        cfg = E.Config(seg_len=8)
        W,y,g = E.segment(ds,cfg)
        Wr,yr,rows = E.segment(ds,cfg,grouping='row')
        np.testing.assert_array_equal(W,Wr)
        np.testing.assert_array_equal(y,yr)
        self.assertEqual(set(g), {'person'})
        self.assertEqual(len(set(rows)), 4)

    def test_missing_patient_ids_fail_instead_of_using_rows(self):
        ds = ECGDataset(np.ones((1,16)),np.array(['ARR']),np.array([0]),128,'synthetic')
        with self.assertRaisesRegex(ValueError,'Verified patient IDs are missing'):
            E.segment(ds,E.Config(seg_len=8))

    def test_subject_map_rejects_reordered_same_class_rows(self):
        data = np.array([[1.,2.,3.],[4.,5.,6.]])
        labels = np.array(['ARR','ARR'])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'map.csv'
            with path.open('w',newline='') as stream:
                w = csv.DictWriter(stream,fieldnames=['row','label','row_sha256','source_id','patient_id'])
                w.writeheader()
                for i,x in enumerate(data):
                    w.writerow(dict(row=i,label='ARR',row_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
                                    source_id='record',patient_id='patient'))
            source,patients = load_subject_map(data,labels,path)
            self.assertEqual(set(patients), {'patient'})
            with self.assertRaisesRegex(ValueError,'does not match ECGData row'):
                load_subject_map(data[::-1],labels,path)

    def test_patient_splits_keep_multiple_sources_together(self):
        y = np.repeat(E.CLASS_ORDER, 40)
        groups = np.concatenate([np.repeat([f'{c}/{i}' for i in range(10)],4) for c in E.CLASS_ORDER])
        X = np.arange(len(y))[:,None]
        for train,test in checked_splits(X,y,groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    @unittest.skipUnless(Path('ECGData.mat').exists(),'local ECGData.mat required')
    def test_real_map_merges_both_leads_and_repeat_subject(self):
        ds = E.load_ecgdata()
        self.assertEqual(len(set(ds.source_ids)),81)
        self.assertEqual(len(set(ds.patient_ids)),80)
        repeated = np.isin(ds.source_ids,['mitdb/201','mitdb/202'])
        self.assertEqual(repeated.sum(),4)
        self.assertEqual(len(set(ds.patient_ids[repeated])),1)
        self.assertEqual(ds.patient_ids[0],ds.patient_ids[43])


if __name__ == '__main__':
    unittest.main()
