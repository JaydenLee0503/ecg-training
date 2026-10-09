"""Patient isolation and explicit invalid-input handling for the Lead-I study."""
import tempfile
import unittest
import numpy as np
import joblib
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc import lead_i_study as S


class LeadIStudyTests(unittest.TestCase):
    def test_sampling_ignores_labels_excludes_old_and_validation_and_separates_patients(self):
        rows=[dict(record_id=str(i),patient_id=str(i//2),partition='fit' if i<200 else 'validation',label=str(i%2)) for i in range(240)]
        spec=dict(patients=60,model_fit_patients=40,model_evaluation_patients=20,engineering_patients=10,
            selection_id='a',split_id='b',engineering_id='c')
        a=S.select(rows,{'0','1'},spec)
        b=S.select([dict(r,label=str(1-int(r['label']))) for r in reversed(rows)],{'0','1'},spec)
        self.assertEqual([(r['record_id'],r['study_partition'],r['engineering']) for r in a],
                         [(r['record_id'],r['study_partition'],r['engineering']) for r in b])
        self.assertEqual(len({r['patient_id'] for r in a}),60)
        self.assertFalse({r['patient_id'] for r in a}&{'0','1'})
        self.assertTrue(all(r['partition']=='fit' for r in a))
        self.assertEqual(sum(r['engineering']=='1' for r in a),10)
        self.assertTrue(all(r['study_partition']=='model_fit' for r in a if r['engineering']=='1'))

    def test_scaler_uses_only_valid_training_rows_and_fallback_preserves_evaluation_cohort(self):
        rng=np.random.default_rng(25);X=rng.normal(size=(100,12));y=np.arange(100)%2
        valid=np.ones(100,dtype=bool);valid[-2:]=False;X[-2:]=np.nan
        settings=C.read(S.SPEC)['classifiers']['shared']
        models,_=S.fit_models(X,y,valid,settings)
        np.testing.assert_allclose(models['full_geometry'][0].mean_,X[valid].mean(0))
        evaluation=np.full((3,12),10.);evaluation[-1]=np.nan
        scores=S.predict(models,evaluation,[True,True,False],.07)
        self.assertEqual(scores['full_geometry'].shape,(3,))
        self.assertEqual(scores['full_geometry'][-1],.07)
        self.assertEqual(scores['without_geometry'][-1],.07)
        np.testing.assert_allclose(models['full_geometry'][0].mean_,X[valid].mean(0))
        with tempfile.TemporaryDirectory() as tmp:
            p=C.Path(tmp)/'model.joblib';S.save_models(p,models)
            actual=S.predict(joblib.load(p),evaluation,[True,True,False],.07)
            for name in scores:np.testing.assert_array_equal(actual[name],scores[name])

    def test_paired_bootstrap_identical_models_have_zero_difference(self):
        y=np.tile([0,1],10);p=np.linspace(.05,.95,20)
        result,draws=S.resample_metrics(y,np.arange(20),{'a':p,'b':p.copy()},
            {'same':('a','b')},30,12)
        self.assertEqual(result['comparisons']['same']['intervals']['average_precision'],[0.,0.])
        np.testing.assert_array_equal(draws['a'],draws['b'])
        self.assertEqual(draws['patient_indices'].shape,(30,20))


if __name__=='__main__':unittest.main()
