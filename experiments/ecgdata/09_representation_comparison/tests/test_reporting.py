"""Synthetic end-to-end report generation; contains no real ECG predictions."""
import contextlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
p=Path(__file__).resolve().parents[1]/'study.py';spec=importlib.util.spec_from_file_location('ecg09_reporting_test',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class ReportingTest(unittest.TestCase):
 def test_synthetic_complete_report_and_predictions(self):
  n=30;y=np.tile(np.repeat(np.arange(3),2),5).astype(np.int64);fold=np.repeat(np.arange(5),6);patients=np.array([f'patient{i:02d}' for i in range(n)])
  data={'y':y,'patients':patients,'fold':fold,'row_ids':np.arange(n),'window_start':np.zeros(n,dtype=int),'v1_index':np.arange(n)}
  baseline={'swin':[np.roll(y,k) for k in (0,1,2)],'logistic':[np.roll(y,1)]}
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);run=root/'results';run.mkdir();cache=root/'features';cache.mkdir();report=root/'reports'
   (run/'manifest.json').write_text('{}');(cache/'extraction.json').write_text(json.dumps({'shape':[n,13,1,32],'exclusions':0}))
   for fold_id in range(5):
    idx=np.flatnonzero(fold==fold_id)
    for arm in ('band_swin','band_logistic','raw_cnn'):
     for seed in ([None] if arm=='band_logistic' else (0,1,2)):
      name=f'{arm}_fold{fold_id}' if seed is None else f'{arm}_seed{seed}_fold{fold_id}';folder=run/name;folder.mkdir()
      pred=np.roll(y,0 if seed is None else seed)[idx]
      np.savez(folder/'predictions.npz',indices=idx,patients=patients[idx],y=y[idx],pred=pred)
      (folder/'fit.json').write_text(json.dumps({'seconds':1.,'epochs':40 if seed is not None else None,'warnings':[]}))
   with patch.object(s,'RUN',run),patch.object(s,'CACHE',cache),patch.object(s,'REPORT',report),patch.object(s,'read_data',return_value=data),patch.object(s,'verify_trial',return_value=True),patch.object(s.old,'verify_cache',return_value=data),patch.object(s,'collect_predictions',return_value=baseline):
    with contextlib.redirect_stdout(io.StringIO()):s.report()
    print("Synthetic report and 2,000-draw bootstrap generated successfully; no real ECG data.",flush=True)
   a=json.loads((report/'analysis.json').read_text());self.assertEqual(len(a['fits']),35)
   self.assertIn('band_swin_minus_lfcc_swin',a['results']['v1_windows']['bootstrap']['paired_differences'])
   self.assertTrue((report/'REPORT.md').exists());self.assertTrue((run/'predictions_export.npz').exists())
if __name__=='__main__':unittest.main()
