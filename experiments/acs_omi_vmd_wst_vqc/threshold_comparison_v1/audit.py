"""Independently reconstruct operating points from parent report and bootstrap draws."""
import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent; parent=BASE.parent/'undersampled_v1';a=json.loads((BASE/'analysis.json').read_text());old=json.loads((parent/'analysis.json').read_text())
fields=('sensitivity','specificity','precision','tp','fp','fn','tn')
for kind in ('undersampled','oversampled'):
 for arm in ('logistic','swin'):
  g=kind+'_'+arm;members=[g] if arm=='logistic' else [g+'_seed'+str(i) for i in range(3)]
  for row in range(4):
   reference=[old['metrics'][m] if row==0 else old['operating'][m][row-1] for m in members]
   for f in fields:np.testing.assert_allclose(a['results'][g][row]['point'][f],np.mean([v[f] for v in reference]),atol=1e-11)
with np.load(BASE/'results/bootstrap.npz',allow_pickle=False) as z:
 for g in z.files:
  assert z[g].shape==(a['bootstrap']['valid'],4,7)
  lo,hi=np.quantile(z[g],[.025,.975],axis=0)
  for i,row in enumerate(a['results'][g]):
   for j,f in enumerate(fields):np.testing.assert_allclose(row['intervals'][f],[lo[i,j],hi[i,j]],atol=1e-11)
result={'status':'passed','operating_points':16,'metrics_per_point':7,'bootstrap_draws':a['bootstrap']['valid'],'comparisons_note':'individual fixed thresholds, seeds averaged; no new models or calibration fitted'}
(BASE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
