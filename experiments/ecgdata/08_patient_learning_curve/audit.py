"""Independent checks of saved patient subsets, fit artifacts and pooled metrics."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
BASE=Path(__file__).resolve().parent; REPO=BASE.parents[2]; ATT=REPO/'attention method'
sys.path[:0]=[str(ATT),str(ATT/'.venv/torch-cu128')]
from attention_ecg.ecgdata import verify_cache,fit_normalizer,read_json,file_hash
from attention_ecg.ecgdata_training import verify_trial,DEFAULT_RUN
_,data=verify_cache(ATT/'features/ecgdata_lfcc_v1'); run=BASE/'results'; result=read_json(run/'analysis.json')
checked=0; pooled={}
for rep in range(3):
 for fold in range(5):
  before=set(); test=np.flatnonzero(data['fold']==fold)
  for pct in (25,50,75):
   parent=run/f'fold{fold}/rep{rep}/pct{pct}'; sub=read_json(parent/'subset.json'); train=np.array(sub['train_indices']); assert before<=set(train);before=set(train)
   assert not set(data['patients'][train])&set(data['patients'][test]); assert set(data['y'][train])=={0,1,2}; np.testing.assert_array_equal(sub['test_indices'],test)
   for arm in ('swin','logistic'):
    folder=parent/arm; assert verify_trial(folder); checked+=1
    path=folder/('epoch_040.npz' if arm=='swin' else 'predictions.npz')
    with np.load(path,allow_pickle=False) as z:
     np.testing.assert_array_equal(z['indices'],test);np.testing.assert_array_equal(z['y'],data['y'][test]);np.testing.assert_array_equal(z['patients'],data['patients'][test])
     pooled.setdefault((arm,pct,rep),np.full(len(data['y']),-1))[test]=z['pred']
    if arm=='swin':
     with np.load(folder/'preprocessing.npz',allow_pickle=False) as z:
      mean,scale=fit_normalizer(data['X'],train);np.testing.assert_array_equal(z['train_indices'],train);np.testing.assert_array_equal(z['mean'],mean);np.testing.assert_array_equal(z['scale'],scale)
     fit=read_json(folder/'fit.json'); np.testing.assert_allclose(fit['class_weights'],len(train)/(3*np.bincount(data['y'][train],minlength=3)))
  for arm in ('swin','logistic'):
   folder=DEFAULT_RUN/(f'swin_seed{rep}_fold{fold}' if arm=='swin' else f'logistic_fold{fold}'); assert verify_trial(folder)
   with np.load(folder/'predictions.npz',allow_pickle=False) as z:
    np.testing.assert_array_equal(z['indices'],test);pooled.setdefault((arm,100,rep),np.full(len(data['y']),-1))[test]=z['pred']
for (arm,pct) in [(a,p) for a in ('swin','logistic') for p in (25,50,75,100)]:
 pred=[pooled[(arm,pct,rep)] for rep in range(3)]; assert all(np.all(x>=0) for x in pred)
 np.testing.assert_allclose(np.mean([accuracy_score(data['y'],v) for v in pred]),result['metrics'][f'{arm}_{pct}']['window_accuracy'])
 np.testing.assert_allclose(np.mean([f1_score(data['y'],v,average='macro') for v in pred]),result['metrics'][f'{arm}_{pct}']['window_macro_f1'])
audit=dict(status='passed',new_fit_markers_checked=checked,outer_test_patients_disjoint=True,nested_subsets_verified=True,normalization_and_class_weights_verified=45,pooled_accuracy_and_macro_f1_verified=8,analysis_sha256=file_hash(run/'analysis.json'),manifest_sha256=file_hash(run/'manifest.json'))
(BASE/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
