"""Audit saved representations, patient boundaries, reloads, and primary bootstrap."""
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from scipy.fft import dct
BASE=Path(__file__).resolve().parent;REPO=BASE.parents[2];ATT=REPO/'attention method'
sys.path[:0]=[str(REPO),str(ATT),str(ATT/'.venv/torch-cu128')]
spec=importlib.util.spec_from_file_location('ecg09_audit_source',BASE/'study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
from attention_ecg.ecgdata import fit_normalizer,normalize
from attention_ecg.ecgdata_report import collect_predictions
from attention_ecg.model import CepstralSwin
from sklearn.metrics import accuracy_score,f1_score
import joblib, torch

def grouped_confusion(y,pred,patients):
 unique,inverse=np.unique(patients,return_inverse=True); out=np.zeros((len(unique),3,3),dtype=np.int64)
 np.add.at(out,(inverse,y,pred),1);return out

def main():
 start=time.perf_counter(); torch.set_num_threads(1)
 if not s.verify_trial(s.RUN):raise ValueError('Incomplete training')
 data=s.read_data(); olddata=s.old.verify_cache();p=s.old.protocol();baseline=collect_predictions(s.old.RUN,olddata,p)
 np.testing.assert_array_equal(data['patients'],olddata['patients']);np.testing.assert_array_equal(data['fold'],olddata['fold'])
 recovered=dct(data['bands'].astype(np.float64),type=2,norm='ortho',axis=-1)[...,:16].astype(np.float32)
 maxerror=float(np.max(abs(recovered-olddata['X'])));assert maxerror<=2e-5
 checked=0;reload_checks=0;predictions={a:[] for a in ('band_swin','band_logistic','raw_cnn')}
 for arm in predictions:
  for seed in ([None] if arm=='band_logistic' else (0,1,2)):
   full=np.full(len(data['y']),-1)
   for fold in range(5):
    name=f'{arm}_fold{fold}' if seed is None else f'{arm}_seed{seed}_fold{fold}';folder=s.RUN/name
    if not s.verify_trial(folder):raise ValueError('Missing fit '+name)
    fit=s.read_json(folder/'fit.json');tr,te=s.partition(data,fold)
    np.testing.assert_array_equal(fit['train_indices'],tr);np.testing.assert_array_equal(fit['test_indices'],te)
    assert not set(data['patients'][tr])&set(data['patients'][te])
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
     np.testing.assert_array_equal(z['indices'],te);np.testing.assert_array_equal(z['patients'],data['patients'][te]);np.testing.assert_array_equal(z['y'],data['y'][te]);probs=z['probabilities'];pred=z['pred'];np.testing.assert_array_equal(pred,probs.argmax(1));assert np.isfinite(probs).all();full[te]=pred
    if arm=='band_swin':
     with np.load(folder/'preprocessing.npz',allow_pickle=False) as z:
      mean,scale=fit_normalizer(data['bands'],tr);np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['test_indices'],te);np.testing.assert_allclose(z['mean'],mean,rtol=0,atol=0);np.testing.assert_allclose(z['scale'],scale,rtol=0,atol=0)
     config=dict(p['model']);config['n_ceps']=32;model=CepstralSwin(**config);sample=torch.from_numpy(normalize(data['bands'][te[:8]],mean,scale))
    elif arm=='raw_cnn': model=s.RawCNN();sample=torch.from_numpy(data['raw'][te[:8],None,:])
    else:
     model=joblib.load(folder/'model.joblib');X=s.pooled_features(data['bands']) if hasattr(s,'pooled_features') else np.concatenate([data['bands'].mean(1).reshape(len(data['y']),-1),data['bands'].std(1).reshape(len(data['y']),-1)],axis=1)
     np.testing.assert_allclose(model[0].mean_,X[tr].astype(np.float64).mean(axis=0),rtol=1e-6,atol=1e-6)
     np.testing.assert_allclose(model.predict_proba(X[te[:8]]),probs[:8],rtol=1e-6,atol=1e-7);reload_checks+=1;checked+=1;continue
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);assert state['epoch']==40;model.load_state_dict(state['model']);model.eval()
    with torch.inference_mode():actual=model(sample).softmax(-1).numpy()
    np.testing.assert_allclose(actual,probs[:8],rtol=1e-4,atol=2e-6);np.testing.assert_array_equal(actual.argmax(1),pred[:8]);reload_checks+=1;checked+=1
   assert np.all(full>=0);predictions[arm].append(full)
 analysis=s.read_json(s.REPORT/'analysis.json');positions=np.flatnonzero(data['v1_index']>=0);positions=positions[np.argsort(data['v1_index'][positions])]
 allarms={'lfcc_swin':baseline['swin'],'lfcc_logistic':baseline['logistic'],**predictions}
 for label,ix in (('v1_windows',positions),('all_windows',np.arange(len(data['y'])))):
  for arm,items in allarms.items():
   accuracy=np.mean([accuracy_score(data['y'][ix],v[ix]) for v in items]);macro=np.mean([f1_score(data['y'][ix],v[ix],average='macro') for v in items]);saved=analysis['results'][label]['arms'][arm]['mean']['window'];np.testing.assert_allclose([accuracy,macro],[saved['accuracy'],saved['macro_f1']],rtol=1e-12,atol=1e-12)
 # Independently reconstruct the primary paired bootstrap with per-patient confusion counts.
 y=data['y'][positions];patients=data['patients'][positions];grouped={name:[grouped_confusion(y,v[positions],patients) for v in allarms[name]] for name in ('band_swin','lfcc_swin')};n=len(np.unique(patients));rng=np.random.default_rng(20260925);draws=[];rejected=0
 def macro(cm):
  tp=np.diag(cm);denom=cm.sum(axis=0)+cm.sum(axis=1);return np.divide(2*tp,denom,out=np.zeros(3),where=denom!=0).mean()
 for i in range(2000):
  count=np.bincount(rng.integers(n,size=n),minlength=n)
  if np.any((grouped['lfcc_swin'][0]*count[:,None,None]).sum((0,2))==0):rejected+=1;continue
  values={g:np.mean([macro((m*count[:,None,None]).sum(0)) for m in z]) for g,z in grouped.items()};draws.append(values['band_swin']-values['lfcc_swin'])
 ci=np.quantile(draws,[.025,.975]);expected=analysis['results']['v1_windows']['bootstrap']['paired_differences']['band_swin_minus_lfcc_swin']['window_macro_f1']['interval_95'];np.testing.assert_allclose(ci,expected,atol=1e-12)
 s.atomic_npz(s.RUN/'primary_bootstrap_audit.npz',difference=np.asarray(draws));audit=dict(status='passed',feature_lfcc_max_error=maxerror,fit_markers=checked,model_reload_spot_checks=reload_checks,normalizers_checked=15,patient_isolation=True,metrics_checked=10,primary_bootstrap_draws=len(draws),rejected=rejected,seconds=time.perf_counter()-start,training_manifest_sha256=s.file_hash(s.RUN/'manifest.json'),analysis_sha256=s.file_hash(s.REPORT/'analysis.json'))
 s.atomic_json(s.REPORT/'audit.json',audit);print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
