"""ECG-08: fixed-protocol, patient-stratified learning curve."""
import csv, hashlib, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[2]
ATT=REPO/'attention method'
for p in (REPO,ATT,ATT/'.venv/torch-cu128'):
    sys.path.insert(0,str(p))
from attention_ecg.ecgdata import verify_cache,read_json,file_hash,atomic_json,atomic_npz,writer_lock
from attention_ecg.ecgdata_training import partition, verify_trial, pooled_features, complete_trial, DEFAULT_RUN
from attention_ecg.ecgdata_regularization import fit, setup
from attention_ecg.ecgdata_report import patient_confusions,scores,metric_vector

RUN=BASE/'results'
DRAFT=ATT/'ecgdata_learning_curve_protocol.json'

def subset(data,fold,rep,fraction):
    train,test=partition(data,fold)
    selected=[]
    for label in range(3):
        patients=np.unique(data['patients'][train][data['y'][train]==label])
        rng=np.random.default_rng(20260929+1000*rep+10*fold+label)
        selected.extend(rng.permutation(patients)[:max(2,int(np.ceil(fraction*len(patients))))])
    chosen=train[np.isin(data['patients'][train],selected)]
    assert not set(data['patients'][chosen]) & set(data['patients'][test])
    return chosen,test

def logistic(folder,data,tr,te,base):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    if verify_trial(folder): return
    folder.mkdir(parents=True,exist_ok=True)
    X=pooled_features(data['X']); c=base['control']; start=time.perf_counter()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],class_weight=c['class_weight'],max_iter=c['max_iter'],solver=c['solver'],random_state=0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); model.fit(X[tr],data['y'][tr])
    p=model.predict_proba(X[te]); train=model.predict(X[tr])
    joblib.dump(model,folder/'model.joblib')
    np.testing.assert_allclose(joblib.load(folder/'model.joblib').predict_proba(X[te]),p)
    atomic_npz(folder/'predictions.npz',indices=te,y=data['y'][te],patients=data['patients'][te],pred=p.argmax(1),probabilities=p)
    from attention_ecg.ecgdata_report import confusion
    atomic_json(folder/'fit.json',dict(train_indices=tr.tolist(),test_indices=te.tolist(),seconds=time.perf_counter()-start,warnings=[str(x.message) for x in w],training=scores(confusion(data['y'][tr],train))))
    complete_trial(folder,['model.joblib','predictions.npz','fit.json'])

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--execute-training',action='store_true'); p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); args=p.parse_args()
    if not args.execute_training: raise ValueError('Pass --execute-training')
    setup(args.device)
    manifest,data=verify_cache(ATT/'features/ecgdata_lfcc_v1'); base=manifest['protocol']; spec=read_json(DRAFT)
    sources=[Path(__file__),DRAFT,ATT/'attention_ecg/ecgdata_regularization.py',ATT/'attention_ecg/ecgdata_training.py',ATT/'attention_ecg/ecgdata_report.py']
    context={'sources':{str(x.relative_to(REPO)):file_hash(x) for x in sources},'base_cache':file_hash(ATT/'features/ecgdata_lfcc_v1/completed.json'),'base_marker':file_hash(DEFAULT_RUN/'completed.json'),'device':args.device,'spec':spec}
    signature=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    with writer_lock(RUN):
        path=RUN/'manifest.json'
        if path.exists() and read_json(path)!=context: raise ValueError('Manifest drift')
        atomic_json(path,context)
        for fold in range(5):
            for rep in range(3):
                prior=set()
                for pct in (25,50,75):
                    tr,te=subset(data,fold,rep,pct/100)
                    assert prior <= set(tr); prior=set(tr)
                    parent=RUN/f'fold{fold}/rep{rep}/pct{pct}'; parent.mkdir(parents=True,exist_ok=True)
                    details=dict(fold=fold,replicate=rep,fraction=pct/100,train_indices=tr.tolist(),test_indices=te.tolist(),patients=np.unique(data['patients'][tr]).tolist())
                    if (parent/'subset.json').exists() and read_json(parent/'subset.json')!=details: raise ValueError('Subset changed')
                    atomic_json(parent/'subset.json',details)
                    logistic(parent/'logistic',data,tr,te,base)
                    fit(parent/'swin',data,tr,te,rep,0.,40,[40],base,args.device,signature,'outer')
                    print(f'completed fold={fold} replicate={rep} fraction={pct}%',flush=True)
        report(data,base)
        atomic_json(RUN/'completed.json',dict(status='complete',manifest_sha256=file_hash(path),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def report(data,base):
    arms={}
    for pct in (25,50,75,100):
        for arm in ('swin','logistic'):
            preds=[]
            for rep in range(3):
                full=np.full(len(data['y']),-1)
                for fold in range(5):
                    if pct==100:
                        folder=DEFAULT_RUN/(f'swin_seed{rep}_fold{fold}' if arm=='swin' else f'logistic_fold{fold}')
                        if not verify_trial(folder): raise ValueError('Missing original fit')
                        path=folder/'predictions.npz'
                    else:
                        folder=RUN/f'fold{fold}/rep{rep}/pct{pct}'/arm
                        if not verify_trial(folder): raise ValueError('Missing fit')
                        path=folder/('epoch_040.npz' if arm=='swin' else 'predictions.npz')
                    with np.load(path,allow_pickle=False) as z:
                        expected=np.flatnonzero(data['fold']==fold)
                        np.testing.assert_array_equal(z['indices'],expected)
                        full[expected]=z['pred']
                assert np.all(full>=0); preds.append(full)
            arms[f'{arm}_{pct}']=preds
    names=['window_macro_f1','window_accuracy','patient_vote_macro_f1','patient_vote_accuracy']
    grouped={name:[patient_confusions(data['y'],pred,data['patients']) for pred in preds] for name,preds in arms.items()}
    def measure(items,mult): return np.mean([metric_vector((w*mult[:,None,None]).sum(0),(v*mult[:,None,None]).sum(0)) for w,v in items],axis=0)
    n=len(np.unique(data['patients'])); points={k:measure(v,np.ones(n)) for k,v in grouped.items()}; rng=np.random.default_rng(20260925)
    draws={k:[] for k in arms}; rejected=0
    for _ in range(2000):
        mult=np.bincount(rng.integers(n,size=n),minlength=n)
        ref=next(iter(grouped.values()))[0][0]
        if np.any((ref*mult[:,None,None]).sum((0,2))==0): rejected+=1; continue
        for k,v in grouped.items(): draws[k].append(measure(v,mult))
    results={k:dict(zip(names,v.tolist())) for k,v in points.items()}
    diffs={}
    for pct in (25,50,75):
        k=f'swin_{pct}'; d=np.asarray(draws['swin_100'])-np.asarray(draws[k])
        diffs[f'swin_100_minus_{pct}']=dict(point=dict(zip(names,(points['swin_100']-points[k]).tolist())),intervals={n:np.quantile(d[:,j],[.025,.975]).tolist() for j,n in enumerate(names)})
    result=dict(metrics=results,paired=diffs,bootstrap=dict(seed=20260925,draws=2000,rejected=rejected),caveat='Patient bootstrap conditional on predictions; subset and initialization vary together; fewer patients also means fewer optimizer steps.')
    atomic_json(RUN/'analysis.json',result)
    lines=['# ECG-08 patient-count learning curve','','Three patient-subset draws paired with seeds 0/1/2; five patient folds; 40 epochs. Logistic uses the same subsets.','', '| Fraction | Swin window accuracy | Swin macro-F1 | Swin patient-vote accuracy | Logistic window accuracy |','|---|---:|---:|---:|---:|']
    for pct in (25,50,75,100):
        s=results[f'swin_{pct}']; l=results[f'logistic_{pct}']; lines.append(f'| {pct}% | {s["window_accuracy"]:.2%} | {s["window_macro_f1"]:.4f} | {s["patient_vote_accuracy"]:.2%} | {l["window_accuracy"]:.2%} |')
    d=diffs['swin_100_minus_75']; ci=d['intervals']['window_macro_f1']; lines += ['',f'Primary 100% minus 75% macro-F1: {d["point"]["window_macro_f1"]:+.4f}, 95% patient-bootstrap interval [{ci[0]:+.4f}, {ci[1]:+.4f}].','',result['caveat'],'The 80-patient cohort is repeatedly studied and confounds diagnosis with source. No external validation.']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__': main()
