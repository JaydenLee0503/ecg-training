"""Reduced negative-patient training with natural-prevalence evaluation."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[2]
for p in (REPO,REPO/'attention method',REPO/'attention method/.venv/torch-cu128'): sys.path.insert(0,str(p))
from experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2 import common as c, engine as e
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.common import lock
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.metrics import Curve
from experiments.acs_omi_vmd_wst_vqc.reporting import binary_metrics
RUN=BASE/'results'
PARENT=BASE.parent/'sensitivity_study_v1/results/models/lfcc'

def select(data, seed=20261031):
    sub=np.flatnonzero(data['partitions']=='subfit'); y=data['y']; patients=data['patients']
    positive=set(patients[sub[y[sub]==1]])
    mandatory=sub[np.isin(patients[sub],list(positive))]
    target=int(y[mandatory].sum())-int((y[mandatory]==0).sum())
    candidates=np.array(sorted(set(patients[sub])-positive))
    candidates=np.random.default_rng(seed).permutation(candidates)
    groups=[sub[patients[sub]==patient] for patient in candidates]
    # First reachable path for each total, traversed downward so a group is used once.
    paths={0:()}
    for i,g in enumerate(groups):
        for total,path in sorted(list(paths.items()),reverse=True):
            new=total+len(g)
            if new<=target and new not in paths: paths[new]=path+(i,)
        if target in paths: break
    if target not in paths: raise ValueError('Cannot form equal record counts without splitting patients')
    chosen=np.sort(np.concatenate([mandatory]+[groups[i] for i in paths[target]]))
    assert len(np.unique(chosen))==len(chosen)
    assert np.array_equal(np.bincount(y[chosen],minlength=2),[int(y[chosen].sum())]*2)
    for patient in np.unique(patients[chosen]):
        assert set(chosen[patients[chosen]==patient])==set(sub[patients[sub]==patient])
    return chosen

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--execute-training',action='store_true');parser.add_argument('--device',default='cuda',choices=['cuda','cpu']);args=parser.parse_args()
    if not args.execute_training: raise ValueError('Pass --execute-training')
    e.setup(args.device)
    old,oldctx=c.manifest()
    if not c.verified(c.RUN/'prepared',oldctx): raise ValueError('Missing verified preparation')
    data=c.cohort(); X=np.load(c.RUN/'prepared/lfcc.npy',mmap_mode='r'); P=np.load(c.RUN/'prepared/pooled.npy',mmap_mode='r')
    spec=old['protocol']; selected=select(data); cal=np.flatnonzero(data['partitions']=='calibration'); val=np.flatnonzero(data['partitions']=='validation')
    assert not (set(data['patients'][selected]) & (set(data['patients'][cal])|set(data['patients'][val])))
    manifest=dict(protocol_sha=c.sha(BASE/'PROTOCOL.md'),source_sha=c.sha(Path(__file__)),parent_manifest_sha=oldctx,train_indices=selected.tolist(),seed=20261031,device=args.device)
    ctx=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    with lock(RUN):
        c.frozen_write(RUN/'manifest.json',manifest)
        c.npz(RUN/'cohort.npz',indices=selected,patient=data['patients'][selected],record_id=data['record_ids'][selected],y=data['y'][selected])
        print('Selected ECG counts',np.bincount(data['y'][selected]),'patients',len(set(data['patients'][selected])),flush=True)
        allscores={}; thresholds={}
        for arm,seed in [('logistic',None)]+[('swin0',x) for x in (0,1,2)]:
            name=arm if seed is None else f'swin_seed{seed}'; folder=RUN/name
            if arm=='logistic':
                import joblib
                e.fit_logistic(folder,data,P,selected,cal,1.,ctx)
                scaler,clf=joblib.load(folder/'model.joblib'); get=lambda idx: clf.predict_proba(scaler.transform(P[idx]))[:,1]
                cal_score=c.arrays(folder/'held.npz')['score']
            else:
                e.fit_neural(folder,data,X,selected,cal,'swin0',seed,40,spec,ctx,args.device)
                norm=c.arrays(folder/'normalization.npz'); net=e.model('swin0',spec).to(args.device)
                net.load_state_dict(e.torch.load(folder/'epoch40/weights.pt',map_location=args.device,weights_only=True))
                get=lambda idx: e.predict(net,X,idx,norm['mean'],norm['scale'],spec['training']['batch_size'],args.device)
                cal_score=c.arrays(folder/'epoch40/held.npz')['score']
            points=Curve(data['y'][cal],cal_score).operating_points([80,90,95]); c.frozen_write(folder/'thresholds.json',points)
            score=get(val); e.predictions(folder/'validation.npz',data,val,score); allscores['undersampled_'+name]=score;thresholds['undersampled_'+name]=[p['threshold'] for p in points]
        for arm,seed in [('logistic',None)]+[('swin',x) for x in (0,1,2)]:
            folder=PARENT/(arm if seed is None else f'swin_seed{seed}')
            if not c.verified(folder): raise ValueError('Missing parent model')
            z=c.arrays(folder/'validation_predictions.npz')
            np.testing.assert_array_equal(z['record_ids'],data['record_ids'][val]); name='oversampled_'+(arm if seed is None else f'swin_seed{seed}')
            allscores[name]=z['score']; thresholds[name]=[x['threshold'] for x in c.read(folder/'thresholds.json')['calibration_points']]
        from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.metrics import at_threshold
        y=data['y'][val]; patients=data['patients'][val]; unique,inv=np.unique(patients,return_inverse=True); rng=np.random.default_rng(20261031)
        point={k:binary_metrics(y,v) for k,v in allscores.items()}; draws=[]
        for i in range(2000):
            w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
            if not all(w[y==j].sum()>0 for j in (0,1)): continue
            draws.append(binary_metrics(y,allscores['undersampled_logistic'],w)['average_precision']-binary_metrics(y,allscores['oversampled_logistic'],w)['average_precision'])
        delta=point['undersampled_logistic']['average_precision']-point['oversampled_logistic']['average_precision']; ci=np.quantile(draws,[.025,.975]).tolist()
        result=dict(metrics=point,thresholds=thresholds,operating={k:[at_threshold(y,v,t) for t in thresholds[k]] for k,v in allscores.items()},primary_ap_difference=float(delta),primary_ci=ci,bootstrap_valid=len(draws),training_counts=np.bincount(data['y'][selected]).tolist(),training_patients=len(set(data['patients'][selected])))
        c.write(RUN/'analysis.json',result)
        lines=['# OMI undersampling v1 — internal validation','','Training ECG counts: '+str(result['training_counts'])+'; patients: '+str(result['training_patients'])+'.','Primary AP difference, undersampled minus oversampled logistic: '+f'{delta:+.4f}, 95% patient bootstrap [{ci[0]:+.4f}, {ci[1]:+.4f}].','','| Model | AP | Sensitivity at 0.5 | Specificity at 0.5 |','|---|---:|---:|---:|']
        for k,m in point.items(): lines.append(f'| {k} | {m["average_precision"]:.4f} | {m["sensitivity"]:.2%} | {m["specificity"]:.2%} |')
        lines+=['','All three Swin seeds shown individually; natural-prevalence validation. Bootstrap is conditional on saved fits and selected patients; no external or clinical validation. The official test set was not processed.']
        (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n'); c.complete(RUN,ctx)
        print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
