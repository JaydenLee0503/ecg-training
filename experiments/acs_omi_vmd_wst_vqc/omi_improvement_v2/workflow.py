"""Inner model selection precedes calibration and unchanged validation evaluation."""
import time
import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from . import common as c
from . import engine as e
from .data import prepare, prepare_raw
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.fit import freeze_thresholds


def choose(rows, key='AP'):
    # Caller supplies prospective tie order; exact ties retain first entry.
    return max(rows,key=lambda row: row[key])


def selection(data,spec,ctx):
    out=c.RUN/'selection'
    if c.verified(out,ctx):
        return c.read(out/'selected.json')
    neural=[]
    for arm in ('swin0','swin1','raw'):
        for epoch in spec['epochs']:
            scores=[]
            for seed in spec['seeds']:
                entries=[c.arrays(c.RUN/'inner'/arm/f'fold{fold}'/f'seed{seed}'/f'epoch{epoch}/held.npz')
                         for fold in range(spec['inner']['folds'])]
                indices=np.concatenate([d['indices'] for d in entries])
                sub=np.flatnonzero(data['partitions']=='subfit')
                np.testing.assert_array_equal(np.sort(indices),sub)
                score=np.concatenate([d['score'] for d in entries])
                y=np.concatenate([d['y'] for d in entries])
                np.testing.assert_array_equal(y,data['y'][indices])
                scores.append(float(average_precision_score(y,score)))
            neural.append(dict(arm=arm,epoch=epoch,AP=float(np.mean(scores)),seed_AP=scores))
    logistic=[]
    for C in spec['logistic_C']:
        entries=[c.arrays(c.RUN/'inner/logistic'/f'C{C:g}'/f'fold{f}'/'held.npz') for f in range(spec['inner']['folds'])]
        indices=np.concatenate([d['indices'] for d in entries])
        np.testing.assert_array_equal(np.sort(indices),np.flatnonzero(data['partitions']=='subfit'))
        logistic.append(dict(C=C,AP=float(average_precision_score(np.concatenate([d['y'] for d in entries]),
                                    np.concatenate([d['score'] for d in entries])))))
    candidates=sorted([r for r in neural if r['arm'].startswith('swin')],key=lambda r:(r['epoch'],r['arm']))
    selected=dict(context=ctx,swin=choose(candidates),logistic=choose(logistic),
        arms={arm:choose([r for r in neural if r['arm']==arm]) for arm in ('swin0','swin1','raw')},
        chosen_on='subfit inner predictions only',calibration_used=False,validation_used=False)
    c.frozen_write(out/'selected.json',selected)
    c.write(out/'all_candidates.json',dict(neural=neural,logistic=logistic))
    c.complete(out,ctx)
    print('All candidate choices frozen:',selected,flush=True)
    return selected


def copy_control(source,out,ctx,identity):
    if c.verified(out,ctx):
        return
    if not c.verified(source):
        raise ValueError('Baseline incomplete')
    for name in ('calibration_predictions.npz','validation_predictions.npz','thresholds.json'):
        c.write_bytes(out/name,(source/name).read_bytes())
    train=source/'training_predictions.npz'
    if not train.exists():
        train=source/'fit/training_predictions.npz'
    c.write_bytes(out/'training.npz',train.read_bytes())
    c.write(out/'reuse.json',dict(source=str(source.relative_to(c.ROOT)),source_marker_sha=c.sha(source/'completed.json'),
        identity=identity,seconds=0,official_test_processed=False))
    c.complete(out,ctx,reused=True)


def finalize(data,spec,ctx,selected,Xlfcc,Xraw,Xpooled,device):
    sampled=data['sampled_indices']
    cal=np.flatnonzero(data['partitions']=='calibration')
    val=np.flatnonzero(data['partitions']=='validation')
    parent=c.PARENT/'results/models/lfcc'
    copy_control(parent/'logistic',c.RUN/'final/logistic_fixed',ctx,'C=1 matched subfit')
    for seed in spec['seeds']:
        copy_control(parent/f'swin_seed{seed}',c.RUN/'final/swin_fixed'/f'seed{seed}',ctx,'zero dropout epoch40 matched subfit')
    C=selected['logistic']['C']
    out=c.RUN/'final/logistic_tuned'
    if not c.verified(out,ctx):
        if C==1.:
            copy_control(parent/'logistic',out,ctx,'inner selected C=1')
        else:
            e.fit_logistic(out/'fit',data,Xpooled,sampled,cal,C,ctx)
            saved=c.arrays(out/'fit/held.npz')
            freeze_thresholds(out,data['y'][cal],saved['score'],spec['evaluation']['targets_percent'],ctx)
            c.write_bytes(out/'calibration_predictions.npz',(out/'fit/held.npz').read_bytes())
            scaler,clf=joblib.load(out/'fit/model.joblib')
            score=clf.predict_proba(scaler.transform(Xpooled[val]))[:,1]
            scaler2,clf2=joblib.load(out/'fit/model.joblib')
            np.testing.assert_array_equal(score,clf2.predict_proba(scaler2.transform(Xpooled[val]))[:,1])
            e.predictions(out/'validation_predictions.npz',data,val,score)
            c.write_bytes(out/'training.npz',(out/'fit/training.npz').read_bytes())
            c.complete(out,ctx,thresholds_frozen_before_validation=True,reload_exact=True)
    for arm in ('swin0','swin1','raw'):
        epoch=selected['arms'][arm]['epoch']
        X=Xraw if arm=='raw' else Xlfcc
        for seed in spec['seeds']:
            out=c.RUN/'final'/arm/f'seed{seed}'
            if c.verified(out,ctx):
                continue
            if arm=='swin0' and epoch==40:
                copy_control(parent/f'swin_seed{seed}',out,ctx,'inner-selected epoch40; matched zero dropout')
                continue
            if arm.startswith('swin'):
                norm=c.read(c.PARENT/'results/prepared/lfcc_normalization.json')
                expected=sorted(data['record_ids'][np.unique(sampled)].tolist())
                if norm['record_ids']!=expected:
                    raise ValueError('Final normalization source identity differs')
                c.npz(out/'fit/normalization.npz',mean=np.array(norm['mean']),scale=np.array(norm['scale']),fit_indices=np.unique(sampled))
            e.fit_neural(out/'fit',data,X,sampled,cal,arm,seed,epoch,spec,ctx,device)
            held=c.arrays(out/f'fit/epoch{epoch}/held.npz')
            np.testing.assert_array_equal(held['record_ids'],data['record_ids'][cal])
            freeze_thresholds(out,data['y'][cal],held['score'],spec['evaluation']['targets_percent'],ctx)
            c.write_bytes(out/'calibration_predictions.npz',(out/f'fit/epoch{epoch}/held.npz').read_bytes())
            norm=c.arrays(out/'fit/normalization.npz')
            net=e.model(arm,spec).to(device)
            weights=e.torch.load(out/f'fit/epoch{epoch}/weights.pt',map_location=device,weights_only=True)
            net.load_state_dict(weights)
            score=e.predict(net,X,val,norm['mean'],norm['scale'],spec['training']['batch_size'],device)
            net.load_state_dict(e.torch.load(out/f'fit/epoch{epoch}/weights.pt',map_location=device,weights_only=True))
            np.testing.assert_array_equal(score,e.predict(net,X,val,norm['mean'],norm['scale'],spec['training']['batch_size'],device))
            e.predictions(out/'validation_predictions.npz',data,val,score)
            c.write_bytes(out/'training.npz',(out/'fit/training.npz').read_bytes())
            c.complete(out,ctx,thresholds_frozen_before_validation=True,reload_exact=True)
            print(f'Final {arm} seed{seed} complete',flush=True)


def run(device):
    prepare()
    manifest,ctx=c.manifest()
    spec=manifest['protocol']
    c.frozen_write(c.RUN/'environment.json',e.setup(device))
    if not c.verified(c.RUN/'prepared',ctx):
        raise ValueError('Missing preparation')
    data=c.cohort()
    Xlfcc=np.load(c.RUN/'prepared/lfcc.npy',mmap_mode='r')
    Xpooled=np.load(c.RUN/'prepared/pooled.npy',mmap_mode='r')
    # The raw arm is prespecified, regardless of the Swin results.
    prepare_raw(ctx)
    Xraw=np.load(c.RUN/'raw/signals.npy',mmap_mode='r')
    for fold in range(spec['inner']['folds']):
        split=c.arrays(c.RUN/'prepared'/f'inner{fold}.npz')
        for C in spec['logistic_C']:
            e.fit_logistic(c.RUN/'inner/logistic'/f'C{C:g}'/f'fold{fold}',data,Xpooled,
                           split['sampled'],split['held'],C,ctx)
    for arm in ('swin0','swin1','raw'):
        X=Xraw if arm=='raw' else Xlfcc
        for fold in range(spec['inner']['folds']):
            split=c.arrays(c.RUN/'prepared'/f'inner{fold}.npz')
            for seed in spec['seeds']:
                e.fit_neural(c.RUN/'inner'/arm/f'fold{fold}'/f'seed{seed}',data,X,split['sampled'],
                    split['held'],arm,seed,spec['training']['epochs'],spec,ctx,device,inner=True)
    selected=selection(data,spec,ctx)
    finalize(data,spec,ctx,selected,Xlfcc,Xraw,Xpooled,device)
    from .report import report
    report()
