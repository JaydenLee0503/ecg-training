"""All-seed paired patient uncertainty, operating points and completion audit."""
import os
import time
import numpy as np
from . import common as c
from experiments.acs_omi_vmd_wst_vqc.reporting import binary_metrics, METRICS
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.metrics import Curve,at_threshold,FIELDS


def inputs(ctx):
    data=c.cohort()
    val=np.flatnonzero(data['partitions']=='validation')
    cal=np.flatnonzero(data['partitions']=='calibration')
    selected=c.read(c.RUN/'selection/selected.json')
    spec=c.read(c.BASE/'protocol.json')
    scores,thresholds,groups={},{},{}
    for arm in ('logistic_fixed','logistic_tuned','swin_fixed','swin0','swin1','raw'):
        members=[]
        for seed in ([None] if arm.startswith('logistic') else spec['seeds']):
            name=arm if seed is None else f'{arm}_seed{seed}'
            folder=c.RUN/'final'/arm
            if seed is not None:
                folder=folder/f'seed{seed}'
            if not c.verified(folder,ctx):
                raise ValueError(f'Incomplete final {folder}')
            pred=c.arrays(folder/'validation_predictions.npz')
            calibration=c.arrays(folder/'calibration_predictions.npz')
            for key in ('y','patients','record_ids'):
                np.testing.assert_array_equal(pred[key],data[key][val])
                np.testing.assert_array_equal(calibration[key],data[key][cal])
            frozen=c.read(folder/'thresholds.json')
            points=Curve(calibration['y'],calibration['score']).operating_points(spec['evaluation']['targets_percent'])
            if points!=frozen['calibration_points']:
                raise ValueError('Calibration thresholds fail independent reconstruction')
            scores[name]=pred['score']
            thresholds[name]=[p['threshold'] for p in points]
            members.append(name)
        groups[arm]=members
    groups['swin_selected']=groups[selected['swin']['arm']]
    return data,val,scores,thresholds,groups


def report():
    manifest,ctx=c.manifest()
    spec=manifest['protocol']
    out=c.RUN/'analysis'
    if c.verified(out,ctx):
        publish(out)
        return
    start=time.perf_counter()
    data,val,scores,thresholds,groups=inputs(ctx)
    y,patients=data['y'][val],data['patients'][val]
    def calculate(weights=None):
        single={n:binary_metrics(y,s,weights) for n,s in scores.items()}
        points={n:[at_threshold(y,s,t,weights) for t in thresholds[n]] for n,s in scores.items()}
        grouped={g:np.array([np.mean([single[n][k] for n in names]) for k in METRICS]) for g,names in groups.items()}
        operating={g:np.mean([[[p[k] for k in FIELDS] for p in points[n]] for n in names],axis=0) for g,names in groups.items()}
        return single,points,grouped,operating
    single,points,grouped,operating=calculate()
    draws={g:[] for g in groups}
    opdraws={g:[] for g in groups}
    unique,inverse=np.unique(patients,return_inverse=True)
    cfg=spec['evaluation']
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    skipped=0
    for draw in range(cfg['bootstrap_draws']):
        weights=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse]
        if any(weights[y==label].sum()==0 for label in (0,1)):
            skipped+=1
            continue
        _,_,sample,opsample=calculate(weights)
        for g in groups:
            draws[g].append(sample[g])
            opdraws[g].append(opsample[g])
        if (draw+1)%200==0:
            print(f'Report bootstrap {draw+1}/{cfg["bootstrap_draws"]}',flush=True)
    if skipped==cfg['bootstrap_draws']:
        raise ValueError('No valid bootstrap draws')
    draws={g:np.asarray(d) for g,d in draws.items()}
    opdraws={g:np.asarray(d) for g,d in opdraws.items()}
    c.npz(out/'bootstrap.npz',**{g+'_metrics':d for g,d in draws.items()},**{g+'_operating':d for g,d in opdraws.items()})
    pairs={'primary_swin_selected_minus_fixed':('swin_selected','swin_fixed'),
           'raw_minus_swin_selected':('raw','swin_selected')}
    pairs.update({g+'_minus_logistic_tuned':(g,'logistic_tuned') for g in groups if g!='logistic_tuned'})
    def interval(d,fields):
        lo,hi=np.percentile(d,[2.5,97.5],axis=0)
        return {k:[float(a),float(b)] for k,a,b in zip(fields,lo,hi)}
    result=dict(per_seed=single,per_seed_operating=points,thresholds=thresholds,groups={},pairs={},
        targets_percent=cfg['targets_percent'],seeds=spec['seeds'],records=len(y),patients=len(unique),positive=int(y.sum()),
        bootstrap=dict(requested=cfg['bootstrap_draws'],valid=cfg['bootstrap_draws']-skipped,
                       single_class_discarded=skipped,seed=cfg['bootstrap_seed'],thresholds_fixed=True),
        selection=c.read(c.RUN/'selection/selected.json'),seconds=time.perf_counter()-start,completed_at_utc=c.now())
    for g in groups:
        result['groups'][g]=dict(metrics=dict(zip(METRICS,grouped[g].tolist())),intervals=interval(draws[g],METRICS),
            operating=[dict(zip(FIELDS,v.tolist())) for v in operating[g]],
            operating_intervals=[interval(opdraws[g][:,j],FIELDS) for j in range(len(cfg['targets_percent']))])
    for name,(a,b) in pairs.items():
        result['pairs'][name]=dict(point=dict(zip(METRICS,(grouped[a]-grouped[b]).tolist())),
            intervals=interval(draws[a]-draws[b],METRICS),left=a,right=b)
    c.write(out/'analysis.json',result)
    c.csv_write(out/'predictions.csv',['record_id','patient_id','y']+list(scores),
        ([data['record_ids'][i],data['patients'][i],int(data['y'][i])]+[float(s[j]) for s in scores.values()] for j,i in enumerate(val)))
    c.csv_write(out/'metrics.csv',['model']+list(single[next(iter(single))]),
        ([n]+list(m.values()) for n,m in single.items()))
    c.csv_write(out/'operating_points.csv',['model','calibration_target','threshold']+list(FIELDS),
        ([n,target,thresholds[n][j]]+[points[n][j][k] for k in FIELDS] for n in scores for j,target in enumerate(cfg['targets_percent'])))
    fits,histories=[],[]
    for path in sorted(c.RUN.rglob('fit.json')):
        if 'source' in path.parts:
            continue
        record=c.read(path)
        fits.append(dict(path=str(path.relative_to(c.RUN)),**record))
        if (path.parent/'history.json').exists():
            histories.extend(dict(fit=str(path.parent.relative_to(c.RUN)),**r) for r in c.read(path.parent/'history.json'))
    c.write(out/'fits.json',fits)
    c.write(out/'histories.json',histories)
    primary=result['pairs']['primary_swin_selected_minus_fixed']
    delta=primary['point']['average_precision']
    lo,hi=primary['intervals']['average_precision']
    lines=['# OMI improvement v2 — completed','',
        f'All neural seeds 0/1/2; three inner patient folds. Validation: {len(y):,} ECGs, {len(unique):,} patients, {int(y.sum())} OMI ECGs.',
        '',f'Primary selected-Swin minus matched fixed-40 Swin AP: **{delta:+.4f}**, paired 95% interval **[{lo:+.4f}, {hi:+.4f}]**.',
        ('The interval includes zero; improvement is not established.' if lo<=0<=hi else
         'The interval favors the selected procedure in this internal comparison.' if lo>0 else
         'The interval favors the original fixed-40 procedure; the attempted improvement was unsuccessful.'),'',
        '| Pipeline | AP | ROC AUC | Accuracy at 0.5 | Sensitivity at 0.5 | Specificity at 0.5 |',
        '|---|---:|---:|---:|---:|---:|']
    for g,value in result['groups'].items():
        p=value['metrics']
        lines.append(f'| {g} | {p["average_precision"]:.4f} | {p["roc_auc"]:.4f} | {p["accuracy"]:.2%} | {p["sensitivity"]:.2%} | {p["specificity"]:.2%} |')
    lines+=['','| Pipeline | Calibration target | Achieved sensitivity | Specificity | Precision |', '|---|---:|---:|---:|---:|']
    for g,value in result['groups'].items():
        for target,p in zip(cfg['targets_percent'],value['operating']):
            lines.append(f'| {g} | {target}% | {p["sensitivity"]:.2%} | {p["specificity"]:.2%} | {p["precision"]:.2%} |')
    lines+=['','Model/epoch choices used only inner subfit predictions. Calibration cutoffs were fixed before new validation inference; no subsequent refitting.',
        'The fixed logistic and fixed-40 Swin controls reuse verified sensitivity-study fits on exactly the same subfit/calibration patients. Swin0 reuses a final fit only when epoch40 was selected.',
        'Neural rows are means of metrics across all seeds, not probability ensembles. The selected Swin reuses its chosen arm; it is not an extra fit.',
        f'New saved fits: {len(fits)}; neural epoch records: {len(histories)}. Sum of recorded fit seconds: {sum(f["seconds"] for f in fits):.1f}.',
        f'Patient bootstrap: {result["bootstrap"]["valid"]}/{cfg["bootstrap_draws"]} valid draws, seed {cfg["bootstrap_seed"]}.',
        'Intervals are pointwise and conditional on saved models, selected settings and calibration thresholds; no multiplicity adjustment, retraining or selection uncertainty. The primary was not replaced by the best observed secondary arm.',
        'Validation has been repeatedly examined. Official test waveforms remain reserved. Accuracy alone is misleading at the natural low prevalence. Calibration targets need not transfer to validation. Raw CNN changes both representation and classifier; this is not evidence of transform superiority, clinical validity or quantum advantage.',
        'All candidate scores, splits, sampling indices, fitted preprocessing, predictions, weights, optimizer/RNG checkpoints, warnings, failures and source snapshots are retained in this study. Large artifacts are local and gitignored.',
        '', '[Full results and intervals](analysis.json) · [Every seed](metrics.csv) · [Thresholds and counts](operating_points.csv) · [Fits](fits.json) · [Histories](histories.json)', '']
    c.write_bytes(out/'SUMMARY.md','\n'.join(lines).encode())
    os.environ.setdefault('MPLCONFIGDIR','/tmp/omi-improvement-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for j,(g,v) in enumerate(result['groups'].items()):
        p=v['metrics']['average_precision']; low,high=v['intervals']['average_precision']
        axes[0].plot([low,high],[j,j],color='steelblue')
        axes[0].scatter(p,j,color='navy')
        ops=v['operating']
        axes[1].plot([r['sensitivity'] for r in ops],[r['specificity'] for r in ops],'-o',label=g)
    axes[0].set_yticks(range(len(groups)),list(groups)); axes[0].invert_yaxis()
    axes[0].set_xlabel('Average precision; 95% patient-bootstrap interval')
    axes[1].set_xlabel('Achieved validation sensitivity'); axes[1].set_ylabel('Validation specificity')
    axes[1].legend(fontsize=8); axes[1].set_title('Cutoffs selected on calibration patients')
    fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    c.complete(out,ctx)
    publish(out)
    audit()


def publish(out):
    for name in ('analysis.json','SUMMARY.md','metrics.csv','operating_points.csv','fits.json','histories.json','comparison.png'):
        c.write_bytes(c.REPORTS/name,(out/name).read_bytes())


def audit():
    manifest,ctx=c.manifest()
    protected=c.protect_check(manifest)
    checked=0
    for marker in sorted(c.RUN.rglob('completed.json')):
        if marker.parent==c.RUN or 'source' in marker.parts:
            continue
        if not c.verified(marker.parent,ctx):
            raise ValueError('Incomplete output')
        checked+=len(c.read(marker)['files'])
    data,val,scores,thresholds,groups=inputs(ctx)
    result=c.read(c.RUN/'analysis/analysis.json')
    for name,score in scores.items():
        if binary_metrics(data['y'][val],score)!=result['per_seed'][name]:
            raise ValueError('Reported metrics differ from saved predictions')
    for path in (c.RUN/'inner').rglob('indices.npz'):
        split=c.arrays(path)
        if set(data['patients'][split['unique']]) & set(data['patients'][split['held']]):
            raise ValueError('Inner patient overlap')
        if np.any(data['partitions'][np.r_[split['unique'],split['held']]]!='subfit'):
            raise ValueError('Inner fit used calibration/validation')
    c.write(c.REPORTS/'audit.json',dict(status='passed',time_utc=c.now(),artifact_entries=checked,
        protected_files=protected,per_seed_metrics_verified=len(scores),thresholds_reconstructed=len(scores)*3,
        official_test_processed=False,context_sha=ctx))
    print(f'Audit passed: {checked} output entries; {protected} protected files',flush=True)
