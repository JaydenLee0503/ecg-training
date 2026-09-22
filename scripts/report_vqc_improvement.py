#!/usr/bin/env python
"""Validate and document every completed nested VQC trial and its comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ecgvmd as E
from scripts.improve_vqc import CANDIDATES, EPOCHS, nested_splits, choose_candidate, array_hash
from scripts.summarize_patient_vqc import accuracy, macro_f1


def patient_confusions(y,pred,groups,voting=False):
    patients,indices = np.unique(groups,return_inverse=True)
    mapping = {v:i for i,v in enumerate(E.CLASS_ORDER)}
    cm = np.zeros((len(patients),3,3),dtype=int)
    if voting:
        truth,votes = E.record_vote(pred,groups,y)
        cm[np.arange(len(patients)),[mapping[v] for v in truth],[mapping[v] for v in votes]]=1
    else:
        np.add.at(cm,(indices,[mapping[v] for v in y],[mapping[v] for v in pred]),1)
    return cm


def paired_changes(y,groups,new,controls,n_boot=5000):
    n_patients = len(np.unique(groups))
    multiplicities = np.random.default_rng(0).multinomial(n_patients,
                            np.full(n_patients,1/n_patients),size=n_boot)
    result = {}
    for control,reference in controls.items():
        for voting in (False,True):
            for name,metric in [('accuracy',accuracy),('macro_f1',macro_f1)]:
                values,boots = [],[]
                for predictions in (new,reference):
                    scores,samples = [],[]
                    for p in predictions:
                        cm = patient_confusions(y,p,groups,voting)
                        scores.append(float(metric(cm.sum(0))))
                        sampled = (multiplicities @ cm.reshape(n_patients,9)).reshape(-1,3,3)
                        samples.append(metric(sampled))
                    values.append(float(np.mean(scores)))
                    boots.append(np.mean(samples,axis=0))
                key = f'{control}/{"patient" if voting else "window"}_{name}'
                result[key]=dict(new_minus_control=values[0]-values[1],
                                 ci95=np.percentile(boots[0]-boots[1],[2.5,97.5]).tolist())
    return result


def validate(run,source):
    if not (run/'completed.json').exists():
        raise ValueError('Run is incomplete; do not publish partial outer scores')
    meta = json.loads((run/'manifest.json').read_text())
    for relative,digest in meta['code_and_protocol_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Code/protocol changed since training: {relative}')
    if hashlib.sha256((source/'features.npz').read_bytes()).hexdigest()!=meta['source_features_sha256']:
        raise ValueError('Input features changed')
    with np.load(source/'features.npz',allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    with np.load(run/'predictions.npz',allow_pickle=False) as z:
        predictions = {k:z[k] for k in z.files}
    for key in ('y','patients','fold'):
        np.testing.assert_array_equal(data[key],predictions[key])
    splits = nested_splits(data['y'],data['patients'],data['fold'])
    selected = pd.read_csv(run/'selected.csv')
    scores = pd.read_csv(run/'inner_scores.csv',float_precision='round_trip')
    ledger = pd.read_csv(run/'trial_ledger.csv')
    if len(ledger)!=120 or not (ledger.status=='complete').all():
        raise ValueError('Expected 90 completed inner fits and 30 completed outer fits')
    if len(scores)!=120 or len(selected)!=10:
        raise ValueError('Missing candidate/epoch score rows or selected configurations')
    model_checks,inner_checks = [],[]
    for arm in ('vmd','wst'):
        X = data[arm]
        for fold,(development,test,inner) in splits.items():
            candidates = scores[(scores.arm==arm)&(scores.outer_fold==fold)].to_dict('records')
            for candidate in candidates:
                truth,output = [],[]
                for i,(_,val) in enumerate(inner):
                    path = run/'inner'/f'{arm}_o{fold}_i{i}_{candidate["candidate"]}_predictions.npz'
                    with np.load(path,allow_pickle=False) as z:
                        truth.extend(data['y'][val])
                        output.extend(z[f'epoch_{candidate["epochs"]}'])
                actual = E.full_metrics(np.asarray(truth),np.asarray(output))
                for metric in ('accuracy','macro_f1'):
                    np.testing.assert_allclose(candidate[metric],actual[metric],atol=1e-14,rtol=0)
            winner = choose_candidate(candidates)
            choice = selected[(selected.arm==arm)&(selected.outer_fold==fold)].iloc[0]
            assert choice.candidate==winner['candidate'] and choice.epochs==winner['epochs']
            for i,(train,val) in enumerate(inner):
                prep = joblib.load(run/'preprocessing'/f'{arm}_o{fold}_i{i}.joblib')
                np.testing.assert_allclose(prep[1].scaler_.mean_,
                    X[train][:,prep[0].idx_].astype(float).mean(0),atol=1e-12,rtol=0)
                for candidate in CANDIDATES:
                    stem = f'{arm}_o{fold}_i{i}_{candidate["name"]}'
                    model = joblib.load(run/'inner'/f'{stem}.joblib')
                    history = pd.read_csv(run/'inner'/f'{stem}_history.csv')
                    assert len(history)==80 and np.isfinite(history[['loss','gradient_norm','lr']]).all().all()
                    assert model.n_steps_==80*int(np.ceil(len(train)/32))
                    with np.load(run/'inner'/f'{stem}_predictions.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['validation_indices'],val)
                        for epoch in EPOCHS:
                            np.testing.assert_array_equal(z[f'epoch_{epoch}'],model.validation_predictions_[epoch])
                            # Restore each saved checkpoint and verify a few predictions.
                            model.w_,model.W_,model.b_ = model.validation_weights_[epoch]
                            np.testing.assert_array_equal(model.predict(prep.transform(X[val[:3]])),z[f'epoch_{epoch}'][:3])
                    inner_checks.append(stem)
            for seed in range(3):
                stem = f'{arm}_o{fold}_seed{seed}'
                pipeline = joblib.load(run/'outer'/f'{stem}.joblib')
                original = joblib.load(source/f'{arm}_seed{seed}_fold{fold}.joblib')
                np.testing.assert_array_equal(pipeline[0].idx_,original[0].idx_)
                for attribute in ('mean_','scale_'):
                    np.testing.assert_array_equal(getattr(pipeline[1].scaler_,attribute),
                                                  getattr(original[1].scaler_,attribute))
                model = pipeline[-1]
                history = pd.read_csv(run/'outer'/f'{stem}_history.csv')
                assert len(history)==choice.epochs==model.epochs
                assert np.isfinite(history[['loss','gradient_norm','lr']]).all().all()
                assert model.n_steps_==choice.epochs*int(np.ceil(len(development)/32))
                assert model.validation_predictions_=={}
                np.testing.assert_allclose(pipeline[1].scaler_.mean_,
                    X[development][:,pipeline[0].idx_].astype(float).mean(0),atol=1e-12,rtol=0)
                assert len(pipeline[0].idx_)==12 and model.n_qubits==6
                assert model.n_layers==choice.n_layers
                expected = CANDIDATES[int(choice.candidate_order)]
                assert model.reupload==expected['reupload']
                for parameters in (model.w_,model.W_,model.b_):
                    assert np.isfinite(parameters).all()
                with np.load(run/'outer'/f'{stem}_predictions.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['train_indices'],development)
                    np.testing.assert_array_equal(z['test_indices'],test)
                    np.testing.assert_array_equal(z['predictions'],predictions[f'{arm}_seed{seed}'][test])
                    np.testing.assert_array_equal(pipeline.predict(X[test[:6]]),z['predictions'][:6])
                model_checks.append(stem)
    result=dict(inner_models_checked=len(inner_checks),inner_checkpoints_checked=4*len(inner_checks),
                outer_models_checked=len(model_checks),patients=len(np.unique(data['patients'])),
                windows=len(data['y']),status='passed',
                checks=['code/input hashes','patient separation','training-only selection/scaling',
                        'all candidate histories','checkpoint prediction spot checks','recomputed inner scores',
                        'selected inner maxima','identical outer features/scaling to original VQC',
                        'finite training states','selected epoch counts','no outer eval_set',
                        'outer saved-model prediction spot checks','exact prediction coverage'])
    (run/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return data,predictions,ledger,selected,result


def table(frame):
    # Avoid an optional tabulate dependency in the pinned environment.
    lines=['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'results/vqc_improvement')
    parser.add_argument('--source',type=Path,default=ROOT/'results/patient_vmd_wst_vqc')
    args=parser.parse_args()
    data,pred,ledger,selected,validation=validate(args.run,args.source)
    previous=np.load(args.source/'predictions.npz',allow_pickle=False)
    knn=np.load(ROOT/'results/patient_knn_diagnostic/predictions.npz',allow_pickle=False)
    for old_key,key in [('__y','y'),('__groups','patients'),('__fold','fold')]:
        np.testing.assert_array_equal(previous[old_key],data[key])
    np.testing.assert_array_equal(knn['y'],data['y'])
    np.testing.assert_array_equal(knn['patients'],data['patients'])
    np.testing.assert_array_equal(knn['fold'],data['fold'])
    comparisons={}
    rows=[]
    for arm,baseline in [('vmd','VMD descriptors'),('wst','WST log')]:
        updated=[pred[f'{arm}_seed{seed}'] for seed in range(3)]
        original=[previous[f'{baseline} + VQC seed={seed}'] for seed in range(3)]
        control=[knn[f'{arm}/selected12_angles/inverse']]
        comparisons[arm]=paired_changes(data['y'],data['patients'],updated,
                                       {'original_vqc':original,'weighted_knn':control})
        for model,output in [('Original VQC',original),('Nested compact VQC',updated),('Weighted KNN (1/d)',control)]:
            for seed,p in enumerate(output):
                row=E.metrics_row(f'{arm}/{model}/seed{seed}',data['y'],p,data['patients'],
                                  arm=arm,classifier=model,seed=seed)
                rows.append({k.replace('record_','patient_'):v for k,v in row.items()})
    all_metrics=pd.DataFrame(rows)
    all_metrics.to_csv(args.run/'comparison_metrics.csv',index=False)
    front_end_changes=paired_changes(data['y'],data['patients'],
        [pred[f'wst_seed{seed}'] for seed in range(3)],
        {'vmd':[pred[f'vmd_seed{seed}'] for seed in range(3)]})
    intervals=dict(n_patients=80,n_boot=5000,bootstrap_seed=0,
                   scope='Paired patient bootstrap conditional on saved predictions; no retraining or nested reselection',
                   comparisons=comparisons,wst_minus_vmd=front_end_changes)
    (args.run/'paired_comparisons.json').write_text(json.dumps(intervals,indent=2)+'\n')
    summaries=[]
    for (arm,model),frame in all_metrics.groupby(['arm','classifier'],sort=False):
        row={'Front end':arm.upper(),'Classifier':model}
        for label,key,scale,places in [('Window accuracy','accuracy',100,2),('Macro-F1','macro_f1',1,4),
                                       ('Patient accuracy','patient_accuracy',100,2),('Patient macro-F1','patient_macro_f1',1,4)]:
            mean=frame[key].mean()*scale
            suffix='%' if scale==100 else ''
            row[label]=f'{mean:.{places}f}{suffix}'
            if len(frame)>1:
                sd=frame[key].std()*scale
                row[label]+=f' ± {sd:.{places}f}'
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    changes=[]
    for arm,items in comparisons.items():
        for name,value in items.items():
            control,metric=name.split('/')
            scale=100 if metric.endswith('accuracy') else 1
            places=2 if scale==100 else 4
            unit=' pp' if scale==100 else ''
            lo,hi=np.array(value['ci95'])*scale
            changes.append({'Front end':arm.upper(),'Control':control,'Metric':metric,
                            'New minus control':f'{value["new_minus_control"]*scale:+.{places}f}{unit}',
                            'Paired 95% interval':f'[{lo:+.{places}f}, {hi:+.{places}f}]{unit}'})
    front_end_rows=[]
    for name,value in front_end_changes.items():
        metric=name.split('/')[1]
        scale=100 if metric.endswith('accuracy') else 1
        places=2 if scale==100 else 4
        unit=' pp' if scale==100 else ''
        lo,hi=np.array(value['ci95'])*scale
        front_end_rows.append({'Metric':metric,
            'WST minus VMD':f'{value["new_minus_control"]*scale:+.{places}f}{unit}',
            'Paired 95% interval':f'[{lo:+.{places}f}, {hi:+.{places}f}]{unit}'})
    train_rows=[]
    for path in sorted((args.run/'outer').glob('*_predictions.npz')):
        arm=path.name.split('_')[0]
        stem=path.name.removesuffix('_predictions.npz')
        metadata=json.loads((args.run/'outer'/f'{stem}.json').read_text())
        history=pd.read_csv(args.run/'outer'/f'{stem}_history.csv')
        with np.load(args.run/'outer'/f'{stem}_weights.npz',allow_pickle=False) as weights:
            initial=np.random.default_rng(metadata['seed']).normal(0,.1,weights['w'].shape)
            circuit_change=float(np.linalg.norm(weights['w']-initial))
        with np.load(path,allow_pickle=False) as z:
            truth=data['y'][z['train_indices']]
            m=E.full_metrics(truth,z['training_predictions'])
            train_rows.append(dict(model=stem,arm=arm,outer_fold=metadata['outer_fold'],
                                   seed=metadata['seed'],candidate=metadata['choice']['candidate'],
                                   epochs=metadata['choice']['epochs'],accuracy=m['accuracy'],
                                   macro_f1=m['macro_f1'],first_loss=history.loss.iloc[0],
                                   final_loss=history.loss.iloc[-1],
                                   final_gradient_norm=history.gradient_norm.iloc[-1],
                                   circuit_weight_change_norm=circuit_change))
    training=pd.DataFrame(train_rows)
    training.to_csv(args.run/'training_fit.csv',index=False)
    first=min(datetime.fromisoformat(v) for v in ledger.started)
    last=max(datetime.fromisoformat(v) for v in ledger.finished)
    duration=(last-first).total_seconds()
    stats=dict(validation=validation,fit_phase_wall_seconds=duration,
               inner_fits=int((ledger.phase=='inner').sum()),outer_fits=int((ledger.phase=='outer').sum()),
               training_means=training.groupby('arm')[['accuracy','macro_f1']].mean().to_dict('index'))
    (args.run/'analysis_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    provenance={}
    for path in [args.source/'predictions.npz',args.source/'manifest.json',
                 ROOT/'results/patient_knn_diagnostic/predictions.npz',
                 ROOT/'scripts/report_vqc_improvement.py',ROOT/'tests/test_reupload_vqc.py']:
        provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.run/'reporting_inputs.json').write_text(json.dumps(provenance,indent=2)+'\n')
    recorded=ROOT/'architects/vqc_improvement_tables'
    recorded.mkdir(exist_ok=True)
    for name in ('inner_scores.csv','selected.csv','metrics.csv','summary.csv','comparison_metrics.csv','trial_ledger.csv','training_fit.csv'):
        (recorded/name).write_bytes((args.run/name).read_bytes())
    for name in ('manifest.json','paired_comparisons.json','validation.json','analysis_summary.json','reporting_inputs.json'):
        (recorded/name).write_bytes((args.run/name).read_bytes())
    per_seed=pd.read_csv(args.run/'metrics.csv')[['arm','seed','accuracy','balanced_accuracy','macro_f1','patient_accuracy','patient_macro_f1']].round(4)
    choices=selected[['arm','outer_fold','candidate','epochs','macro_f1','accuracy']].round(4)
    measured='\n\n'.join([
        '<!-- measured-results:start -->',
        '**Completed nested experiment.** All 90 inner and 30 outer fits finished. '
        f'The parallel fit phase took {duration/60:.1f} minutes, excluding preparation and reporting. '
        'Every planned candidate was retained. No outer score was used to choose an architecture or epoch.',
        '**Observed results.** VQC values are means ± sample standard deviation across three initialization seeds. '
        'The deterministic KNN control has one score. Accuracy variability is expressed in percentage points. '
        'Seed spread is not a confidence interval for unseen patients.',
        table(summary),
        '**Per-seed outer results.**',table(per_seed),
        '**Training-fit diagnostic.** These are unweighted means across the 15 outer-training '
        'models per front end; they are not held-out scores. The full table also records each '
        'fit\'s selected epoch, first/final loss, and final gradient norm.',
        table(training.groupby('arm')[['accuracy','macro_f1','first_loss','final_loss','final_gradient_norm']].mean().round(4).reset_index()),
        '**Settings selected using inner patients only.** The last two columns are pooled inner-validation scores, '
        'not outer performance estimates.',table(choices),
        '**Paired patient-bootstrap changes.** Positive values favor the new VQC. These 5,000 paired resamples '
        'condition on saved predictions; they do not include retraining or repeating hyperparameter selection. '
        'Intervals crossing zero leave the direction uncertain and do not establish equivalence.',table(pd.DataFrame(changes)),
        '**Matched front-end comparison with nested classifier tuning.** Both front ends received '
        'the same candidate family and search budget, with settings chosen separately. Positive values '
        'favor WST. These differences describe the full tuned pipelines.',table(pd.DataFrame(front_end_rows)),
        '**Verification.** '+f'{validation["inner_models_checked"]} inner models, {validation["inner_checkpoints_checked"]} '
        f'early checkpoints, and {validation["outer_models_checked"]} outer models passed the saved-artifact checks. '
        'This includes patient boundaries, scaler training means, numerical finiteness, selected epoch counts, '
        'checkpoint/saved-model prediction spot checks, and exact outer prediction coverage.',
        'All pooled candidate/epoch scores, selections, per-class metrics, confusion matrices, the complete trial ledger, '
        'and paired intervals are preserved in [the tracked tables](vqc_improvement_tables/). '
        'Epoch-level histories, checkpoint weights, fitted pipelines, and predictions remain in `results/vqc_improvement/`.',
        '<!-- measured-results:end -->'])
    doc=ROOT/'architects/vqc_improvement_results.md'
    text=doc.read_text()
    start,end='<!-- measured-results:start -->','<!-- measured-results:end -->'
    if start in text:
        a=text.index(start)
        b=text.index(end,a)+len(end)
        text=text[:a]+measured+text[b:]
    else:
        stale='The VQC improvement experiment is running. No improved accuracy is claimed until\nall outer-fold predictions have been produced and checked.\n'
        text=text.replace(stale,'',1)
        text=measured+'\n\n'+text.lstrip()
    doc.write_text(text)
    (args.run/'report.md').write_text(text)
    print(summary.to_string(index=False))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
