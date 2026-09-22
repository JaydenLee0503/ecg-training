#!/usr/bin/env python
"""Nested patient-grouped VQC improvement experiment; see the declared protocol.

All trials, including failures, are retained. The source baseline is read-only.
Completed inner/outer fits resume under an identical content-hashed manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ecgvmd as E
from ecgvmd.reupload import ReuploadVQC

CANDIDATES = (
    dict(name='compact_once',n_layers=2,reupload=False),
    dict(name='reupload_2',n_layers=2,reupload=True),
    dict(name='reupload_3',n_layers=3,reupload=True),
)
EPOCHS = (20,40,60,80)
ARMS = ('vmd','wst')


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def atomic_npz(path, **values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)


def nested_splits(y,patients,outer):
    result = {}
    for fold in range(5):
        development = np.flatnonzero(outer!=fold)
        test = np.flatnonzero(outer==fold)
        if set(patients[development]) & set(patients[test]):
            raise ValueError('Outer patient overlap')
        cv = StratifiedGroupKFold(3,shuffle=True,random_state=100+fold)
        inner = []
        coverage = np.zeros(len(y),dtype=int)
        for itr,iva in cv.split(np.zeros((len(development),1)),y[development],patients[development]):
            train,val = development[itr],development[iva]
            if set(patients[train]) & set(patients[val]):
                raise ValueError('Inner patient overlap')
            for indices in (train,val,test):
                if set(y[indices])!=set(E.CLASS_ORDER):
                    raise ValueError('All partitions must contain all classes')
            coverage[val] += 1
            inner.append((train,val))
        if not np.all(coverage[development]==1) or np.any(coverage[test]):
            raise ValueError('Invalid nested validation coverage')
        result[fold] = (development,test,inner)
    return result


def prepare_context(out,arm,fold,inner,X,y,train,val,names):
    stem = f'{arm}_o{fold}_i{inner}'
    folder = out/'preprocessing'
    path = folder/f'{stem}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['train'],train) or not np.array_equal(z['val'],val):
                raise ValueError('Preprocessing cache has different split rows')
        return path
    model = make_pipeline(E.MRMRSelector(k=12),E.TanhAngleScaler())
    A = model.fit_transform(X[train],y[train])
    B = model.transform(X[val])
    if A.shape[1]!=12:
        raise ValueError('Expected 12 selected input features')
    selected = model[0].idx_
    np.testing.assert_allclose(model[1].scaler_.mean_,X[train][:,selected].astype(float).mean(0),atol=1e-12,rtol=0)
    joblib.dump(model,folder/f'{stem}.joblib',compress=3)
    atomic_npz(path,train=train,val=val,X_train=A,X_val=B,y_train=y[train],y_val=y[val],
               selected_indices=selected,selected_names=names[selected])
    return path


def inner_trial(out,arm,fold,inner,candidate,context_path):
    stem = f'{arm}_o{fold}_i{inner}_{candidate["name"]}'
    folder = out/'inner'
    result_path = folder/f'{stem}.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    record = dict(phase='inner',arm=arm,outer_fold=fold,inner_fold=inner,
                  candidate=candidate['name'],seed=0,started=now(),status='running')
    atomic_json(folder/f'{stem}_started.json',record)
    print(f'START inner {stem}',flush=True)
    started = time.monotonic()
    try:
        with np.load(context_path,allow_pickle=False) as z:
            context = {k:z[k] for k in z.files}
        model = ReuploadVQC(n_layers=candidate['n_layers'],reupload=candidate['reupload'],epochs=80,seed=0)
        model.fit(context['X_train'],context['y_train'],
                  eval_set=(context['X_val'],context['y_val']),eval_epochs=EPOCHS)
        pd.DataFrame(model.history_).to_csv(folder/f'{stem}_history.csv',index=False)
        joblib.dump(model,folder/f'{stem}.joblib',compress=3)
        model.save_weights(folder/f'{stem}_weights.npz')
        atomic_npz(folder/f'{stem}_checkpoints.npz',classes=model.classes_,
                   **{f'epoch_{ep}_{name}':value for ep,values in model.validation_weights_.items()
                      for name,value in zip(('w','W','b'),values)})
        atomic_npz(folder/f'{stem}_predictions.npz',validation_indices=context['val'],
                   **{f'epoch_{ep}':model.validation_predictions_[ep].astype(str) for ep in EPOCHS})
        record.update(status='complete',finished=now(),seconds=time.monotonic()-started,
                      train_n=len(context['y_train']),val_n=len(context['y_val']),
                      parameters=model.parameter_count_,steps=model.n_steps_,
                      checkpoints=model.validation_metrics_)
        atomic_json(result_path,record)
        print(f'DONE inner {stem} {record["seconds"]:.1f}s',flush=True)
        return record
    except Exception:
        record.update(status='failed',finished=now(),seconds=time.monotonic()-started,
                      traceback=traceback.format_exc())
        atomic_json(folder/f'{stem}_failure.json',record)
        raise


def choose_candidate(scores):
    """Scores must contain only pooled inner-validation metrics."""
    return sorted(scores,key=lambda row:(-row['macro_f1'],row['n_layers'],row['epochs'],row['candidate_order']))[0]


def select_all(out,y,splits):
    choices,scores = {},[]
    for arm in ARMS:
        for fold,(development,test,inner) in splits.items():
            local = []
            for order,candidate in enumerate(CANDIDATES):
                for ep in EPOCHS:
                    truth,pred = [],[]
                    for i,(_,val) in enumerate(inner):
                        path = out/'inner'/f'{arm}_o{fold}_i{i}_{candidate["name"]}_predictions.npz'
                        with np.load(path,allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['validation_indices'],val)
                            truth.extend(y[val])
                            pred.extend(z[f'epoch_{ep}'])
                    metrics = E.full_metrics(np.asarray(truth),np.asarray(pred))
                    row = dict(arm=arm,outer_fold=fold,candidate=candidate['name'],
                               candidate_order=order,n_layers=candidate['n_layers'],epochs=ep,
                               macro_f1=metrics['macro_f1'],accuracy=metrics['accuracy'])
                    local.append(row)
                    scores.append(row)
            choices[(arm,fold)] = choose_candidate(local)
    pd.DataFrame(scores).to_csv(out/'inner_scores.csv',index=False)
    pd.DataFrame(choices.values()).to_csv(out/'selected.csv',index=False)
    atomic_json(out/'selection.json',{f'{arm}/fold{fold}':row for (arm,fold),row in choices.items()})
    return choices


def outer_trial(out,arm,fold,seed,choice,context_path):
    stem = f'{arm}_o{fold}_seed{seed}'
    folder = out/'outer'
    result_path = folder/f'{stem}.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    record = dict(phase='outer',arm=arm,outer_fold=fold,seed=seed,choice=choice,
                  started=now(),status='running')
    atomic_json(folder/f'{stem}_started.json',record)
    print(f'START outer {stem} {choice["candidate"]} epochs={choice["epochs"]}',flush=True)
    started = time.monotonic()
    try:
        with np.load(context_path,allow_pickle=False) as z:
            context = {k:z[k] for k in z.files}
        candidate = CANDIDATES[choice['candidate_order']]
        model = ReuploadVQC(n_layers=candidate['n_layers'],reupload=candidate['reupload'],
                            epochs=choice['epochs'],seed=seed)
        # No validation data and no outer test labels enter the fit.
        model.fit(context['X_train'],context['y_train'])
        prediction = model.predict(context['X_val']).astype(str)
        training_prediction = model.predict(context['X_train']).astype(str)
        prep = joblib.load(context_path.with_suffix('.joblib'))
        pipeline = Pipeline(prep.steps+[('vqc',model)])
        joblib.dump(pipeline,folder/f'{stem}.joblib',compress=3)
        model.save_weights(folder/f'{stem}_weights.npz')
        pd.DataFrame(model.history_).to_csv(folder/f'{stem}_history.csv',index=False)
        atomic_npz(folder/f'{stem}_predictions.npz',test_indices=context['val'],train_indices=context['train'],
                   predictions=prediction,training_predictions=training_prediction,
                   selected_indices=context['selected_indices'],selected_names=context['selected_names'])
        record.update(status='complete',finished=now(),seconds=time.monotonic()-started,
                      train_n=len(context['y_train']),test_n=len(context['y_val']),
                      parameters=model.parameter_count_,steps=model.n_steps_)
        atomic_json(result_path,record)
        print(f'DONE outer {stem} {record["seconds"]:.1f}s',flush=True)
        return record
    except Exception:
        record.update(status='failed',finished=now(),seconds=time.monotonic()-started,
                      traceback=traceback.format_exc())
        atomic_json(folder/f'{stem}_failure.json',record)
        raise


def write_ledger(out):
    records = []
    for phase in ('inner','outer'):
        for path in sorted((out/phase).glob('*.json')):
            if path.name.endswith('_started.json'):
                continue
            row = json.loads(path.read_text())
            row['artifact'] = str(path.relative_to(out))
            records.append(row)
    pd.DataFrame(records).to_csv(out/'trial_ledger.csv',index=False)


def summarize(out,data):
    rows,payload = [],dict(y=data['y'],patients=data['patients'],fold=data['fold'])
    for arm in ARMS:
        for seed in range(3):
            pred = np.empty_like(data['y'])
            coverage = np.zeros(len(pred),dtype=int)
            for fold in range(5):
                with np.load(out/'outer'/f'{arm}_o{fold}_seed{seed}_predictions.npz',allow_pickle=False) as z:
                    te = z['test_indices']
                    coverage[te] += 1
                    pred[te] = z['predictions']
            if not np.all(coverage==1):
                raise ValueError('Every outer window must be predicted exactly once')
            name = f'{arm}_seed{seed}'
            payload[name] = pred
            row = E.metrics_row(name,data['y'],pred,data['patients'],arm=arm,seed=seed)
            rows.append({k.replace('record_','patient_'):v for k,v in row.items()})
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'metrics.csv',index=False)
    frame.groupby('arm')[['accuracy','balanced_accuracy','macro_f1','patient_accuracy','patient_macro_f1']].agg(
        ['mean','std']).to_csv(out/'summary.csv')
    atomic_npz(out/'predictions.npz',**payload)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/patient_vmd_wst_vqc')
    parser.add_argument('--out',type=Path,default=ROOT/'results/vqc_improvement')
    parser.add_argument('--n-jobs',type=int,default=10)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    for folder in ('preprocessing','inner','outer'):
        (args.out/folder).mkdir(parents=True,exist_ok=True)
    with np.load(args.source/'features.npz',allow_pickle=False) as f:
        data = {k:f[k] for k in f.files}
    source_meta = json.loads((args.source/'manifest.json').read_text())
    for arm,name in [('vmd','VMD descriptors'),('wst','WST log')]:
        if array_hash(data[arm])!=source_meta['feature_sha256'][name]:
            raise ValueError('Input feature digest differs from baseline manifest')
    if array_hash(data['fold'])!=source_meta['fold_sha256']:
        raise ValueError('Outer folds differ from baseline manifest')
    splits = nested_splits(data['y'],data['patients'],data['fold'])
    import pennylane,scipy,sklearn
    sources = ['ecgvmd/reupload.py','ecgvmd/select.py','ecgvmd/quantum.py',
               'scripts/improve_vqc.py','architects/vqc_improvement_protocol.md']
    manifest = dict(protocol_version=1,source_manifest_sha256=file_hash(args.source/'manifest.json'),
                    source_features_sha256=file_hash(args.source/'features.npz'),
                    code_and_protocol_sha256={p:file_hash(ROOT/p) for p in sources},
                    candidates=CANDIDATES,checkpoints=EPOCHS,inner_seed=0,outer_seeds=[0,1,2],
                    inner_folds=3,inner_split_seed='100 + outer fold',outer_folds=5,
                    metric='pooled inner window macro-F1',k=12,n_qubits=6,
                    optimizer=dict(name='Adam',lr=.02,min_lr=.002,schedule_epochs=80,batch_size=32),
                    device='default.qubit',diff_method='backprop',shots=None,
                    versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                                  pennylane=pennylane.__version__,joblib=joblib.__version__))
    # JSON normalization makes tuples round-trip identically to saved lists.
    manifest = json.loads(json.dumps(manifest))
    path = args.out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Manifest differs; preserve this run and choose a new output directory')
    atomic_json(path,manifest)
    protocol_copy = args.out/'declared_protocol.md'
    protocol_copy.write_text((ROOT/'architects/vqc_improvement_protocol.md').read_text())
    if not (args.out/'started.json').exists():
        atomic_json(args.out/'started.json',dict(started=now(),status='prepared'))
    assignments=[]
    contexts={}
    for fold,(development,test,inner) in splits.items():
        for i,(train,val) in enumerate(inner):
            assignments.extend(dict(outer_fold=fold,inner_fold=i,row=int(j),patient=data['patients'][j],
                                    role=role) for role,indices in [('train',train),('validation',val),('test',test)]
                               for j in indices)
            for arm in ARMS:
                contexts[(arm,fold,i)] = prepare_context(args.out,arm,fold,i,data[arm],data['y'],
                                                         train,val,data[arm+'_names'])
        for arm in ARMS:
            contexts[(arm,fold,'outer')] = prepare_context(args.out,arm,fold,'outer',data[arm],data['y'],
                                                           development,test,data[arm+'_names'])
    pd.DataFrame(assignments).to_csv(args.out/'nested_splits.csv',index=False)
    if args.prepare_only:
        print('Prepared nested splits, preprocessing, and immutable manifest.',flush=True)
        return
    tasks = [(arm,fold,i,c) for fold in range(5) for i in range(3) for c in CANDIDATES for arm in ARMS]
    iterator = joblib.Parallel(n_jobs=args.n_jobs,return_as='generator_unordered')(
        joblib.delayed(inner_trial)(args.out,arm,fold,i,c,contexts[(arm,fold,i)]) for arm,fold,i,c in tasks)
    for completed,record in enumerate(iterator,1):
        write_ledger(args.out)
        print(f'INNER {completed}/{len(tasks)} complete',flush=True)
    choices = select_all(args.out,data['y'],splits)
    print(pd.DataFrame(choices.values()).to_string(index=False),flush=True)
    tasks = [(arm,fold,seed) for seed in range(3) for fold in range(5) for arm in ARMS]
    iterator = joblib.Parallel(n_jobs=args.n_jobs,return_as='generator_unordered')(
        joblib.delayed(outer_trial)(args.out,arm,fold,seed,choices[(arm,fold)],contexts[(arm,fold,'outer')])
        for arm,fold,seed in tasks)
    for completed,record in enumerate(iterator,1):
        write_ledger(args.out)
        print(f'OUTER {completed}/{len(tasks)} complete',flush=True)
    result = summarize(args.out,data)
    atomic_json(args.out/'completed.json',dict(finished=now(),inner_fits=90,outer_fits=30,status='complete'))
    print(result[['arm','seed','accuracy','macro_f1','patient_accuracy']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
