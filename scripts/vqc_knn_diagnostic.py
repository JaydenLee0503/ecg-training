#!/usr/bin/env python
"""Fixed KNN controls on the completed patient-grouped VQC folds.

This is a classifier/input diagnostic, not a replication of the SPAR paper.
No parameters are chosen from the held-out results and no VQC is retrained.
"""
from pathlib import Path
import argparse
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ecgvmd as E


def inverse_square(distances):
    zero = distances == 0
    weights = 1 / np.maximum(distances, 1e-12)**2
    exact = zero.any(axis=1)
    weights[exact] = zero[exact]
    return weights


def training_fit(run, arm, seed, fold, X, y, folds):
    model = joblib.load(run/f'{arm}_seed{seed}_fold{fold}.joblib')
    train = np.flatnonzero(folds != fold)
    pred = model.predict(X[train])
    return dict(arm=arm,seed=seed,fold=fold,train_n=len(train),
                train_accuracy=accuracy_score(y[train],pred),
                train_macro_f1=f1_score(y[train],pred,average='macro'),
                first_loss=model[-1].loss_[0],final_loss=model[-1].loss_[-1],
                tail_mean_loss=float(np.mean(model[-1].loss_[-5:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-scores',action='store_true',
                        help='Also score the saved VQCs on their own training folds; no retraining')
    parser.add_argument('--n-jobs',type=int,default=5)
    args = parser.parse_args()
    run = ROOT/'results/patient_vmd_wst_vqc'
    out = ROOT/'results/patient_knn_diagnostic'
    out.mkdir(exist_ok=True)
    f = np.load(run/'features.npz', allow_pickle=False)
    y, groups, folds = f['y'], f['patients'], f['fold']
    results, predictions = [], {}
    for arm in ('vmd', 'wst'):
        X = f[arm]
        conditions = ('all_standard', 'selected12_standard', 'selected12_angles')
        preds = {(c,w):np.empty_like(y) for c in conditions for w in ('inverse', 'inverse_square')}
        for fold in range(5):
            tr, te = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
            assert not set(groups[tr]) & set(groups[te])
            vqc = joblib.load(run/f'{arm}_seed0_fold{fold}.joblib')
            selected = vqc[0].transform(X)
            full_scaler = StandardScaler().fit(X[tr])
            matrices = dict(all_standard=full_scaler.transform(X),
                            selected12_standard=vqc[1].scaler_.transform(selected),
                            selected12_angles=vqc[1].transform(selected))
            np.testing.assert_allclose(vqc[1].scaler_.mean_, selected[tr].astype(float).mean(0), atol=1e-12, rtol=0)
            for condition, Z in matrices.items():
                for weight, weights in [('inverse','distance'), ('inverse_square',inverse_square)]:
                    model = KNeighborsClassifier(n_neighbors=10, weights=weights,
                                                 algorithm='brute', metric='euclidean', n_jobs=1)
                    preds[(condition,weight)][te] = model.fit(Z[tr],y[tr]).predict(Z[te])
        for (condition,weight), pred in preds.items():
            name = f'{arm}/{condition}/{weight}'
            m = E.full_metrics(y,pred,groups)
            row = dict(arm=arm,inputs=condition,weights=weight,k=10,
                       accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                       patient_accuracy=m['record']['accuracy'],
                       patient_macro_f1=m['record']['macro_f1'])
            results.append(row)
            predictions[name] = pred
            print(row,flush=True)
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    np.savez_compressed(out/'predictions.npz', y=y,patients=groups,fold=folds,**predictions)
    (out/'protocol.json').write_text(json.dumps(dict(
        source_run=str(run.relative_to(ROOT)),k=10,metric='euclidean',
        weights=['inverse distance','inverse squared distance'],
        selected_features='Exact saved in-fold VQC selections, identical across VQC seeds',
        normalization='All fit on training fold; angle condition uses exact VQC preprocessing',
        interpretation='Fixed diagnostic configurations, not tuned KNN and not a SPAR reproduction'),indent=2)+'\n')
    if args.train_scores:
        tasks = [joblib.delayed(training_fit)(run,arm,seed,fold,f[arm],y,folds)
                 for arm in ('vmd','wst') for seed in range(3) for fold in range(5)]
        train_results = joblib.Parallel(n_jobs=args.n_jobs)(tasks)
        df = pd.DataFrame(train_results)
        df.to_csv(out/'vqc_training_fit.csv',index=False)
        print(df.groupby('arm')[['train_accuracy','train_macro_f1','first_loss','final_loss']].mean())


if __name__ == '__main__':
    main()
