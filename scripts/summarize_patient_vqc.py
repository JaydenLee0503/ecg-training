#!/usr/bin/env python
"""Summarize the matched VQC run with a paired bootstrap over patients.

Intervals resample patients and condition on the saved out-of-fold predictions;
they do not include retraining or tuning variability. Seeds share the same data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ecgvmd as E
from scripts.matched_vqc import ARMS


def accuracy(cm):
    return np.trace(cm,axis1=-2,axis2=-1) / cm.sum(axis=(-2,-1))


def macro_f1(cm):
    tp = np.diagonal(cm,axis1=-2,axis2=-1)
    denominator = cm.sum(-2) + cm.sum(-1)
    return np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),
                     where=denominator>0).mean(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,default=ROOT/'results/patient_vmd_wst_vqc')
    ap.add_argument('--n-boot',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=0)
    args = ap.parse_args()
    meta = json.loads((args.run_dir/'manifest.json').read_text())
    z = np.load(args.run_dir/'predictions.npz',allow_pickle=False)
    y,groups,fold = z['__y'],z['__groups'],z['__fold']
    patients,pi = np.unique(groups,return_inverse=True)
    if any(len(set(fold[groups==p]))!=1 for p in patients):
        raise ValueError('Patient crosses outer folds')
    labels = {c:i for i,c in enumerate(E.CLASS_ORDER)}
    yi = np.asarray([labels[v] for v in y])
    matrices = {}
    rows = []
    for arm in ARMS:
        for seed in meta['seeds']:
            name = f'{arm} + VQC seed={seed}'
            pred = z[name]
            if len(pred)!=len(y):
                raise ValueError('Prediction rows differ')
            pj = np.asarray([labels[v] for v in pred])
            cm = np.zeros((len(patients),3,3),dtype=int)
            np.add.at(cm,(pi,yi,pj),1)
            matrices[(arm,seed)] = cm
            truth,vote = E.record_vote(pred,groups,y)
            pcm = np.zeros_like(cm)
            pcm[np.arange(len(patients)),[labels[v] for v in truth],[labels[v] for v in vote]] = 1
            matrices[(arm,seed,'patient')] = pcm
            metrics = E.full_metrics(y,pred,groups)
            np.testing.assert_allclose(macro_f1(cm.sum(0)),metrics['macro_f1'])
            np.testing.assert_allclose(macro_f1(pcm.sum(0)),metrics['record']['macro_f1'])
            rows.append(dict(arm=arm,seed=seed,accuracy=metrics['accuracy'],
                             macro_f1=metrics['macro_f1'],
                             patient_accuracy=metrics['record']['accuracy'],
                             patient_macro_f1=metrics['record']['macro_f1']))
    # The same patient multiplicities are applied to every arm and seed.
    rng = np.random.default_rng(args.seed)
    multiplicities = rng.multinomial(len(patients),np.full(len(patients),1/len(patients)),
                                    size=args.n_boot)
    comparisons = {}
    for level in ('window','patient'):
        for metric,func in [('accuracy',accuracy),('macro_f1',macro_f1)]:
            samples, observed = {}, {}
            for arm in ARMS:
                boot_scores, observed_scores = [], []
                for seed in meta['seeds']:
                    key=(arm,seed) if level=='window' else (arm,seed,'patient')
                    cm=matrices[key]
                    sampled=(multiplicities @ cm.reshape(len(patients),9)).reshape(-1,3,3)
                    boot_scores.append(func(sampled))
                    observed_scores.append(float(func(cm.sum(0))))
                samples[arm]=np.mean(boot_scores,axis=0)
                observed[arm]=float(np.mean(observed_scores))
            delta=samples[ARMS[1]]-samples[ARMS[0]]
            comparisons[f'{level}_{metric}']=dict(
                wst_minus_vmd=observed[ARMS[1]]-observed[ARMS[0]],
                ci95=np.percentile(delta,[2.5,97.5]).tolist())
    result=dict(n_patients=len(patients),n_windows=len(y),n_boot=args.n_boot,
                bootstrap_seed=args.seed,initialization_seeds=meta['seeds'],
                interval_scope='Paired patient bootstrap, conditional on saved predictions; no retraining',
                comparisons=comparisons)
    (args.run_dir/'paired_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    df=pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
