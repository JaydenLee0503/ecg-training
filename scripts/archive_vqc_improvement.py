#!/usr/bin/env python
"""Export complete VQC trial histories, figures, predictions, and artifact hashes.

Run after report_vqc_improvement.py and again after final narrative edits.
This script reads saved results only; it neither selects nor retrains models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ecgvmd as E


def main():
    run = ROOT/'results/vqc_improvement'
    recorded = ROOT/'architects/vqc_improvement_tables'
    figures = ROOT/'architects/vqc_improvement_figures'
    figures.mkdir(exist_ok=True)
    assert json.loads((run/'validation.json').read_text())['status']=='passed'
    ledger = pd.read_csv(run/'trial_ledger.csv')
    histories = []
    for row in ledger.to_dict('records'):
        phase,arm,fold = row['phase'],row['arm'],row['outer_fold']
        if phase=='inner':
            stem = f'{arm}_o{fold}_i{int(row["inner_fold"])}_{row["candidate"]}'
            candidate = row['candidate']
        else:
            stem = f'{arm}_o{fold}_seed{row["seed"]}'
            candidate = json.loads((run/phase/f'{stem}.json').read_text())['choice']['candidate']
        history = pd.read_csv(run/phase/f'{stem}_history.csv')
        for key,value in dict(trial=stem,phase=phase,arm=arm,outer_fold=fold,
                              inner_fold=row['inner_fold'],seed=row['seed'],candidate=candidate).items():
            history[key] = value
        histories.append(history)
    history = pd.concat(histories,ignore_index=True)
    history.to_csv(recorded/'epoch_histories.csv',index=False)

    metrics = pd.read_csv(run/'comparison_metrics.csv')
    classifiers = ['Original VQC','Nested compact VQC','Weighted KNN (1/d)']
    colors = ['#777777','#006d9c','#c26000']
    fig,axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','macro_f1','patient_accuracy'],
                                ['Window accuracy','Window macro-F1','Patient-vote accuracy']):
        for j,(classifier,color) in enumerate(zip(classifiers,colors)):
            for i,arm in enumerate(('vmd','wst')):
                values = metrics[(metrics.arm==arm)&(metrics.classifier==classifier)][metric]
                ax.errorbar(i+(j-1)*.20,values.mean(),
                            yerr=values.std() if len(values)>1 else None,
                            fmt='o',color=color,capsize=4,label=classifier if i==0 else None)
        ax.set(xticks=[0,1],xticklabels=['VMD','WST'],ylim=(0,1),title=title)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Patient-held-out scores; VQC error bars show initialization SD, not patient uncertainty',fontsize=11)
    for suffix in ('png','svg'):
        fig.savefig(figures/f'comparison.{suffix}',dpi=180)
    plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for i,arm in enumerate(('vmd','wst')):
        for candidate,color in zip(('compact_once','reupload_2','reupload_3'),colors):
            frame = history[(history.phase=='inner')&(history.arm==arm)&(history.candidate==candidate)]
            means = frame.groupby('epoch')[['loss','gradient_norm','val_macro_f1']].mean()
            for ax,metric in zip(axes[i],('loss','gradient_norm','val_macro_f1')):
                values = means[metric].dropna()
                ax.plot(values.index,values,label=candidate,color=color,
                        marker='o' if metric=='val_macro_f1' else None,markersize=4)
                ax.set_xlabel('Epoch')
                ax.grid(alpha=.2)
        axes[i,0].set_ylabel(arm.upper())
    for ax,title in zip(axes[0],('Mean minibatch weighted loss','Mean gradient norm','Inner-validation macro-F1')):
        ax.set_title(title)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Descriptive means over 15 overlapping inner fits per candidate and front end',fontsize=11)
    for suffix in ('png','svg'):
        fig.savefig(figures/f'learning_curves.{suffix}',dpi=180)
    plt.close(fig)

    with np.load(run/'predictions.npz',allow_pickle=False) as new, \
         np.load(ROOT/'results/patient_vmd_wst_vqc/predictions.npz',allow_pickle=False) as old, \
         np.load(ROOT/'results/patient_knn_diagnostic/predictions.npz',allow_pickle=False) as knn:
        predictions = pd.DataFrame(dict(window_index=np.arange(len(new['y'])),
                                        patient=new['patients'],fold=new['fold'],truth=new['y']))
        fold_rows = []
        for arm,label in [('vmd','VMD descriptors'),('wst','WST log')]:
            outputs = [(f'{arm}_knn',knn[f'{arm}/selected12_angles/inverse'])]
            for seed in range(3):
                outputs.extend([(f'{arm}_original_seed{seed}',old[f'{label} + VQC seed={seed}']),
                                (f'{arm}_new_seed{seed}',new[f'{arm}_seed{seed}'])])
            for name,pred in outputs:
                predictions[name]=pred
                for fold in range(5):
                    take = new['fold']==fold
                    row = E.metrics_row(name,new['y'][take],pred[take],new['patients'][take],fold=fold)
                    fold_rows.append({k.replace('record_','patient_'):v for k,v in row.items()})
        predictions.to_csv(recorded/'window_predictions.csv',index=False)
        pd.DataFrame(fold_rows).to_csv(recorded/'fold_metrics.csv',index=False)

    # Retain the initial snapshot unchanged; record final reporting code separately.
    reporting_sources = ['scripts/report_vqc_improvement.py','scripts/archive_vqc_improvement.py']
    for relative in reporting_sources:
        destination = run/'reporting_source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    shutil.copyfile(ROOT/'architects/vqc_improvement_results.md',run/'report.md')
    for name in ('environment.json','nested_splits.csv','selection.json','started.json','completed.json'):
        shutil.copyfile(run/name,recorded/name)
    inventory = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name!='artifact_inventory.csv':
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            inventory.append(dict(path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=digest))
    pd.DataFrame(inventory).to_csv(recorded/'artifact_inventory.csv',index=False)
    shutil.copyfile(recorded/'artifact_inventory.csv',run/'artifact_inventory.csv')
    print(json.dumps(dict(trials=history.trial.nunique(),epochs=len(history),
                          artifacts=len(inventory),figures=str(figures)),indent=2))


if __name__=='__main__':
    main()
