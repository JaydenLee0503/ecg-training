"""Validate saved ACS predictions and report fixed internal-validation results."""
import argparse
import csv
import json
from pathlib import Path

from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as training
from experiments.acs_omi_vmd_wst_vqc.reporting import bootstrap,METRICS
import numpy as np


def report(out):
    m,sha=training.read_run(out)
    completed=json.loads((out/'completed.json').read_text())
    if completed['context_sha']!=sha or completed['status']!='complete' or completed['fits']!=len(training.TRIALS):
        raise ValueError('Training must complete before reporting')
    if (out/'report_completed.json').exists():
        training.checked_files(out,json.loads((out/'report_completed.json').read_text()),sha)
        print('Verified completed statistical report; no bootstrap recomputation',flush=True)
        return
    scores={}; groups={}; reference=None
    for arm,name in training.TRIALS:
        folder=out/arm/name
        training.checked_files(folder,json.loads((folder/'completed.json').read_text()),sha)
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            saved={k:z[k] for k in z.files}
        if reference is None:
            reference=saved
        for field in ('y','patients','record_ids'):
            np.testing.assert_array_equal(reference[field],saved[field])
        np.testing.assert_array_equal(saved['prediction'],(saved['score']>=0.5).astype(int))
        key=f'{arm}/{name}'
        scores[key]=saved['score']
        if not name.startswith('vqc'):
            groups[key]=[key]
    for arm in ('vmd','wst'):
        groups[f'{arm}/vqc']=[f'{arm}/vqc_seed{seed}' for seed in m['protocol']['vqc']['seeds']]
    with np.load(out/'vmd/inputs.npz',allow_pickle=False) as z:
        prevalence=float(z['y_fit'].mean())
        np.testing.assert_array_equal(z['y_validation'],reference['y'])
        np.testing.assert_array_equal(z['validation_record_ids'],reference['record_ids'])
        np.testing.assert_array_equal(z['validation_patients'],reference['patients'])
    scores['constant_prevalence']=np.full(len(reference['y']),prevalence)
    groups['constant_prevalence']=['constant_prevalence']
    pairs={f'vmd_minus_wst/{name}':(f'vmd/{name}',f'wst/{name}') for name in ('vqc','logistic','weighted_knn')}
    for arm in ('vmd','wst'):
        for control in ('logistic','weighted_knn','constant_prevalence'):
            pairs[f'{arm}/vqc_minus_{control}']=(f'{arm}/vqc',
                control if control=='constant_prevalence' else f'{arm}/{control}')
    cfg=m['protocol']['evaluation']
    summary=bootstrap(reference['y'],reference['patients'],scores,groups,pairs,
                      cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    summary.update(context_sha=sha,fit_prevalence=prevalence,validation_records=len(reference['y']),
        validation_patients=len(np.unique(reference['patients'])),official_test_processed=False,
        completed_at_utc=training.old.serial.now(),threshold=0.5)
    training.old.atomic_json(out/'analysis.json',summary)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model']+list(next(iter(summary['per_model'].values()))))
        writer.writeheader()
        writer.writerows(dict(model=name,**values) for name,values in summary['per_model'].items())
    with (out/'predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['record_id','patient_id','label']+list(scores))
        for i in range(len(reference['y'])):
            writer.writerow([reference['record_ids'][i],reference['patients'][i],reference['y'][i]]+
                            [float(s[i]) for s in scores.values()])
    lines=['# ACS OMI fixed internal-validation results','',
        'All eligible development records use the frozen patient split. Official test data remain reserved.',
        'VQC rows average metrics over seeds 0/1/2. Intervals resample patients and retain their ECGs together.',
        '', '| Model | Average precision (95% interval) | ROC AUC | Sensitivity | Specificity | Accuracy |',
        '|---|---:|---:|---:|---:|---:|']
    for name,values in summary['groups'].items():
        lo,hi=summary['group_intervals'][name]['average_precision']
        lines.append(f'| {name} | {values["average_precision"]:.4f} [{lo:.4f}, {hi:.4f}] | '
            f'{values["roc_auc"]:.4f} | {values["sensitivity"]:.4f} | {values["specificity"]:.4f} | {values["accuracy"]:.4f} |')
    lines+=['','## Paired average-precision differences','']
    for name,values in summary['paired_differences'].items():
        lo,hi=values['intervals']['average_precision']
        lines.append(f'- {name}: {values["point"]["average_precision"]:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].')
    lines+=['','These intervals condition on saved predictions. They do not include retraining uncertainty.',
            'This fixed comparison does not establish clinical validity or quantum advantage.','']
    (out/'report.md').write_text('\n'.join(lines))
    training.old.atomic_json(out/'report_completed.json',dict(status='complete',context_sha=sha,
        files={name:training.old.serial.file_hash(out/name) for name in
               ('analysis.json','metrics.csv','predictions.csv','report.md')}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=training.BASE/'results/omi_models_v1_retry128k')
    args=parser.parse_args()
    with training.old.run_lock(args.out):
        report(args.out)
