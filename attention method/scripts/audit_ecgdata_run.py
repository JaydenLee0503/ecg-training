"""Reload all completed ECGData models on CPU, verify predictions, export evidence."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from attention_ecg.ecgdata import (verify_cache, fit_normalizer, normalize, read_json,
                                   file_hash, atomic_json)
from attention_ecg.ecgdata_training import (DEFAULT_RUN, load_checkpoint, verify_trial,
                                            partition, pooled_features)
from attention_ecg.ecgdata_report import make_report, confusion, scores
from attention_ecg.model import CepstralSwin


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    run = DEFAULT_RUN
    manifest, data = verify_cache()
    report = make_report()
    protocol = manifest['protocol']
    out = ROOT/'reports/ecgdata_lfcc_swin_v1'
    out.mkdir(exist_ok=True)
    histories, predictions, fits = [], [], []
    for fold in range(5):
        tr, te = partition(data, fold)
        for seed in [None] + protocol['training']['seeds']:
            name = f'logistic_fold{fold}' if seed is None else f'swin_seed{seed}_fold{fold}'
            folder = run/name
            assert verify_trial(folder)
            fit = read_json(folder/'fit.json')
            assert fit['train_indices'] == tr.tolist() and fit['test_indices'] == te.tolist()
            if seed is None:
                model = joblib.load(folder/'model.joblib')
                pooled = pooled_features(data['X'])
                np.testing.assert_allclose(model[0].mean_, pooled[tr].mean(axis=0), rtol=1e-5, atol=1e-5)
                test_prob = model.predict_proba(pooled[te])
                train_pred = model.predict(pooled[tr])
            else:
                mean, scale = fit_normalizer(data['X'], tr)
                with np.load(folder/'preprocessing.npz') as z:
                    np.testing.assert_array_equal(z['train_indices'], tr)
                    np.testing.assert_array_equal(z['test_indices'], te)
                    np.testing.assert_array_equal(z['mean'], mean)
                    np.testing.assert_array_equal(z['scale'], scale)
                weights = len(tr)/(3*np.bincount(data['y'][tr], minlength=3))
                np.testing.assert_array_equal(fit['class_weights'], weights)
                state = load_checkpoint(folder)
                assert state['epoch'] == protocol['training']['epochs'] == 40
                assert state['seed'] == seed and state['fold'] == fold
                assert len(state['history']) == 40
                assert state['history'] == read_json(folder/'history.json')
                assert state['scheduler']['last_epoch'] == 40
                model = CepstralSwin(**protocol['model']).eval()
                model.load_state_dict(state['model'])
                assert all(torch.isfinite(p).all() for p in model.parameters())
                X = normalize(data['X'], mean, scale)
                with torch.inference_mode():
                    # Match the saved inference batch size, on a second device.
                    test_prob = np.concatenate([model(torch.from_numpy(X[te[i:i+32]])).softmax(-1).numpy()
                                                for i in range(0, len(te), 32)])
                    train_pred = np.concatenate([model(torch.from_numpy(X[tr[i:i+32]])).argmax(-1).numpy()
                                                 for i in range(0, len(tr), 32)])
                for row in state['history']:
                    assert np.isfinite(row['loss']) and row['loss'] >= 0
                    epoch = row['epoch'] - 1
                    settings = protocol['training']
                    expected_lr = settings['min_lr'] + (settings['lr']-settings['min_lr']) * (1+np.cos(np.pi*epoch/40))/2
                    np.testing.assert_allclose(row['lr'], expected_lr, rtol=1e-12)
                    histories.append(dict(fold=fold, seed=seed, **row))
            with np.load(folder/'predictions.npz') as saved:
                np.testing.assert_allclose(test_prob, saved['probabilities'], rtol=2e-4, atol=2e-5)
                np.testing.assert_array_equal(test_prob.argmax(1), saved['pred'])
                max_error = float(np.abs(test_prob-saved['probabilities']).max())
                for j, idx in enumerate(te):
                    predictions.append(dict(arm='logistic' if seed is None else 'swin',
                        seed='' if seed is None else seed, fold=fold, window_id=int(idx),
                        patient=data['patients'][idx], source=data['sources'][idx],
                        row_id=int(data['row_ids'][idx]), window_start=int(data['window_start'][idx]),
                        label=protocol['classes'][data['y'][idx]],
                        prediction=protocol['classes'][saved['pred'][j]],
                        **{f'p_{label}':float(saved['probabilities'][j,k]) for k,label in enumerate(protocol['classes'])}))
            train_metrics = scores(confusion(data['y'][tr], train_pred))
            fits.append(dict(trial=name, seed=seed, fold=fold, train_windows=len(tr), test_windows=len(te),
                train_patients=len(np.unique(data['patients'][tr])), test_patients=len(np.unique(data['patients'][te])),
                seconds=fit['seconds'], peak_cuda_bytes=fit.get('peak_cuda_bytes'),
                training_accuracy=train_metrics['accuracy'], training_macro_f1=train_metrics['macro_f1'],
                max_reloaded_probability_error=max_error, warnings=fit.get('warnings', [])))
    metrics = []
    for arm, result in report['results'].items():
        for i, row in enumerate(result['per_seed']):
            metrics.append(dict(arm=arm, seed='' if arm=='logistic' else report['seed_order'][i],
                **{f'{scope}_{metric}':row[scope][metric] for scope in ('window','patient_vote')
                   for metric in ('accuracy','macro_f1','balanced_accuracy')}))
    write_csv(out/'predictions.csv', predictions)
    write_csv(out/'histories.csv', histories)
    write_csv(out/'metrics.csv', metrics)
    write_csv(out/'fits.csv', fits)
    atomic_json(out/'report.json', report)
    validation = dict(status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        fits=len(fits), history_rows=len(histories), new_prediction_rows=len(predictions),
        all_saved_models_reloaded_on='cpu', same_predicted_classes=True,
        max_probability_absolute_error=max(f['max_reloaded_probability_error'] for f in fits),
        sum_fit_seconds=sum(f['seconds'] for f in fits),
        max_peak_cuda_bytes=max(f['peak_cuda_bytes'] or 0 for f in fits),
        failed_real_trials=[p.name for p in run.glob('failure_*.json')],
        logistic_warnings={f['trial']:f['warnings'] for f in fits if f['warnings']},
        checked=['patient separation and saved indices', 'fit-only normalization', 'class weights',
                 'all 40 epochs and cosine learning rates', 'finite model parameters and losses',
                 'saved probabilities reproduced', 'exactly-once out-of-fold coverage'],
        script_sha256=file_hash(__file__))
    atomic_json(out/'validation.json', validation)
    # Independent inventory also covers the report and all final model artifacts.
    inventory=[dict(path=str(p.relative_to(run)), bytes=p.stat().st_size, sha256=file_hash(p))
               for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('.writer.lock','artifact_inventory.csv')]
    write_csv(out/'artifact_inventory.csv', inventory)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ecg-attention-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed in protocol['training']['seeds']:
        loss=np.array([[r['loss'] for r in histories if r['fold']==fold and r['seed']==seed] for fold in range(5)])
        ax.plot(np.arange(1,41), loss.mean(axis=0), label=f'Seed {seed}')
    ax.set(xlabel='Epoch', ylabel='Balanced training cross-entropy',
           title='LFCC + temporal Swin: mean training loss across five folds')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'learning_curves.png', dpi=180)
    fig.savefig(out/'learning_curves.svg')
    plt.close(fig)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
