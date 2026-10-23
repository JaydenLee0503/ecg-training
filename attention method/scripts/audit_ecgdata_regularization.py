"""Audit completed v2 artifacts and predictions; never extract features or fit models."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RUN = ROOT / 'results/ecgdata_lfcc_swin_regularization_v2'
OUT = ROOT / 'reports/ecgdata_lfcc_swin_regularization_v2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def metric(y, prediction):
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0)),
                balanced_accuracy=float(recall_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist())


def main():
    completed = read(RUN / 'completed.json')
    check(completed['status'] == 'complete', 'Run incomplete')
    manifest = read(RUN / 'manifest.json')
    counts, inventory = {}, []
    for category, base, entries in [('v2', RUN, completed['files']),
            ('protected', REPO, manifest['protected_files']),
            ('sources', REPO, manifest['identity']['sources'])]:
        for name, expected in entries.items():
            path = base / name
            check(path.is_file() and sha(path) == expected, f'{category} artifact changed: {name}')
            if category == 'v2':
                inventory.append(dict(path=name, bytes=path.stat().st_size, sha256=expected))
        counts[category] = len(entries)
    for name, expected in manifest['identity']['sources'].items():
        check(sha(RUN / 'sources' / Path(name).name) == expected, f'Snapshot changed: {name}')
    for name in ('analysis.json', 'fits.csv', 'histories.csv', 'metrics.csv', 'predictions.csv',
                 'report.md', 'selection.csv', 'splits.csv'):
        check(sha(OUT / name) == sha(RUN / name), f'Report export changed: {name}')
    with np.load(ROOT / 'features/ecgdata_lfcc_v1/features.npz', allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    n = len(data['y'])
    spec = manifest['identity']['spec']
    selections = read(RUN / 'selections.json')
    saved_splits = read(RUN / 'splits.json')
    check(len(selections) == 5, 'Expected five selections')
    expected_splits = {}
    for outer in range(5):
        outer_train = np.flatnonzero(data['fold'] != outer)
        patients = np.unique(data['patients'][outer_train])
        labels = [data['y'][np.flatnonzero(data['patients'] == p)[0]] for p in patients]
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=spec['inner_split_seed'] + outer)
        for inner, (a, b) in enumerate(splitter.split(patients, labels)):
            train = outer_train[np.isin(data['patients'][outer_train], patients[a])]
            val = outer_train[np.isin(data['patients'][outer_train], patients[b])]
            expected_splits[outer, inner] = (train, val)
            np.testing.assert_array_equal(train, saved_splits[str(outer)][inner]['fit_indices'])
            np.testing.assert_array_equal(val, saved_splits[str(outer)][inner]['validation_indices'])
    inner_predictions, outer_predictions = {}, {}
    histories, fit_count, candidate_count = [], 0, 0
    for kind in ('inner', 'outer'):
        folders = sorted((RUN / kind).glob('*/completed.json'))
        check(len(folders) == (90 if kind == 'inner' else 30), 'Unexpected fit count')
        for marker in folders:
            folder = marker.parent
            match = re.fullmatch(r'outer(\d)_(R[01])(?:_inner(\d))?_seed(\d)', folder.name)
            outer, arm, inner, seed = match.groups()
            outer, seed = int(outer), int(seed)
            identity, info = read(folder / 'identity.json'), read(folder / 'fit.json')
            check(not info.get('reused_v1', False), 'Audit currently expects this run without reused outer fits')
            if kind == 'inner':
                train, evaluation = expected_splits[outer, int(inner)]
                epochs = [10, 20, 40]
            else:
                train, evaluation = np.flatnonzero(data['fold'] != outer), np.flatnonzero(data['fold'] == outer)
                epochs = [selections[outer]['per_candidate'][arm]['epoch']]
            np.testing.assert_array_equal(identity['train'], train)
            np.testing.assert_array_equal(identity['evaluation'], evaluation)
            check(not (set(data['patients'][train]) & set(data['patients'][evaluation])), 'Patient leakage')
            check(identity['retained'] == epochs and info['epochs'] == epochs[-1], 'Epoch mismatch')
            values = data['X'][train].astype(np.float64)
            mean, scale = values.mean((0, 1)), values.std((0, 1))
            scale[scale < 1e-8] = 1
            with np.load(folder / 'preprocessing.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['mean'], mean)
                np.testing.assert_array_equal(z['scale'], scale)
                np.testing.assert_array_equal(z['train_indices'], train)
                np.testing.assert_array_equal(z['evaluation_indices'], evaluation)
            np.testing.assert_array_equal(info['class_weights'], len(train) / (3 * np.bincount(data['y'][train], minlength=3)))
            history = read(folder / 'history.json')
            check(len(history) == info['epochs'], 'History length mismatch')
            for index, row in enumerate(history):
                check(row['epoch'] == index + 1 and np.isfinite(row['loss']) and row['loss'] >= 0, 'Invalid history')
                np.testing.assert_allclose(row['lr'], .0001 + .0009 * (1 + np.cos(np.pi * index / 40)) / 2, rtol=1e-12)
                histories.append(dict(arm=arm, kind=kind, seed=seed, **row))
            for epoch in epochs:
                with np.load(folder / f'epoch_{epoch:03d}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['indices'], evaluation)
                    np.testing.assert_array_equal(z['y'], data['y'][evaluation])
                    np.testing.assert_array_equal(z['patients'], data['patients'][evaluation])
                    probabilities = z['probabilities']
                    check(probabilities.shape == (len(evaluation), 3) and np.isfinite(probabilities).all()
                          and np.all(probabilities >= 0), 'Invalid probabilities')
                    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
                    np.testing.assert_array_equal(z['pred'], probabilities.argmax(1))
                    metrics = metric(z['y'], z['pred'])
                    np.testing.assert_allclose(metrics['macro_f1'], history[epoch - 1]['evaluation_macro_f1'], atol=1e-14)
                    if kind == 'inner':
                        inner_predictions[outer, arm, seed, int(inner), epoch] = (evaluation.copy(), z['pred'].copy())
                    else:
                        outer_predictions[outer, arm, seed] = (evaluation.copy(), z['pred'].copy(), probabilities.copy())
                candidate_count += 1
            fit_count += 1
    for outer, selection in enumerate(selections):
        expected = np.flatnonzero(data['fold'] != outer)
        rows = []
        for arm in ('R0', 'R1'):
            for epoch in (10, 20, 40):
                f1s = []
                for seed in (0, 1, 2):
                    predictions, coverage = np.full(n, -1), np.zeros(n, dtype=int)
                    for inner in range(3):
                        indices, pred = inner_predictions[outer, arm, seed, inner, epoch]
                        predictions[indices] = pred
                        coverage[indices] += 1
                    np.testing.assert_array_equal(np.flatnonzero(coverage), expected)
                    check(np.all(coverage[expected] == 1), 'Inner coverage differs')
                    f1s.append(metric(data['y'][expected], predictions[expected])['macro_f1'])
                rows.append(dict(candidate=arm, epoch=epoch, mean_macro_f1=float(np.mean(f1s)), seed_macro_f1=f1s))
        for actual, recorded in zip(rows, selection['rows']):
            check(actual['candidate'] == recorded['candidate'] and actual['epoch'] == recorded['epoch'], 'Candidate order changed')
            np.testing.assert_allclose(actual['seed_macro_f1'], recorded['seed_macro_f1'], atol=1e-14)
        best = lambda items: min(items, key=lambda r: (-r['mean_macro_f1'], r['epoch'], r['candidate']))
        for chosen, recorded in [(best(rows), selection['selected'])] + [
                (best([r for r in rows if r['candidate'] == arm]), selection['per_candidate'][arm]) for arm in ('R0','R1')]:
            check((chosen['candidate'], chosen['epoch']) == (recorded['candidate'], recorded['epoch']), 'Selection differs')
    report = read(OUT / 'analysis.json')
    checked_metrics = {}
    patients = np.unique(data['patients'])
    for arm in ('R0', 'R1', 'swin'):
        checked_metrics[arm] = []
        for seed in (0, 1, 2):
            full = np.full(n, -1)
            for outer in range(5):
                chosen = selections[outer]['selected']['candidate'] if arm == 'swin' else arm
                indices, prediction, _ = outer_predictions[outer, chosen, seed]
                full[indices] = prediction
            check(np.all(full >= 0), 'Outer coverage missing')
            y_patient, pred_patient = [], []
            for patient in patients:
                idx = data['patients'] == patient
                check(len(np.unique(data['y'][idx])) == 1, 'Inconsistent patient label')
                y_patient.append(data['y'][idx][0])
                pred_patient.append(np.bincount(full[idx], minlength=3).argmax())
            metrics = dict(window=metric(data['y'], full), patient_vote=metric(y_patient, pred_patient))
            for scope, values in metrics.items():
                for key, value in values.items():
                    np.testing.assert_allclose(value, report['results'][arm]['per_seed'][seed][scope][key], atol=1e-14)
            checked_metrics[arm].append(metrics)
    with (OUT / 'predictions.csv').open(newline='') as stream:
        prediction_rows = list(csv.DictReader(stream))
    check(len(prediction_rows) == 3 * 3 * n, 'Prediction export row count differs')
    labels = manifest['identity']['base_protocol']['classes']
    seen = set()
    for row in prediction_rows:
        outer, seed, idx = int(row['outer_fold']), int(row['seed']), int(row['window_id'])
        key = (row['arm'], seed, idx)
        check(key not in seen, 'Duplicate exported prediction')
        seen.add(key)
        arm = selections[outer]['selected']['candidate'] if row['arm'] == 'selected' else row['arm']
        indices, pred, probabilities = outer_predictions[outer, arm, seed]
        j = int(np.flatnonzero(indices == idx)[0])
        check(row['patient'] == data['patients'][idx] and row['label'] == labels[data['y'][idx]]
              and row['prediction'] == labels[pred[j]], 'Export identity differs')
        np.testing.assert_allclose([float(row[f'p_{label}']) for label in labels], probabilities[j], rtol=0, atol=0)
    with (OUT / 'artifact_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(inventory)
    result = dict(status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  artifact_checks=counts, source_snapshots=10, matching_report_exports=8,
                  fits=fit_count, candidate_prediction_sets=candidate_count, history_rows=len(histories),
                  exported_prediction_rows=len(prediction_rows), independently_recomputed_metrics=checked_metrics,
                  checked=['sizes/hashes via frozen completion marker', 'protected original files', 'current sources and snapshots',
                           'patient splits and training-only normalization/class weights', 'learning-rate schedule and histories',
                           'all candidate probabilities and pooled inner selection', 'window/patient metrics using sklearn',
                           'exported predictions against original arrays'],
                  new_training=False, new_inference=False, script_sha256=sha(Path(__file__)))
    (OUT / 'completion_audit_2026-10-23.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot(histories)
    print(json.dumps({k:v for k,v in result.items() if k != 'independently_recomputed_metrics'}, indent=2))


def plot(histories):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ecgdata-regularization-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, color in [('R0', '#3465a4'), ('R1', '#c75b12')]:
        for seed in (0, 1, 2):
            rows = [r for r in histories if r['kind'] == 'inner' and r['arm'] == arm and r['seed'] == seed]
            for ax, key in zip(axes, ['loss', 'evaluation_macro_f1']):
                values = [np.mean([r[key] for r in rows if r['epoch'] == epoch]) for epoch in range(1, 41)]
                ax.plot(range(1, 41), values, color=color, linestyle=['-', '--', ':'][seed], label=f'{arm}, seed {seed}')
    axes[0].set(title='Training loss', ylabel='Class-weighted cross-entropy')
    axes[1].set(title='Inner validation performance', ylabel='Macro-F1')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Mean across 15 inner partitions per seed; descriptive curves')
    fig.tight_layout()
    fig.savefig(OUT / 'inner_learning_curves.png', dpi=180)
    fig.savefig(OUT / 'inner_learning_curves.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
