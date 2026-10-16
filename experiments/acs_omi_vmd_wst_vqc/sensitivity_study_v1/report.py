"""Descriptive saved curves and independent calibration-threshold evaluation."""
import csv
import json
import os
import time

import numpy as np

from . import common as c
from .metrics import analyze, groups_for, FIELDS, Curve
from experiments.acs_omi_vmd_wst_vqc.reporting import binary_metrics


def csv_scores(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames[3:]
        rows = list(reader)
    reference = dict(record_ids=np.array([r['record_id'] for r in rows]),
                     patients=np.array([r['patient_id'] for r in rows]),
                     y=np.array([int(r['label']) for r in rows]))
    return reference, {n: np.array([float(r[n]) for r in rows]) for n in names}


def figure(out, result, prefix):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/omi-sensitivity-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in result['groups'] if n.startswith(prefix + '/')]
    labels = [n.split('/', 1)[1].replace('/', ' + ').replace('weighted_knn', 'KNN').upper() for n in names]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for j, target in enumerate(result['targets_percent']):
        for i, metric in enumerate(('specificity', 'precision')):
            ax = axes[i, j]
            for k, name in enumerate(names):
                point = 100 * result['groups'][name][j][metric]
                lo, hi = np.array(result['group_intervals'][name][j][metric]) * 100
                # A percentile interval need not contain the estimate; draw separately.
                ax.plot([lo, hi], [k, k], color='#32658a', linewidth=2)
                ax.scatter(point, k, color='#153b58', s=28, zorder=3)
            ax.set_yticks(range(len(names)), labels if j == 0 else [''] * len(names))
            ax.invert_yaxis()
            ax.set_xlim(0, 100 if metric == 'specificity' else 30)
            ax.set_xlabel(metric.capitalize() + ' (%)')
            ax.set_title(f'{target}% sensitivity target')
            ax.grid(axis='x', alpha=.2)
    label = ('Cutoffs chosen on validation: descriptive curves' if prefix == 'balanced' else
             'Cutoffs chosen on calibration; evaluated on separate validation patients')
    fig.suptitle(label + '\nPoints: all-seed means; lines: 95% patient-bootstrap intervals', fontsize=12)
    for ext in ('png', 'svg'):
        fig.savefig(out / f'operating_points.{ext}', dpi=160)
    plt.close(fig)


def export(out, result, reference, scores, mode, seconds):
    fields = ('threshold',) + FIELDS
    rows = []
    for name, points in result['per_model'].items():
        for j, point in enumerate(points):
            rows.append([name, result['targets_percent'][j]] + [point[k] for k in fields] +
                        [v for k in fields for v in result['per_model_intervals'][name][j][k]])
    header = ['model', 'target_percent'] + list(fields) + [k + suffix for k in fields for suffix in ('_lo', '_hi')]
    c.csv_write(out / 'per_seed_metrics.csv', header, rows)
    group_rows = []
    for name, points in result['groups'].items():
        for j, point in enumerate(points):
            group_rows.append([name, result['targets_percent'][j]] + [point[k] for k in FIELDS] +
                [v for k in FIELDS for v in result['group_intervals'][name][j][k]])
    c.csv_write(out / 'group_metrics.csv', ['model', 'target_percent'] + list(FIELDS) +
        [k + suffix for k in FIELDS for suffix in ('_lo', '_hi')], group_rows)
    baseline = {n: binary_metrics(reference['y'], s) for n, s in scores.items()}
    result['fixed_0_5_per_model'] = baseline
    result['fixed_0_5_groups'] = {name: {k: float(np.mean([baseline[m][k] for m in members]))
        for k in baseline[members[0]]} for name, members in groups_for(scores).items()}
    result.update(mode=mode, records=len(reference['y']), patients=len(np.unique(reference['patients'])),
        positive_ecgs=int(reference['y'].sum()), seconds=seconds, completed_at_utc=c.now(), official_test_processed=False)
    c.write(out / 'analysis.json', result)
    c.csv_write(out / 'predictions.csv', ['record_id', 'patient_id', 'label'] + list(scores),
        ([reference['record_ids'][i], reference['patients'][i], int(reference['y'][i])] +
         [float(s[i]) for s in scores.values()] for i in range(len(reference['y']))))
    saved = mode == 'saved_score_descriptive'
    lines = ['# ACS/OMI sensitivity study: ' + ('saved-score exploration' if saved else 'calibration-selected thresholds'), '',
        ('Thresholds are chosen from the same validation labels used to describe these curves. '
         'These are exploratory operating points, not independently evaluated cutoffs.' if saved else
         'Each threshold was selected using only the held-out calibration patients within the original fit cohort. '
         'These tables evaluate those frozen thresholds on the unchanged validation patients. '
         'A calibration target is not a guarantee of validation sensitivity.'), '',
        f"Validation: {len(reference['y']):,} ECGs, {len(np.unique(reference['patients'])):,} patients, "
        f"{int(reference['y'].sum())} OMI-positive ECGs. Natural prevalence retained; official test untouched.", '',
        'VQC/Swin results average metrics across all initialization seeds 0/1/2, not probabilities. '
        'Every seed has its own cutoff. No seed is selected. All classical controls are included.', '',
        '| Model | Target | Achieved sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | False alarms / 1,000 non-OMI |',
        '|---|---:|---:|---:|---:|---:|']
    for name, points in result['groups'].items():
        for j, p in enumerate(points):
            cells = []
            for metric in ('sensitivity', 'specificity', 'precision'):
                lo, hi = result['group_intervals'][name][j][metric]
                cells.append(f'{100*p[metric]:.2f}% [{100*lo:.2f}, {100*hi:.2f}]')
            lines.append(f'| {name} | {result["targets_percent"][j]}% | ' + ' | '.join(cells) +
                         f' | {p["false_alarms_per_1000_negative"]:.1f} |')
    lines += ['', '## Fixed 0.5 cutoff and ranking metrics', '',
        '| Model | AP | ROC AUC | Sensitivity | Specificity | Precision | Accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, p in result['fixed_0_5_groups'].items():
        lines.append(f'| {name} | {p["average_precision"]:.4f} | {p["roc_auc"]:.4f} | '
            f'{p["sensitivity"]:.2%} | {p["specificity"]:.2%} | {p["precision"]:.2%} | {p["accuracy"]:.2%} |')
    lines += ['', '## Statistical interpretation', '',
        f"- {result['bootstrap']['valid']:,}/{result['bootstrap']['requested']:,} paired patient-bootstrap draws; "
        f"seed {result['bootstrap']['seed']}; {result['bootstrap']['single_class_discarded']} single-class draws discarded.",
        '- Thresholds use score >= cutoff, keep ties together, and choose the largest observed score meeting the target. '
        'No interpolated or randomized decisions. Discrete KNN scores can overshoot a target substantially.',
        ('- Thresholds are reselected in each bootstrap draw: intervals describe the empirical curve at the requested sensitivity. '
         'This resampling does not remove selection optimism or independently validate a cutoff.' if saved else
         '- Bootstrap thresholds stay fixed. Intervals condition on the fitted models and calibration cutoffs; '
         'they exclude training, calibration-selection and split uncertainty.'),
        '- Intervals are pointwise, without multiplicity correction. ECG outcomes remain clustered by patient; '
        'mixed-label patients are retained without inventing a patient-level diagnosis.',
        '- AP and ROC AUC are unchanged by moving a cutoff. Higher sensitivity can come with many false positives.',
        '- The validation cohort has informed earlier research decisions. No external/clinical validation or quantum advantage is established.',
        '- All model-minus-LFCC-logistic paired differences and intervals are retained in analysis.json. '
        'For new fits, achieved sensitivities may differ, so those differences are not a comparison at identical achieved sensitivity.', '',
        '## Artifacts', '',
        '- [All seed cutoffs, confusion counts and intervals](per_seed_metrics.csv)',
        '- [All group metrics and intervals](group_metrics.csv)',
        '- [Complete analysis and paired comparisons](analysis.json)',
        '- [Operating-point figure](operating_points.png)',
        '- Full prediction CSV and bootstrap draws are retained in the study results folder (gitignored).', '']
    c.write_bytes(out / 'report.md', '\n'.join(lines).encode())
    figure(out, result, 'balanced' if saved else 'new')


def publish(out):
    target = c.REPORTS / out.name
    for name in ('report.md', 'analysis.json', 'per_seed_metrics.csv', 'group_metrics.csv',
                 'operating_points.png', 'operating_points.svg'):
        c.write_bytes(target / name, (out / name).read_bytes())


def saved():
    started = time.perf_counter()
    manifest, ctx = c.manifest()
    out = c.RUN / 'saved_scores'
    if c.verified(out, ctx):
        publish(out)
        print('Verified completed saved-score analysis', flush=True)
        return
    spec, paths = c.source_paths()
    source = paths['balanced'] / 'report'
    if not c.verified(source):
        raise ValueError('Saved source report is incomplete')
    reference, scores = csv_scores(source / 'predictions.csv')
    # Match every CSV probability/identity to its frozen per-trial NPZ.
    for name, score in scores.items():
        if name == 'constant_natural_prior':
            continue
        prefix, arm, model = name.split('/')
        folder = (paths['original'] / arm / model if prefix == 'original' else
                  paths['balanced'] / ('lfcc_models' if arm == 'lfcc' else f'balanced/{arm}') / model)
        marker = c.read(folder / 'completed.json')
        if c.sha(folder / 'predictions.npz') != marker['files']['predictions.npz']:
            raise ValueError('Changed saved predictions')
        data = c.arrays(folder / 'predictions.npz')
        for key in ('y', 'record_ids', 'patients'):
            np.testing.assert_array_equal(data[key], reference[key])
        np.testing.assert_array_equal(data['score'], score)
    cfg = spec['bootstrap']
    result, draws = analyze(reference['y'], reference['patients'], scores,
        spec['targets_percent'], cfg['draws'], cfg['seed'])
    out.mkdir(parents=True, exist_ok=True)
    c.npz(out / 'bootstrap_draws.npz', **draws)
    export(out, result, reference, scores, 'saved_score_descriptive', time.perf_counter()-started)
    c.complete(out, ctx)
    publish(out)
    print('Saved-score comparison complete', flush=True)


def new_report():
    started = time.perf_counter()
    manifest, ctx = c.manifest()
    out = c.RUN / 'calibrated'
    if c.verified(out, ctx):
        publish(out)
        print('Verified completed calibrated report', flush=True)
        return
    from .run import trial_names
    scores, thresholds, calibration = {}, {}, {}
    cohort = c.arrays(c.RUN / 'prepared/cohort.npz')
    idx = np.flatnonzero(cohort['partitions'] == 'validation')
    reference = {key: cohort[key][idx] for key in ('record_ids', 'patients', 'y')}
    for arm, model in trial_names():
        folder = c.RUN / 'models' / arm / model
        if not c.verified(folder, ctx):
            raise ValueError(f'Incomplete trial: {folder}')
        name = f'new/{arm}/{model}'
        data = c.arrays(folder / 'validation_predictions.npz')
        for key in reference:
            np.testing.assert_array_equal(data[key], reference[key])
        scores[name] = data['score']
        frozen = c.read(folder / 'thresholds.json')
        cal = c.arrays(folder / 'calibration_predictions.npz')
        cal_idx = np.flatnonzero(cohort['partitions'] == 'calibration')
        for key in reference:
            np.testing.assert_array_equal(cal[key], cohort[key][cal_idx])
        points = Curve(cal['y'], cal['score']).operating_points(manifest['protocol']['targets_percent'])
        if points != frozen['calibration_points']:
            raise ValueError('Frozen calibration thresholds cannot be reproduced')
        thresholds[name] = [p['threshold'] for p in points]
        np.testing.assert_array_equal(data['target_predictions'], data['score'][:, None] >= np.array(thresholds[name]))
        calibration[name] = frozen
    scores['constant_natural_prior'] = np.full(len(idx), cohort['y'][cohort['partitions'] == 'subfit'].mean())
    thresholds['constant_natural_prior'] = [float(scores['constant_natural_prior'][0])] * 3
    spec = manifest['protocol']
    result, draws = analyze(reference['y'], reference['patients'], scores,
        spec['targets_percent'], spec['bootstrap']['draws'], spec['bootstrap']['seed'], thresholds)
    result['calibration'] = calibration
    result['preparation'] = c.read(c.RUN / 'prepared/audit.json')
    out.mkdir(parents=True, exist_ok=True)
    c.npz(out / 'bootstrap_draws.npz', **draws)
    export(out, result, reference, scores, 'calibration_selected_validation_evaluated', time.perf_counter()-started)
    c.complete(out, ctx)
    publish(out)
    print('Independent calibration-threshold evaluation complete', flush=True)


def audit():
    from .run import trial_names
    manifest, ctx = c.manifest()
    count = 0
    for name, digest in manifest['protected_sha256'].items():
        if c.sha(c.ROOT / name) != digest:
            raise ValueError(f'Protected original artifact changed: {name}')
        count += 1
    folders = [c.RUN / 'prepared', c.RUN / 'saved_scores', c.RUN / 'calibrated'] + [
        c.RUN / 'models' / arm / name for arm, name in trial_names()]
    for folder in folders:
        if not c.verified(folder, ctx):
            raise ValueError(f'Incomplete study artifact: {folder}')
    entries = sum(len(c.read(f / 'completed.json')['files']) for f in folders)
    for name, digest in manifest['code_sha256'].items():
        if c.sha(c.RUN / 'source' / name) != digest:
            raise ValueError('Source snapshot mismatch')
    for part in ('saved_scores', 'calibrated'):
        for path in (c.REPORTS / part).iterdir():
            if c.sha(path) != c.sha(c.RUN / part / path.name):
                raise ValueError('Report export mismatch')
    result = dict(status='complete', protected_files_verified=count, new_artifact_entries_verified=entries,
        source_snapshots_verified=len(manifest['code_sha256']), fits=14, mismatches=0,
        official_test_processed=False, completed_at_utc=c.now())
    c.write(c.RUN / 'audit.json', result)
    c.write(c.REPORTS / 'audit.json', result)
    # Inventory excludes large raw caches only insofar as their hashes are already in manifest.
    c.complete(c.RUN / 'completion', ctx, audit_sha256=c.sha(c.RUN / 'audit.json'), fits=14)
    print(json.dumps(result), flush=True)
