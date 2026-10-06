"""Paired patient-level uncertainty on the unchanged natural validation cohort."""
import common as c
import numpy as np
from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as original
from experiments.acs_omi_vmd_wst_vqc.reporting import bootstrap, METRICS


def collect():
    spec, ctx, source, _, _ = c.context()
    scores, groups, pairs = {}, {}, {}
    reference = c.arrays(source / 'vmd/logistic/predictions.npz')

    def add(key, path):
        saved = c.arrays(path)
        for field in ('record_ids', 'patients', 'y'):
            np.testing.assert_array_equal(saved[field], reference[field])
        np.testing.assert_array_equal(saved['prediction'], (saved['score'] >= .5).astype(int))
        scores[key] = saved['score']

    for prefix, folder in [('original', source), ('balanced', c.RUN / 'balanced')]:
        _, digest = original.read_run(folder)
        marker = c.read(folder / 'completed.json')
        if marker['status'] != 'complete' or marker['context_sha'] != digest or marker['fits'] != 10:
            raise ValueError('All ten VMD/WST trials must finish')
        for arm, name in original.TRIALS:
            trial = folder / arm / name
            original.checked_files(trial, c.read(trial / 'completed.json'), digest)
            key = f'{prefix}/{arm}/{name}'
            add(key, trial / 'predictions.npz')
            if not name.startswith('vqc'):
                groups[key] = [key]
        for arm in ('vmd', 'wst'):
            groups[f'{prefix}/{arm}/vqc'] = [f'{prefix}/{arm}/vqc_seed{s}' for s in (0, 1, 2)]
    folder = c.RUN / 'lfcc_models'
    if not c.verified(folder, ctx):
        raise ValueError('LFCC models are incomplete')
    digest = c.sha(folder / 'manifest.json')
    for name in ('logistic', 'swin_seed0', 'swin_seed1', 'swin_seed2'):
        if not c.verified(folder / name, digest):
            raise ValueError('Incomplete LFCC trial')
        add(f'balanced/lfcc/{name}', folder / name / 'predictions.npz')
    groups['balanced/lfcc/logistic'] = ['balanced/lfcc/logistic']
    groups['balanced/lfcc/swin'] = [f'balanced/lfcc/swin_seed{s}' for s in (0, 1, 2)]
    data = c.arrays(source / 'vmd/inputs.npz')
    for key, field in [('validation_record_ids', 'record_ids'), ('validation_patients', 'patients'),
                       ('y_validation', 'y')]:
        np.testing.assert_array_equal(data[key], reference[field])
    # The reference prior is from the original unique fit records, never the duplicated 50:50 sample.
    prevalence = float(data['y_fit'].mean())
    scores['constant_natural_prior'] = np.full(len(reference['y']), prevalence)
    groups['constant_natural_prior'] = ['constant_natural_prior']
    for arm in ('vmd', 'wst'):
        for name in ('vqc', 'logistic', 'weighted_knn'):
            pairs[f'{arm}/{name}: balanced minus original'] = (f'balanced/{arm}/{name}', f'original/{arm}/{name}')
        for name in ('logistic', 'weighted_knn'):
            pairs[f'balanced/{arm}: vqc minus {name}'] = (f'balanced/{arm}/vqc', f'balanced/{arm}/{name}')
        pairs[f'swin minus balanced/{arm}/vqc'] = ('balanced/lfcc/swin', f'balanced/{arm}/vqc')
    for name in ('vqc', 'logistic', 'weighted_knn'):
        pairs[f'balanced: vmd minus wst/{name}'] = (f'balanced/vmd/{name}', f'balanced/wst/{name}')
    pairs['lfcc: swin minus logistic'] = ('balanced/lfcc/swin', 'balanced/lfcc/logistic')
    pairs['swin minus constant prior'] = ('balanced/lfcc/swin', 'constant_natural_prior')
    return spec, ctx, reference, scores, groups, pairs, prevalence


def report():
    spec, ctx, reference, scores, groups, pairs, prevalence = collect()
    out = c.RUN / 'report'
    if c.verified(out, ctx):
        export(out)
        print('Verified completed comparison report', flush=True)
        return
    cfg = spec['evaluation']
    result = bootstrap(reference['y'], reference['patients'], scores, groups, pairs,
                       cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    result.update(context_sha=ctx, natural_fit_prevalence=prevalence,
                  validation_records=len(reference['y']), validation_positive=int(reference['y'].sum()),
                  validation_patients=len(np.unique(reference['patients'])), threshold=.5,
                  official_test_processed=False, completed_at_utc=c.now())
    c.write(out / 'analysis.json', result)
    fields = list(next(iter(result['per_model'].values())))
    c.write_bytes(out / 'metrics.csv', c.csv_bytes(['model'] + fields,
        ([key] + [values[f] for f in fields] for key, values in result['per_model'].items())))
    c.write_bytes(out / 'predictions.csv', c.csv_bytes(['record_id', 'patient_id', 'label'] + list(scores),
        ([reference['record_ids'][i], reference['patients'][i], reference['y'][i]] +
         [float(s[i]) for s in scores.values()] for i in range(len(reference['y'])))))
    lines = ['# Balanced OMI training: complete internal-validation comparison', '',
             'Validation retains its natural prevalence and the original patient split. Official test ECGs remain reserved.',
             'VQC/Swin rows average all seed 0/1/2 metrics, not probabilities. AP is primary.', '',
             '| Model | AP (95% interval) | ROC AUC | Balanced accuracy | Sensitivity | Specificity | Accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in result['groups'].items():
        lo, hi = result['group_intervals'][name]['average_precision']
        lines.append(f'| {name} | {m["average_precision"]:.4f} [{lo:.4f}, {hi:.4f}] | '
            f'{m["roc_auc"]:.4f} | {m["balanced_accuracy"]:.4f} | {m["sensitivity"]:.4f} | '
            f'{m["specificity"]:.4f} | {m["accuracy"]:.4f} |')
    lines += ['', '## Paired average-precision differences', '']
    for name, values in result['paired_differences'].items():
        lo, hi = values['intervals']['average_precision']
        lines.append(f'- {name}: {values["point"]["average_precision"]:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].')
    lines += ['', '## Limits', '',
        '- Exploratory follow-up on a previously examined validation set; no threshold tuning or early stopping.',
        '- Repetition creates no independent patients. The fixed sampling seed is not a sampling-uncertainty analysis.',
        '- Original VQC/logistic used balanced class weights. New fits use duplicated data and no class weights.',
        '- VQC updates increase from 17,920 to 33,520; duplicated logistic rows also change effective regularization at C=1.',
        '- LFCC + Swin changes the representation and classifier together; this is a complete-pipeline comparison.',
        '- Patient-bootstrap intervals condition on saved predictions and exclude retraining uncertainty.',
        '- No external validation, clinical validity, transform superiority or quantum advantage is established by this design.', '']
    c.write_bytes(out / 'report.md', '\n'.join(lines).encode())
    checked = 0
    for name, digest in c.read(c.RUN / 'baseline_inventory.json').items():
        if c.sha(c.ROOT / name) != digest:
            raise ValueError(f'Protected baseline changed: {name}')
        checked += 1
    c.write(out / 'integrity.json', dict(protected_baseline_files_verified=checked, mismatches=0,
                                        completed_at_utc=c.now()))
    c.complete(out, ['analysis.json', 'metrics.csv', 'predictions.csv', 'report.md', 'integrity.json'], ctx)
    export(out)
    print('\n'.join(lines), flush=True)


def export(out):
    for name in ('analysis.json', 'metrics.csv', 'report.md', 'integrity.json'):
        c.write_bytes(c.BASE / 'reports/v1' / name, (out / name).read_bytes())
