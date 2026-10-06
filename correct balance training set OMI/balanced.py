"""Build an explicitly oversampled copy; reuse the frozen numerical trainer."""
import copy
import time
import common as c
import numpy as np
from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as original


def resample(data, indices):
    c.validate_inputs(data)
    result = dict(data)
    for key in ('X_fit', 'y_fit', 'fit_record_ids', 'fit_patients'):
        result[key] = data[key][indices]
    return result


def prepare():
    spec, ctx, source, baseline, baseline_sha = c.context(create=True)
    out = c.RUN / 'balanced'
    out.mkdir(exist_ok=True)
    manifest = copy.deepcopy(baseline)
    manifest['protocol']['id'] = spec['id']
    manifest['protocol']['vqc']['class_weight'] = None
    manifest['protocol']['controls']['logistic']['class_weight'] = None
    manifest.update(balance_context_sha=ctx, baseline_manifest_sha256=baseline_sha,
                    evaluation='Exploratory oversampling follow-up; unchanged natural validation; official test reserved')
    if (out / 'manifest.json').exists():
        if c.read(out / 'manifest.json') != manifest:
            raise ValueError('Changed balanced manifest')
    else:
        c.write(out / 'manifest.json', manifest)
        c.write_bytes(out / 'manifest.sha256', (c.sha(out / 'manifest.json')+'\n').encode())
    _, run_sha = original.read_run(out)
    reference = None
    inventory = {}
    for arm, name in original.TRIALS:
        folder = source / arm / name
        marker = c.read(folder / 'completed.json')
        original.checked_files(folder, marker, baseline_sha)
        for filename, digest in marker['files'].items():
            inventory[str((folder / filename).relative_to(c.ROOT))] = digest
        inventory[str((folder / 'completed.json').relative_to(c.ROOT))] = c.sha(folder / 'completed.json')
    for arm in ('vmd', 'wst'):
        folder = source / arm
        original.checked_files(folder, c.read(folder / 'preprocessing.json'), baseline_sha)
        data = c.arrays(folder / 'inputs.npz')
        c.validate_inputs(data)
        if reference is None:
            reference = data
            idx = c.oversample_indices(data['y_fit'], spec['sampling']['seed'])
            if (np.bincount(data['y_fit']).tolist() != spec['sampling']['original_counts'] or
                    np.bincount(data['y_fit'][idx]).tolist() != spec['sampling']['balanced_counts'] or
                    len(data['y_validation']) != spec['sampling']['validation_records']):
                raise ValueError('Unexpected original/balanced counts')
            sample = c.RUN / 'sampling'
            if not c.verified(sample, ctx):
                c.npz(sample / 'indices.npz', indices=idx, original_y=data['y_fit'],
                      original_record_ids=data['fit_record_ids'], original_patients=data['fit_patients'])
                c.write_bytes(sample / 'ledger.csv', c.csv_bytes(
                    ['sample_index', 'original_fit_index', 'record_id', 'patient_id', 'label'],
                    ((i, j, data['fit_record_ids'][j], data['fit_patients'][j], data['y_fit'][j])
                     for i, j in enumerate(idx))))
                c.write(sample / 'audit.json', dict(original_counts=np.bincount(data['y_fit']).tolist(),
                    balanced_counts=np.bincount(data['y_fit'][idx]).tolist(),
                    unique_fit_records=len(np.unique(idx)), all_originals_retained=True,
                    unique_fit_patients=len(np.unique(data['fit_patients'])),
                    validation_records=len(data['y_validation']), validation_positive=int(data['y_validation'].sum()),
                    validation_patients=len(np.unique(data['validation_patients'])),
                    patient_overlap=0, seed=spec['sampling']['seed'],
                    natural_fit_prevalence=float(data['y_fit'].mean()), official_test_processed=False))
                c.complete(sample, ['indices.npz', 'ledger.csv', 'audit.json'], ctx)
            saved = c.arrays(sample / 'indices.npz')
            np.testing.assert_array_equal(saved['indices'], idx)
            for key, src in [('original_y', 'y_fit'), ('original_record_ids', 'fit_record_ids'),
                             ('original_patients', 'fit_patients')]:
                np.testing.assert_array_equal(saved[key], data[src])
        else:
            for key in ('y_fit', 'y_validation', 'fit_record_ids', 'validation_record_ids',
                        'fit_patients', 'validation_patients'):
                np.testing.assert_array_equal(data[key], reference[key])
        for filename in ('inputs.npz', 'preprocessor.joblib', 'preprocessing.json'):
            inventory[str((folder / filename).relative_to(c.ROOT))] = c.sha(folder / filename)
        target = out / arm
        target.mkdir(exist_ok=True)
        started = time.perf_counter()
        expected = resample(data, idx)
        if (target / 'preprocessing.json').exists():
            original.checked_files(target, c.read(target / 'preprocessing.json'), run_sha)
        else:
            c.write_bytes(target / 'preprocessor.joblib', (folder / 'preprocessor.joblib').read_bytes())
            c.npz(target / 'inputs.npz', **expected)
            c.write(target / 'preprocessing.json', dict(status='complete', context_sha=run_sha,
                files={n: c.sha(target / n) for n in ('inputs.npz', 'preprocessor.joblib')},
                seconds=time.perf_counter()-started, warnings=[], fit_records=len(idx),
                validation_records=len(data['y_validation']), selected_names=data['selected_names'].tolist(),
                preprocessing_reused_from=str(folder.relative_to(c.ROOT))))
        for key, value in c.arrays(target / 'inputs.npz').items():
            np.testing.assert_array_equal(value, expected[key])
        if c.sha(target / 'preprocessor.joblib') != c.sha(folder / 'preprocessor.joblib'):
            raise ValueError('Preprocessor differs from frozen original')
    inventory[str((source / 'manifest.json').relative_to(c.ROOT))] = baseline_sha
    if (c.RUN / 'baseline_inventory.json').exists():
        if c.read(c.RUN / 'baseline_inventory.json') != inventory:
            raise ValueError('Protected baseline inventory changed')
    else:
        c.write(c.RUN / 'baseline_inventory.json', inventory)
    c.write(out / 'preparation.json', dict(status='complete', context_sha=run_sha, completed_at_utc=c.now()))
    print(f'Balanced inputs verified: {len(idx)//2:,} examples/class; every original retained; validation unchanged', flush=True)


def train(jobs=3):
    _, ctx, _, _, baseline_sha = c.context()
    manifest, _ = original.read_run(c.RUN / 'balanced')
    if manifest.get('balance_context_sha') != ctx or manifest.get('baseline_manifest_sha256') != baseline_sha:
        raise ValueError('Balanced training manifest belongs to another experiment')
    if not c.verified(c.RUN / 'sampling', ctx):
        raise ValueError('Missing sampling provenance')
    original.train(c.RUN / 'balanced', jobs)
