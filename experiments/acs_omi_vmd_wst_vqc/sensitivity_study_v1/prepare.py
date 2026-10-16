"""Pin inputs, retain all patients, and refit preprocessing below calibration."""
import importlib.metadata
import json
import platform
import time
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import common as c
from experiments.acs_omi_vmd_wst_vqc.pipeline import fit_preprocessor
from experiments.acs_omi_vmd_wst_vqc.training import atomic_joblib


def partition(y, patients, original_parts, spec):
    y, patients, original_parts = map(np.asarray, (y, patients, original_parts))
    if set(original_parts) != {'fit', 'validation'}:
        raise ValueError('Official test or unknown partition')
    fit = np.flatnonzero(original_parts == 'fit')
    cfg = spec['calibration_split']
    folds = list(StratifiedGroupKFold(cfg['n_splits'], shuffle=True, random_state=cfg['seed']).split(
                 np.zeros((len(fit), 1)), y[fit], patients[fit]))
    train, cal = (fit[idx] for idx in folds[cfg['fold']])
    parts = np.full(len(y), 'validation', dtype='<U11')
    parts[train], parts[cal] = 'subfit', 'calibration'
    for a in ('subfit', 'calibration', 'validation'):
        if set(y[parts == a]) != {0, 1}:
            raise ValueError(f'Missing class in {a}')
        for b in ('subfit', 'calibration', 'validation'):
            if a != b and set(patients[parts == a]) & set(patients[parts == b]):
                raise ValueError('Patient overlap')
    return parts


def oversample(y, seed):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes required for oversampling')
    counts = np.bincount(y, minlength=2)
    rng = np.random.default_rng(seed)
    return np.concatenate([np.arange(len(y))] +
        [rng.choice(np.flatnonzero(y == k), counts.max()-counts[k], replace=True) for k in (0, 1)])


def protected_inventory(paths):
    """Verify every artifact referenced by previous completed model/report markers."""
    inventory = {}
    roots = [paths['original'], paths['balanced'] / 'balanced', paths['balanced'] / 'lfcc_models',
             paths['balanced'] / 'report']
    for root in roots:
        for marker in sorted(root.rglob('completed.json')):
            value = c.read(marker)
            if value['status'] != 'complete':
                raise ValueError(f'Incomplete source: {marker}')
            inventory[str(marker.relative_to(c.ROOT))] = c.sha(marker)
            for name, digest in value.get('files', {}).items():
                path = marker.parent / name
                if c.sha(path) != digest:
                    raise ValueError(f'Changed protected artifact: {path}')
                inventory[str(path.relative_to(c.ROOT))] = digest
    return inventory


def prepare():
    started = time.perf_counter()
    spec, paths = c.source_paths()
    if (c.RUN / 'manifest.json').exists():
        _, ctx = c.manifest()
        if c.verified(c.RUN / 'prepared', ctx):
            print('Verified completed preparation; no recomputation', flush=True)
            return
    print('Verifying protected models, predictions and feature caches', flush=True)
    inventory = protected_inventory(paths)
    original = c.read(paths['original'] / 'manifest.json')
    balanced = c.read(c.ROOT / 'correct balance training set OMI/protocol.json')
    expected = {paths['features']: original['features_sha256'],
                paths['split']: spec['sources']['split_sha256'],
                paths['balanced'] / 'lfcc/features.npz': c.read(paths['balanced'] / 'lfcc/completed.json')['files']['features.npz']}
    for path, digest in expected.items():
        if c.sha(path) != digest:
            raise ValueError(f'Changed feature/split source: {path}')
        inventory[str(path.relative_to(c.ROOT))] = digest
    for path in (paths['original'] / 'manifest.json', paths['balanced'] / 'manifest.json',
                 c.ROOT / 'correct balance training set OMI/protocol.json'):
        inventory[str(path.relative_to(c.ROOT))] = c.sha(path)
    code = set(original['code_sha256'])
    code.update(str(p.relative_to(c.ROOT)) for p in c.BASE.glob('*.py'))
    code.update(str(p.relative_to(c.ROOT)) for p in (c.BASE / 'tests').glob('*.py'))
    code.add(str((c.BASE / 'protocol.json').relative_to(c.ROOT)))
    code.update('correct balance training set OMI/' + name for name in ('common.py', 'swin.py'))
    code.update('attention method/attention_ecg/' + name for name in ('model.py', 'data.py', 'cepstral.py'))
    value = dict(protocol=spec, original_settings=original['protocol'], balanced_settings=balanced,
        code_sha256={name: c.sha(c.ROOT / name) for name in sorted(code)}, protected_sha256=inventory,
        environment=dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'pennylane', 'joblib', 'matplotlib')}))
    path = c.RUN / 'manifest.json'
    if path.exists() and c.read(path) != value:
        raise ValueError('Changed study manifest; use a new study folder')
    c.write(path, value)
    ctx = c.sha(path)
    for name in code:
        c.write_bytes(c.RUN / 'source' / name, (c.ROOT / name).read_bytes())
    out = c.RUN / 'prepared'
    out.mkdir(parents=True, exist_ok=True)
    data = c.arrays(paths['features'])
    import csv
    with paths['split'].open(newline='') as stream:
        split = {r['record_id']: r for r in csv.DictReader(stream)}
    if len(set(data['record_ids'])) != len(data['record_ids']) or set(data['record_ids']) != set(split):
        raise ValueError('Feature cohort differs from frozen eligible split')
    for i, rid in enumerate(data['record_ids']):
        row = split[rid]
        if (row['patient_id'] != data['patients'][i] or int(row['label']) != data['y'][i]
                or row['partition'] != data['partitions'][i] or row['split'] != 'train'):
            raise ValueError('Feature metadata differs from split')
    parts = partition(data['y'], data['patients'], data['partitions'], spec)
    tr = np.flatnonzero(parts == 'subfit')
    sampled = tr[oversample(data['y'][tr], spec['sampling']['seed'])]
    meta = dict(y=data['y'], record_ids=data['record_ids'], patients=data['patients'],
                original_partitions=data['partitions'], partitions=parts, sampled_indices=sampled)
    c.npz(out / 'cohort.npz', **meta)
    c.csv_write(out / 'split.csv', ['record_id', 'patient_id', 'label', 'original_partition', 'partition'],
                zip(meta['record_ids'], meta['patients'], meta['y'], meta['original_partitions'], parts))
    c.csv_write(out / 'sampling.csv', ['sample_index', 'cohort_index', 'record_id', 'patient_id', 'label'],
                ((i, j, meta['record_ids'][j], meta['patients'][j], meta['y'][j]) for i, j in enumerate(sampled)))
    audit = dict(partitions={part: dict(records=int(np.sum(parts == part)),
                patients=len(np.unique(meta['patients'][parts == part])),
                counts=np.bincount(meta['y'][parts == part], minlength=2).tolist())
                for part in ('subfit', 'calibration', 'validation')},
                balanced_counts=np.bincount(meta['y'][sampled]).tolist(),
                patient_overlap=0, exclusions_added=0, official_test_processed=False)
    caught_all = []
    for arm in ('vmd', 'wst'):
        print(f'Refitting {arm} feature selection and scaling on unique subfit records', flush=True)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always')
            prep, angles, _, _ = fit_preprocessor(data[arm], meta['y'], meta['patients'],
                np.where(parts == 'subfit', 'fit', 'validation'), original['protocol'])
        np.testing.assert_allclose(prep[1].scaler_.mean_,
            data[arm][tr][:, prep[0].idx_].astype(float).mean(0), rtol=0, atol=1e-12)
        atomic_joblib(out / f'{arm}_preprocessor.joblib', prep)
        c.npz(out / f'{arm}.npz', X=angles, selected_indices=prep[0].idx_,
              selected_names=data[arm + '_names'][prep[0].idx_])
        caught_all.extend(str(w.message) for w in caught)
    del data
    print('Refitting LFCC scaling on unique subfit records', flush=True)
    lfcc = c.arrays(paths['balanced'] / 'lfcc/features.npz')
    lookup = {rid: i for i, rid in enumerate(lfcc['record_ids'])}
    order = np.array([lookup[rid] for rid in meta['record_ids']])
    if len(lookup) != len(meta['y']) or set(lookup) != set(meta['record_ids']):
        raise ValueError('LFCC cohort mismatch')
    for name, expected_name in [('y', 'y'), ('patients', 'patients'), ('partitions', 'original_partitions')]:
        np.testing.assert_array_equal(lfcc[name][order], meta[expected_name])
    legacy = c.legacy_swin()
    from attention_ecg.data import FitStandardizer
    rows = {rid: dict(partition='fit' if part == 'subfit' else 'validation')
            for rid, part in zip(meta['record_ids'], parts)}
    normalizer = FitStandardizer().fit(((str(meta['record_ids'][i]), lfcc['X'][order[i]]) for i in tr), rows)
    c.write(out / 'lfcc_normalization.json', normalizer.state_dict())
    normalized = np.lib.format.open_memmap(out / 'lfcc_normalized.npy', mode='w+', dtype='float32',
                                          shape=lfcc['X'].shape)
    pooled = []
    for first in range(0, len(order), 256):
        raw = lfcc['X'][order[first:first+256]]
        x = raw.astype(np.float64)
        normalized[first:first+len(x)] = ((x-normalizer.mean_) / normalizer.scale_).astype(np.float32)
        pooled.append(np.concatenate([raw.mean(axis=1).reshape(len(x), -1),
                                      raw.std(axis=1).reshape(len(x), -1)], axis=1))
    normalized.flush()
    pooled = np.concatenate(pooled)
    scaler = StandardScaler().fit(pooled[tr])
    c.npz(out / 'lfcc.npz', X=scaler.transform(pooled))
    atomic_joblib(out / 'lfcc_preprocessor.joblib', scaler)
    audit.update(warnings=caught_all, seconds=time.perf_counter()-started, completed_at_utc=c.now(),
                 protected_files=len(inventory), learned_preprocessing_fit_on='unique subfit only')
    c.write(out / 'audit.json', audit)
    c.complete(out, ctx)
    c.write(c.REPORTS / 'preparation.json', audit)
    print(json.dumps(audit), flush=True)
