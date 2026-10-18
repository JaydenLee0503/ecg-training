"""ECGData-only adapter and resumable CPU cache; never reads or writes ACS data."""
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .cepstral import CepstralConfig, cepstral_features

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / 'ecgdata_protocol.json'
DEFAULT_CACHE = ROOT / 'features/ecgdata_lfcc_v1'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def safe_output(path, category):
    path = Path(path).resolve()
    base = (ROOT / category).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f'Output must be a child of {base}; original experiments are read-only')
    return path


@contextmanager
def writer_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another writer owns this run') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def feature_sources():
    paths = [PROTOCOL, Path(__file__), ROOT/'attention_ecg/cepstral.py',
             REPO/'ecgvmd/data.py', REPO/'ecgvmd/segment.py', REPO/'ecgvmd/config.py',
             REPO/'ecgvmd/ecgdata_subjects.csv', REPO/'ecgvmd/ecgdata_subjects.json']
    return {str(p.relative_to(REPO)): file_hash(p) for p in paths}


def validate_fold_arrays(y, patients, folds):
    if not (len(y) == len(patients) == len(folds)) or set(folds) != set(range(5)):
        raise ValueError('Expected aligned five-fold assignments')
    for patient in np.unique(patients):
        idx = patients == patient
        if len(np.unique(folds[idx])) != 1 or len(np.unique(y[idx])) != 1:
            raise ValueError('Patient crosses folds or has inconsistent diagnosis')
    for fold in range(5):
        if set(y[folds == fold]) != {0, 1, 2} or set(y[folds != fold]) != {0, 1, 2}:
            raise ValueError('All classes required in each training and test fold')


def matched_windows(mat, protocol):
    from ecgvmd.data import load_ecgdata
    from ecgvmd.segment import segment
    from ecgvmd.config import Config

    ref = REPO / protocol['reference']['directory']
    manifest = read_json(ref/'manifest.json')
    cfg = Config(**manifest['config'])
    expected = protocol['input']
    if (cfg.fs != expected['fs'] or cfg.seg_len != expected['seg_len'] or
            cfg.n_per_record != expected['n_per_record'] or cfg.seed != expected['seed']
            or cfg.seg_mode != 'fixed' or cfg.seg_stride != 1 or not expected['zscore_each_window']):
        raise ValueError('Baseline segmentation differs from protocol')
    dataset = load_ecgdata(str(mat))
    windows, labels, row_ids = segment(dataset, cfg, grouping='row')
    patients = dataset.patient_ids[row_ids].astype(str)
    sources = dataset.source_ids[row_ids].astype(str)
    actual_window_hash = array_hash(windows)
    if actual_window_hash != protocol['reference']['windows_sha256'] or actual_window_hash != manifest['windows_sha256']:
        raise ValueError('Windows do not exactly match the corrected baseline')
    if file_hash(REPO/'ecgvmd/ecgdata_subjects.csv') != protocol['reference']['subject_map_sha256']:
        raise ValueError('Patient mapping changed')
    with np.load(ref/'features.npz', allow_pickle=False) as baseline:
        for key, actual in [('y', labels), ('patients', patients), ('sources', sources), ('row_ids', row_ids)]:
            if not np.array_equal(actual, baseline[key]):
                raise ValueError(f'Baseline {key} alignment differs')
        folds = baseline['fold'].copy()
    if array_hash(folds) != protocol['reference']['fold_sha256'] or array_hash(folds) != manifest['fold_sha256']:
        raise ValueError('Frozen fold assignments changed')
    with (ref/'folds.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(labels):
        raise ValueError('Fold CSV count differs')
    for i, row in enumerate(rows):
        if (int(row['row']) != row_ids[i] or row['patient'] != patients[i] or row['source'] != sources[i]
                or row['label'] != labels[i] or int(row['fold']) != folds[i]):
            raise ValueError('Fold CSV differs from baseline arrays')
    y = np.array([protocol['classes'].index(str(label)) for label in labels], dtype=np.int64)
    validate_fold_arrays(y, patients, folds)
    # Recover exact within-row window positions from the same sampling procedure.
    rng = np.random.default_rng(cfg.seed)
    starts = np.concatenate([np.sort(rng.choice(dataset.n_samples // cfg.seg_len,
                                cfg.n_per_record, replace=False)) * cfg.seg_len
                             for _ in range(dataset.n_records)])
    return windows, dict(y=y, labels=labels.astype(str), patients=patients, sources=sources,
                         row_ids=row_ids, fold=folds, window_start=starts,
                         window_id=np.arange(len(y), dtype=np.int64))


def verify_cache(folder=DEFAULT_CACHE, check_sources=True):
    folder = safe_output(folder, 'features')
    manifest = read_json(folder/'manifest.json')
    completed = read_json(folder/'completed.json')
    if completed['status'] != 'complete' or completed['manifest_sha256'] != file_hash(folder/'manifest.json'):
        raise ValueError('Incomplete or altered cache manifest')
    for name, digest in completed['files'].items():
        path = folder/name
        if folder not in path.resolve().parents or file_hash(path) != digest:
            raise ValueError(f'Cache artifact changed: {name}')
    if check_sources and manifest['sources'] != feature_sources():
        raise ValueError('Feature source changed: use a new protocol/cache')
    with np.load(folder/'features.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if not np.isfinite(data['X']).all() or data['X'].shape != tuple(completed['shape']):
        raise ValueError('Invalid feature matrix')
    validate_fold_arrays(data['y'], data['patients'], data['fold'])
    return manifest, data


def prepare_cache(mat=REPO/'ECGData.mat', folder=DEFAULT_CACHE):
    folder = safe_output(folder, 'features')
    protocol = read_json(PROTOCOL)
    mat = Path(mat).resolve()
    start = time.perf_counter()
    with writer_lock(folder):
        if (folder/'completed.json').exists():
            manifest, data = verify_cache(folder)
            if manifest['mat_sha256'] != file_hash(mat):
                raise ValueError('ECGData.mat differs from completed cache')
            return dict(status='reused', shape=list(data['X'].shape), training_started=False)
        windows, arrays = matched_windows(mat, protocol)
        ref = REPO/protocol['reference']['directory']
        manifest = dict(protocol=protocol, sources=feature_sources(), mat_sha256=file_hash(mat),
                        reference_files={name:file_hash(ref/name) for name in ('manifest.json','folds.csv','features.npz')},
                        windows_sha256=array_hash(windows), versions=dict(python=platform.python_version(),
                        numpy=np.__version__, scipy=scipy.__version__))
        if (folder/'manifest.json').exists():
            if read_json(folder/'manifest.json') != manifest:
                raise ValueError('Interrupted cache inputs changed: use a new directory')
        else:
            atomic_json(folder/'manifest.json', manifest)
        shard_dir = folder/'shards'
        shard_dir.mkdir(exist_ok=True)
        config = CepstralConfig(**protocol['cepstral'])
        all_features, file_inventory, reused = [], {}, 0
        for row in np.unique(arrays['row_ids']):
            indices = np.flatnonzero(arrays['row_ids'] == row)
            artifact = shard_dir/f'row_{row:03d}.npz'
            marker = artifact.with_suffix('.json')
            if marker.exists():
                saved = read_json(marker)
                if saved['sha256'] != file_hash(artifact) or saved['windows_sha256'] != array_hash(windows[indices]):
                    raise ValueError(f'Corrupt or mismatched shard {row}')
                with np.load(artifact, allow_pickle=False) as z:
                    block = z['X']
                    if not np.array_equal(z['indices'], indices):
                        raise ValueError('Shard window order changed')
                reused += 1
            else:
                began = time.perf_counter()
                extracted = [cepstral_features(windows[i, :, None], config) for i in indices]
                block = np.stack([item[0] for item in extracted])
                atomic_npz(artifact, X=block, indices=indices)
                atomic_json(marker, dict(sha256=file_hash(artifact), windows_sha256=array_hash(windows[indices]),
                                         seconds=time.perf_counter()-began, metadata=extracted[0][1]))
            if block.shape != (len(indices), 13, 1, 16) or not np.isfinite(block).all():
                raise ValueError('Unexpected or nonfinite shard')
            all_features.append(block)
            for path in (artifact, marker):
                file_inventory[str(path.relative_to(folder))] = file_hash(path)
        X = np.concatenate(all_features)
        atomic_npz(folder/'features.npz', X=X, **arrays)
        file_inventory['features.npz'] = file_hash(folder/'features.npz')
        summary = dict(status='complete', manifest_sha256=file_hash(folder/'manifest.json'), files=file_inventory,
                       shape=list(X.shape), patients=len(np.unique(arrays['patients'])),
                       rows=len(np.unique(arrays['row_ids'])), windows_sha256=array_hash(windows),
                       fold_sha256=array_hash(arrays['fold']), reused_shards=reused,
                       seconds=time.perf_counter()-start, training_started=False)
        atomic_json(folder/'completed.json', summary)
        verify_cache(folder)
        return {k:v for k,v in summary.items() if k != 'files'}


def fit_normalizer(X, indices):
    """Fit per lead/cepstrum over supplied training windows and time only."""
    fit = np.asarray(X[indices], dtype=np.float64)
    mean = fit.mean(axis=(0,1))
    scale = fit.std(axis=(0,1))
    scale[scale < 1e-8] = 1
    return mean, scale


def normalize(X, mean, scale):
    out = ((X - mean) / scale).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite normalized features')
    return out
