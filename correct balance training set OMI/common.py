"""Isolated paths, provenance, atomic artifacts and deterministic oversampling."""
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
ATTENTION = ROOT / 'attention method'
for path in (ROOT, ATTENTION, ATTENTION / '.venv/torch-cu128', Path('/tmp/acs-gpu-deps-v1')):
    sys.path.insert(0, str(path))
import numpy as np

RUN = BASE / 'results/v1'
SPEC = BASE / 'protocol.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def write(path, obj):
    write_bytes(path, (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())


def npz(path, **data):
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    write_bytes(path, stream.getvalue())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


@contextlib.contextmanager
def lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.writer.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def complete(folder, files, context, **extra):
    write(folder / 'completed.json', dict(status='complete', context_sha=context,
          files={str(name): sha(folder / name) for name in files}, **extra))


def verified(folder, context):
    path = folder / 'completed.json'
    if not path.exists():
        return False
    marker = read(path)
    if marker['status'] != 'complete' or marker['context_sha'] != context:
        raise ValueError(f'Changed completion context: {folder}')
    for name, digest in marker['files'].items():
        if sha(folder / name) != digest:
            raise ValueError(f'Changed completed artifact: {folder / name}')
    return True


def oversample_indices(y, seed):
    y = np.asarray(y)
    if y.ndim != 1 or set(y.tolist()) != {0, 1}:
        raise ValueError('Expected a one-dimensional binary training label array with both classes')
    counts = np.bincount(y.astype(int), minlength=2)
    rng = np.random.default_rng(seed)
    parts = [np.arange(len(y))]
    for label in (0, 1):
        parts.append(rng.choice(np.flatnonzero(y == label), int(counts.max()-counts[label]), replace=True))
    return np.concatenate(parts).astype(np.int64)


def validate_inputs(data):
    for part in ('fit', 'validation'):
        n = len(data['y_' + part])
        ids, patients = data[part + '_record_ids'], data[part + '_patients']
        if len(ids) != n or len(patients) != n or len(np.unique(ids)) != n:
            raise ValueError('Invalid original record identities')
        if set(data['y_' + part].tolist()) != {0, 1}:
            raise ValueError('Missing class')
        if len(data['X_' + part]) != n or not np.isfinite(data['X_' + part]).all():
            raise ValueError('Invalid original feature matrix')
    if (set(data['fit_patients']) & set(data['validation_patients']) or
            set(data['fit_record_ids']) & set(data['validation_record_ids'])):
        raise ValueError('Fit/validation leakage')


def code_hashes():
    from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as original
    files = list(BASE.glob('*.py')) + list((BASE / 'tests').glob('*.py')) + [SPEC]
    files += [ATTENTION / 'attention_ecg' / n for n in ('cepstral.py', 'data.py', 'model.py')]
    files += [ROOT / 'experiments/acs_omi_vmd_wst_vqc/loader.py']
    return dict(sorted({**original.code_hashes(), **{str(p.relative_to(ROOT)): sha(p) for p in files}}.items()))


def context(create=False):
    from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as original
    spec = read(SPEC)
    source = ROOT / spec['baseline']
    m, digest = original.read_run(source)
    if digest != spec['baseline_manifest_sha256'] or sha(ROOT / spec['split']) != spec['split_sha256']:
        raise ValueError('Changed frozen baseline/split')
    done = read(source / 'completed.json')
    if done['status'] != 'complete' or done['context_sha'] != digest or done['fits'] != 10:
        raise ValueError('Baseline is incomplete')
    candidate = dict(protocol=spec, code_sha256=code_hashes(), environment=original.old.environment(),
                     baseline_manifest_sha256=digest)
    path = RUN / 'manifest.json'
    if path.exists():
        if read(path) != candidate or (RUN / 'manifest.sha256').read_text().strip() != sha(path):
            raise ValueError('Experiment context changed; preserve this run and use a new version')
    elif create:
        RUN.mkdir(parents=True, exist_ok=True)
        write(path, candidate)
        write_bytes(RUN / 'manifest.sha256', (sha(path)+'\n').encode())
        for name in candidate['code_sha256']:
            write_bytes(RUN / 'source' / name, (ROOT / name).read_bytes())
    else:
        raise ValueError('Prepare the experiment first')
    return spec, sha(path), source, m, digest
