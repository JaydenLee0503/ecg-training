"""Separate rotational-morphology provenance and resumable artifact operations."""
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DEPS = BASE / 'results/eyeball_dependencies_v1'
sys.path.insert(0, str(DEPS))
import numpy as np

SPEC = BASE / 'protocols/eyeball_omi_engineering_v1.json'
SOURCE = BASE / 'results/omi_v1_gpu_retry256k'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def put(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write(path, obj):
    put(path, (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())


def npz(path, **values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    put(path, stream.getvalue())


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def csv_write(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    put(path, stream.getvalue().encode())


def event(folder, status, **details):
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(time_utc=now(), status=status, **details)
    with (folder / 'attempts.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(payload, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.writer.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def complete(folder, names, context, **extra):
    write(folder / 'completed.json', dict(status='complete', context_sha=context,
          files={str(n): sha(folder / n) for n in names}, completed_utc=now(), **extra))


def verified(folder, context):
    p = folder / 'completed.json'
    if not p.exists():
        return False
    obj = read(p)
    if obj['status'] != 'complete' or obj['context_sha'] != context:
        raise ValueError(f'Changed completion context: {folder}')
    for name, digest in obj['files'].items():
        if sha(folder / name) != digest:
            raise ValueError(f'Changed artifact: {folder / name}')
    return True


def split_rows():
    manifest = read(SOURCE / 'manifest.json')
    if sha(SOURCE / 'manifest.json') != (SOURCE / 'manifest.sha256').read_text().strip():
        raise ValueError('Changed source manifest')
    if sha(SOURCE / 'splits.csv') != manifest['splits_sha256']:
        raise ValueError('Changed source split')
    with (SOURCE / 'splits.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 17905 or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected cohort or duplicate records')
    fit = [r for r in rows if r['partition'] == 'fit']
    val = [r for r in rows if r['partition'] == 'validation']
    if (len(fit) != 14324 or len(val) != 3581 or
            {r['patient_id'] for r in fit} & {r['patient_id'] for r in val}):
        raise ValueError('Unexpected split or patient overlap')
    return rows


def selection(rows, count, identifier):
    patients = {}
    for row in rows:
        if row['partition'] == 'fit':
            patients.setdefault(row['patient_id'], []).append(row)
    def key(value):
        return hashlib.sha256(f'{identifier}|{value}'.encode()).hexdigest(), value
    return [min(patients[p], key=lambda r: key(r['record_id']))
            for p in sorted(patients, key=key)[:count]]


def source_hashes():
    paths = [ROOT / 'ecgvmd/rotational.py', BASE / 'eyeball_common.py',
             BASE / 'scripts/eyeball_pilot.py', BASE / 'tests/test_rotational.py',
             BASE / 'loader.py', SPEC]
    paths += sorted((DEPS / 'PyEMD').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def environment():
    return dict(python=sys.version, packages={p: importlib.metadata.version(p)
          for p in ('numpy', 'scipy', 'EMD-signal', 'matplotlib', 'scikit-learn')})


def context(folder, create=False):
    rows = split_rows()
    spec = read(SPEC)
    value = dict(protocol=spec, code_sha256=source_hashes(), environment=environment(),
                 source_manifest_sha256=sha(SOURCE / 'manifest.json'),
                 splits_sha256=sha(SOURCE / 'splits.csv'))
    path = folder / 'manifest.json'
    if path.exists():
        if read(path) != value or sha(path) != (folder / 'manifest.sha256').read_text().strip():
            raise ValueError('Engineering context changed; use a new run')
    elif create:
        write(path, value)
        put(folder / 'manifest.sha256', (sha(path) + '\n').encode())
        for name in value['code_sha256']:
            put(folder / 'source' / name, (ROOT / name).read_bytes())
        put(folder / 'splits.csv', (SOURCE / 'splits.csv').read_bytes())
        put(folder / 'source_records.csv', (SOURCE / 'source_records.csv').read_bytes())
    else:
        raise ValueError('Prepare the pilot first')
    return spec, sha(path), rows
