"""Study-local provenance and atomic artifacts; never write to source runs."""
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
RUN = BASE / 'results'
REPORTS = BASE / 'reports'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def write(path, value):
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def npz(path, **values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    write_bytes(path, stream.getvalue())


def csv_write(path, header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    write_bytes(path, stream.getvalue().encode())


def complete(folder, context, **extra):
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name not in ('completed.json', '.writer.lock', 'status.json')
             and not p.name.endswith('.tmp')}
    write(folder / 'completed.json', dict(status='complete', context_sha=context,
          completed_at_utc=now(), files=files, **extra))


def verified(folder, context=None):
    marker = folder / 'completed.json'
    if not marker.exists():
        return False
    value = read(marker)
    if value['status'] != 'complete' or (context is not None and value['context_sha'] != context):
        raise ValueError(f'Incomplete or changed context: {folder}')
    for name, digest in value['files'].items():
        if sha(folder / name) != digest:
            raise ValueError(f'Changed artifact: {folder / name}')
    return True


@contextmanager
def lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.writer.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_paths():
    spec = read(BASE / 'protocol.json')
    return spec, {k: ROOT / v for k, v in spec['sources'].items() if k != 'split_sha256'}


def manifest():
    """Reject code/settings drift; compare data hashes during preparation/audits."""
    path = RUN / 'manifest.json'
    value = read(path)
    for name, digest in value['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Frozen study code changed: {name}')
    return value, sha(path)


def legacy_swin():
    # Only reuse numerical fit/predict helpers; do not invoke its run/context entrypoints.
    import importlib.util
    for path in (ROOT / 'attention method', ROOT / 'attention method/.venv/torch-cu128',
                 ROOT / 'correct balance training set OMI'):
        sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('sensitivity_legacy_swin',
                                                ROOT / 'correct balance training set OMI/swin.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
