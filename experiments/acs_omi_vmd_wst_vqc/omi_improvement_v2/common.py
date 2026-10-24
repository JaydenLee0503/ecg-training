"""Local state with immutable sources and completed-artifact verification."""
from pathlib import Path
import sys

from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.common import (
    read, sha, now, write, write_bytes, arrays, npz, csv_write, complete, verified, lock)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
RUN = BASE / 'results'
REPORTS = BASE / 'reports'
PARENT = BASE.parent / 'sensitivity_study_v1'
BALANCED = ROOT / 'correct balance training set OMI'


def imports():
    for path in (ROOT / 'attention method', ROOT / 'attention method/.venv/torch-cu128'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def manifest():
    value = read(RUN / 'manifest.json')
    for name, digest in value['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    return value, sha(RUN / 'manifest.json')


def protect_check(value):
    for name, digest in value['protected_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Protected input changed: {name}')
    return len(value['protected_sha256'])


def cohort():
    return arrays(RUN / 'prepared/cohort.npz')


def frozen_write(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen decision changed: {path}')
    write(path, value)
