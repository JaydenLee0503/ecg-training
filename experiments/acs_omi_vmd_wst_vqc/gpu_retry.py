"""Strict checkpoint reuse for an appended VMD retry policy.

Neither existing manifests nor numerical implementations are modified. A result
that already converged before the old cap never reaches the appended attempts.
"""
import copy
import json
from pathlib import Path

from .loader import LEADS
from .scripts import extract_gpu as original


def assert_compatible_extension(parent, candidate):
    normalized = copy.deepcopy(candidate)
    normalized['id'] = parent['id']
    normalized['vmd']['iteration_limits'] = parent['vmd']['iteration_limits']
    if (parent['vmd']['iteration_limits'] != [2000, 4000, 8000, 16000, 32000]
            or candidate['vmd']['iteration_limits'] != parent['vmd']['iteration_limits'] + [64000, 128000]
            or candidate['id'] != 'acs-omi-v1-retry128k'
            or normalized != parent):
        raise ValueError('Reuse permits only the declared appended retry limits and protocol ID')


def assert_converged(meta, protocol):
    attempts = meta['vmd_attempts']
    if set(a['lead'] for a in attempts) != set(LEADS):
        raise ValueError('Missing/unknown convergence lead')
    if len(meta['vmd_iterations']) != 12 or len(meta['vmd_limits']) != 12:
        raise ValueError('Missing per-lead convergence metadata')
    for i, lead in enumerate(LEADS):
        history = [a for a in attempts if a['lead'] == lead]
        limits = protocol['vmd']['iteration_limits']
        if [a['limit'] for a in history] != limits[:len(history)]:
            raise ValueError('Retries differ from frozen policy')
        for j, a in enumerate(history):
            last = j == len(history)-1
            if (not 0 < a['iterations'] <= a['limit']
                    or bool(a['capped']) != (a['iterations'] >= a['limit'])
                    or bool(a['capped']) == last):
                raise ValueError('Only fully converged checkpoints can be reused')
        final = history[-1]
        if (final['iterations'] != meta['vmd_iterations'][i]
                or final['limit'] != meta['vmd_limits'][i]):
            raise ValueError('Inconsistent final convergence metadata')


def inventory_entry(folder, row, manifest_sha, protocol):
    _, meta = original.checked(folder, row, manifest_sha)
    assert_converged(meta, protocol)
    rid = row['record_id']
    logs = meta['log_sha256']
    if not logs or any(Path(name).name != name for name in logs):
        raise ValueError('Require local attempt-log file names')
    return dict(source_manifest_sha256=manifest_sha,
                files={f'{rid}.npz': meta['features_sha256'],
                       f'{rid}.json': original.serial.file_hash(folder/f'{rid}.json'), **logs},
                original_execution_origin=meta.get('execution_origin', 'cpu'))


def copy_checkpoint(source, destination, row, entry, manifest_sha, source_run):
    rid = row['record_id']
    metadata_path = destination/f'{rid}.json'
    if metadata_path.exists():
        _, meta = original.checked(destination, row, manifest_sha)
        if (meta.get('parent_checkpoint') != dict(run=source_run, **entry)
                or meta['features_sha256'] != entry['files'][f'{rid}.npz']):
            raise ValueError('Changed imported checkpoint provenance')
        return meta
    for name, expected in entry['files'].items():
        if original.serial.file_hash(source/name) != expected:
            raise ValueError(f'Changed source checkpoint: {name}')
    for name in entry['files']:
        if name != f'{rid}.json':
            original.atomic_bytes(destination/name, (source/name).read_bytes())
    meta = json.loads((source/f'{rid}.json').read_text())
    meta.update(manifest_sha256=manifest_sha, parent_checkpoint=dict(run=source_run, **entry),
                reused_at_utc=original.serial.now())
    original.atomic_json(metadata_path, meta)
    original.checked(destination, row, manifest_sha)
    return meta
