"""Bounded process execution of the frozen ACS feature extractor.

The coordinator owns the source archives and atomic feature checkpoints. Workers
receive one decoded ECG and write only its append-only convergence-attempt log.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from .pipeline import extract_record
from .scripts import experiment as serial


@contextmanager
def run_lock(folder):
    """A crashed process cannot leave a stale lock that blocks the next run."""
    with (folder / '.run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is using this parallel run directory') from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({'pid': os.getpid(), 'started_at_utc': serial.now()}) + '\n')
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def worker_init():
    # Retain the limiter for the worker lifetime, including libraries imported by sklearn.
    global _thread_limiter
    _thread_limiter = threadpool_limits(limits=1)


def feature_worker(record, protocol, attempt_path, reference):
    started = time.perf_counter()
    try:
        with Path(attempt_path).open('a') as stream:
            def log_attempt(attempt):
                stream.write(json.dumps(dict(time_utc=serial.now(), pid=os.getpid(), **attempt)) + '\n')
                stream.flush()
            vectors, details = extract_record(record, protocol, reference=reference, on_attempt=log_attempt)
        details.update(worker_pid=os.getpid(), record_wall_seconds=time.perf_counter() - started)
        return {'ok': True, 'vectors': vectors, 'details': details}
    except Exception as exc:
        return {'ok': False, 'record_id': record.info.record_id, 'worker_pid': os.getpid(),
                'error_type': type(exc).__name__, 'error': str(exc),
                'traceback': traceback.format_exc(), 'record_wall_seconds': time.perf_counter() - started}


def pilot_ids(rows, scientific_protocol, execution_protocol):
    return sorted((r['record_id'] for r in rows if r['partition'] == 'fit'),
                  key=lambda rid: hashlib.sha256((scientific_protocol['id'] + ':' + rid).encode()).digest()
                  )[:execution_protocol['pilot']['records']]


def check_loaded(record, row):
    if (record.info.record_id != row['record_id'] or record.info.split != 'train'
            or record.info.patient_id != row['patient_id'] or record.info.label != int(row['label'])
            or record.decision.waveform_sha256 != row['waveform_sha256']):
        raise ValueError('Loaded ECG differs from the frozen patient split')


def compare_serial(vectors, details, old_vectors, old_details):
    """Reference check on the same ECG; no label- or accuracy-based selection."""
    for key in ('vmd', 'wst', 'vmd_names', 'wst_names'):
        if not np.array_equal(vectors[key], old_vectors[key]):
            raise ValueError(f'Parallel/serial mismatch: {details["record_id"]}/{key}')
    for key in ('physical_input_sha256', 'vmd_iterations', 'vmd_limits', 'vmd_attempts'):
        if details[key] != old_details[key]:
            raise ValueError(f'Parallel/serial diagnostic mismatch: {key}')
    return {'status': 'passed', 'feature_max_abs_error': {'vmd': 0.0, 'wst': 0.0},
            'input_and_convergence_equal': True}


def save_feature(folder, row, result, manifest_sha, parent_sha, reference=None):
    vectors, details = result['vectors'], dict(result['details'])
    if (details['record_id'] != row['record_id'] or details['patient_id'] != row['patient_id']
            or details['label'] != int(row['label']) or details['waveform_sha256'] != row['waveform_sha256']):
        raise ValueError('Worker returned a different record identity')
    if reference is not None:
        details['serial_comparison'] = compare_serial(vectors, details, *reference)
    rid = row['record_id']
    destination = folder / f'{rid}.npz'
    temp = folder / f'{rid}.tmp'
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **vectors)
    temp.replace(destination)
    details.update(status='complete', manifest_sha256=manifest_sha, parent_manifest_sha256=parent_sha,
                   features_sha256=serial.file_hash(destination), completed_at_utc=serial.now())
    serial.atomic_json(folder / f'{rid}.json', details)
    serial.checked_feature(folder, row, manifest_sha)
    return details


def bounded_execute(rows, load, accept, attempt_folder, protocol, workers, reference_ids, *, worker=feature_worker):
    """Decode at most one pending ECG per worker; drain and save in-flight results.

    The injected load/accept callbacks run only in the coordinator. This keeps ZIP
    handles out of workers and makes failure/resume behavior independently testable.
    """
    if workers < 1:
        raise ValueError('Need at least one worker')
    iterator = iter(rows)
    failures = []
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=worker_init)
    futures = {}
    exhausted = False
    try:
        while futures or not exhausted:
            while not failures and not exhausted and len(futures) < workers:
                row = next(iterator, None)
                if row is None:
                    exhausted = True
                    break
                try:
                    record = load(row)
                    future = pool.submit(worker, record, protocol,
                                         str(attempt_folder / (row['record_id'] + '_attempts.jsonl')),
                                         row['record_id'] in reference_ids)
                    futures[future] = row
                except Exception as exc:
                    failures.append(dict(record_id=row['record_id'], error_type=type(exc).__name__,
                                         error=str(exc), traceback=traceback.format_exc()))
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                row = futures.pop(future)
                try:
                    result = future.result()
                    if not result['ok']:
                        failures.append(result)
                    else:
                        accept(row, result)
                except Exception as exc:
                    failures.append(dict(record_id=row['record_id'], error_type=type(exc).__name__,
                                         error=str(exc), traceback=traceback.format_exc()))
            if failures:
                exhausted = True
        return failures
    finally:
        # At most workers records are active; never leave an unbounded task queue.
        pool.shutdown(wait=True, cancel_futures=True)
