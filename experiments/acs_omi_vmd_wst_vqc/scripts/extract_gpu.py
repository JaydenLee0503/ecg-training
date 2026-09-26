#!/usr/bin/env python3
"""Prepare, validate and resume the separate GPU VMD / CPU WST feature run.

Examples: python -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu prepare
          python -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu pilot
          python -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu extract
No command trains a model or reads official-test waveforms.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/tmp/acs-gpu-deps-v1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/acs-matplotlib')
os.environ.setdefault('CUPY_CACHE_DIR', '/tmp/acs-cupy-production-cache')

import numpy as np
from threadpoolctl import threadpool_limits
from experiments.acs_omi_vmd_wst_vqc.gpu_extraction import extract_records, compare_cpu
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, DEFAULT_DATA_DIR
from experiments.acs_omi_vmd_wst_vqc.parallel import run_lock, check_loaded
from experiments.acs_omi_vmd_wst_vqc.pipeline import digest_json
from experiments.acs_omi_vmd_wst_vqc.layout import verify_known_migration
from experiments.acs_omi_vmd_wst_vqc.scripts import experiment as serial, extract_parallel as parallel

SPEC = BASE/'protocols/gpu_extraction_v1.json'
CPU = BASE/'results/omi_v1_parallel'
BENCH = BASE/'results/gpu_vmd_benchmark_v1'
CODE = sorted(set(serial.CODE + parallel.NEW_CODE + tuple(str((BASE/name).relative_to(ROOT)) for name in (
    'gpu_vmd.py', 'gpu_extraction.py', 'scripts/extract_gpu.py', 'protocols/gpu_extraction_v1.json',
    'tests/test_gpu_extraction.py', 'tests/test_gpu_vmd.py'))))


def atomic_bytes(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def environment():
    result = serial.environment()
    result['gpu_packages'] = {n: importlib.metadata.version(n) for n in ('cupy-cuda12x', 'cuda-pathfinder')}
    if any(v != '1' for v in result['threads'].values()):
        raise ValueError('GPU extraction requires OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1')
    return result


def hardware():
    import cupy as cp
    cp.cuda.Device(0).use()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    cp.zeros(1).sum().get()
    return dict(device=0, name=properties['name'].decode(), total_memory=int(properties['totalGlobalMem']),
                compute_capability=cp.cuda.Device(0).compute_capability,
                runtime_version=cp.cuda.runtime.runtimeGetVersion(), driver_version=cp.cuda.runtime.driverGetVersion())


def parent_context():
    spec = json.loads(SPEC.read_text())
    parent, rows, parent_sha = parallel.read_parallel(SimpleNamespace(
        parent=BASE/'results/omi_v1', out=CPU, workers=8))
    if (parent_sha != spec['parent_parallel_manifest_sha256']
            or digest_json(parent['scientific_protocol']) != spec['scientific_protocol_sha256']):
        raise ValueError('Unexpected scientific protocol or parent')
    if (serial.file_hash(BENCH/'manifest.json') != spec['benchmark_manifest_sha256']
            or serial.file_hash(BENCH/'summary.json') != spec['benchmark_summary_sha256']):
        raise ValueError('Changed benchmark evidence')
    bm = json.loads((BENCH/'manifest.json').read_text())
    if not verify_known_migration(bm, spec['benchmark_manifest_sha256']):
        raise ValueError('Benchmark implementation has no verified migration')
    summary = json.loads((BENCH/'summary.json').read_text())
    if (json.loads((BENCH/'status.json').read_text())['status'] != 'complete'
            or not summary['numerical_checks_passed']):
        raise ValueError('GPU numerical benchmark did not pass')
    for artifact in summary['artifacts']:
        if serial.file_hash(BENCH/artifact['path']) != artifact['sha256']:
            raise ValueError('Changed benchmark artifact')
    return spec, parent, rows


def read_run(out, *, prepared=True):
    spec, parent, rows = parent_context()
    sha = serial.file_hash(out/'manifest.json')
    if sha != (out/'manifest.sha256').read_text().strip():
        raise ValueError('Changed GPU manifest')
    m = json.loads((out/'manifest.json').read_text())
    if (m['execution_protocol'] != spec or m['scientific_protocol'] != parent['scientific_protocol']
            or m['source_manifest'] != parent['source_manifest'] or m['environment'] != environment()
            or m['pilot_record_ids'] != parent['pilot_record_ids']):
        raise ValueError('Changed GPU settings, inputs, environment or pilot')
    if m['code_sha256'] != {name: serial.file_hash(ROOT/name) for name in CODE}:
        raise ValueError('Changed GPU implementation; use a new documented run')
    for filename, key in (('splits.csv','splits_sha256'), ('source_records.csv','source_records_sha256')):
        if m[key] != parent[key] or serial.file_hash(out/filename) != parent[key]:
            raise ValueError('Changed split or source ledger')
    if json.loads((out/'protocol.json').read_text()) != m['scientific_protocol']:
        raise ValueError('Changed saved scientific protocol')
    if prepared and json.loads((out/'preparation.json').read_text())['status'] != 'complete':
        raise ValueError('GPU preparation must finish before extraction')
    return m, rows, sha


def checked(folder, row, sha):
    values, meta = serial.checked_feature(folder, row, sha)
    if values['vmd'].shape != (2688,) or values['wst'].shape != (2808,):
        raise ValueError('Unexpected feature count')
    if values['vmd'].dtype != np.float32 or values['wst'].dtype != np.float32:
        raise ValueError('Unexpected stored feature precision')
    for name, expected in meta.get('log_sha256', {}).items():
        if serial.file_hash(folder/name) != expected:
            raise ValueError('Changed convergence attempt log')
    return values, meta


def scan_checkpoints(folder, rows, sha):
    pending, done = [], []
    for row in rows:
        if (folder/(row['record_id']+'.json')).exists():
            _, meta = checked(folder, row, sha)
            done.append(meta)
        else:
            # An orphaned NPZ is incomplete; a crash cannot turn it into a valid checkpoint.
            pending.append(row)
    return pending, done


def prepare(args):
    spec, parent, rows = parent_context()
    args.out.mkdir(parents=True, exist_ok=True)
    with run_lock(args.out), (CPU/'.run.lock').open('r') as cpu_lock:
        fcntl.flock(cpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.out/'manifest.json').exists():
            m, rows, sha = read_run(args.out, prepared=False)
        else:
            inventory = {}
            for row in rows:
                rid = row['record_id']
                if (CPU/'features'/f'{rid}.json').exists():
                    _, meta = serial.checked_feature(CPU/'features', row, spec['parent_parallel_manifest_sha256'])
                    inventory[rid] = {name: serial.file_hash(CPU/'features'/name)
                                      for name in (f'{rid}.npz', f'{rid}.json', f'{rid}_attempts.jsonl')}
            for name in ('protocol.json', 'splits.csv', 'source_records.csv'):
                atomic_bytes(args.out/name, (CPU/name).read_bytes())
            m = dict(execution_protocol=spec, scientific_protocol=parent['scientific_protocol'],
                     source_manifest=parent['source_manifest'], splits_sha256=parent['splits_sha256'],
                     source_records_sha256=parent['source_records_sha256'], environment=environment(),
                     hardware=hardware(), code_sha256={name: serial.file_hash(ROOT/name) for name in CODE},
                     pilot_record_ids=parent['pilot_record_ids'], imported_cpu_checkpoints=inventory,
                     counts=serial.counts(rows), created_at_utc=serial.now(), models_trained=0,
                     official_test_processed=False)
            atomic_json(args.out/'manifest.json', m)
            sha = serial.file_hash(args.out/'manifest.json')
            atomic_bytes(args.out/'manifest.sha256', (sha+'\n').encode())
        folder = args.out/'features'
        folder.mkdir(exist_ok=True)
        lookup = {r['record_id']: r for r in rows}
        for rid, files in m['imported_cpu_checkpoints'].items():
            if (folder/f'{rid}.json').exists():
                checked(folder, lookup[rid], sha)
                continue
            for name, expected in files.items():
                if serial.file_hash(CPU/'features'/name) != expected:
                    raise ValueError(f'Changed source checkpoint: {name}')
            _, info = serial.checked_feature(CPU/'features', lookup[rid], spec['parent_parallel_manifest_sha256'])
            atomic_bytes(folder/f'{rid}.npz', (CPU/'features'/f'{rid}.npz').read_bytes())
            log_name = f'{rid}_source_attempts.jsonl'
            atomic_bytes(folder/log_name, (CPU/'features'/f'{rid}_attempts.jsonl').read_bytes())
            info.update(manifest_sha256=sha, execution_origin='imported_cpu_checkpoint',
                        source_checkpoint=dict(run='omi_v1_parallel', sha256=files),
                        log_sha256={log_name: files[f'{rid}_attempts.jsonl']}, imported_at_utc=serial.now())
            atomic_json(folder/f'{rid}.json', info)
            checked(folder, lookup[rid], sha)
        atomic_json(args.out/'preparation.json', dict(status='complete', manifest_sha256=sha,
                    records=len(rows), imported_cpu_records=len(m['imported_cpu_checkpoints']),
                    completed_at_utc=serial.now()))
        print(f'Prepared {len(rows)} ECGs; retained {len(m["imported_cpu_checkpoints"])} CPU checkpoints', flush=True)


@contextmanager
def graceful_stop():
    state = {'requested': False, 'signal': None}
    def handler(number, frame):
        state.update(requested=True, signal=number)
        print('Stop requested; finishing and checkpointing the current batch', flush=True)
    previous = {n: signal.signal(n, handler) for n in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def save_feature(folder, row, vectors, details, sha):
    if any(details[k] != row[k] for k in ('record_id', 'patient_id', 'waveform_sha256')) or details['label'] != int(row['label']):
        raise ValueError('GPU result identity differs from the frozen row')
    rid = row['record_id']
    path = folder/f'{rid}.npz'
    temporary = folder/f'{rid}.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **vectors)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    log_name = f'{rid}_attempts.jsonl'
    details.update(status='complete', manifest_sha256=sha, features_sha256=serial.file_hash(path),
                   log_sha256={log_name: serial.file_hash(folder/log_name)}, completed_at_utc=serial.now())
    atomic_json(folder/f'{rid}.json', details)
    checked(folder, row, sha)


def validate_pilot(out, m, rows, sha):
    path = out/'pilot_status.json'
    status = json.loads(path.read_text())
    if status['status'] != 'complete' or status['manifest_sha256'] != sha:
        raise ValueError('Integrated GPU pilot must finish first')
    selected = [r for r in rows if r['record_id'] in m['pilot_record_ids']]
    if len(selected) != 16:
        raise ValueError('Unexpected pilot size')
    reference_ids = set(m['pilot_record_ids'][:2])
    for row in selected:
        vectors, details = checked(out/'pilot', row, sha)
        reference = serial.checked_feature(CPU/'features', row, m['execution_protocol']['parent_parallel_manifest_sha256'])
        compare_cpu(vectors, details, *reference)
        if row['record_id'] in reference_ids and details['reference_max_abs_error'] is None:
            raise ValueError('Missing Kymatio reference check')


def run(args):
    m, all_rows, sha = read_run(args.out)
    is_pilot = args.command == 'pilot'
    rows = ([next(r for r in all_rows if r['record_id'] == rid) for rid in m['pilot_record_ids']]
            if is_pilot else all_rows)
    if not is_pilot:
        validate_pilot(args.out, m, all_rows, sha)
    folder = args.out/('pilot' if is_pilot else 'features')
    folder.mkdir(exist_ok=True)
    status_path = args.out/('pilot_status.json' if is_pilot else 'extraction_status.json')
    with run_lock(args.out), graceful_stop() as stop:
        pending, done = scan_checkpoints(folder, rows, sha)
        if not pending and status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete':
            if is_pilot:
                validate_pilot(args.out, m, all_rows, sha)
            print(f'Already complete: {len(rows)} verified {args.command} records; no waveform or GPU work', flush=True)
            return
        started, started_at = time.perf_counter(), serial.now()
        resumed, batches, active = len(done), 0, []
        def progress(status='running', **extra):
            elapsed = time.perf_counter()-started
            new = len(done)-resumed
            result = dict(status=status, stage=args.command, pid=os.getpid(), started_at_utc=started_at,
                          updated_at_utc=serial.now(), manifest_sha256=sha, records_total=len(rows),
                          records_complete=len(done), records_resumed=resumed, records_new=new,
                          elapsed_seconds=elapsed, records_per_hour=3600*new/elapsed if elapsed else 0,
                          active_record_ids=active, models_trained=0, official_test_processed=False, **extra)
            atomic_json(status_path, result)
            return result
        progress()
        try:
            if pending:
                if hardware() != m['hardware']:
                    raise ValueError('Changed CUDA hardware/runtime; use a new documented execution run')
                with ACSDataset(args.data_dir, target=m['scientific_protocol']['target'],
                                leads=m['scientific_protocol']['input']['leads']) as dataset:
                    if dataset.source_manifest != m['source_manifest']:
                        raise ValueError('Changed source archives')
                    size = m['execution_protocol']['records_per_gpu_batch']
                    for start in range(0, len(pending), size):
                        if stop['requested'] or (args.max_batches is not None and batches >= args.max_batches):
                            break
                        selected = pending[start:start+size]
                        active = [r['record_id'] for r in selected]
                        progress()
                        batch_start = time.perf_counter()
                        records = []
                        for row in selected:
                            record = dataset.load_record(row['record_id'])
                            check_loaded(record, row)
                            records.append(record)
                        load_seconds = time.perf_counter()-batch_start
                        batch_id = f'{os.getpid()}-{time.time_ns()}'
                        def attempt(rid, entry):
                            with (folder/f'{rid}_attempts.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(time_utc=serial.now(), batch_id=batch_id, **entry))+'\n')
                                stream.flush()
                                os.fsync(stream.fileno())
                        results = extract_records(records, m['scientific_protocol'], on_attempt=attempt,
                                                  reference_ids=m['pilot_record_ids'][:2] if is_pilot else ())
                        for row, (vectors, details) in zip(selected, results, strict=True):
                            if is_pilot:
                                reference = serial.checked_feature(CPU/'features', row,
                                            m['execution_protocol']['parent_parallel_manifest_sha256'])
                                details['cpu_comparison'] = compare_cpu(vectors, details, *reference)
                            details.update(batch_id=batch_id, batch_record_ids=active)
                            save_feature(folder, row, vectors, details, sha)
                            done.append(details)
                            progress()
                        elapsed = time.perf_counter()-batch_start
                        event = dict(time_utc=serial.now(), batch_id=batch_id, record_ids=active,
                                     load_seconds=load_seconds, wall_seconds=elapsed,
                                     records_per_hour=3600*len(selected)/elapsed)
                        with (args.out/f'{args.command}_batches.jsonl').open('a') as stream:
                            stream.write(json.dumps(event)+'\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        batches += 1
                        active = []
                        state = progress()
                        print(f'{args.command}: {len(done)}/{len(rows)}; batch {elapsed:.2f}s; '
                              f'{state["records_per_hour"]:.1f} ECG/hour including startup', flush=True)
            complete = len(done) == len(rows)
            state = progress('complete' if complete else 'interrupted', stop_signal=stop['signal'],
                             stopped_after_max_batches=args.max_batches)
            if complete and is_pilot:
                validate_pilot(args.out, m, all_rows, sha)
            with (args.out/'sessions.jsonl').open('a') as stream:
                stream.write(json.dumps(state)+'\n')
            print(json.dumps(state), flush=True)
        except BaseException as exc:
            failure = dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            atomic_json(args.out/f'failure_{time.time_ns()}.json', progress('failed', **failure))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pilot', 'extract'))
    parser.add_argument('--out', type=Path, default=BASE/'results/omi_v1_gpu')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--max-batches', type=int, help='Stop cleanly after this many new batches; for recovery checks')
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    with threadpool_limits(limits=1):
        (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
