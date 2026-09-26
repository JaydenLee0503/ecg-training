#!/usr/bin/env python3
"""Separate, resumable parallel extraction run derived from frozen ACS OMI v1.

prepare copies the verified split/protocol without rerandomizing. pilot compares
two existing serial checkpoints and times 16 fit ECGs. extract processes every
eligible development ECG, reusing pilot checkpoints. No models are trained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, DEFAULT_DATA_DIR
from experiments.acs_omi_vmd_wst_vqc.pipeline import digest_json
from experiments.acs_omi_vmd_wst_vqc.parallel import bounded_execute, check_loaded, pilot_ids, run_lock, save_feature
from experiments.acs_omi_vmd_wst_vqc.scripts import experiment as serial
from experiments.acs_omi_vmd_wst_vqc.layout import verify_known_migration

EXECUTION = BASE / 'protocols/parallel_extraction_v1.json'
NEW_CODE = ('experiments/acs_omi_vmd_wst_vqc/parallel.py', 'experiments/acs_omi_vmd_wst_vqc/scripts/extract_parallel.py',
            'experiments/acs_omi_vmd_wst_vqc/protocols/parallel_extraction_v1.json',
            'experiments/acs_omi_vmd_wst_vqc/layout.py', 'experiments/acs_omi_vmd_wst_vqc/provenance/descriptive_rename_v1.json')


def parent_context(parent):
    manifest, rows = serial.read_run(parent)
    execution = json.loads(EXECUTION.read_text())
    parent_sha = serial.file_hash(parent / 'manifest.json')
    if (parent_sha != execution['parent_manifest_sha256']
            or digest_json(manifest['protocol']) != execution['scientific_protocol_sha256']):
        raise ValueError('Parent differs from the declared parallel execution protocol')
    status = json.loads((parent / 'smoke/status.json').read_text())
    if status['status'] != 'complete' or status['manifest_sha256'] != parent_sha:
        raise ValueError('Parent engineering check is incomplete')
    lookup = {r['record_id']: r for r in rows}
    for rid in execution['pilot']['serial_comparison_records']:
        serial.checked_feature(parent / 'smoke', lookup[rid], parent_sha)
    return manifest, rows, execution, parent_sha


def prepare(args):
    if args.out.exists():
        raise ValueError('Existing directory refused; use pilot/extract to resume')
    parent, rows, execution, parent_sha = parent_context(args.parent)
    if args.workers < 1 or args.workers > len(os.sched_getaffinity(0)):
        raise ValueError('Worker count must be positive and no greater than available logical CPUs')
    args.out.mkdir(parents=True, exist_ok=False)
    with run_lock(args.out):
        for name in ('protocol.json', 'splits.csv', 'source_records.csv'):
            (args.out / name).write_bytes((args.parent / name).read_bytes())
        serial.atomic_json(args.out / 'execution_protocol.json', execution)
        manifest = dict(scientific_protocol=parent['protocol'], execution_protocol=execution,
                        parent_manifest_sha256=parent_sha, workers=args.workers,
                        source_manifest=parent['source_manifest'], environment=serial.environment(),
                        splits_sha256=parent['splits_sha256'], source_records_sha256=parent['source_records_sha256'],
                        code_sha256={name: serial.file_hash(ROOT / name) for name in NEW_CODE},
                        pilot_record_ids=pilot_ids(rows, parent['protocol'], execution),
                        created_at_utc=serial.now())
        serial.atomic_json(args.out / 'manifest.json', manifest)
        (args.out / 'manifest.sha256').write_text(serial.file_hash(args.out / 'manifest.json') + '\n')
        (args.out / 'features').mkdir()
        serial.atomic_json(args.out / 'preparation.json', dict(status='complete', counts=serial.counts(rows),
                           pilot_record_ids=manifest['pilot_record_ids'], workers=args.workers,
                           parent_manifest_sha256=parent_sha, completed_at_utc=serial.now()))
    print(json.dumps({'status': 'prepared', 'workers': args.workers, 'records': len(rows),
                      'pilot_record_ids': manifest['pilot_record_ids'], 'out': str(args.out)}, indent=2), flush=True)


def read_parallel(args):
    parent, rows, execution, parent_sha = parent_context(args.parent)
    path = args.out / 'manifest.json'
    sha = serial.file_hash(path)
    if sha != (args.out / 'manifest.sha256').read_text().strip():
        raise ValueError('Changed parallel manifest')
    manifest = json.loads(path.read_text())
    if (manifest['parent_manifest_sha256'] != parent_sha or manifest['scientific_protocol'] != parent['protocol']
            or manifest['execution_protocol'] != execution or manifest['source_manifest'] != parent['source_manifest']
            or manifest['environment'] != serial.environment() or manifest['workers'] != args.workers
            or manifest['pilot_record_ids'] != pilot_ids(rows, parent['protocol'], execution)):
        raise ValueError('Changed parent, settings, environment or worker count; use a new output directory')
    if not verify_known_migration(manifest, sha):
        if set(manifest['code_sha256']) != set(NEW_CODE):
            raise ValueError('Unexpected parallel code manifest')
        for name, checksum in manifest['code_sha256'].items():
            if serial.file_hash(ROOT / name) != checksum:
                raise ValueError(f'Changed code: {name}; use a separate run')
    if json.loads((args.out / 'preparation.json').read_text())['status'] != 'complete':
        raise ValueError('Parallel preparation incomplete')
    for name, key in (('splits.csv', 'splits_sha256'), ('source_records.csv', 'source_records_sha256')):
        if manifest[key] != parent[key] or serial.file_hash(args.out / name) != parent[key]:
            raise ValueError('Changed split/ledger')
    if (json.loads((args.out / 'protocol.json').read_text()) != parent['protocol']
            or json.loads((args.out / 'execution_protocol.json').read_text()) != execution):
        raise ValueError('Changed saved protocol')
    return manifest, rows, sha


def run(args):
    manifest, all_rows, sha = read_parallel(args)
    is_pilot = args.command == 'pilot'
    wanted = set(manifest['pilot_record_ids']) if is_pilot else {r['record_id'] for r in all_rows}
    rows = [r for r in all_rows if r['record_id'] in wanted]
    if not is_pilot:
        pilot = json.loads((args.out / 'pilot_status.json').read_text())
        if pilot['status'] != 'complete' or pilot['manifest_sha256'] != sha:
            raise ValueError('Pilot must complete successfully before full extraction')
    folder = args.out / 'features'
    status_path = args.out / ('pilot_status.json' if is_pilot else 'extraction_status.json')
    with run_lock(args.out):
        pending, details = [], []
        for row in rows:
            if (folder / (row['record_id'] + '.json')).exists():
                _, info = serial.checked_feature(folder, row, sha)
                details.append(info)
            else:
                pending.append(row)
        if not pending and status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete':
            print(f'Already complete: {len(rows)} verified {args.command} records', flush=True)
            return
        started, started_at = time.perf_counter(), serial.now()
        resumed = len(details)
        reference_ids = set(manifest['execution_protocol']['pilot']['serial_comparison_records'])
        def progress(status='running', **extra):
            elapsed = time.perf_counter() - started
            completed_now = len(details) - resumed
            record = dict(status=status, stage=args.command, pid=os.getpid(), workers=args.workers,
                          started_at_utc=started_at, updated_at_utc=serial.now(), records_total=len(rows),
                          records_complete=len(details), records_resumed=resumed,
                          records_new=completed_now, elapsed_seconds=elapsed, manifest_sha256=sha,
                          records_per_hour=3600 * completed_now / elapsed if elapsed > 0 else None,
                          **extra)
            serial.atomic_json(status_path, record)
            return record
        progress()
        try:
            with ACSDataset(args.data_dir, target=manifest['scientific_protocol']['target']) as dataset:
                if dataset.source_manifest != manifest['source_manifest']:
                    raise ValueError('Source archive identity changed')
                def load(row):
                    record = dataset.load_record(row['record_id'])
                    check_loaded(record, row)
                    return record
                def accept(row, result):
                    reference = None
                    if row['record_id'] in reference_ids:
                        reference = serial.checked_feature(args.parent / 'smoke', row, manifest['parent_manifest_sha256'])
                    info = save_feature(folder, row, result, sha, manifest['parent_manifest_sha256'], reference)
                    details.append(info)
                    status = progress()
                    print(f'{args.command}: {len(details)}/{len(rows)} {row["record_id"]}; '
                          f'{info["record_wall_seconds"]:.1f}s; {status["records_per_hour"]:.1f} records/hour', flush=True)
                failures = bounded_execute(pending, load, accept, folder, manifest['scientific_protocol'],
                                           args.workers, reference_ids)
            if failures:
                serial.atomic_json(args.out / ('failures_' + str(time.time_ns()) + '.json'), failures)
                progress('failed', failures=failures)
                raise RuntimeError(f'{len(failures)} record(s) failed; no further tasks submitted; completed checkpoints retained')
            if len(details) != len(rows):
                raise RuntimeError('Incomplete record coverage')
            comparisons = {d['record_id']: d['serial_comparison'] for d in details if 'serial_comparison' in d}
            if is_pilot and set(comparisons) != reference_ids:
                raise ValueError('Pilot missing required serial comparisons')
            summary = progress('complete', serial_comparisons=comparisons,
                               completed_at_utc=serial.now(), final_capped_leads=0,
                               max_vmd_iterations=max(max(d['vmd_iterations']) for d in details),
                               capped_attempts=sum(a['capped'] for d in details for a in d['vmd_attempts']),
                               record_ids=[r['record_id'] for r in rows] if is_pilot else None,
                               models_trained=0)
            print(json.dumps(summary, indent=2), flush=True)
        except BaseException as exc:
            existing = json.loads(status_path.read_text())
            if existing['status'] != 'failed':
                failure = dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                serial.atomic_json(args.out / ('failure_' + str(time.time_ns()) + '.json'), failure)
                progress('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', **failure)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pilot', 'extract'))
    parser.add_argument('--parent', type=Path, default=BASE / 'results/omi_v1')
    parser.add_argument('--out', type=Path, default=BASE / 'results/omi_v1_parallel')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if any(os.environ.get(name) != '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 for the frozen environment')
    prepare(args) if args.command == 'prepare' else run(args)


if __name__ == '__main__':
    main()
