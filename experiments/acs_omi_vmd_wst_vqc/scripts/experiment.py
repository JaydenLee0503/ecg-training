#!/usr/bin/env python3
"""Prepare ACS patient splits and resumable matched VMD/WST feature archives.

Commands are explicit: prepare, smoke, extract, bundle. None trains a model or
reads official test waveforms. A changed protocol or implementation needs a new run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset
from experiments.acs_omi_vmd_wst_vqc.pipeline import digest_json, engineering_ids, extract_record, patient_split
from experiments.acs_omi_vmd_wst_vqc.layout import verify_known_migration

PROTOCOL = EXPERIMENT / 'protocols/acs_omi_protocol_v1.json'
AUDIT = EXPERIMENT / 'results/loader/2026-09-24_all_leads_omi_v2'
LAYOUT = EXPERIMENT / 'provenance/layout_2026-09-24.json'
CODE = ('experiments/acs_omi_vmd_wst_vqc/loader.py', 'experiments/acs_omi_vmd_wst_vqc/pipeline.py', 'ecgvmd/vmd.py',
        'ecgvmd/features.py', 'ecgvmd/scatter.py', 'ecgvmd/config.py',
        'ecgvmd/select.py', 'ecgvmd/quantum.py', 'ecgvmd/reupload.py',
        'experiments/acs_omi_vmd_wst_vqc/scripts/experiment.py', 'experiments/acs_omi_vmd_wst_vqc/provenance/layout_2026-09-24.json',
        'experiments/acs_omi_vmd_wst_vqc/layout.py', 'experiments/acs_omi_vmd_wst_vqc/provenance/descriptive_rename_v1.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    return {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'kymatio', 'pennylane')},
            'threads': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(rows):
    return {part: dict(records=len(selected), patients=len({r['patient_id'] for r in selected}),
                       labels=dict(Counter(str(r['label']) for r in selected)))
            for part in ('fit', 'validation')
            for selected in [[r for r in rows if r['partition'] == part]]}


def prepare(args):
    if args.out.exists():
        raise ValueError('Existing run refused; inspect its status and resume another command if complete')
    protocol = json.loads(PROTOCOL.read_text())
    audit = json.loads((AUDIT / 'summary.json').read_text())
    archived = json.loads((EXPERIMENT / 'reports/acs_loader_2026-09-24.json').read_text())
    if (audit != archived or audit['status'] != 'complete'
            or json.loads((AUDIT / 'status.json').read_text())['status'] != 'complete'
            or file_hash(AUDIT / 'records.csv') != audit['records_csv_sha256']):
        raise ValueError('Saved loader audit is incomplete or changed')
    ledger = read_csv(AUDIT / 'records.csv')
    with ACSDataset(args.data_dir, target=protocol['target'], leads=protocol['input']['leads']) as ds:
        if ds.policy != audit['policy'] or ds.source_manifest != audit['source_manifest']:
            raise ValueError('Protocol/source does not match the audited loader policy')
        expected_ids = set(ds.record_ids('train')) | set(ds.record_ids('test'))
        if len(ledger) != len(expected_ids) or {r['record_id'] for r in ledger} != expected_ids:
            raise ValueError('Audit record coverage differs from source')
        usable = []
        for row in ledger:
            info = ds.info(row['record_id'])
            if (row['patient_id'] != info.patient_id or row['split'] != info.split
                    or row['label'] != ('' if info.label is None else str(info.label))):
                raise ValueError('Audited patient/split/label differs from source')
            if row['eligible'] not in ('True', 'False'):
                raise ValueError('Invalid audit eligibility')
            if info.split == 'train' and row['eligible'] == 'True':
                usable.append(dict(record_id=info.record_id, patient_id=info.patient_id,
                                   split=info.split, label=info.label, waveform_sha256=row['waveform_sha256']))
        rows = patient_split(usable, protocol)
        sources = ds.source_manifest
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / 'prepare_status.json', {'status': 'running', 'started_at_utc': now()})
    try:
        write_csv(args.out / 'splits.csv', rows)
        (args.out / 'source_records.csv').write_bytes((AUDIT / 'records.csv').read_bytes())
        atomic_json(args.out / 'protocol.json', protocol)
        manifest = dict(protocol=protocol, protocol_sha256=digest_json(protocol),
                        source_manifest=sources, source_records_sha256=audit['records_csv_sha256'],
                        splits_sha256=file_hash(args.out / 'splits.csv'),
                        code_sha256={name: file_hash(ROOT / name) for name in CODE},
                        environment=environment(), created_at_utc=now())
        atomic_json(args.out / 'manifest.json', manifest)
        (args.out / 'manifest.sha256').write_text(file_hash(args.out / 'manifest.json') + '\n')
        summary = dict(status='complete', stage='patient split preparation', protocol_id=protocol['id'],
                       counts=counts(rows), excluded_training_records=audit['splits']['train']['excluded_records'],
                       reserved_official_test_records=audit['splits']['test']['records'],
                       reserved_test_policy_exclusions=audit['splits']['test']['excluded_records'],
                       engineering_record_ids=engineering_ids(rows, protocol),
                       manifest_sha256=file_hash(args.out / 'manifest.json'),
                       splits_sha256=manifest['splits_sha256'], models_trained=0, completed_at_utc=now())
        atomic_json(args.out / 'preparation.json', summary)
        atomic_json(args.out / 'prepare_status.json', {'status': 'complete', 'completed_at_utc': now()})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        atomic_json(args.out / 'prepare_status.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


def check_implementation(manifest, manifest_sha):
    """Allow only the recorded path/import migration for the existing frozen run.

    Its original manifest stays byte-identical. Verify both original source
    snapshots and the reviewed relocated implementation; any later edit fails.
    New runs record the new paths directly, including this migration record.
    """
    recorded = manifest['code_sha256']
    if verify_known_migration(manifest, manifest_sha):
        return
    if set(recorded) == set(CODE):
        for name, sha in recorded.items():
            if file_hash(ROOT / name) != sha:
                raise ValueError(f'Changed implementation: {name}; start a separate run')
        return
    layout = json.loads(LAYOUT.read_text())
    if (manifest_sha != layout['legacy_manifest_sha256']
            or recorded != layout['legacy_code_sha256']):
        raise ValueError('No verified layout migration for this manifest')
    for name, sha in recorded.items():
        moved = layout['code_moves'].get(name)
        if moved is None:
            if file_hash(ROOT / name) != sha:
                raise ValueError(f'Changed shared implementation: {name}')
        elif (file_hash(ROOT / moved['snapshot']) != sha
              or file_hash(ROOT / moved['current']) != moved['current_sha256']):
            raise ValueError(f'Changed migrated implementation: {name}; start a separate run')


def read_run(out):
    manifest_path = out / 'manifest.json'
    if json.loads((out / 'prepare_status.json').read_text())['status'] != 'complete':
        raise ValueError('Split preparation has not completed')
    if file_hash(manifest_path) != (out / 'manifest.sha256').read_text().strip():
        raise ValueError('Changed manifest')
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest['protocol']
    if (digest_json(protocol) != manifest['protocol_sha256']
            or json.loads((out / 'protocol.json').read_text()) != protocol
            or json.loads(PROTOCOL.read_text()) != protocol):
        raise ValueError('Changed protocol; start a separate run')
    check_implementation(manifest, file_hash(manifest_path))
    if environment() != manifest['environment']:
        raise ValueError('Changed environment; restore it or start a separate run')
    if (file_hash(out / 'splits.csv') != manifest['splits_sha256']
            or file_hash(out / 'source_records.csv') != manifest['source_records_sha256']):
        raise ValueError('Changed split or source ledger')
    return manifest, read_csv(out / 'splits.csv')


def checked_feature(folder, row, manifest_sha):
    """Validate a completed per-record checkpoint without recomputing features."""
    rid = row['record_id']
    meta = json.loads((folder / f'{rid}.json').read_text())
    if (meta['status'] != 'complete' or meta['manifest_sha256'] != manifest_sha
            or meta['record_id'] != rid or meta['patient_id'] != row['patient_id']
            or meta['label'] != int(row['label']) or meta['waveform_sha256'] != row['waveform_sha256']
            or meta['features_sha256'] != file_hash(folder / f'{rid}.npz')):
        raise ValueError(f'Invalid feature checkpoint: {rid}')
    with np.load(folder / f'{rid}.npz', allow_pickle=False) as z:
        values = {name: z[name] for name in z.files}
    for arm in ('vmd', 'wst'):
        if (values[arm].ndim != 1 or len(values[arm]) != meta[f'{arm}_features']
                or len(values[arm + '_names']) != len(values[arm]) or not np.isfinite(values[arm]).all()):
            raise ValueError(f'Invalid feature vector: {rid}/{arm}')
    return values, meta


def extract(args):
    manifest, rows = read_run(args.out)
    protocol = manifest['protocol']
    is_smoke = args.command == 'smoke'
    if not is_smoke:
        smoke = json.loads((args.out / 'smoke/status.json').read_text())
        if smoke['status'] != 'complete' or smoke['manifest_sha256'] != file_hash(args.out / 'manifest.json'):
            raise ValueError('Complete the declared engineering check before full extraction')
        selected = set(engineering_ids(rows, protocol))
        for row in rows:
            if row['record_id'] in selected:
                _, meta = checked_feature(args.out / 'smoke', row, smoke['manifest_sha256'])
                if meta['reference_max_abs_error'] is None:
                    raise ValueError('Engineering checkpoint lacks a WST reference check')
    if is_smoke:
        ids = set(engineering_ids(rows, protocol))
        rows = [r for r in rows if r['record_id'] in ids]
    folder = args.out / ('smoke' if is_smoke else 'features')
    folder.mkdir(exist_ok=True)
    manifest_sha = file_hash(args.out / 'manifest.json')
    status_path = folder / 'status.json'
    # All completed records are verified; a complete restart does not reopen ECG ZIPs.
    pending, metas = [], []
    for row in rows:
        if (folder / (row['record_id'] + '.json')).exists():
            _, meta = checked_feature(folder, row, manifest_sha)
            metas.append(meta)
        else:
            pending.append(row)
    if not pending and status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete':
        print(f'Already complete: {len(rows)} verified records in {folder}', flush=True)
        return
    atomic_json(status_path, {'status': 'running', 'started_at_utc': now(), 'pending': len(pending)})
    started = time.perf_counter()
    active = None
    try:
        if pending:
            with ACSDataset(args.data_dir, target=protocol['target'], leads=protocol['input']['leads']) as ds:
                if ds.source_manifest != manifest['source_manifest']:
                    raise ValueError('Changed source archives')
                for row in pending:
                    active = row['record_id']
                    print(f'START {active} ({row["partition"]})', flush=True)
                    record = ds.load_record(active)
                    if (record.info.split != 'train' or record.info.patient_id != row['patient_id']
                            or record.info.label != int(row['label'])
                            or record.decision.waveform_sha256 != row['waveform_sha256']):
                        raise ValueError('Loaded waveform differs from prepared row')
                    with (folder / f'{active}_attempts.jsonl').open('a') as stream:
                        def log_attempt(attempt):
                            stream.write(json.dumps(dict(time_utc=now(), **attempt)) + '\n')
                            stream.flush()
                        vectors, details = extract_record(record, protocol, reference=is_smoke, on_attempt=log_attempt)
                    temp = folder / f'{active}.tmp'
                    with temp.open('wb') as stream:
                        np.savez_compressed(stream, **vectors)
                    temp.replace(folder / f'{active}.npz')
                    details.update(status='complete', manifest_sha256=manifest_sha,
                                   features_sha256=file_hash(folder / f'{active}.npz'), completed_at_utc=now())
                    atomic_json(folder / f'{active}.json', details)
                    checked_feature(folder, row, manifest_sha)
                    metas.append(details)
                    print(f'DONE {active}: VMD {details["vmd_seconds"]:.2f}s, WST {details["wst_seconds"]:.2f}s', flush=True)
        summary = dict(status='complete', scope='engineering check only' if is_smoke else 'all eligible official-training features',
                       records=len(rows), record_ids=[r['record_id'] for r in rows] if is_smoke else None,
                       newly_extracted=len(pending), resumed=len(rows) - len(pending),
                       elapsed_seconds=time.perf_counter() - started,
                       summed_vmd_seconds=sum(m['vmd_seconds'] for m in metas),
                       summed_wst_seconds=sum(m['wst_seconds'] for m in metas),
                       max_vmd_iterations=max(max(m['vmd_iterations']) for m in metas),
                       retried_leads=sum(a['capped'] for m in metas for a in m['vmd_attempts']),
                       final_capped_leads=0,
                       feature_dimensions={arm: metas[0][arm + '_features'] for arm in ('vmd', 'wst')},
                       reference_max_abs_error=max(m['reference_max_abs_error'] for m in metas) if is_smoke else None,
                       manifest_sha256=manifest_sha, models_trained=0, completed_at_utc=now())
        atomic_json(status_path, summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        failure = dict(status='failed', record_id=active, time_utc=now(), traceback=traceback.format_exc())
        atomic_json(folder / ('failure_' + str(time.time_ns()) + '.json'), failure)
        atomic_json(status_path, failure)
        raise


def bundle(args):
    manifest, rows = read_run(args.out)
    manifest_sha = file_hash(args.out / 'manifest.json')
    folder = args.out / 'features'
    if json.loads((folder / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('Complete full feature extraction before bundling')
    target = args.out / 'features.npz'
    if target.exists():
        raise ValueError('Existing bundle refused; inspect bundle.json instead of overwriting')
    blocks = {arm: [] for arm in ('vmd', 'wst')}
    names = {}
    for row in rows:
        vectors, _ = checked_feature(folder, row, manifest_sha)
        for arm in blocks:
            if arm in names and not np.array_equal(names[arm], vectors[arm + '_names']):
                raise ValueError('Feature column names changed between records')
            names[arm] = vectors[arm + '_names']
            blocks[arm].append(vectors[arm])
    temp = args.out / 'features.tmp'
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **{arm: np.stack(block) for arm, block in blocks.items()},
                            **{arm + '_names': value for arm, value in names.items()},
                            y=np.array([int(r['label']) for r in rows]),
                            patients=np.array([r['patient_id'] for r in rows]),
                            record_ids=np.array([r['record_id'] for r in rows]),
                            partitions=np.array([r['partition'] for r in rows]),
                            manifest_sha256=np.array(manifest_sha))
    temp.replace(target)
    atomic_json(args.out / 'bundle.json', dict(status='complete', records=len(rows),
                                             features_sha256=file_hash(target), manifest_sha256=manifest_sha))
    print(f'Bundled {len(rows)} aligned ECG rows; no preprocessing or classifiers fitted.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'smoke', 'extract', 'bundle'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=EXPERIMENT / 'data')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command in ('smoke', 'extract'):
        extract(args)
    else:
        bundle(args)


if __name__ == '__main__':
    main()
