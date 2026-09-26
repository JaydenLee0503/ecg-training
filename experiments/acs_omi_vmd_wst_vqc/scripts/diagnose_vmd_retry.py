"""Validate the bounded VMD retry extension on the failing fit ECG only."""
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
from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as old
from experiments.acs_omi_vmd_wst_vqc.pipeline import extract_record
from experiments.acs_omi_vmd_wst_vqc.gpu_extraction import extract_records, compare_cpu
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset
from threadpoolctl import threadpool_limits

SOURCE = BASE/'results/omi_v1_gpu'
SPEC = BASE/'protocols/vmd_retry_extension_diagnostic_v1.json'
PROTOCOL = BASE/'protocols/acs_omi_protocol_v1_retry128k.json'
OUT = BASE/'results/vmd_retry_extension_diagnostic_v1'
CODE = sorted(set(old.CODE + [str(p.relative_to(ROOT)) for p in (Path(__file__).resolve(), SPEC, PROTOCOL)]))


def context():
    parent, rows, parent_sha = old.read_run(SOURCE)
    spec = json.loads(SPEC.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if parent_sha != spec['parent_manifest_sha256']:
        raise ValueError('Changed source run')
    normalized = json.loads(json.dumps(protocol))
    normalized['id'] = parent['scientific_protocol']['id']
    normalized['vmd']['iteration_limits'] = parent['scientific_protocol']['vmd']['iteration_limits']
    if (normalized != parent['scientific_protocol']
            or protocol['vmd']['iteration_limits'] != [2000,4000,8000,16000,32000,64000,128000]):
        raise ValueError('Only the declared appended retry limits and protocol ID may change')
    row = next(r for r in rows if r['record_id'] == spec['record_id'])
    if row['partition'] != spec['required_partition']:
        raise ValueError('Diagnostic record must be in fit partition')
    return spec, protocol, parent, row, parent_sha


def run(backend):
    spec, protocol, parent, row, parent_sha = context()
    OUT.mkdir(parents=True, exist_ok=True)
    with old.run_lock(OUT):
        fingerprint = {name: old.serial.file_hash(ROOT/name) for name in CODE}
        manifest_path = OUT/'manifest.json'
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            if (old.serial.file_hash(manifest_path) != (OUT/'manifest.sha256').read_text().strip()
                    or m['code_sha256'] != fingerprint or m['environment'] != old.environment()
                    or m['spec'] != spec or m['scientific_protocol'] != protocol):
                raise ValueError('Changed diagnostic implementation/settings; use another run')
        else:
            m = dict(spec=spec, scientific_protocol=protocol, parent_manifest_sha256=parent_sha,
                     code_sha256=fingerprint, environment=old.environment(), created_at_utc=old.serial.now())
            old.atomic_json(manifest_path, m)
            old.atomic_bytes(OUT/'manifest.sha256', (old.serial.file_hash(manifest_path)+'\n').encode())
        sha = old.serial.file_hash(manifest_path)
        folder = OUT/backend
        folder.mkdir(exist_ok=True)
        status_path = OUT/f'{backend}_status.json'
        if (folder/(row['record_id']+'.json')).exists():
            old.checked(folder, row, sha)
            if status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete':
                print(f'Already complete: verified {backend} diagnostic', flush=True)
                return
        old.atomic_json(status_path, dict(status='running', started_at_utc=old.serial.now()))
        started = time.perf_counter()
        try:
            device = old.hardware() if backend == 'gpu' else None
            with ACSDataset(BASE/'data', target=protocol['target'], leads=protocol['input']['leads']) as dataset:
                if dataset.source_manifest != parent['source_manifest']:
                    raise ValueError('Changed source archives')
                record = dataset.load_record(row['record_id'])
                old.check_loaded(record, row)
            def attempt(rid, entry):
                with (folder/f'{rid}_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time_utc=old.serial.now(), **entry))+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if entry['lead'] == 'V5':
                    print(f'{backend} V5: limit={entry["limit"]}, iterations={entry["iterations"]}, capped={entry["capped"]}', flush=True)
            if backend == 'cpu':
                vectors, detail = extract_record(record, protocol, reference=True,
                                                on_attempt=lambda a: attempt(row['record_id'], a))
                detail['execution_origin'] = 'cpu_retry_extension_diagnostic'
            else:
                reference = old.checked(OUT/'cpu', row, sha)
                vectors, detail = next(extract_records([record], protocol, on_attempt=attempt,
                                                       reference_ids=[row['record_id']]))
                detail['cpu_comparison'] = compare_cpu(vectors, detail, *reference)
            old.save_feature(folder, row, vectors, detail, sha)
            summary = dict(status='complete', backend=backend, manifest_sha256=sha,
                           record_id=row['record_id'], hardware=device, details=detail,
                           elapsed_seconds=time.perf_counter()-started, completed_at_utc=old.serial.now(),
                           production_resumed=False, models_trained=0, official_test_processed=False)
            old.atomic_json(status_path, summary)
            print(json.dumps(summary, indent=2), flush=True)
        except BaseException:
            old.atomic_json(status_path, dict(status='failed', elapsed_seconds=time.perf_counter()-started,
                                             traceback=traceback.format_exc(), time_utc=old.serial.now()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=('cpu','gpu'))
    with threadpool_limits(limits=1):
        run(parser.parse_args().backend)
