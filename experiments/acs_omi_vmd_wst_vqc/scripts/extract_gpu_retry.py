"""Resume ACS extraction in a new run with validated, bounded extra VMD retries.

The original run and all numerical implementations remain frozen. This runner
imports verified original checkpoints and the validated extension diagnostic.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as old
from experiments.acs_omi_vmd_wst_vqc.gpu_retry import assert_compatible_extension, inventory_entry, copy_checkpoint
from experiments.acs_omi_vmd_wst_vqc.gpu_extraction import extract_records, compare_cpu
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, DEFAULT_DATA_DIR
from threadpoolctl import threadpool_limits

SOURCE = BASE/'results/omi_v1_gpu'
DIAGNOSTIC = BASE/'results/vmd_retry_extension_diagnostic_v1'
SPEC = BASE/'protocols/gpu_extraction_retry128k_v1.json'
PROTOCOL = BASE/'protocols/acs_omi_protocol_v1_retry128k.json'
CODE = sorted(set(old.CODE + [str((BASE/name).relative_to(ROOT)) for name in (
    'scripts/extract_gpu_retry.py', 'gpu_retry.py', 'tests/test_gpu_retry.py',
    'scripts/diagnose_vmd_retry.py', 'protocols/vmd_retry_extension_diagnostic_v1.json',
    'protocols/gpu_extraction_retry128k_v1.json', 'protocols/acs_omi_protocol_v1_retry128k.json')]))


def context():
    parent, rows, parent_sha = old.read_run(SOURCE)
    spec = json.loads(SPEC.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert_compatible_extension(parent['scientific_protocol'],protocol)
    if (parent_sha != spec['parent_manifest_sha256']
            or old.serial.file_hash(SOURCE/'extraction_status.json') != spec['parent_stop_sha256']):
        raise ValueError('Changed original run or stop record')
    diagnostic_sha = old.serial.file_hash(DIAGNOSTIC/'manifest.json')
    if (diagnostic_sha != spec['diagnostic_manifest_sha256']
            or diagnostic_sha != (DIAGNOSTIC/'manifest.sha256').read_text().strip()):
        raise ValueError('Changed diagnostic manifest')
    diagnostic = json.loads((DIAGNOSTIC/'manifest.json').read_text())
    if (diagnostic['scientific_protocol'] != protocol or diagnostic['parent_manifest_sha256'] != parent_sha
            or diagnostic['environment'] != old.environment()):
        raise ValueError('Diagnostic settings/environment differ')
    for name, expected in diagnostic['code_sha256'].items():
        if old.serial.file_hash(ROOT/name) != expected:
            raise ValueError('Changed diagnostic code')
    row = next(r for r in rows if r['record_id'] == spec['diagnostic_record_id'])
    if row['partition'] != 'fit':
        raise ValueError('Retry diagnostic must be a fit record')
    for backend in ('cpu','gpu'):
        status = json.loads((DIAGNOSTIC/f'{backend}_status.json').read_text())
        if status['status'] != 'complete' or status['manifest_sha256'] != diagnostic_sha:
            raise ValueError('CPU and GPU retry diagnostics must complete')
    cpu = old.checked(DIAGNOSTIC/'cpu',row,diagnostic_sha)
    gpu = old.checked(DIAGNOSTIC/'gpu',row,diagnostic_sha)
    compare_cpu(*gpu,*cpu)
    if cpu[1]['reference_max_abs_error'] is None or gpu[1]['reference_max_abs_error'] is None:
        raise ValueError('Missing Kymatio reference checks')
    return spec, protocol, parent, rows, diagnostic_sha


def read_run(out, *, prepared=True):
    spec, protocol, parent, rows, diagnostic_sha = context()
    sha = old.serial.file_hash(out/'manifest.json')
    if sha != (out/'manifest.sha256').read_text().strip():
        raise ValueError('Changed amended-run manifest')
    m = json.loads((out/'manifest.json').read_text())
    if (m['execution_protocol'] != spec or m['scientific_protocol'] != protocol
            or m['source_manifest'] != parent['source_manifest'] or m['environment'] != old.environment()
            or m['hardware'] != parent['hardware']
            or m['pilot_record_ids'] != parent['pilot_record_ids']+[spec['diagnostic_record_id']]
            or m['code_sha256'] != {name:old.serial.file_hash(ROOT/name) for name in CODE}):
        raise ValueError('Changed implementation/settings/environment; use a new run')
    for filename,key in (('splits.csv','splits_sha256'),('source_records.csv','source_records_sha256')):
        if m[key] != parent[key] or old.serial.file_hash(out/filename) != parent[key]:
            raise ValueError('Changed patient split/source ledger')
    if json.loads((out/'protocol.json').read_text()) != protocol:
        raise ValueError('Changed saved scientific protocol')
    if prepared and json.loads((out/'preparation.json').read_text())['status'] != 'complete':
        raise ValueError('Checkpoint imports must complete')
    return m, rows, sha


def prepare(args):
    spec, protocol, parent, rows, diagnostic_sha = context()
    args.out.mkdir(parents=True,exist_ok=True)
    with old.run_lock(args.out), (SOURCE/'.run.lock').open('r') as source_lock:
        fcntl.flock(source_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lookup = {r['record_id']:r for r in rows}
        if (args.out/'manifest.json').exists():
            m,rows,sha = read_run(args.out,prepared=False)
        else:
            checkpoint_inventory = {}
            for row in rows:
                rid = row['record_id']
                if (SOURCE/'features'/f'{rid}.json').exists():
                    checkpoint_inventory[rid] = inventory_entry(SOURCE/'features',row,
                        spec['parent_manifest_sha256'],parent['scientific_protocol'])
                    if len(checkpoint_inventory)%200 == 0:
                        print(f'Verified {len(checkpoint_inventory)} source checkpoints',flush=True)
            if len(checkpoint_inventory) != spec['expected_parent_records']:
                raise ValueError('Unexpected source checkpoint coverage')
            pilot_inventory = {rid: inventory_entry(SOURCE/'pilot',lookup[rid],
                spec['parent_manifest_sha256'],parent['scientific_protocol']) for rid in parent['pilot_record_ids']}
            rid = spec['diagnostic_record_id']
            pilot_inventory[rid] = inventory_entry(DIAGNOSTIC/'gpu',lookup[rid],diagnostic_sha,protocol)
            if rid in checkpoint_inventory:
                raise ValueError('Diagnostic unexpectedly overlaps completed original features')
            m = dict(scientific_protocol=protocol, execution_protocol=spec, environment=old.environment(),
                     hardware=parent['hardware'],source_manifest=parent['source_manifest'],
                     splits_sha256=parent['splits_sha256'],source_records_sha256=parent['source_records_sha256'],
                     code_sha256={name:old.serial.file_hash(ROOT/name) for name in CODE},
                     imported_checkpoints=checkpoint_inventory,pilot_checkpoints=pilot_inventory,
                     pilot_record_ids=parent['pilot_record_ids']+[rid],created_at_utc=old.serial.now(),
                     counts=old.serial.counts(rows),models_trained=0,official_test_processed=False)
            for filename in ('splits.csv','source_records.csv'):
                old.atomic_bytes(args.out/filename,(SOURCE/filename).read_bytes())
            old.atomic_json(args.out/'protocol.json',protocol)
            old.atomic_json(args.out/'manifest.json',m)
            sha = old.serial.file_hash(args.out/'manifest.json')
            old.atomic_bytes(args.out/'manifest.sha256',(sha+'\n').encode())
        for name in ('features','pilot'):
            (args.out/name).mkdir(exist_ok=True)
        for count,(rid,entry) in enumerate(m['imported_checkpoints'].items(),1):
            copy_checkpoint(SOURCE/'features',args.out/'features',lookup[rid],entry,sha,'omi_v1_gpu/features')
            if count%200 == 0:
                print(f'Imported {count} checkpoints without recomputation',flush=True)
        for rid,entry in m['pilot_checkpoints'].items():
            extension = rid == spec['diagnostic_record_id']
            source = DIAGNOSTIC/'gpu' if extension else SOURCE/'pilot'
            origin = 'vmd_retry_extension_diagnostic_v1/gpu' if extension else 'omi_v1_gpu/pilot'
            copy_checkpoint(source,args.out/'pilot',lookup[rid],entry,sha,origin)
            if extension:
                copy_checkpoint(source,args.out/'features',lookup[rid],entry,sha,origin)
        old.atomic_json(args.out/'pilot_status.json',dict(status='complete',manifest_sha256=sha,
            records=17,original_pilot_records_reused=16,extension_diagnostic_record=spec['diagnostic_record_id'],
            method='Reuse independently verified pilot/CPU-GPU diagnostic artifacts; no pilot recomputation',
            completed_at_utc=old.serial.now()))
        old.atomic_json(args.out/'preparation.json',dict(status='complete',manifest_sha256=sha,
            records=len(rows),imported_parent_records=len(m['imported_checkpoints']),imported_diagnostic_records=1,
            completed_at_utc=old.serial.now()))
        validate_pilot(args.out,m,rows,sha)
        print(f'Prepared {len(rows)} ECGs; reused {len(m["imported_checkpoints"])} parent records plus the corrected diagnostic ECG',flush=True)


def validate_pilot(out,m,rows,sha):
    status = json.loads((out/'pilot_status.json').read_text())
    if status['status'] != 'complete' or status['manifest_sha256'] != sha:
        raise ValueError('Missing amended-run validation gate')
    lookup = {r['record_id']:r for r in rows}
    for rid,entry in m['pilot_checkpoints'].items():
        vectors,meta = old.checked(out/'pilot',lookup[rid],sha)
        if meta['features_sha256'] != entry['files'][f'{rid}.npz']:
            raise ValueError('Changed inherited validation feature bytes')
        extension = rid == m['execution_protocol']['diagnostic_record_id']
        source = DIAGNOSTIC/'cpu' if extension else SOURCE/'pilot'
        source_sha = entry['source_manifest_sha256']
        reference = old.checked(source,lookup[rid],source_sha)
        compare_cpu(vectors,meta,*reference)


def verify(args):
    m,rows,sha = read_run(args.out)
    with old.run_lock(args.out):
        validate_pilot(args.out,m,rows,sha)
        pending,done = old.scan_checkpoints(args.out/'features',rows,sha)
        for rid,entry in m['imported_checkpoints'].items():
            if old.serial.file_hash(args.out/'features'/f'{rid}.npz') != entry['files'][f'{rid}.npz']:
                raise ValueError('Changed reused feature bytes')
        summary = dict(status='passed',records_verified=len(done),records_pending=len(pending),
                       reused_features_unchanged=True,manifest_sha256=sha,verified_at_utc=old.serial.now())
        old.atomic_json(args.out/'verification.json',summary)
        print(json.dumps(summary),flush=True)


def run(args):
    m,rows,sha = read_run(args.out)
    validate_pilot(args.out,m,rows,sha)
    folder = args.out/'features'
    status_path = args.out/'extraction_status.json'
    with old.run_lock(args.out), old.graceful_stop() as stop:
        pending,done = old.scan_checkpoints(folder,rows,sha)
        if not pending and status_path.exists() and json.loads(status_path.read_text())['status']=='complete':
            print(f'Already complete: {len(rows)} verified records; no GPU/waveform work',flush=True)
            return
        started,started_at = time.perf_counter(),old.serial.now()
        resumed,batches,active = len(done),0,[]
        def progress(status='running',**extra):
            elapsed = time.perf_counter()-started
            new = len(done)-resumed
            result = dict(status=status,stage='extract',pid=os.getpid(),started_at_utc=started_at,
                updated_at_utc=old.serial.now(),manifest_sha256=sha,records_total=len(rows),
                records_complete=len(done),records_resumed=resumed,records_new=new,
                elapsed_seconds=elapsed,records_per_hour=3600*new/elapsed if elapsed else 0,
                active_record_ids=active,models_trained=0,official_test_processed=False,**extra)
            old.atomic_json(status_path,result)
            return result
        progress()
        try:
            if pending:
                if old.hardware()!=m['hardware']:
                    raise ValueError('Changed GPU hardware/runtime')
                with ACSDataset(args.data_dir,target=m['scientific_protocol']['target'],
                                leads=m['scientific_protocol']['input']['leads']) as dataset:
                    if dataset.source_manifest!=m['source_manifest']:
                        raise ValueError('Changed source archives')
                    size = m['execution_protocol']['records_per_gpu_batch']
                    for start in range(0,len(pending),size):
                        if stop['requested'] or (args.max_batches is not None and batches>=args.max_batches):
                            break
                        selected = pending[start:start+size]
                        active = [r['record_id'] for r in selected]
                        progress()
                        batch_start = time.perf_counter()
                        records = []
                        for row in selected:
                            record = dataset.load_record(row['record_id'])
                            old.check_loaded(record,row)
                            records.append(record)
                        load_seconds = time.perf_counter()-batch_start
                        batch_id = f'{os.getpid()}-{time.time_ns()}'
                        def attempt(rid,entry):
                            with (folder/f'{rid}_attempts.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(time_utc=old.serial.now(),batch_id=batch_id,**entry))+'\n')
                                stream.flush()
                                os.fsync(stream.fileno())
                        results = extract_records(records,m['scientific_protocol'],on_attempt=attempt)
                        for row,(vectors,details) in zip(selected,results,strict=True):
                            details.update(batch_id=batch_id,batch_record_ids=active)
                            old.save_feature(folder,row,vectors,details,sha)
                            done.append(details)
                            progress()
                        elapsed = time.perf_counter()-batch_start
                        event = dict(time_utc=old.serial.now(),batch_id=batch_id,record_ids=active,
                            load_seconds=load_seconds,wall_seconds=elapsed,records_per_hour=3600*len(selected)/elapsed)
                        with (args.out/'extract_batches.jsonl').open('a') as stream:
                            stream.write(json.dumps(event)+'\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        batches+=1
                        active=[]
                        state=progress()
                        print(f'extract: {len(done)}/{len(rows)}; batch {elapsed:.2f}s; '
                              f'{state["records_per_hour"]:.1f} ECG/hour including startup',flush=True)
            state=progress('complete' if len(done)==len(rows) else 'interrupted',stop_signal=stop['signal'],
                           stopped_after_max_batches=args.max_batches)
            with (args.out/'sessions.jsonl').open('a') as stream:
                stream.write(json.dumps(state)+'\n')
            print(json.dumps(state),flush=True)
        except BaseException as exc:
            failure=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            old.atomic_json(args.out/f'failure_{time.time_ns()}.json',progress('failed',**failure))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','extract'))
    parser.add_argument('--out',type=Path,default=BASE/'results/omi_v1_gpu_retry128k')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--max-batches',type=int)
    args=parser.parse_args()
    if args.max_batches is not None and args.max_batches<1:
        parser.error('--max-batches must be positive')
    with threadpool_limits(limits=1):
        {'prepare':prepare,'verify':verify,'extract':run}[args.command](args)


if __name__=='__main__':
    main()
