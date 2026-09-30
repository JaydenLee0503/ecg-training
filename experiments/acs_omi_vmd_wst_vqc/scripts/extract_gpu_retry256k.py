"""Separate 256k retry run; preserve and verify the frozen 128k parent."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu_retry as parent_runner
from experiments.acs_omi_vmd_wst_vqc.gpu_retry import inventory_entry,copy_checkpoint
from experiments.acs_omi_vmd_wst_vqc.gpu_extraction import extract_records,compare_cpu
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset,DEFAULT_DATA_DIR
from threadpoolctl import threadpool_limits

old=parent_runner.old
BASE=old.BASE
ROOT=old.ROOT
SOURCE=BASE/'results/omi_v1_gpu_retry128k'
DIAGNOSTIC=BASE/'results/vmd_04124_gpu_check_v1'
SPEC=BASE/'protocols/gpu_extraction_retry256k_v1.json'
PROTOCOL=BASE/'protocols/acs_omi_protocol_v1_retry256k.json'
CODE=sorted(set(parent_runner.CODE+[str((BASE/name).relative_to(ROOT)) for name in (
    'scripts/extract_gpu_retry256k.py','scripts/verify_04124_gpu.py','scripts/diagnose_04124.py',
    'vmd_convergence_diagnostic.py','tests/test_convergence_diagnostic.py','tests/test_retry256k.py',
    'protocols/gpu_extraction_retry256k_v1.json','protocols/acs_omi_protocol_v1_retry256k.json',
    'protocols/vmd_04124_diagnostic_v1.json')]))


def assert_extension(parent,candidate):
    normalized=json.loads(json.dumps(candidate))
    normalized['id']=parent['id']
    normalized['vmd']['iteration_limits']=parent['vmd']['iteration_limits']
    if (parent['id']!='acs-omi-v1-retry128k' or candidate['id']!='acs-omi-v1-retry256k'
            or candidate['vmd']['iteration_limits']!=parent['vmd']['iteration_limits']+[256000]
            or normalized!=parent):
        raise ValueError('Only the declared 256000 retry and protocol ID may change')


def context():
    parent,rows,parent_sha=parent_runner.read_run(SOURCE)
    spec=json.loads(SPEC.read_text());protocol=json.loads(PROTOCOL.read_text())
    assert_extension(parent['scientific_protocol'],protocol)
    if (parent_sha!=spec['parent_manifest_sha256']
            or old.serial.file_hash(SOURCE/'extraction_status.json')!=spec['parent_stop_sha256']):
        raise ValueError('Changed parent run or stop')
    if (old.serial.file_hash(DIAGNOSTIC/'manifest.json')!=spec['gpu_diagnostic_manifest_sha256']
            or old.serial.file_hash(DIAGNOSTIC/'summary.json')!=spec['gpu_diagnostic_summary_sha256']):
        raise ValueError('Changed GPU convergence diagnostic')
    diagnostic=json.loads((DIAGNOSTIC/'manifest.json').read_text())
    result=json.loads((DIAGNOSTIC/'summary.json').read_text())
    if (result['status']!='complete' or not result['passed'] or result['iterations']>=256000
            or diagnostic['max_iter']!=256000 or diagnostic['hardware']!=spec['hardware']
            or diagnostic['environment']!=old.environment()
            or old.serial.file_hash(DIAGNOSTIC/'result.npz')!=result['result_sha256']
            or old.serial.file_hash(BASE/'scripts/verify_04124_gpu.py')!=diagnostic['probe_sha256']):
        raise ValueError('Invalid diagnostic result or device context')
    cpu=BASE/'results/vmd_04124_diagnostic_v1'
    if old.serial.file_hash(cpu/'summary.json')!=diagnostic['cpu_summary_sha256']:
        raise ValueError('Changed CPU convergence evidence')
    cpu_manifest=json.loads((cpu/'manifest.json').read_text())
    for name,digest in cpu_manifest['code_sha256'].items():
        if old.serial.file_hash(ROOT/name)!=digest:
            raise ValueError('Changed convergence diagnostic implementation')
    for name,digest in json.loads((cpu/'summary.json').read_text())['artifacts'].items():
        if old.serial.file_hash(cpu/name)!=digest:
            raise ValueError('Changed CPU diagnostic artifact')
    return spec,protocol,parent,rows


def read_run(out,*,prepared=True):
    spec,protocol,parent,rows=context()
    sha=old.serial.file_hash(out/'manifest.json')
    if sha!=(out/'manifest.sha256').read_text().strip():
        raise ValueError('Changed amended manifest')
    m=json.loads((out/'manifest.json').read_text())
    if (m['scientific_protocol']!=protocol or m['execution_protocol']!=spec
            or m['environment']!=old.environment() or m['hardware']!=spec['hardware']
            or m['code_sha256']!={name:old.serial.file_hash(ROOT/name) for name in CODE}
            or m['source_manifest']!=parent['source_manifest']
            or m['pilot_record_ids']!=parent['pilot_record_ids']):
        raise ValueError('Changed run context')
    for filename,key in [('splits.csv','splits_sha256'),('source_records.csv','source_records_sha256')]:
        if m[key]!=parent[key] or old.serial.file_hash(out/filename)!=parent[key]:
            raise ValueError('Changed split/source ledger')
    if json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('Changed saved scientific protocol')
    if prepared:
        preparation=json.loads((out/'preparation.json').read_text())
        if preparation['status']!='complete' or preparation['manifest_sha256']!=sha:
            raise ValueError('Incomplete preparation')
    return m,rows,sha


def validate_pilot(out,m,rows,sha):
    marker=json.loads((out/'pilot_status.json').read_text())
    if marker['status']!='complete' or marker['manifest_sha256']!=sha:
        raise ValueError('Missing pilot validation marker')
    lookup={r['record_id']:r for r in rows}
    for rid,entry in m['pilot_checkpoints'].items():
        values,meta=old.checked(out/'pilot',lookup[rid],sha)
        if meta['features_sha256']!=entry['files'][f'{rid}.npz']:
            raise ValueError('Changed imported validation features')
        compare_cpu(values,meta,*old.checked(SOURCE/'pilot',lookup[rid],entry['source_manifest_sha256']))


def prepare(args):
    spec,protocol,parent,rows=context()
    args.out.mkdir(parents=True,exist_ok=True)
    lookup={r['record_id']:r for r in rows}
    with old.run_lock(args.out),(SOURCE/'.run.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.out/'manifest.json').exists():
            m,rows,sha=read_run(args.out,prepared=False)
        else:
            inventory={}
            for row in rows:
                if (SOURCE/'features'/f'{row["record_id"]}.json').exists():
                    inventory[row['record_id']]=inventory_entry(SOURCE/'features',row,
                        spec['parent_manifest_sha256'],parent['scientific_protocol'])
                    if len(inventory)%500==0:
                        print(f'Verified {len(inventory)} parent checkpoints',flush=True)
            if len(inventory)!=spec['expected_parent_records']:
                raise ValueError('Unexpected parent checkpoint count')
            pilot={rid:inventory_entry(SOURCE/'pilot',lookup[rid],spec['parent_manifest_sha256'],
                       parent['scientific_protocol']) for rid in parent['pilot_record_ids']}
            m=dict(scientific_protocol=protocol,execution_protocol=spec,source_manifest=parent['source_manifest'],
                environment=old.environment(),hardware=spec['hardware'],code_sha256={n:old.serial.file_hash(ROOT/n) for n in CODE},
                imported_checkpoints=inventory,pilot_checkpoints=pilot,pilot_record_ids=parent['pilot_record_ids'],
                splits_sha256=parent['splits_sha256'],source_records_sha256=parent['source_records_sha256'],
                counts=old.serial.counts(rows),created_at_utc=old.serial.now(),models_trained=0,official_test_processed=False)
            for filename in ('splits.csv','source_records.csv'):
                old.atomic_bytes(args.out/filename,(SOURCE/filename).read_bytes())
            old.atomic_json(args.out/'protocol.json',protocol)
            old.atomic_json(args.out/'manifest.json',m)
            sha=old.serial.file_hash(args.out/'manifest.json')
            old.atomic_bytes(args.out/'manifest.sha256',(sha+'\n').encode())
        for name in ('features','pilot'):
            folder=args.out/name;folder.mkdir(exist_ok=True)
            entries=m['imported_checkpoints'] if name=='features' else m['pilot_checkpoints']
            for i,(rid,entry) in enumerate(entries.items(),1):
                copy_checkpoint(SOURCE/name,folder,lookup[rid],entry,sha,f'omi_v1_gpu_retry128k/{name}')
                if i%500==0:
                    print(f'Imported {i} {name} checkpoints',flush=True)
        old.atomic_json(args.out/'pilot_status.json',dict(status='complete',manifest_sha256=sha,
            records=len(m['pilot_record_ids']),gpu_diagnostic_manifest_sha256=spec['gpu_diagnostic_manifest_sha256']))
        validate_pilot(args.out,m,rows,sha)
        old.atomic_json(args.out/'preparation.json',dict(status='complete',manifest_sha256=sha,
            records=len(rows),imported_parent_records=len(m['imported_checkpoints']),completed_at_utc=old.serial.now()))
        print(f'Prepared {len(rows)} ECGs; reused {len(m["imported_checkpoints"])} checkpoints',flush=True)


def verify(args):
    m,rows,sha=read_run(args.out)
    with old.run_lock(args.out):
        validate_pilot(args.out,m,rows,sha)
        pending,done=old.scan_checkpoints(args.out/'features',rows,sha)
        for rid,entry in m['imported_checkpoints'].items():
            if old.serial.file_hash(args.out/'features'/f'{rid}.npz')!=entry['files'][f'{rid}.npz']:
                raise ValueError('Changed imported feature bytes')
        result=dict(status='passed',records_verified=len(done),records_pending=len(pending),
                    reused_features_unchanged=True,manifest_sha256=sha,verified_at_utc=old.serial.now())
        old.atomic_json(args.out/'verification.json',result)
        print(json.dumps(result),flush=True)


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
    parser.add_argument('--out',type=Path,default=BASE/'results/omi_v1_gpu_retry256k')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--max-batches',type=int)
    args=parser.parse_args()
    if args.max_batches is not None and args.max_batches<1:
        parser.error('--max-batches must be positive')
    with threadpool_limits(limits=1):
        {'prepare':prepare,'verify':verify,'extract':run}[args.command](args)


if __name__=='__main__':
    main()
