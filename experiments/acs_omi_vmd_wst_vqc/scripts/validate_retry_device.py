"""Revalidate saved fit-record features on the current GPU before a new run."""
import json
from pathlib import Path
import time
import traceback

from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as old
from experiments.acs_omi_vmd_wst_vqc.gpu_extraction import extract_records, compare_cpu
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'results/retry_device_validation_20260927'


def main():
    parent, rows, parent_sha = old.read_run(BASE/'results/omi_v1_gpu')
    diagnostic = BASE/'results/vmd_retry_extension_diagnostic_v1'
    diagnostic_sha = old.serial.file_hash(diagnostic/'manifest.json')
    protocol = json.loads((BASE/'protocols/acs_omi_protocol_v1_retry128k.json').read_text())
    ids = parent['pilot_record_ids'] + ['01985']
    lookup = {r['record_id']: r for r in rows}
    OUT.mkdir(exist_ok=True)
    with old.run_lock(OUT):
        manifest = dict(parent_manifest_sha256=parent_sha, diagnostic_manifest_sha256=diagnostic_sha,
                        protocol=protocol, hardware=old.hardware(), environment=old.environment(),
                        records=ids, code_sha256={n: old.serial.file_hash(old.ROOT/n) for n in
                        old.CODE+[str(Path(__file__).resolve().relative_to(old.ROOT))]})
        if (OUT/'manifest.json').exists():
            if json.loads((OUT/'manifest.json').read_text()) != manifest:
                raise ValueError('Validation context changed; use a separate validation run')
        else:
            old.atomic_json(OUT/'manifest.json',manifest)
        sha = old.serial.file_hash(OUT/'manifest.json')
        if (OUT/'status.json').exists():
            status = json.loads((OUT/'status.json').read_text())
            if status['status']=='complete':
                for rid in ids:
                    old.checked(OUT,lookup[rid],sha)
                print('Verified complete device validation',flush=True)
                return
        started = time.perf_counter()
        try:
            with ACSDataset(BASE/'data',target=protocol['target'],leads=protocol['input']['leads']) as ds:
                if ds.source_manifest != parent['source_manifest']:
                    raise ValueError('Source archives changed')
                for batch_ids in (ids[:16],ids[16:]):
                    records=[]
                    for rid in batch_ids:
                        if lookup[rid]['partition']!='fit':
                            raise ValueError('Device validation accepts fit patients only')
                        record=ds.load_record(rid)
                        old.check_loaded(record,lookup[rid])
                        records.append(record)
                    def attempt(rid,entry):
                        with (OUT/f'{rid}_attempts.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(time_utc=old.serial.now(),**entry))+'\n')
                    for rid,(vectors,details) in zip(batch_ids,extract_records(records,protocol,
                            on_attempt=attempt,reference_ids=batch_ids),strict=True):
                        reference = old.checked(diagnostic/'cpu',lookup[rid],diagnostic_sha) if rid=='01985' else old.checked(
                            BASE/'results/omi_v1_gpu/pilot',lookup[rid],parent_sha)
                        details['saved_reference_comparison']=compare_cpu(vectors,details,*reference)
                        old.save_feature(OUT,lookup[rid],vectors,details,sha)
                        print(f'Current-device validation passed: {rid}',flush=True)
            old.atomic_json(OUT/'status.json',dict(status='complete',manifest_sha256=sha,
                hardware=manifest['hardware'],records=len(ids),elapsed_seconds=time.perf_counter()-started,
                completed_at_utc=old.serial.now(),official_test_processed=False,models_trained=0))
        except BaseException:
            old.atomic_json(OUT/'status.json',dict(status='failed',traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
