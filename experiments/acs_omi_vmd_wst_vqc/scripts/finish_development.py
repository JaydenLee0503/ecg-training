"""Finish the authorized fixed ACS comparison after successful extraction.

Stops on extraction failure/interruption or any downstream error. Does not start
GPU extraction, alter protocols, tune models, or access official test waveforms.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.acs_omi_vmd_wst_vqc.scripts import train_models as training

BASE=training.BASE
ROOT=training.ROOT
OUT=BASE/'results/omi_development_workflow_v1'
SOURCE=BASE/'results/omi_v1_gpu_retry128k'
MODELS=BASE/'results/omi_models_v1_retry128k'
old=training.old


def main():
    OUT.mkdir(exist_ok=True)
    with old.run_lock(OUT):
        manifest=dict(source_manifest_sha256=old.serial.file_hash(SOURCE/'manifest.json'),
            training_code_sha256=training.code_hashes(),environment=old.environment(),jobs=3,
            workflow_sha256=old.serial.file_hash(Path(__file__)),official_test_processed=False)
        if (OUT/'manifest.json').exists():
            if json.loads((OUT/'manifest.json').read_text())!=manifest:
                raise ValueError('Changed workflow context; use a new workflow run')
        else:
            old.atomic_json(OUT/'manifest.json',manifest)
        def status(stage,**extra):
            old.atomic_json(OUT/'status.json',dict(stage=stage,updated_at_utc=old.serial.now(),**extra))
        try:
            while True:
                extraction=json.loads((SOURCE/'extraction_status.json').read_text())
                if extraction['manifest_sha256']!=manifest['source_manifest_sha256']:
                    raise ValueError('Extraction manifest changed')
                if extraction['status']=='complete':
                    break
                if extraction['status']!='running':
                    raise RuntimeError(f'Extraction stopped: {extraction["status"]}; inspect its logs before resuming')
                status('waiting_for_extraction',records_complete=extraction['records_complete'],
                       records_total=extraction['records_total'])
                time.sleep(30)
            stages=[('bundle','bundle_gpu',[]),('prepare_models','train_models',['prepare']),
                    ('train_models','train_models',['train','--jobs','3']),('report','report_models',[])]
            for stage,module,args in stages:
                if training.code_hashes()!=manifest['training_code_sha256']:
                    raise ValueError('Downstream implementation changed after workflow was declared')
                status(stage,status='running')
                print(f'START {stage}',flush=True)
                with (OUT/f'{stage}.log').open('a') as log:
                    result=subprocess.run([sys.executable,'-B','-m',
                        f'experiments.acs_omi_vmd_wst_vqc.scripts.{module}',*args],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{stage} failed with exit {result.returncode}; see {stage}.log')
                print(f'DONE {stage}',flush=True)
            status('complete',status='complete',report=str(MODELS/'report.md'),official_test_processed=False)
        except BaseException:
            failure=dict(status='failed',traceback=traceback.format_exc())
            status('stopped',**failure)
            old.atomic_json(OUT/f'failure_{time.time_ns()}.json',failure)
            raise


if __name__=='__main__':
    main()
