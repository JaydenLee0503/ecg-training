"""Bounded fit-lead CPU diagnosis; no production changes, exclusions or fitting."""
from collections import deque
import json
from pathlib import Path
import time
import traceback

from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as old
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, LEADS
from experiments.acs_omi_vmd_wst_vqc.vmd_convergence_diagnostic import traced_vmd_batch
from ecgvmd.vmd import vmd_batch
import numpy as np
from threadpoolctl import threadpool_limits

BASE=old.BASE
OUT=BASE/'results/vmd_04124_diagnostic_v1'
SOURCE=BASE/'results/omi_v1_gpu_retry128k'
SPEC=BASE/'protocols/vmd_04124_diagnostic_v1.json'


def save_arrays(path,**arrays):
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    path.with_suffix('.tmp').replace(path)


def main():
    spec=json.loads(SPEC.read_text())
    parent=json.loads((SOURCE/'manifest.json').read_text())
    if old.serial.file_hash(SOURCE/'manifest.json')!=spec['parent_manifest_sha256']:
        raise ValueError('Changed source manifest')
    for name,digest in parent['code_sha256'].items():
        if old.serial.file_hash(old.ROOT/name)!=digest:
            raise ValueError(f'Changed production source: {name}')
    if old.serial.file_hash(SOURCE/'splits.csv')!=parent['splits_sha256']:
        raise ValueError('Changed source split')
    rows=old.serial.read_csv(SOURCE/'splits.csv')
    row=next(r for r in rows if r['record_id']==spec['record_id'])
    if row['partition']!='fit':
        raise ValueError('Diagnosis must use a fit record')
    paths=[SPEC,Path(__file__).resolve(),BASE/'vmd_convergence_diagnostic.py',BASE/'tests/test_convergence_diagnostic.py']
    manifest=dict(spec=spec,environment=old.environment(),
        code_sha256={str(p.relative_to(old.ROOT)):old.serial.file_hash(p) for p in paths},
        production_code_sha256=parent['code_sha256'])
    OUT.mkdir(exist_ok=True)
    with old.run_lock(OUT):
        if (OUT/'manifest.json').exists():
            if json.loads((OUT/'manifest.json').read_text())!=manifest:
                raise ValueError('Changed diagnostic context; use a new output')
        else:
            old.atomic_json(OUT/'manifest.json',manifest)
        if (OUT/'summary.json').exists():
            summary=json.loads((OUT/'summary.json').read_text())
            for name,digest in summary['artifacts'].items():
                if old.serial.file_hash(OUT/name)!=digest:
                    raise ValueError('Changed completed diagnostic')
            print(json.dumps(summary),flush=True)
            return
        old.atomic_json(OUT/'status.json',dict(status='running',started_at_utc=old.serial.now()))
        try:
            with ACSDataset(BASE/'data',target=parent['scientific_protocol']['target'],
                            leads=parent['scientific_protocol']['input']['leads']) as ds:
                if ds.source_manifest!=parent['source_manifest']:
                    raise ValueError('Changed data archives')
                record=ds.load_record(row['record_id'])
                old.check_loaded(record,row)
            x=np.asarray(record.signal_mV[:,LEADS.index(spec['lead'])],dtype=float)[None,:]
            settings={k:parent['scientific_protocol']['vmd'][k] for k in ('K','alpha','tau','dc','init','tol')}
            settings['fs']=record.fs
            amplitude=dict(min_mV=float(x.min()),max_mV=float(x.max()),mean_mV=float(x.mean()),
                std_mV=float(x.std()),peak_to_peak_mV=float(np.ptp(x)),finite=bool(np.isfinite(x).all()))
            print('CPU reproduction at 128000; unchanged production settings',flush=True)
            before=time.perf_counter()
            ref=vmd_batch(x,max_iter=spec['reproduction_cap'],**settings)
            reproduction=dict(iterations=int(ref.iters[0]),capped=bool(ref.capped[0]),seconds=time.perf_counter()-before,
                finite_modes=bool(np.isfinite(ref.modes).all()),finite_centres=bool(np.isfinite(ref.omega).all()))
            save_arrays(OUT/'cpu_128000.npz',modes=ref.modes,omega_hz=ref.omega_hz,iters=ref.iters,signal=x)
            old.atomic_json(OUT/'cpu_reproduction.json',reproduction)
            print(json.dumps(reproduction),flush=True)
            samples=[];tail=deque(maxlen=4096)
            def observe(n,d,o):
                value=float(d[0]);tail.append((n,value))
                if n==1 or n%100==0 or value<=settings['tol']:
                    samples.append((n,value,*o[0].tolist()))
                if n%32000==0 or value<=settings['tol']:
                    print(f'CPU trace iteration {n}: stopping statistic={value:.12g}; tolerance={settings["tol"]}',flush=True)
                    old.atomic_json(OUT/'progress.json',dict(iteration=n,stopping_statistic=value,tolerance=settings['tol'],
                                                           omega_hz=(o[0]*record.fs).tolist(),time_utc=old.serial.now()))
            before=time.perf_counter()
            result=traced_vmd_batch(x,max_iter=spec['diagnostic_cap'],on_iteration=observe,**settings)
            duration=time.perf_counter()-before
            save_arrays(OUT/'trace.npz',sampled_trace=np.asarray(samples),last_4096=np.asarray(tail),
                modes=result.modes,omega_hz=result.omega_hz,iters=result.iters)
            last=np.asarray(tail)
            summary=dict(status='complete',record_id=row['record_id'],lead=spec['lead'],partition='fit',
                amplitude=amplitude,cpu_reproduction=reproduction,
                diagnostic=dict(iterations=int(result.iters[0]),capped=bool(result.capped[0]),seconds=duration,
                    finite_modes=bool(np.isfinite(result.modes).all()),finite_centres=bool(np.isfinite(result.omega).all()),
                    final_stopping_statistic=float(last[-1,1]),tail_min=float(last[:,1].min()),
                    tail_max=float(last[:,1].max()),omega_hz=result.omega_hz[0].tolist(),
                    residual_energy_fraction=float(result.residual_energy_fraction(x)[0])),
                settings=settings,diagnostic_cap=spec['diagnostic_cap'],waveform_sha256=row['waveform_sha256'],
                manifest_sha256=old.serial.file_hash(OUT/'manifest.json'),
                artifacts={n:old.serial.file_hash(OUT/n) for n in ('cpu_128000.npz','trace.npz','cpu_reproduction.json')},
                production_resumed=False,production_settings_changed=False,models_trained=0,
                official_test_processed=False,completed_at_utc=old.serial.now())
            old.atomic_json(OUT/'summary.json',summary)
            old.atomic_json(OUT/'status.json',dict(status='complete',completed_at_utc=old.serial.now()))
            print(json.dumps(summary,indent=2),flush=True)
        except BaseException:
            old.atomic_json(OUT/'status.json',dict(status='failed',traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
