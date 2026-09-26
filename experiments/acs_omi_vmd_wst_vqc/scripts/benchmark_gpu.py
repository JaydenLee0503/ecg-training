#!/usr/bin/env python3
"""Separate, resumable ACS VMD GPU engineering benchmark; no classifiers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
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

import numpy as np
from threadpoolctl import threadpool_limits
from experiments.acs_omi_vmd_wst_vqc.gpu_vmd import extract_vmd, vmd_gpu_batch
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, LEADS
from experiments.acs_omi_vmd_wst_vqc.parallel import check_loaded
from experiments.acs_omi_vmd_wst_vqc.scripts import extract_parallel as parallel, experiment as serial
from experiments.acs_omi_vmd_wst_vqc.layout import verify_known_migration

SPEC = BASE/'protocols/gpu_vmd_benchmark_v1.json'
CODE = ('experiments/acs_omi_vmd_wst_vqc/gpu_vmd.py', 'experiments/acs_omi_vmd_wst_vqc/scripts/benchmark_gpu.py',
        'experiments/acs_omi_vmd_wst_vqc/tests/test_gpu_vmd.py', 'experiments/acs_omi_vmd_wst_vqc/protocols/gpu_vmd_benchmark_v1.json',
        'experiments/acs_omi_vmd_wst_vqc/layout.py', 'experiments/acs_omi_vmd_wst_vqc/provenance/descriptive_rename_v1.json')


def cpu_worker_init():
    global limiter
    limiter = threadpool_limits(limits=1)


def cpu_worker(task):
    rid, signals, protocol = task
    result = extract_vmd(signals, protocol, backend='cpu', lead_batch=4)
    result.pop('modes')
    result.pop('omega_hz')
    return rid, result


def array_check(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, error='shape mismatch', actual=list(actual.shape), expected=list(expected.shape))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    delta = np.abs(actual.astype(float)-expected.astype(float))
    return dict(passed=finite and bool(np.all(delta <= atol+rtol*np.abs(expected))),
                exact=bool(np.array_equal(actual, expected)),
                max_abs_error=float(delta.max(initial=0)),
                violations=int(np.count_nonzero(delta > atol+rtol*np.abs(expected))),
                atol=atol, rtol=rtol)


def save_result(out, name, result, *, modes=False):
    arrays = {k: result[k] for k in ('features', 'names', 'iterations', 'limits')}
    if modes:
        arrays.update(modes=result['modes'], omega_hz=result['omega_hz'])
    np.savez_compressed(out/f'{name}.npz', **arrays)
    metadata = {k: v for k, v in result.items() if k not in arrays and k not in ('modes', 'omega_hz')}
    metadata['artifact_sha256'] = serial.file_hash(out/f'{name}.npz')
    serial.atomic_json(out/f'{name}.json', metadata)


def reference_checks(result, ids, references, spec):
    checks = []
    c = spec['checks']
    for i, rid in enumerate(ids):
        vectors, detail = references[rid]
        selection = slice(i*12, (i+1)*12)
        names = [f'{lead}:{n}' for lead in LEADS for n in result['names']]
        feature = array_check(result['features'][selection].reshape(-1), vectors['vmd'],
                              c['features_atol'], c['features_rtol'])
        check = dict(record_id=rid, features=feature,
                     names_equal=bool(np.array_equal(names, vectors['vmd_names'])),
                     iterations_equal=bool(np.array_equal(result['iterations'][selection], detail['vmd_iterations'])),
                     limits_equal=bool(np.array_equal(result['limits'][selection], detail['vmd_limits'])))
        check['passed'] = feature['passed'] and all(check[k] for k in ('names_equal', 'iterations_equal', 'limits_equal'))
        checks.append(check)
    return checks


def run(args):
    spec = json.loads(SPEC.read_text())
    source = BASE/'results/omi_v1_parallel'
    parent_args = SimpleNamespace(parent=BASE/'results/omi_v1', out=source, workers=8)
    manifest, rows, parent_sha = parallel.read_parallel(parent_args)
    if parent_sha != spec['parent_parallel_manifest_sha256']:
        raise ValueError('Unexpected parent run')
    fingerprint = {name: serial.file_hash(ROOT/name) for name in CODE}
    if args.out.exists():
        old = json.loads((args.out/'manifest.json').read_text())
        status = json.loads((args.out/'status.json').read_text())
        code_ok = old['code_sha256'] == fingerprint or verify_known_migration(
            old, serial.file_hash(args.out/'manifest.json'))
        if code_ok and status['status'] == 'complete':
            summary = json.loads((args.out/'summary.json').read_text())
            for item in summary['artifacts']:
                if serial.file_hash(args.out/item['path']) != item['sha256']:
                    raise ValueError('Completed benchmark artifact changed')
            print('Already complete; verified saved benchmark without recomputation', flush=True)
            print(json.dumps(summary['timing_summary'], indent=2), flush=True)
            return
        raise ValueError('Existing incomplete/changed benchmark retained; use a new output directory')
    # Read-only open; lock excludes a competing production writer without changing it.
    with (source/'.run.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.out.mkdir(parents=True, exist_ok=False)
        os.environ['CUPY_CACHE_DIR'] = str(args.out/'cuda_cache')
        def event(stage, **kw):
            entry=dict(time_utc=serial.now(), stage=stage, **kw)
            with (args.out/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(entry)+'\n')
            print(json.dumps(entry), flush=True)
        serial.atomic_json(args.out/'status.json', dict(status='running', started_at_utc=serial.now()))
        ids=manifest['pilot_record_ids']
        protocol=manifest['scientific_protocol']
        bench_manifest=dict(protocol=spec, scientific_protocol=protocol, record_ids=ids,
                            parent_manifest_sha256=parent_sha, code_sha256=fingerprint,
                            environment=serial.environment(), started_at_utc=serial.now(),
                            dependencies={n: importlib.metadata.version(n) for n in ('cupy-cuda12x', 'cuda-pathfinder')},
                            source_manifest=manifest['source_manifest'], official_test_processed=False)
        serial.atomic_json(args.out/'manifest.json', bench_manifest)
        try:
            import cupy as cp
            started=time.perf_counter()
            cp.cuda.Device(0).use()
            cp.zeros(1).sum().get()
            context_seconds=time.perf_counter()-started
            gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version,pstate,utilization.gpu',
                                               '--format=csv,noheader'], text=True).strip()
            with (args.out/'cupy_config.txt').open('w') as stream:
                from contextlib import redirect_stdout
                with redirect_stdout(stream):
                    cp.show_config()
            serial.atomic_json(args.out/'hardware.json', dict(gpu=gpu_info,
                               compute_capability=cp.cuda.Device(0).compute_capability,
                               driver_version=cp.cuda.runtime.driverGetVersion(),
                               runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                               context_seconds=context_seconds))
            install=Path('/tmp/acs-gpu-install-v1.json')
            if install.is_file():
                shutil.copyfile(install, args.out/'dependency_install.json')
            started=time.perf_counter()
            warm=np.random.default_rng(20260925).normal(size=(4, 128))
            vmd_gpu_batch(warm, K=4, max_iter=3)
            warmup_seconds=time.perf_counter()-started
            event('gpu_ready', gpu=gpu_info, context_seconds=context_seconds, warmup_seconds=warmup_seconds)
            lookup={r['record_id']:r for r in rows}
            references={rid:serial.checked_feature(source/'features', lookup[rid], parent_sha) for rid in ids}
            records=[]
            started=time.perf_counter()
            with ACSDataset(target='OMI') as dataset:
                if dataset.source_manifest != manifest['source_manifest']:
                    raise ValueError('Changed source archives')
                for rid in ids:
                    if lookup[rid]['partition'] != 'fit':
                        raise ValueError('Benchmark may use only fit records')
                    record=dataset.load_record(rid)
                    check_loaded(record, lookup[rid])
                    records.append(record)
            load_seconds=time.perf_counter()-started
            X=np.vstack([r.signal_mV.T for r in records])
            serial.atomic_json(args.out/'inputs.json', dict(record_ids=ids, load_seconds=load_seconds,
                               physical_input_sha256={r.info.record_id:hashlib.sha256(np.ascontiguousarray(r.signal_mV.T).tobytes()).hexdigest() for r in records}))
            event('inputs_loaded', records=len(ids), seconds=load_seconds)

            event('cpu_two_record_reference_started')
            cpu=extract_vmd(X[:24], protocol, backend='cpu', lead_batch=4)
            cpu_checks=reference_checks(cpu, ids[:2], references, spec)
            if not all(c['passed'] and c['features']['exact'] for c in cpu_checks):
                raise AssertionError(f'CPU benchmark differs from frozen CPU features: {cpu_checks}')
            save_result(args.out,'cpu_two_records',cpu,modes=True)
            event('cpu_two_record_reference_complete', seconds=cpu['total_seconds'])
            gpu=extract_vmd(X[:24], protocol, backend='gpu', lead_batch=4)
            gpu_checks=reference_checks(gpu, ids[:2], references, spec)
            c=spec['checks']
            mode_check=array_check(gpu['modes'],cpu['modes'],c['modes_atol'],c['modes_rtol'])
            omega_check=array_check(gpu['omega_hz'],cpu['omega_hz'],c['omega_hz_atol'],c['omega_hz_rtol'])
            save_result(args.out,'gpu_two_records',gpu,modes=True)
            serial.atomic_json(args.out/'two_record_comparison.json',dict(cpu=cpu_checks,gpu=gpu_checks,modes=mode_check,omega_hz=omega_check))
            event('gpu_two_record_reference_complete', seconds=gpu['total_seconds'],
                  passed=all(c['passed'] for c in gpu_checks) and mode_check['passed'] and omega_check['passed'])

            event('cpu_eight_worker_baseline_started')
            started=time.perf_counter()
            with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=cpu_worker_init) as pool:
                cpu_results=list(pool.map(cpu_worker, [(r.info.record_id,r.signal_mV.T,protocol) for r in records]))
            cpu_wall=time.perf_counter()-started
            all_cpu_checks=[]
            for rid,result in cpu_results:
                check=reference_checks(result,[rid],references,spec)
                all_cpu_checks.extend(check)
                save_result(args.out,f'cpu_{rid}',result)
            if not all(c['passed'] and c['features']['exact'] for c in all_cpu_checks):
                raise AssertionError('Eight-worker CPU baseline differs from frozen features')
            event('cpu_eight_worker_baseline_complete',seconds=cpu_wall,records=len(ids))

            trials=[]
            for repeat in range(spec['gpu_warm_repetitions']):
                event('gpu_batched_trial_started',repeat=repeat)
                started=time.perf_counter()
                result=extract_vmd(X,protocol,backend='gpu',lead_batch=len(X))
                cp.cuda.get_current_stream().synchronize()
                seconds=time.perf_counter()-started
                checks=reference_checks(result,ids,references,spec)
                save_result(args.out,f'gpu_batch_{repeat}',result)
                trials.append(dict(repeat=repeat,wall_seconds=seconds,solver_seconds=result['solver_seconds'],checks=checks,
                                   passed=all(c['passed'] for c in checks)))
                event('gpu_batched_trial_complete',repeat=repeat,seconds=seconds,passed=trials[-1]['passed'])
            gpu_median=float(np.median([r['wall_seconds'] for r in trials]))
            passed=mode_check['passed'] and omega_check['passed'] and all(c['passed'] for c in gpu_checks) and all(t['passed'] for t in trials)
            timing=dict(cpu_two_records_seconds=cpu['total_seconds'],gpu_two_records_seconds=gpu['total_seconds'],
                        cpu_eight_workers_16_records_seconds=cpu_wall,gpu_16_records_seconds=[r['wall_seconds'] for r in trials],
                        gpu_median_seconds=gpu_median,speedup_vs_eight_workers=cpu_wall/gpu_median,
                        gpu_records_per_hour=3600*len(ids)/gpu_median,
                        extrapolated_17905_records_hours=17905*gpu_median/(len(ids)*3600),
                        includes_wst=False,includes_source_loading=False,includes_artifact_writes=False,
                        cpu_repetitions=1,gpu_repetitions=3)
            summary=dict(status='complete',numerical_checks_passed=passed,timing_summary=timing,
                         startup=dict(context_seconds=context_seconds,warmup_seconds=warmup_seconds,load_seconds=load_seconds),
                         record_ids=ids,cpu_checks=all_cpu_checks,gpu_trials=trials,
                         limitations=['16 fit ECGs only; no model accuracy comparison',
                                      'CPU baseline measured once; GPU trials share one process and warm cache',
                                      'Speedup compares CPU VMD/descriptors on eight workers with hybrid GPU VMD/descriptors; WST and disk/source I/O excluded',
                                      'No full GPU extraction or VQC training performed'],
                         models_trained=0,predictions_available=False,official_test_processed=False,
                         completed_at_utc=serial.now())
            summary['artifacts']=[dict(path=str(p.relative_to(args.out)),sha256=serial.file_hash(p))
                                  for p in sorted(args.out.iterdir()) if p.is_file() and p.name not in ('status.json','events.jsonl')]
            serial.atomic_json(args.out/'summary.json',summary)
            serial.atomic_json(args.out/'status.json',dict(status='complete',numerical_checks_passed=passed,completed_at_utc=serial.now()))
            event('complete',numerical_checks_passed=passed,**timing)
        except BaseException as exc:
            failure=dict(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),time_utc=serial.now())
            serial.atomic_json(args.out/'failure.json',failure)
            serial.atomic_json(args.out/'status.json',failure)
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE/'results/gpu_vmd_benchmark_v1')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
