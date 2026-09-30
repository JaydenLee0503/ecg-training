"""Compare the frozen GPU solver with the completed single-fit-lead CPU trace."""
import json
from pathlib import Path
import time
import traceback

from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu as old
from experiments.acs_omi_vmd_wst_vqc.gpu_vmd import vmd_gpu_batch
import numpy as np

BASE=old.BASE
CPU=BASE/'results/vmd_04124_diagnostic_v1'
OUT=BASE/'results/vmd_04124_gpu_check_v1'


def main():
    summary=json.loads((CPU/'summary.json').read_text())
    if summary['status']!='complete' or summary['diagnostic']['capped'] or summary['partition']!='fit':
        raise ValueError('Require a completed, converged fit-lead CPU diagnostic')
    for name,digest in summary['artifacts'].items():
        if old.serial.file_hash(CPU/name)!=digest:
            raise ValueError('CPU diagnostic artifact changed')
    source_manifest=json.loads((CPU/'manifest.json').read_text())
    for name,digest in source_manifest['production_code_sha256'].items():
        if old.serial.file_hash(old.ROOT/name)!=digest:
            raise ValueError('Frozen production implementation changed')
    OUT.mkdir(exist_ok=True)
    with old.run_lock(OUT):
        manifest=dict(cpu_summary_sha256=old.serial.file_hash(CPU/'summary.json'),
            source_code_sha256=source_manifest['production_code_sha256'],
            probe_sha256=old.serial.file_hash(Path(__file__)),hardware=old.hardware(),environment=old.environment(),
            settings=summary['settings'],max_iter=256000,record_id='04124',lead='V5',partition='fit',
            check='same iteration count and modes/centres atol=1e-6 rtol=1e-5 against CPU',
            production_resumed=False,official_test_processed=False)
        if (OUT/'manifest.json').exists():
            if json.loads((OUT/'manifest.json').read_text())!=manifest:
                raise ValueError('Changed GPU diagnostic context')
        else:
            old.atomic_json(OUT/'manifest.json',manifest)
        if (OUT/'summary.json').exists():
            saved=json.loads((OUT/'summary.json').read_text())
            if old.serial.file_hash(OUT/'result.npz')!=saved['result_sha256']:
                raise ValueError('Changed completed GPU check')
            print(json.dumps(saved),flush=True)
            return
        old.atomic_json(OUT/'status.json',dict(status='running',started_at_utc=old.serial.now()))
        try:
            with np.load(CPU/'cpu_128000.npz',allow_pickle=False) as z:
                signal=z['signal']
            with np.load(CPU/'trace.npz',allow_pickle=False) as z:
                reference={k:z[k] for k in ('modes','omega_hz','iters')}
            before=time.perf_counter()
            result=vmd_gpu_batch(signal,max_iter=256000,**summary['settings'])
            seconds=time.perf_counter()-before
            np.testing.assert_array_equal(result.iters,reference['iters'])
            np.testing.assert_allclose(result.modes,reference['modes'],atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(result.omega_hz,reference['omega_hz'],atol=1e-6,rtol=1e-5)
            if result.capped.any() or not np.isfinite(result.modes).all():
                raise ValueError('GPU diagnostic did not converge to finite modes')
            with (OUT/'result.tmp').open('wb') as stream:
                np.savez_compressed(stream,modes=result.modes,omega_hz=result.omega_hz,iters=result.iters)
            (OUT/'result.tmp').replace(OUT/'result.npz')
            report=dict(status='complete',passed=True,iterations=int(result.iters[0]),cap=256000,
                modes_max_abs_error=float(np.max(np.abs(result.modes-reference['modes']))),
                omega_hz_max_abs_error=float(np.max(np.abs(result.omega_hz-reference['omega_hz']))),
                seconds=seconds,result_sha256=old.serial.file_hash(OUT/'result.npz'),
                manifest_sha256=old.serial.file_hash(OUT/'manifest.json'),completed_at_utc=old.serial.now(),
                production_resumed=False,production_settings_changed=False,models_trained=0,official_test_processed=False)
            old.atomic_json(OUT/'summary.json',report)
            old.atomic_json(OUT/'status.json',dict(status='complete',completed_at_utc=old.serial.now()))
            print(json.dumps(report,indent=2),flush=True)
        except BaseException:
            old.atomic_json(OUT/'status.json',dict(status='failed',traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
