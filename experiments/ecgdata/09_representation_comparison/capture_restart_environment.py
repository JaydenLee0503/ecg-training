"""Record post-restart runtime details without touching training inputs."""
import importlib.metadata,json,os,platform,subprocess,time
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
run=json.loads((BASE/'results/manifest.json').read_text())
command=['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']
result=subprocess.run(command,text=True,capture_output=True,check=False)
record={'captured_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'platform':platform.platform(),
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','pandas','joblib')},
        'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'device_name':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'nvidia_smi':{'command':command,'exit_code':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip()},
        'threads_and_determinism':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},
        'original_manifest':{k:run.get(k) for k in ('device','device_name','python','numpy','torch','cuda')},
        'driver_note':'Original ECG-09 launch manifest did not record GPU driver version; driver equality across restart is not established.'}
(BASE/'restart_environment_2026-11-07.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
