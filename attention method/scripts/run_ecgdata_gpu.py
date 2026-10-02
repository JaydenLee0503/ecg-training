"""Explicit launch of the fixed ECGData GPU experiment with synthetic preflight."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-training', action='store_true', required=True)
    parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root.parent)]
    from attention_ecg.ecgdata import atomic_json, file_hash
    from attention_ecg.ecgdata_training import DEFAULT_RUN, run_training
    from validate_gpu import main as validate_gpu
    validate_gpu()
    stamp = datetime.now(timezone.utc)
    attempt = root/'reports'/f'ecgdata_execution_{stamp.strftime("%Y%m%dT%H%M%SZ")}.json'
    record = dict(status='running', started_at_utc=stamp.isoformat(), device='cuda',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        launch_source_sha256=file_hash(__file__),
        packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                  if d.metadata.get('Name')},
        command=sys.argv)
    atomic_json(attempt, record)
    done = threading.Event()

    def progress():
        while not done.wait(30):
            completed = list(DEFAULT_RUN.glob('*/completed.json'))
            epochs = 0
            for marker in DEFAULT_RUN.glob('swin_*/checkpoint.json'):
                epochs += json.loads(marker.read_text())['epoch']
            print(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                  completed_fits=len(completed), saved_epochs=epochs, planned_epochs=600)), flush=True)

    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    began = time.perf_counter()
    try:
        result = run_training(execute=True, device='cuda')
        record.update(status=result['status'], result=result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        done.set()
        thread.join()
        record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-began)
        atomic_json(attempt, record)


if __name__ == '__main__':
    main()
