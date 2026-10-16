"""Launch the authorized, isolated ECGData regularization study."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import unittest

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent), str(ROOT / '.venv/torch-cu128')]

from attention_ecg.ecgdata_regularization import run, RUN, REPORT
from attention_ecg.ecgdata import atomic_json


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-training', action='store_true', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    log = (REPORT / 'execution.log').open('a', buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    print(json.dumps(dict(event='launch', time=datetime.now(timezone.utc).isoformat(), command=sys.argv)), flush=True)
    if not (RUN / 'completed.json').exists():
        began = time.perf_counter()
        if args.device == 'cuda':
            os.environ['ECG_REG_TEST_CUDA'] = '1'
            sys.path.insert(0, str(ROOT / 'tests'))
            from test_ecgdata_regularization import RegularizationTests
            suite = unittest.TestSuite([RegularizationTests('test_gpu_forward_gradient_and_resume')])
            checked = unittest.TextTestRunner(verbosity=2).run(suite)
            record = dict(status='passed' if checked.wasSuccessful() else 'failed',
                          checks=['GPU/CPU forward and gradient agreement',
                                  'exact interrupted dropout training recovery on GPU',
                                  'candidate checkpoint retention/reload and completed-fit reuse'],
                          seconds=time.perf_counter() - began)
            atomic_json(REPORT / 'gpu_preflight.json', record)
            if not checked.wasSuccessful():
                raise SystemExit('GPU preflight failed; no real training started')
        else:
            raise SystemExit('This frozen execution requires the verified CUDA path')
    run(device=args.device, execute=args.execute_training)
