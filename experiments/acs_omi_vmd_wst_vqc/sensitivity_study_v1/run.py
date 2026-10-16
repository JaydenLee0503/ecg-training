"""Run isolated stages; log every child and stop owned workers on interruption."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from . import common as c

MODULE = 'experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.run'


def trial_names():
    spec, _ = c.source_paths()
    return [(a, n) for a in ('vmd', 'wst') for n in spec['models']['vmd_wst']] + [
        ('lfcc', n) for n in spec['models']['lfcc']]


def workers(tasks, jobs, stop):
    queue, active = list(tasks), []
    try:
        while queue or active:
            if stop.is_set():
                raise RuntimeError('Other worker failed or workflow interrupted; stopping owned children')
            while queue and len(active) < jobs:
                arm, name = queue.pop(0)
                log = c.RUN / 'logs' / f'{arm}_{name}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('a')
                proc = subprocess.Popen([sys.executable, '-B', '-m', MODULE, 'trial', '--arm', arm,
                    '--model', name, '--device', 'cuda' if name.startswith('swin') else 'cpu'],
                    stdout=stream, stderr=subprocess.STDOUT, cwd=c.ROOT)
                active.append((proc, stream, arm, name))
                print(f'Started {arm}/{name}, PID {proc.pid}, log {log}', flush=True)
            for item in list(active):
                proc, stream, arm, name = item
                if proc.poll() is not None:
                    stream.close()
                    active.remove(item)
                    if proc.returncode:
                        raise RuntimeError(f'{arm}/{name} failed with exit {proc.returncode}; see its log')
                    print(f'Finished {arm}/{name}', flush=True)
            time.sleep(1)
    except BaseException:
        stop.set()
        raise
    finally:
        for proc, stream, _, _ in active:
            if proc.poll() is None:
                proc.terminate()
        for proc, stream, _, _ in active:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stream.close()


def all_models(jobs):
    import concurrent.futures
    _, ctx = c.manifest()
    if not c.verified(c.RUN / 'prepared', ctx):
        raise ValueError('Preparation incomplete')
    # Independent CPU and GPU groups; each serial GPU fit retains all three seeds.
    tasks = trial_names()
    stop = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        cpu = pool.submit(workers, [t for t in tasks if not t[1].startswith('swin')], jobs, stop)
        gpu = pool.submit(workers, [t for t in tasks if t[1].startswith('swin')], 1, stop)
        try:
            cpu.result()
            gpu.result()
        except BaseException:
            stop.set()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'saved', 'train', 'trial', 'report', 'audit', 'all'])
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--arm', choices=['vmd', 'wst', 'lfcc'])
    p.add_argument('--model')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if not 1 <= args.jobs <= 6:
        p.error('Use 1 to 6 CPU workers')
    if args.stage == 'trial':
        if (args.arm, args.model) not in trial_names():
            p.error('Unknown trial')
        from .fit import run_trial
        run_trial(args.arm, args.model, args.device)
        return
    with c.lock(c.RUN):
        started = time.perf_counter()
        c.write(c.RUN / 'status.json', dict(status='running', stage=args.stage, started_at_utc=c.now(), pid=os.getpid()))
        try:
            if args.stage in ('prepare', 'all'):
                from .prepare import prepare
                prepare()
            if args.stage in ('saved', 'all'):
                from .report import saved
                saved()
            if args.stage in ('train', 'all'):
                all_models(args.jobs)
            if args.stage in ('report', 'all'):
                from .report import new_report
                new_report()
            if args.stage in ('audit', 'all'):
                from .report import audit
                audit()
            c.write(c.RUN / 'status.json', dict(status='complete', stage=args.stage,
                    seconds=time.perf_counter()-started, completed_at_utc=c.now()))
        except BaseException:
            failure = dict(status='failed', stage=args.stage, time_utc=c.now(), traceback=traceback.format_exc())
            c.write(c.RUN / f'failure_{time.time_ns()}.json', failure)
            c.write(c.RUN / 'status.json', failure)
            raise


if __name__ == '__main__':
    main()
