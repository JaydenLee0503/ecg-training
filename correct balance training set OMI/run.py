"""Execute and resume the explicitly authorized balanced OMI/Swin experiment."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import common as c


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}; preserve committed checkpoints')


def stop_child(child):
    """Stop only the process group created by this workflow, including its workers."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def status(stage, **extra):
    c.write(c.RUN / 'status.json', dict(status='running', stage=stage, pid=os.getpid(),
                                       updated_at_utc=c.now(), **extra))


def all_stages(jobs):
    import balanced
    import lfcc
    import swin
    import report
    swin.setup('cuda')  # Confirm access before launching any fitting.
    balanced.prepare()
    lfcc.engineering_check()
    status('balanced training plus LFCC preparation')
    with (c.RUN / 'balanced.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                  'train-balanced', '--jobs', str(jobs)], stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        c.write(c.RUN / 'processes.json', dict(workflow_pid=os.getpid(), balanced_pid=child.pid,
                                              launched_at_utc=c.now()))
        try:
            with c.lock(c.RUN / 'lfcc_job'):
                lfcc.extract()
            status('balanced training plus Swin')
            with c.lock(c.RUN / 'swin_job'):
                swin.train()
            status('waiting for balanced VQC', balanced_pid=child.pid)
            while child.poll() is None:
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'Balanced training failed (exit {child.returncode}); inspect balanced.log')
        except BaseException:
            stop_child(child)
            c.write(c.RUN / 'child_after_failure.json', dict(pid=child.pid, exit_code=child.poll(),
                                                           stopped=True, time_utc=c.now()))
            raise
    status('patient bootstrap report')
    with c.lock(c.RUN / 'report_job'):
        report.report()
    c.write(c.RUN / 'status.json', dict(status='complete', updated_at_utc=c.now(), new_fits=14,
                                       official_test_processed=False))


def main():
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train-balanced', 'extract-lfcc', 'train-swin', 'report', 'all'])
    parser.add_argument('--jobs', type=int, default=3, choices=range(1, 7))
    args = parser.parse_args()
    if args.command == 'train-balanced':
        import balanced
        balanced.train(args.jobs)
        return
    with c.lock(c.RUN / 'workflow'):
        log = (c.RUN / 'run.log').open('a', buffering=1)
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            if args.command == 'all':
                all_stages(args.jobs)
            elif args.command == 'prepare':
                import balanced
                balanced.prepare()
            elif args.command == 'extract-lfcc':
                import lfcc
                with c.lock(c.RUN / 'lfcc_job'):
                    lfcc.extract()
            elif args.command == 'train-swin':
                import swin
                with c.lock(c.RUN / 'swin_job'):
                    swin.train()
            else:
                import report
                with c.lock(c.RUN / 'report_job'):
                    report.report()
        except BaseException:
            failure = dict(status='interrupted' if isinstance(sys.exception(), KeyboardInterrupt) else 'failed',
                           command=args.command, time_utc=c.now(), traceback=traceback.format_exc())
            c.write(c.RUN / f'failure_{time.time_ns()}.json', failure)
            c.write(c.RUN / 'status.json', failure)
            raise


if __name__ == '__main__':
    main()
