"""Run with python -B -m experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2.run."""
import argparse
import os
import time
import traceback
from . import common as c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','all','report','audit'])
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args()
    with c.lock(c.RUN):
        started=time.perf_counter()
        if args.action=='all' and (c.RUN/'completed.json').exists():
            _,ctx=c.manifest()
            if c.verified(c.RUN,ctx):
                print('Complete study verified; no retraining',flush=True)
                return
        c.write(c.RUN/'status.json',dict(status='running',action=args.action,pid=os.getpid(),started_at_utc=c.now()))
        try:
            if args.action=='prepare':
                from .data import prepare
                prepare()
            elif args.action=='all':
                from .workflow import run
                run(args.device)
                _,ctx=c.manifest()
                c.complete(c.RUN,ctx,seconds_this_session=time.perf_counter()-started)
            elif args.action=='report':
                from .report import report
                report()
            else:
                from .report import audit
                audit()
            c.write(c.RUN/'status.json',dict(status='complete' if args.action=='all' else args.action+'_complete',
                seconds_this_session=time.perf_counter()-started,completed_at_utc=c.now()))
        except BaseException:
            failure=dict(status='failed',action=args.action,time_utc=c.now(),traceback=traceback.format_exc())
            c.write(c.RUN/f'failure_{time.time_ns()}.json',failure)
            c.write(c.RUN/'status.json',failure)
            raise


if __name__=='__main__':
    main()
