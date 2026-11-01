"""One-shot completion monitor: wait for the existing writer, then audit once.

Does not start training or write to existing checkpoints. Runs outside the
training writer lock only after the saved writer process has exited. A missing
report or failed audit remains a recorded failure, never a performance result.
"""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent; RUN=BASE/'results'; REPORT=BASE/'reports'

def write(value):
    path=BASE/'monitor_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def alive(pid):
    try:
        text=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError): return False
    return '09_representation_comparison/study.py' in text and ' train ' in ' '+text+' '

def main():
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());state=json.loads((RUN/'status.json').read_text());pid=state.get('pid')
    if state.get('status')=='running' and not pid: raise ValueError('No writer pid recorded')
    write(dict(status='monitoring',training_pid=pid,started_utc=started))
    while pid and alive(pid): time.sleep(15)
    state=json.loads((RUN/'status.json').read_text())
    if state.get('status')!='complete' or not (RUN/'completed.json').exists() or not (REPORT/'analysis.json').exists() or not (REPORT/'REPORT.md').exists():
        write(dict(status='training_or_reporting_incomplete',training_status=state,observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print('Training/reporting incomplete; inspect status and logs.',flush=True);return 1
    audit=REPORT/'audit.json'
    if audit.exists() and json.loads(audit.read_text()).get('status')=='passed':
        write(dict(status='audit_already_passed',observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print('Existing audit passed; no repeated audit.',flush=True);return 0
    print('Training and report complete; running independent saved-fit audit.',flush=True)
    with (BASE/'audit.log').open('w') as stream:
        result=subprocess.run([sys.executable,'-B',str(BASE/'audit.py')],stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
    passed=result.returncode==0 and audit.exists() and json.loads(audit.read_text()).get('status')=='passed'
    write(dict(status='audit_passed' if passed else 'audit_failed',exit_code=result.returncode,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('Audit passed' if passed else 'Audit failed; inspect audit.log',flush=True);return 0 if passed else 1

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        write(dict(status='monitor_failed',traceback=traceback.format_exc()));raise
