"""Read-only snapshot of completed fit markers and current status, no scores."""
import hashlib,json,os,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;RUN=BASE/'results';OUT=BASE/'progress';OUT.mkdir(exist_ok=True)
status=json.loads((RUN/'status.json').read_text());markers=[]
for p in sorted(RUN.glob('*/completed.json')):
 with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
 marker=json.loads(p.read_text());assert marker['status']=='complete';markers.append({'fit':p.parent.name,'marker_sha256':digest})
text=(BASE/'training.log').read_text();timestamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());stem=OUT/timestamp
meta={'captured_utc':timestamp,'status':status,'completed_fit_markers':markers,'training_log_sha256':hashlib.sha256(text.encode()).hexdigest(),'note':'Progress only; no held-out scores were inspected. Results require all planned fits and final audit.'}
(stem.with_suffix('.json')).write_text(json.dumps(meta,indent=2)+'\n');(stem.with_suffix('.log')).write_text(text)
print('Snapshot:',stem,'| completed fit markers:',len(markers))
