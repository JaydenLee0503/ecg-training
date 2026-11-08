"""Read-only checksum check of completed OMI results after restart."""
import hashlib,json,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
PARENT=BASE.parent/'undersampled_v1/results'
NOTEBOOK=BASE.parent/'notebooks/omi_lfcc_swin_logistic_training.ipynb'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    marker=json.loads((BASE/'results/completed.json').read_text())
    if marker['status']!='complete':raise ValueError('Threshold analysis incomplete')
    items={'protocol_sha256':BASE/'PROTOCOL.md','source_sha256':BASE/'analyze.py','report_sha256':BASE/'REPORT.md','analysis_sha256':BASE/'analysis.json','draws_sha256':BASE/'results/bootstrap.npz'}
    issues=[str(path) for key,path in items.items() if not path.is_file() or sha(path)!=marker[key]]
    parent=json.loads((PARENT/'completed.json').read_text())
    if parent['status']!='complete':raise ValueError('Undersampling run incomplete')
    for name,expected in parent['files'].items():
        f=PARENT/name
        if not f.is_file() or sha(f)!=expected:issues.append(str(f))
    notebook=json.loads(NOTEBOOK.read_text());cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    outputs_ok=bool(cells) and all(c.get('execution_count') is not None and c.get('outputs') and not any(o.get('output_type')=='error' for o in c['outputs']) for c in cells)
    record={'status':'passed' if not issues and outputs_ok else 'failed','checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'threshold_artifacts_checked':len(items),'undersampling_artifacts_checked':len(parent['files']),'notebook_code_cells':len(cells),'notebook_outputs_present':outputs_ok,'notebook_sha256':sha(NOTEBOOK),'issues':issues,'scope':'source artifacts read-only; writes this JSON audit; no inference, fitting or official-test processing'}
    out=BASE/'restart_check_2026-11-07.json';out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if record['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
