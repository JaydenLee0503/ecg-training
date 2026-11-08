"""Refresh ECGData manuscript notebook only after a passed ECG-09 audit.

All code cells are executed and their output saved. Existing notebook bytes are
replaced only after the complete execution and output checks succeed; this never
initiates model training and requires the original local artifacts.
"""
import hashlib,json,os,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[2]
NOTEBOOK=REPO/'attention method/reproduce_swin/lfcc_swin_training.ipynb'


def main():
    audit=BASE/'reports/audit.json'
    if not audit.exists() or json.loads(audit.read_text()).get('status')!='passed':
        raise RuntimeError('ECG-09 must have a passed audit before the notebook refresh')
    if not (BASE/'reports/REPORT.md').exists():
        raise RuntimeError('ECG-09 report is missing')
    os.environ.pop('ECG_RUN_FRESH',None)
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    original_sha=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    NotebookClient(notebook,timeout=600,kernel_name='python3',
                   resources={'metadata':{'path':str(REPO)}}).execute()
    cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    if not cells or any(cell.execution_count is None or not cell.outputs or
                        any(output.output_type=='error' for output in cell.outputs)
                        for cell in cells):
        raise ValueError('Notebook has unexecuted, outputless or failed code cells')
    gate=next(cell for cell in cells if "study = REPO/'experiments/ecgdata/09_representation_comparison'" in cell.source)
    visible=''.join(str(output.get('text','')) for output in gate.outputs)
    if 'ECG-09 representation comparison' not in visible:
        raise ValueError('Audited ECG-09 report did not appear in notebook output')
    fd,name=tempfile.mkstemp(prefix=NOTEBOOK.name+'.',suffix='.tmp',dir=NOTEBOOK.parent)
    os.close(fd); temporary=Path(name)
    try:
        nbformat.write(notebook,temporary)
        if hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()!=original_sha:
            raise RuntimeError('Notebook changed during execution; original changes preserved')
        os.replace(temporary,NOTEBOOK)
    finally:
        if temporary.exists():temporary.unlink()
    print(f'Refreshed {len(cells)} code cells with saved outputs; ECG-09 audit passed.')


if __name__=='__main__':main()
