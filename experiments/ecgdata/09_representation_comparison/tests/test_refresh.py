"""The notebook refresh must not overwrite output before the audit passes."""
import importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'refresh_notebook.py';spec=importlib.util.spec_from_file_location('ecg09_refresh_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RefreshGateTest(unittest.TestCase):
 def test_missing_audit_preserves_notebook(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);nb=base/'original.ipynb';nb.write_bytes(b'original notebook bytes')
   with patch.object(m,'BASE',base),patch.object(m,'NOTEBOOK',nb),patch.object(m,'NotebookClient') as client:
    with self.assertRaisesRegex(RuntimeError,'passed audit'):m.main()
    client.assert_not_called();self.assertEqual(nb.read_bytes(),b'original notebook bytes')
 def test_concurrent_notebook_edit_is_preserved(self):
  import nbformat
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);(base/'reports').mkdir();(base/'reports/audit.json').write_text('{\"status\":\"passed\"}');(base/'reports/REPORT.md').write_text('# ECG-09 representation comparison')
   nb=base/'original.ipynb'; notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("study = REPO/'experiments/ecgdata/09_representation_comparison'")]);nbformat.write(notebook,nb)
   class FakeClient:
    def __init__(self,doc,**kwargs):self.doc=doc
    def execute(self):
     self.doc.cells[0].execution_count=1;self.doc.cells[0].outputs=[nbformat.v4.new_output('stream',name='stdout',text='# ECG-09 representation comparison')]
     nb.write_text('concurrent human edit')
   with patch.object(m,'BASE',base),patch.object(m,'REPO',base),patch.object(m,'NOTEBOOK',nb),patch.object(m,'NotebookClient',FakeClient):
    with self.assertRaisesRegex(RuntimeError,'changed during execution'):m.main()
    self.assertEqual(nb.read_text(),'concurrent human edit')

if __name__=='__main__':unittest.main()
