"""Completion monitor must not audit an incomplete run."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'monitor.py';spec=importlib.util.spec_from_file_location('ecg09_monitor',p);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)

class MonitorTest(unittest.TestCase):
 def test_incomplete_writer_never_starts_audit(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);run=base/'results';run.mkdir();(run/'status.json').write_text(json.dumps({'status':'running','pid':12345}))
   with patch.object(monitor,'BASE',base),patch.object(monitor,'RUN',run),patch.object(monitor,'REPORT',base/'reports'),patch.object(monitor,'alive',return_value=False),patch.object(monitor.subprocess,'run') as execute:
    self.assertEqual(monitor.main(),1);execute.assert_not_called();self.assertEqual(json.loads((base/'monitor_status.json').read_text())['status'],'training_or_reporting_incomplete')
 def test_completed_writer_starts_audit_once(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);run=base/'results';run.mkdir();report=base/'reports';report.mkdir()
   (run/'status.json').write_text(json.dumps({'status':'complete'}));(run/'completed.json').write_text('{}');(report/'analysis.json').write_text('{}');(report/'REPORT.md').write_text('complete')
   class Result: returncode=0
   def fake_run(*args,**kwargs):
    (report/'audit.json').write_text(json.dumps({'status':'passed'}));return Result()
   with patch.object(monitor,'BASE',base),patch.object(monitor,'RUN',run),patch.object(monitor,'REPORT',report),patch.object(monitor.subprocess,'run',side_effect=fake_run) as execute:
    self.assertEqual(monitor.main(),0);self.assertEqual(execute.call_count,1);self.assertEqual(json.loads((base/'monitor_status.json').read_text())['status'],'audit_passed')
    self.assertEqual(monitor.main(),0);self.assertEqual(execute.call_count,1)
if __name__=='__main__':unittest.main()
