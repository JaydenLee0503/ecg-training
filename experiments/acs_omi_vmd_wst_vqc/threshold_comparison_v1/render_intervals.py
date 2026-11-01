"""Readable fixed-threshold operating points from the completed analysis JSON."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent;a=json.loads((p/'analysis.json').read_text())
lines=['# OMI operating points with patient-bootstrap intervals','','All values are validation estimates. Parentheses show pointwise 95% patient-bootstrap intervals; Swin metrics average seeds 0/1/2. Cutoffs were fixed at 0.5 or selected previously on calibration patients.','','| Model | Threshold | Sensitivity (95% interval) | Specificity (95% interval) | Precision (95% interval) |','|---|---|---|---|---|']
for model,rows in a['results'].items():
 for label,row in zip(a['threshold_labels'],rows):
  def metric(name):
   x=row['point'][name];lo,hi=row['intervals'][name];return f'{x:.2%} ({lo:.2%}–{hi:.2%})'
  lines.append(f'| {model} | {label} | {metric("sensitivity")} | {metric("specificity")} | {metric("precision")} |')
lines+=['','Intervals hold fitted models, selected training patients and cutoffs fixed; they omit retraining, threshold-selection and training-subset uncertainty. They are not adjusted for multiple comparisons. Target levels describe calibration, not guaranteed validation sensitivity. Existing validation patients have been repeatedly examined. The results do not establish clinical effectiveness.']
(p/'REPORT_INTERVALS.md').write_text('\n'.join(lines)+'\n');print('Wrote 16 fixed-threshold rows with pointwise bootstrap intervals.')
