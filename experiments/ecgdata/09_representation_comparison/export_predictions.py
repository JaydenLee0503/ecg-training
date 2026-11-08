"""Save a portable, checked table of the completed ECG-09 class predictions."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
analysis_path = HERE / 'reports/analysis.json'
audit = json.loads((HERE / 'reports/audit.json').read_text())
assert audit['status'] == 'passed'
assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == audit['analysis_sha256']
analysis = json.loads(analysis_path.read_text())
with np.load(HERE / 'results/predictions_export.npz', allow_pickle=False) as data:
    keys = data.files
    arrays = {key: data[key] for key in keys}
assert all(a.shape == (21222,) for a in arrays.values())
mask = arrays['v1_index'] >= 0
assert mask.sum() == 1620
assert len(set(arrays['patients'])) == 80
for patient in set(arrays['patients']):
    assert len(set(arrays['fold'][arrays['patients'] == patient])) == 1
for arm, result in analysis['results']['v1_windows']['arms'].items():
    for seed, metrics in enumerate(result['per_seed']):
        pred = arrays[f'{arm}_seed{seed}'][mask]
        assert set(pred) <= {0, 1, 2}
        assert np.isclose(accuracy_score(arrays['y'][mask], pred), metrics['window']['accuracy'])
        assert np.isclose(f1_score(arrays['y'][mask], pred, average='macro'), metrics['window']['macro_f1'])

output = HERE / 'reports/predictions.csv.gz'
with output.open('wb') as raw:
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as text:
            writer = csv.writer(text)
            writer.writerow(keys)
            writer.writerows(zip(*(arrays[key] for key in keys)))
print(f'{output}: {len(arrays["y"])} rows; {len(keys)} columns; SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}')
