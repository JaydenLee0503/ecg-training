"""Independently recompute ranking metrics, cutoffs and patient separation."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

base = Path(__file__).resolve().parents[1]
result = json.loads((base/'reports/analysis.json').read_text())
with (base/'results/analysis/predictions.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
y = np.array([int(r['y']) for r in rows])
for name, reported in result['per_seed'].items():
    score = np.array([float(r[name]) for r in rows])
    np.testing.assert_allclose(average_precision_score(y, score), reported['average_precision'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(roc_auc_score(y, score), reported['roc_auc'], atol=1e-12, rtol=0)
for name, thresholds in result['thresholds'].items():
    if '_seed' in name:
        arm, seed = name.rsplit('_seed', 1)
        folder = base/'results/final'/arm/f'seed{seed}'
    else:
        folder = base/'results/final'/name
    with np.load(folder/'calibration_predictions.npz', allow_pickle=False) as d:
        labels, scores = d['y'], d['score']
    positives = np.sort(scores[labels == 1])[::-1]
    for target, saved in zip(result['targets_percent'], thresholds):
        rank = (len(positives)*target+99)//100
        assert float(positives[rank-1]) == saved, (name, target)
with np.load(base/'results/prepared/cohort.npz', allow_pickle=False) as d:
    pats, parts = d['patients'], d['partitions']
    for i, a in enumerate(('subfit', 'calibration', 'validation')):
        for b in ('subfit', 'calibration', 'validation')[i+1:]:
            assert not set(pats[parts == a]) & set(pats[parts == b])
output = dict(status='passed', models=len(result['per_seed']), thresholds=3*len(result['thresholds']),
              patients_disjoint=True, official_test_processed=False)
(base/'reports/independent_check.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output))
