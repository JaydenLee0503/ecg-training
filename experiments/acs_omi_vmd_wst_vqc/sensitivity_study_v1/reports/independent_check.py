"""Read-only statistical cross-check of the completed study using sklearn."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    start = time.perf_counter()
    manifest = read(BASE / 'results/manifest.json')
    cohort = arrays(BASE / 'results/prepared/cohort.npz')
    parts = cohort['partitions']
    for a in ('subfit', 'calibration', 'validation'):
        for b in ('subfit', 'calibration', 'validation'):
            if a != b:
                assert not set(cohort['patients'][parts == a]) & set(cohort['patients'][parts == b])
    assert np.all(parts[cohort['sampled_indices']] == 'subfit')
    assert set(cohort['sampled_indices']) == set(np.flatnonzero(parts == 'subfit'))
    expected_counts = np.bincount(cohort['y'][cohort['sampled_indices']])
    assert expected_counts[0] == expected_counts[1]
    report = read(BASE / 'reports/calibrated/analysis.json')
    model_count, point_count, epoch_count = 0, 0, 0
    for arm in ('vmd', 'wst', 'lfcc'):
        names = manifest['protocol']['models']['lfcc' if arm == 'lfcc' else 'vmd_wst']
        for name in names:
            folder = BASE / 'results/models' / arm / name
            marker = read(folder / 'completed.json')
            assert marker['status'] == 'complete'
            cal = arrays(folder / 'calibration_predictions.npz')
            val = arrays(folder / 'validation_predictions.npz')
            for source, part in ((cal, 'calibration'), (val, 'validation')):
                for key in ('y', 'patients', 'record_ids'):
                    np.testing.assert_array_equal(source[key], cohort[key][parts == part])
            fpr, tpr, thresholds = roc_curve(cal['y'], cal['score'], drop_intermediate=False)
            result = report['per_model'][f'new/{arm}/{name}']
            for j, target in enumerate(manifest['protocol']['targets_percent']):
                threshold = thresholds[np.flatnonzero(tpr >= target/100)[0]]
                assert threshold == result[j]['threshold'] == val['thresholds'][j]
                prediction = val['score'] >= threshold
                np.testing.assert_array_equal(prediction, val['target_predictions'][:, j])
                tn, fp, fn, tp = confusion_matrix(val['y'], prediction, labels=[0, 1]).ravel()
                assert [tn, fp, fn, tp] == [result[j][k] for k in ('tn', 'fp', 'fn', 'tp')]
                assert result[j]['sensitivity'] == tp/(tp+fn)
                assert result[j]['specificity'] == tn/(tn+fp)
                assert result[j]['precision'] == tp/(tp+fp)
                point_count += 1
            ranking = report['fixed_0_5_per_model'][f'new/{arm}/{name}']
            assert ranking['average_precision'] == average_precision_score(val['y'], val['score'])
            assert ranking['roc_auc'] == roc_auc_score(val['y'], val['score'])
            record = read(folder / 'fit_record.json')
            assert record['threshold_frozen_before_validation_inference']
            assert record['calibration_reload_exact'] and record['validation_reload_exact']
            if name.startswith(('vqc', 'swin')):
                history = read(folder / ('fit/history.json' if name.startswith('swin') else 'history.json'))
                assert [r['epoch'] for r in history] == list(range(1, 41))
                epoch_count += len(history)
            model_count += 1
    assert model_count == 14 and epoch_count == 360 and point_count == 42
    protected = 0
    for name, digest in manifest['protected_sha256'].items():
        with (ROOT / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
        protected += 1
    result = dict(models=14, calibration_selected_operating_points_checked=point_count,
        all_epoch_entries_verified=epoch_count, protected_files_verified=protected,
        patient_overlap=0, split_and_sampling_verified=True, metric_mismatches=0,
        seconds=time.perf_counter()-start, checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (BASE / 'reports/independent_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
