# ECG-07 — LFCC/Swin on every ECGData window

Status: **complete** (2026-10-24 05:11Z, 20/20 fits). Results: [reports/report.md](reports/report.md); full record: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

**Question.** v1 trained on 10 randomly placed 500-sample windows per ECGData
row, 7.6% of each 512-second recording. Does using all 131 non-overlapping
windows per row (21,222 windows) improve held-out patient performance, with
everything else unchanged?

**What is identical to v1.** ECGData.mat, the 80 verified patients and their
five outer folds, per-window z-scoring, LFCC settings, the Swin architecture and
training recipe (40 epochs, AdamW, class weights, seeds 0/1/2), the pooled-LFCC
logistic control, and the metric/bootstrap code. Fitting calls the unchanged
`attention_ecg.ecgdata_training` functions. The 1,620 v1 windows are a verified
bit-identical subset of the new grid.

**What changes.** Training windows per fold rise ~13×, and so do optimizer updates
(fixed 40 epochs). These two effects are not separated.

**Primary outcome.** Window macro-F1, new Swin minus v1 Swin, on the identical
1,620 v1 held-out windows, with a paired patient bootstrap. Secondary: accuracy,
patient vote, and metrics on all 21,222 held-out windows.

## Result

On the identical 1,620 v1 held-out windows (mean of seeds 0/1/2), all-window Swin
reached **86.83% window accuracy, macro-F1 0.8390, 95.83% patient-vote accuracy**,
versus v1's 83.35% / 0.8006 / 94.17%. Primary macro-F1 difference **+0.0384
[+0.0153, +0.0602]**; the window-accuracy gain is +3.48 pp [+1.58, +5.38]. The
patient-vote gain (+1.67 pp [−1.25, +5.00]) is not established. Every seed improved
in window accuracy. Logistic changed little (+0.0120 [−0.0053, +0.0301]). On all
21,222 windows Swin scored 86.06% / 0.8290 / 95.42%. More data and more optimizer
updates are confounded; Swin still fits training windows almost perfectly.

## Files

| Path | Contents |
|---|---|
| [protocol.json](protocol.json) | Frozen settings, outcomes and caveats |
| [study.py](study.py), [run.py](run.py) | Extraction, fitting, reporting |
| [tests/test_study.py](tests/test_study.py) | Synthetic checks |
| [report_markdown.py](report_markdown.py) | Writes report.md from analysis.json (fixes a bug in study.write_markdown) |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Everything done, with times, hashes and failures |
| [logs/](logs) | Test, extraction and training console logs |
| `reports/` | `report.md`, `metrics.csv`, `fits.csv`, `analysis.json` (after reporting) |
| `features/all_windows/` | LFCC cache (gitignored) |
| `results/all_windows/` | Models, checkpoints, histories, predictions (gitignored) |

## Commands (from this folder)

```bash
export PYTHONPATH="/mnt/d/Projects/test-ecg-training/attention method/.venv/torch-cu128"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/jaydenlee/venvs/test-ecg-training/bin/python
$PY -B -m unittest discover -s tests -v
$PY -B run.py prepare            # reuses a completed cache
$PY -B run.py train --execute-training --device cuda   # resumes; skips completed fits
$PY -B run.py report      # saves analysis/CSVs, then hits a known markdown KeyError
$PY -B report_markdown.py # writes reports/report.md from analysis.json
```

Check `results/all_windows/status.json` and completion markers before rerunning;
completed fits are verified and reused, never retrained. Original v1/v2 caches
and runs are read only.

Limits: same 80 patients and folds that informed earlier work (internal evidence,
not an external test); windows within a patient are correlated; source database
is confounded with diagnosis; ARR/CHF/NSR, not MI/OMI.

[All studies](../../README.md).
