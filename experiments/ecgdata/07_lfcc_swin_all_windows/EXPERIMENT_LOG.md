# ECG-07 experiment log

Chronological record of everything done for this study. Times are UTC unless noted.

## 2026-10-23 EDT (2026-10-24 UTC) — design, checks, extraction

**Request.** The user asked to raise ECGData.mat validation accuracy, then
approved the proposed next experiment in a separate, fully recorded folder,
ECGData only (no OMI work).

**Motivation, checked on the real file.** Each ECGData row has 65,536 samples
(512 s at 128 Hz). v1 used `n_per_record: 10` windows of 500 samples per row:
1,620 windows, 7.6% of the signal. The non-overlapping grid has 131 windows per
row, 21,222 in all.

**Design decisions (made before any fitting).**
- One change only: all grid windows instead of 10 sampled per row. LFCC config,
  Swin architecture, optimizer, 40 epochs, class weights, seeds 0/1/2 and the
  logistic control are unchanged. Fitting calls the existing
  `attention_ecg.ecgdata_training.fit_swin` / `fit_control` without modification.
- Folds: each patient keeps its v1 fold, so all 80 patients stay held out exactly
  once and train/test patient sets match v1 fold by fold.
- 40 epochs over ~13× more windows means ~13× more optimizer updates. This is
  accepted as part of the tested change and is recorded as a confound.
- Primary outcome: window macro-F1 of the new Swin minus v1 Swin, on the
  identical 1,620 v1 held-out windows (paired patient bootstrap, 2,000 draws,
  seed 20260925). All-window metrics and patient votes over all windows are secondary.

**Files written.** `protocol.json`, `study.py`, `run.py`, `tests/test_study.py`.
Source hashes at launch preparation (repository HEAD b018f78, files uncommitted):

| File | SHA-256 |
|---|---|
| protocol.json | 8fc29ab53593a14b70052ad17cdc9a5023d504cb84114579bead997aa37bd008 |
| study.py | 12bd72ed2872d3e3994546267a262cbaf91044c48ca4f96810fc09b41883a1b7 |
| run.py | 0347c9e9e903d175a448b348142217986cbd7af2f38320dd20a42b895b8c1ddf |
| tests/test_study.py | 94e9b2e7436d27382f6bfa4de8389664cdec865cdf47e00f2895f0702cb578bf |

**Environment.** `/home/jaydenlee/venvs/test-ecg-training/bin/python` 3.12.3
with `PYTHONPATH=attention method/.venv/torch-cu128` (PyTorch 2.7.1+cu128,
NumPy 2.5.2, scikit-learn 1.9.0). The plain venv has no torch; the first
inspection attempt without that PYTHONPATH failed with `No module named 'torch'`.
pytest is not installed; tests use unittest. RTX 5070, CUDA visible.

**Tests.** 6/6 synthetic unittest checks passed in 33.8 s
([log](logs/tests_2026-10-23.log)): v1-style sampled windows are exact grid
windows; off-grid starts and fold-crossing patients are rejected; identical arms
give zero bootstrap difference and a strictly better arm a positive interval;
a 2-epoch CPU Swin and logistic fit complete with no train/test patient overlap.

**Extraction (03:53:32–03:53:57Z, 17.2 s).** `run.py prepare`
([log](logs/prepare_2026-10-23.log)). 21,222 × 13 frames × 1 lead × 16 LFCC,
80 patients, 162 rows, no exclusions, all finite.

| Class | Windows | Patients |
|---|---:|---:|
| ARR | 12,576 | 47 |
| CHF | 3,930 | 15 |
| NSR | 4,716 | 18 |

Checks passed: ECGData.mat SHA-256 matches v1; the 1,620 v1 windows are exactly
grid windows, with bit-identical z-scored samples (against the hash-checked v1
loader) and bit-identical LFCC features (against the v1 cache); labels, patients,
rows, starts and folds align. Cache manifest SHA-256
22c32be171497c96bdb89ff84cebb0eb3cc6092a92ee8a44a7cff3de720974b9.

**Concurrent work.** A separate OMI workflow (not part of this study, not touched)
was using the same GPU at launch. Sharing the GPU can lengthen both runtimes
but does not change deterministic results.

## 2026-10-24 03:55Z — training launched

Command (from this folder, background):
`CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=…/torch-cu128 python -B run.py train --execute-training --device cuda`,
console to [logs/training.log](logs/training.log). Order: per fold, logistic then
Swin seeds 0/1/2. First logistic fit finished in 0.3 s; first Swin epoch took
10.5 s (~7 min per 40-epoch fit; ~1.75 h expected for 15 Swin fits, longer if
the GPU stays shared). No outer-fold result was inspected during training.

## 2026-10-24 05:11Z — training complete

All 20 fits finished at 05:11:23Z, exit code 0, no failure records.
Session time 4,611 s (1 h 17 min); summed fit time 4,591 s. Swin fits took
298–332 s each; logistic fits 0.1–0.3 s with no warnings. Each fold trained on
~16,800–17,000 windows (v1: ~1,300). Final weighted training loss for every Swin
fit was 1.8e-6 to 3.5e-6, i.e. the models still fit their training windows
essentially perfectly. A separate OMI job shared the GPU throughout.

## 2026-10-24 ~05:13Z — reporting, one bug, fixed without touching guarded code

- `run.py report` wrote `analysis.json`, `metrics.csv`, `fits.csv` and the
  prediction export, then crashed in `study.write_markdown` with
  `KeyError: 'acro_f1'`: it slices `patient_vote_` keys with `[14:]` instead of
  `[13:]` ([log](logs/report_2026-10-24.log)). The numbers were unaffected.
- First fix edited `study.py`; the rerun was then correctly refused by the cache
  guard (`Source changed since extraction`) because `study.py` is hashed.
- The edit was reverted (SHA-256 back to 12bd72ed…a1b7, matching launch). The
  corrected writer lives in the new [report_markdown.py](report_markdown.py),
  which only reads `analysis.json`. `run.py report` was rerun (same KeyError
  after saving the same outputs, [log](logs/report_2026-10-24_rerun.log)), then
  `report_markdown.py` produced [reports/report.md](reports/report.md).
- Sanity check: the v1 rows recomputed here equal the saved v1 report
  (Swin 83.35% / 0.8006 / 94.17%; logistic 76.48% / 0.7342 / 87.50%).

Report file hashes: analysis.json 646143e4…94707, metrics.csv bc3e8349…e1c1b,
fits.csv b1e24…c5097, report.md 6aa45052…14053, report_markdown.py 1d9ef7c9…88603.

## Results

**Primary (identical 1,620 v1 windows, mean of seeds 0/1/2):** window macro-F1
0.8390 vs v1 0.8006, difference **+0.0384 [+0.0153, +0.0602]**; the interval
excludes zero. Window accuracy 86.83% vs 83.35% (+3.48 pp [+1.58, +5.38]).
Patient-vote accuracy 95.83% vs 94.17% (+1.67 pp [−1.25, +5.00]); not established.

| Seed | New window acc. | v1 window acc. | New patient vote | v1 patient vote |
|---|---:|---:|---:|---:|
| 0 | 87.53% | 81.98% | 97.50% | 92.50% |
| 1 | 87.10% | 84.75% | 96.25% | 96.25% |
| 2 | 85.86% | 83.33% | 93.75% | 93.75% |

Window recall ARR / CHF / NSR: 90.94 / 76.11 / 84.81% (v1 88.82 / 72.00 / 78.24%).
CHF remains the weakest class.

**Logistic control:** 77.65% / 0.7462 on the v1 windows (v1 76.48% / 0.7342);
macro-F1 change +0.0120 [−0.0053, +0.0301], not established. New Swin minus new
logistic: +0.0928 macro-F1 [+0.0424, +0.1479].

**All 21,222 held-out windows:** Swin 86.06% window accuracy, macro-F1 0.8290,
patient vote 95.42% (macro-F1 0.9335); logistic 78.07%, 0.7505, 88.75%.

**Interpretation limits.** More windows and ~13× more optimizer updates changed
together and are not separated. Swin still reaches near-zero training loss. The
80 patients and folds have informed earlier decisions; this is internal evidence,
not an external test. Intervals are pointwise, condition on saved predictions and
exclude retraining uncertainty. No clinical claim; ARR/CHF/NSR only.

**Artifacts.** `results/all_windows/` (36 MB: 20 trial folders with models,
checkpoints, histories, predictions; `predictions_export.npz`) and
`features/all_windows/` (98 MB) are local and gitignored.
