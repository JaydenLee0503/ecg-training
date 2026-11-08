# ECG-09 experiment log

## 2026-10-31 EDT / 2026-11-01 UTC — protocol, tests and launch

The user requested the proposed ECGData comparisons and full logbook records.
A separate frozen protocol compares 32 log-filterbank features against the
previously completed all-window LFCC baseline, with matched Swin and logistic
settings and a raw-waveform CNN control. Same 21,222 windows, 80 patients, five
patient folds, 40 epochs, seeds 0/1/2; primary comparison is macro-F1 on the
identical 1,620 v1 windows. The specified comparison requires 35 new fits and
reuses 20 verified ECG-07 fits; no additional patient cohort was requested.

Three synthetic CPU tests passed in 16.615 s: filterbank/LFCC DCT consistency
and batch independence, finite raw-CNN gradients, patient-separated 1-epoch raw
fit with verified checkpoint reuse. See `tests/test_study.py`.

Feature preparation reconstructed the source windows and folds and produced
21,222 × 13 × 1 × 32 log-band features without exclusions in 3.941 s (feature
stage, excludes earlier data loading/checks). Maximum reconstructed LFCC error
versus the baseline was 7.62939453125e-06; see `preparation.log` and
`features/extraction.json`. The host GPU run started with the explicit training
flag; initial timestamp 2026-11-01T03:46:15Z. Device/software and source hashes
are recorded in `results/manifest.json`, source files and input completion hashes
in `features/manifest.json`; output/checkpoint files remain gitignored. Progress
and failure information: `results/status.json`, per-fit completion files, epoch
histories, and `training.log`. Do not launch a duplicate writer or infer results
from partially completed fits. Check the markers for actual progress.

After launch, an additional synthetic test fitted and verified both the 32-band
Swin and logistic arms on patient-disjoint dummy data; all four CPU checks passed
in 17.498 s (`tests/test_run.log`). The original three checks passed before launch.
During the first real fit the 32-band Swin saved epoch checkpoints normally.
The full study is substantially longer than the original small-window tests;
check live status and per-fit markers rather than inferring a completion time.

Two additional post-launch checks passed: an interrupted 2-epoch raw-CNN CPU fit
resumed with bit-identical final weights and predictions (16.423 s), and CPU/GPU
forward/gradient comparisons passed for both the 32-band Swin and raw CNN (3.779 s).
Their exact commands and outputs are in `tests/test_raw_recovery.py`,
`tests/recovery_test.log`, `tests/test_cuda.py`, and `tests/cuda_test.log`. The
first full 32-band Swin fit (fold 0, seed 0) completed in 331.6 s; subsequent
fits continue. This is a runtime observation, not a final result or reliable ETA.

Copied the feature/training manifests, feature completion marker and extraction
summary to the versioned `provenance/` directory while the immutable run was
active. The copies record the original source/input hashes; the large data,
checkpoints and predictions remain in gitignored directories.

The first completed 32-band Swin fit passed an independent check of its
training-only normalizer and a CPU checkpoint reload on eight held-out windows;
maximum probability difference versus saved GPU inference was 2.0861626e-07.
No model selection was performed. `tests/audit_first_fit.py` and
`tests/first_fit_audit.log` retain the reproducible check.

A separate one-shot `monitor.py` was started after the training writer. It only
waits for the writer to exit, verifies that both the run and report finished,
then runs the independent `audit.py` once. It never starts training or touches
checkpoints. `monitor_status.json`, `monitor.log` and eventual `audit.log` record
its actual state; absence of an audit pass must not be interpreted as validation.

Two isolated completion-monitor tests passed (`tests/monitor_test.log`): an
incomplete writer never triggers audit, and a completed run triggers the audit
once then reuses a passed audit. These tests use temporary dummy artifacts; the
real monitor remains active and its final audit has not yet run.

The first full raw-CNN fit (fold 0, seed 0) completed in 60.1 s. The existing
ECGData manuscript notebook gained a status-aware ECG-09 section and all 11 code
cells were re-executed with saved outputs; it displays results only after the
independent full-run audit passes. Initial notebook-output inspection had a JSON
text-list handling TypeError, corrected in a repeat verification; the notebook
execution itself succeeded. No interim scores were used for decisions.

All three fold-0 raw-CNN fits completed at 40 epochs and passed an independent
CPU checkpoint reload and patient-boundary spot check. Maximum CPU/GPU saved
probability differences were 1.28e-08, 1.77e-08 and 1.86e-09 for seeds 0/1/2;
`tests/audit_first_raw.py` and `tests/first_raw_audit.log` preserve the check.
The study continued to later patient folds without inspecting interim scores.

At 2026-11-01T04:24:34Z, a read-only progress snapshot captured 15 complete
marker statuses and SHA-256 digests and the current training log, without inspecting
held-out scores. See `progress/20261101T042434Z.json` and `.log`; the snapshot
records local marker hashes but does not back up the gitignored checkpoints.

At approximately 2026-11-01T04:30Z, a read-only progress check verified the completion markers and recorded
artifact hashes for the first 16 completed fits; no interim scores were inspected.
The run remained active at that snapshot. This check does not replace the final
full-cohort, all-fit and patient-bootstrap audit.

## 2026-11-07 EDT / 2026-11-08 UTC — restart and verified continuation

The user asked to continue after a restart. At inspection the old status said
running with the stale 2026-11-01 PID; no host writer or monitor was alive.
There were 23 completed fit markers, including all three full outer folds and
fold 3 logistic plus Swin seed 0. Read-only SHA-256 checks verified all 14
frozen source files, all 117 files listed by the 23 completed fits, the feature
manifest and all three feature-cache marker entries with zero mismatches. The
interrupted fold 3 seed 1 checkpoint was at epoch 6 and its stored checksum
matched. Python 3.12.3, NumPy 2.5.2, PyTorch 2.7.1+cu128 and the host NVIDIA
GeForce RTX 5070 matched the run manifest. No held-out predictions were inspected
or used to change the protocol.

The initial inspection helper erroneously requested a nonexistent nested `versions`
key after completing its fit/source checks; a corrected separate check verified
the actual top-level version keys and feature checksums. The sandbox did not
expose `/dev/dxg`; the authorized host check found CUDA available. These were
inspection/environment issues, not training failures.

Resumed the unchanged runner on the host GPU. At 2026-11-08T00:04:31Z it
verified and reused the 23 completed fits and began continuing the saved epoch-6
checkpoint. New writer PID at launch: 430581; PID is not durable. Console:
`training_resume_2026-11-07.log`. Restarted the separate one-time audit monitor
at 00:04:46Z; its output is `monitor_resume_2026-11-07.log` and its current
status is `monitor_status.json`. It waits for this writer and then audits only a
completed run/report. No final result or audit pass exists at this launch snapshot.

Prepared `refresh_notebook.py` for the post-audit notebook update. It verifies
that the ECG-09 report and successful audit exist, disables fresh model training,
executes all notebook cells, and only replaces the notebook after checking every
cell output and ensuring the original did not change during execution. Running
it before audit produced the expected `RuntimeError` and did not alter the
notebook. Two synthetic safety tests passed in 0.050 s, including a concurrent
notebook edit; see `tests/refresh_test.log`. The real refresh remains pending the
completed experiment and final audit.

The post-restart host environment inventory is saved in
`restart_environment_2026-11-07.json`; the reproducible helper is
`capture_restart_environment.py`. Current driver is 610.88, GPU RTX 5070 with
12,227 MiB, and the recorded Python, NumPy, SciPy, scikit-learn, pandas, joblib,
PyTorch and CUDA versions match the original run manifest or post-launch
inventory. The original ECG-09 manifest did not record the GPU driver, so its
version cannot be verified against launch. This is a provenance limitation, not
proof of numerical drift. The fold 3 Swin seed 2 fit completed in 383.7 seconds;
its longer runtime is recorded without assigning an unverified cause.

Before the real run finished, exercised the complete ECG-09 reporting path with
30 synthetic examples and 35 simulated fit records. The test generated both
report sets, the prediction export and the full 2,000-draw paired bootstrap;
it passed in 2.086 s (`tests/test_reporting.py`, `tests/reporting_test.log`).
The first successful synthetic run printed dummy scores; the test was rerun
with those scores suppressed to avoid confusing them with research results.
No real held-out scores were inspected or used for decisions in this check.

## 2026-11-07 EDT / 2026-11-08 UTC — completion and audited notebook

The resumed runner reused all 23 completed fits, continued the intact epoch-6
checkpoint, completed the remaining 12 fits, wrote the report and exited 0.
The completion marker reports 35 new fits; fit-time sum 6,343.7 seconds excludes
feature preparation/reporting and spans the interruption. The one-time monitor
finished the independent audit at 2026-11-08T00:47:39Z with exit 0: all 35 fit
markers, 35 reload spot checks, 15 normalizers, patient isolation, 10 metric
sets and 2,000 primary bootstrap draws passed; no draw was rejected. Full
settings, all per-seed results, confusion matrices, per-fit runtimes and recorded
warnings: `reports/analysis.json`. Training, monitor and audit logs are saved.

Primary window macro-F1: log-band Swin 0.8846 versus LFCC Swin 0.8390; paired
difference +0.0457, 95% interval [+0.0202,+0.0800]. Raw CNN 0.5795 was worse;
log-band logistic was 0.8345. Patient-vote accuracy difference has interval
[0.0,+0.0542]. These are conditional, pointwise intervals on a small repeatedly
examined cohort with diagnosis/source confounding; no external validation.
Representations also change projection dimensions and raw CNN changes model.
Read `reports/COMPLETION_REVIEW.md` and the frozen protocol before interpretation.

After the audit, `refresh_notebook.py` executed the ECGData manuscript notebook
on the host, using the existing interpreter and disabling fresh training. All
11 code cells have saved, non-error output, including the ECG-09 results.
Jupyter emitted its local TCP transport warning; the notebook completed. See
`notebook_refresh_2026-11-07.log`. Exported all 21,222 saved out-of-fold window
class predictions to versioned `reports/predictions.csv.gz` (SHA-256
e38a9b7b5238b1c800543d1fd1c27cfbb57db0678b8fa424242a0fd73fe3a9be).
`export_predictions.py` verified every primary window accuracy and macro-F1
against the audited analysis, 80 patients and one held-out fold per patient.
Full probabilities, models, caches and bootstrap draws remain local/gitignored.
No model was retrained during the notebook refresh or prediction export.
