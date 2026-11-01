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
