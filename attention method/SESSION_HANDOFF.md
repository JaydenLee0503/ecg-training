# Attention method handoff — 2026-09-25

## Learning-curve protocol drafted — 2026-09-29 EDT

A draft [learning-curve protocol](ecgdata_learning_curve_protocol.md) is written
(45 new Swin + 45 logistic fits planned at 25/50/75% of training patients; 100%
reuses the completed run read-only). Runner and tests are **not implemented**;
nothing has been fitted. The completed run below is unchanged.

## Training completed — 2026-09-29 EDT

All 15 Swin fits (seeds 0/1/2 × five patient folds) and five logistic controls
completed on the unchanged ECGData protocol. The user authorized training after
repository checkpoint `eba3a78`; the earlier instruction to defer is superseded.
**Do not retrain this completed experiment.** Verify/reuse the saved artifacts.

- RTX 5070, PyTorch 2.7.1+cu128, float32, deterministic algorithms, TF32 disabled.
- Training routine: 407.645 seconds; 600 saved epoch records, no failed real fits.
- Swin means: 83.35% window accuracy, 0.8006 macro-F1, 94.17% patient-vote accuracy.
- Pooled LFCC logistic control: 76.48% window accuracy, 0.7342 macro-F1.
- All 21 CPU tests, GPU forward/gradient and synthetic recovery checks, and
  reload checks of all 20 models passed. No logistic warnings.
- All 134 protected files were unchanged after training. ACS remains paused.

Read [results and limitations](reports/ecgdata_lfcc_swin_v1/results.md) and the
[execution/environment guide](reports/ecgdata_gpu_execution_2026-09-29.md).
Intervals are conditional on saved predictions; this 80-patient cohort has
informed prior decisions and confounds source with diagnosis. All Swin models
fit their training windows perfectly; held-out performance is lower.

Models and completion markers: `results/ecgdata_lfcc_swin_v1/` (gitignored).
The checksum inventory, full report, all seed metrics, 6,480 new predictions,
600 epoch records and figures are in `reports/ecgdata_lfcc_swin_v1/`.
Each final Swin checkpoint includes model/optimizer/scheduler/RNG state and
the full history; older epoch weight files are replaced at each committed save.

The persistent dependency target is `.venv/torch-cu128/` in this directory,
loaded through `PYTHONPATH`, with the original project Python interpreter.
The old `/tmp/ecg-attention-deps` installation is absent. See the execution
guide for exact commands. `report` performs no fitting; `train` on an unchanged
completed run verifies/reuses its fits. No follow-up tuning was performed or scheduled.

The sections below record the earlier preparation state.

## Latest: ECGData preparation

The user approved adapting the attention pipeline to `ECGData.mat`, building its
CPU feature cache and preparing the training runner while leaving ACS intact.
See [ECGDATA.md](ECGDATA.md), [ecgdata_protocol.json](ecgdata_protocol.json) and
[preparation report](reports/ecgdata_preparation.md). The cache contains the exact
1,620 corrected baseline windows and patient folds, represented as 13 × 1 × 16
LFCC values each. The new compact model uses widths 16/32 and three outputs.

The explicit-only ECGData runner now implements future 15 Swin and five logistic
fits, checkpoints, saved predictions and patient-cluster reporting. Its optimizer
path has not run. Preparation and verification are separate commands and do not
start fitting. Keep the current GPU workload undisturbed. The original ACS
protocol and adapter are unchanged. The paragraphs below describe earlier ACS
implementation status; they do not negate the new ECGData runner.

## Earlier ACS implementation

Feature extraction and the temporal Swin model are implemented and pass 11 CPU
checks. Start with [README.md](README.md), [protocol.json](protocol.json), and the
[implementation report](reports/implementation.md). Defaults are LFCC sequences
for 12-lead, 500 Hz, 10-second ACS ECGs, with binary OMI output. All model weights
remain random; there is no trained checkpoint or predictive performance.

The existing split was verified by hash and patient boundaries, not regenerated.
Actual ACS signals have not been extracted for this method. Original VMD/WST
experiments and their frozen artifacts remain unchanged. This separate folder
name/location follows the user's explicit request.

Temporary CPU PyTorch dependencies are at `/tmp/ecg-attention-deps`; the existing
project interpreter supplies NumPy/SciPy. Commands in README run synthetic inference
and tests only. No executable training entry point or scheduled job exists.

Pending for a later training request: real-data engineering verification, resumable
extraction with provenance, persistent normalization packaging, declared training
budget, training runner and classical controls, full prediction/uncertainty reporting.
Keep official test patients reserved and retain all seeds 0/1/2 and failures.
