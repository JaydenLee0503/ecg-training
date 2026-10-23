# Attention method handoff — 2026-09-25

## Regularization v2 completed and audited — 2026-10-23 EDT

The authorized study completed on **2026-10-16T04:32:18Z** (00:32 EDT), with
90 inner fits and 30 outer trials, all seeds 0/1/2, and no recorded training
failures. **Do not retrain this completed run.** No epoch-40 R0 models were
selected, so all 30 outer trials required fitting. See the
[completion review](reports/ecgdata_lfcc_swin_regularization_v2/SUMMARY.md) and
[frozen report](reports/ecgdata_lfcc_swin_regularization_v2/report.md).

The predeclared inner-selected procedure reached **84.12% window accuracy**,
0.8099 macro-F1 and 94.58% patient-vote accuracy, versus v1's 83.35%, 0.8006 and
94.17%. Primary macro-F1 difference +0.0093 has paired 95% interval
[-0.0023, +0.0212], so improvement is not established. Dropout R1 reached
84.49% / 0.8144 / 95.83%; its positive pointwise secondary window intervals are
promising but do not replace the primary comparison or establish external validity.
R0 reached 83.46% / 0.8012 / 93.75%; logistic remains 76.48% / 0.7342 / 87.50%.

The October 23 read-only numerical review verified 1,887 artifact entries,
472 protected original files, 10 sources/snapshots and 8 report exports. It
independently checked all patient splits, fit-only statistics, pooled inner
selections, window/patient metrics and 14,580 prediction export rows. All 4,380
epoch records and 300 retained candidate/final prediction sets are accounted for.
The frozen runner had already reloaded all retained weights during execution;
this review performed no new model inference, extraction or training.

Audit reproduction: `/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/scripts/audit_ecgdata_regularization.py'`.
It writes only new audit/inventory/figure exports beside the report. The first
invocation's export failed because the sandbox allowed lowercase `projects`
while the resolved path used uppercase `Projects`; the approved rerun succeeded.
Numerical checks passed. Existing v1 and unrelated ACS/audit edits remain intact.
Further feature, raw-waveform, longer-window and additional-patient experiments
remain unrun separate follow-ups. Large checkpoints remain local and gitignored.

## LFCC + Swin walkthrough notebook — 2026-10-15

At the user's request, [lfcc_swin_walkthrough.ipynb](lfcc_swin_walkthrough.ipynb)
walks through every stage of the completed v1 experiment with **no training**:
raw windows (hash-matched), patient folds, step-by-step LFCC and full cache
rebuild (bit-exact), fit-only normalization, architecture/shifted-window masks,
the training code (printed only), CPU reload of all 15 Swin + 5 logistic models,
metrics and the 2,000-draw bootstrap. Executed in about 1 minute on CPU: predicted classes
identical, max probability difference 1.85e-6, and metrics/bootstrap equal the
saved `report.json` exactly. It writes nothing to `results/` or `features/`.
The running regularization v2 study is not covered.

## LFCC + Swin full-training notebook — 2026-10-17

At the user's request, [reproduce_swin/lfcc_swin_training.ipynb](reproduce_swin/lfcc_swin_training.ipynb)
trains the unchanged v1 protocol, Swin arm only (15 fits, no logistic control),
from `ECGData.mat` into the new `results/ecgdata_lfcc_swin_retrain/`. The notebook
is intended for Colab and lists its minimal input files. It was not run in full here. A
one-fit CPU smoke test (seed 0, fold 0, from a minimal-file copy, since deleted)
reproduced every original predicted class; max probability difference was
0.0053 versus the GPU fit. Results cells, run read-only on the v1 fits, gave
83.35%/0.8006/94.17% with matching bootstrap intervals. Other hardware
or library versions may change weights; results should be close but are not bit-identical.
Later the same day the notebook moved into the self-contained
[reproduce_swin/](reproduce_swin/README.md) upload folder, which holds copies of the raw data,
loader, frozen reference files and code, and gained a cell showing LFCC extraction.
Run from a Colab-like copy, it extracted everything from scratch and trained one CPU fit
that reproduced the original classes.

## Regularization v2 prepared and authorized — 2026-10-15

The user authorized the suggested improvement work. The first bounded study is
[v2 epoch selection and dropout](ecgdata_regularization_protocol_v2.md), with R0
and dropout-0.1 R1, epoch candidates 10/20/40, three inner patient folds and seeds
0/1/2. It preserves all original windows/outer folds and compares both candidates
plus their inner-selected procedure against saved v1 and logistic controls.
Plan: 90 inner fits and 30 outer trials; R0 epoch-40 selections reuse v1 models.

Six new CPU checks passed (one GPU check deferred to launch), including exact
zero-dropout training equivalence to v1, patient isolation, fit-only statistics,
dropout checkpoint recovery and completed-result reuse. Ten original ECGData
tests also passed. Existing feature cache and completed v1 predictions verified.
The RTX 5070 is accessible outside the sandbox with existing PyTorch 2.7.1+cu128.
No dependencies installed. Real training has not started at this preparation entry.

Launch: `/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/scripts/run_ecgdata_regularization.py' --execute-training --device cuda`.
The launcher first runs CPU/GPU agreement and exact GPU recovery checks. It logs
to `reports/ecgdata_lfcc_swin_regularization_v2/execution.log`. Check the matching
`results/.../status.json` and completion markers before a restart; never duplicate
a running job. Do not change the frozen protocol or source during execution.
Feature/raw-waveform, longer-window and additional-patient comparisons remain
separate follow-ups. An existing modified October-12 audit JSON was left untouched.

## Swin literature and improvement review — 2026-10-12 EDT

Completed the user-requested [research review](reports/swin_research_2026-10-12.md)
of the current ECGData code/results and nine relevant 2024–2026 papers, with
publication/access limits and proposed implementation changes. Priority is a
bounded regularization and inner-patient epoch-selection experiment, followed
by representation controls/raw-waveform fusion and external ECG pretraining.
The proposed v2 experiment is not implemented or frozen; no fitting was started.

The [fresh audit](reports/swin_research_audit_2026-10-12.json) verified all 129
saved v1 artifact entries with no mismatches, and checked exported window and
patient-vote accuracies. All 15 Swin fits have perfect training accuracy; mean
window recall is ARR 88.82%, CHF 72.00%, NSR 78.24%. These are descriptive saved
results, not new model evaluations. The audit script uses PowerShell only.
Completed models, features, protocols and ACS work remain unchanged. The
learning-curve draft remains unimplemented. No new training is pending from
this research-only session.

## ECGData rotational-morphology training completed — 2026-10-10

The frozen 30-second Eyeball experiment completed extraction and all five
patient-separated folds. Balanced RBF SVC achieved **81.25% patient-vote
accuracy** (macro-F1 0.7621); balanced logistic achieved 67.50% (macro-F1
0.6766). Extraction coverage was 1,531/1,620 (94.51%); 89 EMD iteration-cap
failures were retained in the failure and prediction exports. Coverage and
fold/class safeguards passed. All 80 patients retained valid clips. Reload and
artifact audit passed; no refit is pending. See the [full report](reports/ecgdata_eyeball_30s_v3/report.md)
and versioned exports beside it. This is ARR/CHF/NSR classification, not MI/OMI.
It is not a controlled comparison with the LFCC Swin result (94.17%), which
uses a different 3.9-second pipeline. Small cohort and source/diagnosis
confounding limit interpretation. Large resumable outputs remain gitignored
under `results/ecgdata_eyeball_30s_v3/`.

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
