# Attention method experiment log

## 2026-09-29 — Learning-curve protocol drafted; nothing run

At the user's request, wrote [ecgdata_learning_curve_protocol.md](ecgdata_learning_curve_protocol.md)
and its JSON. It asks whether held-out Swin performance still rises as training
patients are added: nested, class-stratified patient subsets of 25/50/75% per fold
(3 draws paired with seeds 0/1/2), same test folds and frozen settings, reusing the
completed 100% trials read-only. Primary contrast is 100% minus 75% window macro-F1
with a paired patient bootstrap and a pre-stated interpretation rule. The runner
is not implemented and no model was fitted.

## 2026-09-29 — ECGData LFCC + Swin GPU training completed

After user authorization and Git checkpoint `eba3a78`, reused the verified
1,620-window LFCC cache and unchanged protocol with five patient folds. All 15 Swin
fits (seeds 0/1/2, 40 epochs each) and five deterministic logistic controls
completed. Swin used the RTX 5070 and isolated PyTorch 2.7.1+cu128; the original
environment and ACS work were unchanged. Training took 407.645 seconds,
excluding setup; no real-data fit failed or generated logistic warnings.

All 21 CPU tests and synthetic GPU/recovery checks passed. All 20 saved models
were reloaded on CPU, reproduced their predicted classes, and passed partition,
normalization, checkpoint and coverage checks. All 134 protected files were
unchanged after fitting; 930 older improvement artifacts passed preflight.

Mean Swin window accuracy/macro-F1: 83.35% / 0.8006; patient-vote accuracy:
94.17%. Logistic: 76.48% / 0.7342 / 87.50%. The primary paired macro-F1 gain
over logistic is +0.0664, 95% patient-bootstrap interval [0.0224, 0.1143].
All Swin fits reached 100% training accuracy, leaving a generalization gap.
This is an exploratory, repeatedly studied 80-patient cohort with source/label
confounding, not external validation. No further tuning was performed.

Setup issues were a sandbox DNS failure and a forward-only invocation without
the PyTorch path; both were resolved before fitting. Windows-drive dependency
copying was slow. All setup issues, seeds, negative findings, controls, histories,
predictions, intervals, runtimes and reproduction commands are recorded in the
[results](reports/ecgdata_lfcc_swin_v1/results.md) and
[execution guide](reports/ecgdata_gpu_execution_2026-09-29.md).

## 2026-09-25 — ECGData preparation completed; training deferred

User approved the ECGData adapter, frozen protocol, CPU cache and training runner
while preserving original ACS data. Exact corrected-baseline window and patient
fold hashes matched. Extracted 1,620 × 13 × 1 × 16 LFCC features in 5.981 seconds;
saved 162 row checkpoints and the combined cache. Small Swin has 33,607 parameters.

All 21 checks passed. CPU forward inference on real fit features left weights
unchanged. Completed-cache reuse, partial-cache recovery and deliberate-corruption
rejection passed. Before/after hashes found zero changes in 134 protected files,
including both ACS archives and original experiment settings/artifacts.

Training/report runners are implemented for later execution, including a logistic
control, three Swin seeds, epoch checkpoints and paired patient-bootstrap reporting.
Neither classifier was fitted; no optimizer steps, CUDA work, trained predictions
or performance evaluation ran. Existing GPU workload was not modified. No preparation
failures occurred. Details: [preparation report](reports/ecgdata_preparation.md),
[validation evidence](reports/ecgdata_validation.json),
[architecture record](../architects/ecgdata_attention_preparation.md).

## 2026-09-25 — architecture documentation

At the user's request, added the consolidated
[architecture record](../architects/attention_method_architecture.md) under
`architects/` and linked it from the repository handoff. It covers the cepstral
extractor and temporal Swin, data safeguards, recorded checks, and pending work.
Documentation only; no extraction, training or test rerun was performed.

## 2026-09-25 — implementation only

Implemented the user-requested cepstral + temporal Swin experiment in its own
`attention method` folder. Current ACS OMI task is the documented default. Reuse
the frozen patient split and verified loader read-only; official test records
are rejected. LFCC is primary; optional mel spacing requires a separate experiment.

All 11 synthetic/data-guard/model checks passed. Full synthetic forward pass
confirmed [37,12,20] input features, [1,2] finite logits and unchanged weights.
The existing split checksum/counts and absence of patient overlap were verified.
No real-data extraction or model training started. There are no predictive results.

Initial pip network access failed in the sandbox; escalated isolated CPU installation
succeeded under `/tmp/ecg-attention-deps`, leaving the existing environment intact.
See [implementation record](reports/implementation.md), [validation evidence](reports/validation.json),
and [declared settings](protocol.json). Future runners, controls and evaluation are
pending; do not infer approval to train from the existence of these files.
