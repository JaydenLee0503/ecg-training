# Attention method experiment log

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
