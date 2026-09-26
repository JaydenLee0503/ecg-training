# ACS OMI detection: VMD + VQC versus WST + VQC

This experiment asks whether either feature pipeline helps a fixed quantum
classifier detect OMI in the ACS dataset, and how both pipelines compare with
classical classifiers on the same patient split and selected features.

The current workspace is `experiments/acs_omi_vmd_wst_vqc/`. The descriptive
rename is complete; explicit hash-checked migration preserves the older runs
without compatibility links. The
[session handoff](SESSION_HANDOFF.md) records actual progress; the
[log guide](LOGS.md) explains where the evidence is saved. The authoritative
scientific settings are in [OMI protocol v1](protocols/acs_omi_protocol_v1.md)
and its [JSON specification](protocols/acs_omi_protocol_v1.json).

## The comparison

| Feature pipeline | Classifiers using the same selected features |
|---|---|
| VMD: eight modes per lead and 28 descriptors per mode | VQC seeds 0/1/2; weighted KNN; logistic regression |
| Standard WST: pooled scattering features | VQC seeds 0/1/2; weighted KNN; logistic regression |

Each ECG supplies all 12 leads, the full ten seconds at native 500 Hz, in mV.
Both arms use the same eligible records and patient assignments. There is no
extra filtering, normalization of the input waveform, or resampling in v1.
Each arm independently fits mRMR to select 12 features and then fits its angle
scaler **using the fit partition only**. The resulting inputs are shared by
that arm's quantum and classical classifiers.

The VQC has six qubits, two circuit blocks, one input encoding, 40 epochs,
batch size 32, balanced loss, and initialization seeds 0, 1 and 2. There are
six VQC fits in total. Weighted KNN uses k=10 and inverse-distance weights;
logistic regression uses C=1 and balanced class weights. Constant-prevalence
scores and always-negative predictions provide baseline controls.

## Data and evaluation

- Task: OMI versus non-OMI at the ECG-record level.
- Development cohort: 17,905 eligible official-training ECGs, with 55 source
  training ECGs excluded under the declared all-12-lead quality policy.
- Fit partition: 14,324 ECGs from 13,576 patients; 917 OMI-positive records.
- Validation partition: 3,581 ECGs from 3,391 patients; 229 OMI-positive records.
- Split seed: 20260924. Five patient folds define the split, with fold 0 held
  out for this fixed internal validation; this is not a five-fold model result.
- Patients never cross partitions. Different ECG labels from the same patient
  remain record-level labels; no patient majority label is invented.
- Official test: all 1,995 records remain reserved. Labels are withheld; two
  records fail the quality policy. Coverage/submission handling and access to
  official scoring must be resolved before any final-test evaluation.

The primary metric is **average precision**, a precision-recall summary.
Always predicting non-OMI already gives 93.605% validation accuracy, so a high
accuracy alone would not demonstrate useful detection. Also report ROC AUC,
balanced accuracy, sensitivity, specificity, precision, F1, accuracy and the
confusion matrix at the fixed 0.5 threshold.

Report all three VQC seeds and paired patient-cluster bootstrap intervals
(2,000 resamples, seed 20260925). The protocol has no hyperparameter search or
early stopping. A later tuned experiment needs a separate declared protocol
and output directory. These are internal-validation results, not evidence of
clinical validity, quantum advantage, or equivalence to a paper's 94% result.

## Work stages and status as of 2026-09-25

| Stage | Status and next action |
|---|---|
| Verify source data, implement loader and document exclusions | Complete |
| Freeze settings and patient split | Complete |
| Validate VMD/WST and extraction checkpoints | Two-record serial check and 16-record parallel pilot complete; 31 ACS tests passed |
| Extract all development features | Stopped at 1,795 verified ECGs: record 01985/V5 reached the 32,000 cap; CPU diagnostic reproduced it |
| Investigate acceleration | Complete: FP64 GPU prototype passed four device tests and all 16-ECG checks; 19.06× faster than the eight-worker CPU baseline for VMD/descriptors |
| Integrate GPU execution for full extraction | Complete: 37 CPU-side and four device tests passed; integrated 16-ECG pilot and stop/resume recovery passed |
| Verify and bundle the complete GPU feature archive | Pending; needs a bundler for the GPU manifest |
| Select/scale features and fit classical controls plus all six VQCs | Pending; fitting runner and durable model/history output still needed |
| Save validation predictions, metrics, all seeds and uncertainty | Pending; statistical reporting runner still needed |
| Official-test evaluation | Reserved; outside the current extraction/training protocol |

The CPU pilot extrapolated to about three days of feature extraction. The
completed [GPU benchmark](reports/gpu_vmd_benchmark_2026-09-25.md) instead
extrapolates to 3.63 hours for VMD/descriptors, excluding WST, source/file
handling and all classifier training. This is a 16-record estimate, not a
full-run measurement. VQC training time remains unmeasured. GPU production
integration and pilot validation are complete; full extraction later stopped at a convergence limit. Read the stop report before resuming.
Extraction does not automatically launch training.

All changes, failures, exclusions and negative findings belong in the
[experiment log](EXPERIMENT_LOG.md), with detailed reports and saved artifacts.
Check completion markers before resuming; a restart is not a reason to recompute
verified completed ECGs or retrain completed models.
