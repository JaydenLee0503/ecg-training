# ACS OMI development comparison — completed

The workflow completed at **2026-10-02T03:57:53 UTC** (October 1, 23:57 EDT).
All **17,905 development ECGs**, **10 model fits** and the final statistical
report are saved. There is no unfinished ACS extraction or model job to resume.
Official test data remain reserved.

## What finished

- Full feature verification/bundling passed: 2,688 VMD descriptors and 2,808 WST
  features per ECG, with zero final iteration-capped leads.
- The frozen patient split contains 14,324 fit ECGs and 3,581 validation ECGs
  from disjoint patient groups. Validation contains 3,391 patients and 229
  OMI-positive ECGs. Training-only preprocessing and the fixed 0.5 decision
  threshold were retained.
- All six VQC fits completed 40 epochs: VMD and WST, each with seeds 0/1/2.
  All 240 epoch records and their checkpoints are saved. Both front ends also
  completed logistic regression and weighted KNN controls.
- All 2,000 paired patient-bootstrap draws were valid (seed 20260925). VQC
  summaries average the three seed metrics. Intervals condition on the saved
  predictions and do not include retraining uncertainty.
- A post-run check verified **516 model/report artifact entries**, with no missing
  files or hash mismatches. All ten trial records contain empty warning lists.

## Main findings

- Primary metric: average precision (AP). **VMD + VQC: 0.1442**; **WST + VQC:
  0.1352**. Their paired difference is **+0.0090**, with a 95% interval of
  **[-0.0190, +0.0353]**. This does not establish transform superiority.
- Logistic regression has higher AP point estimates: **0.1540 with VMD** and
  **0.1569 with WST**. Each VQC-minus-logistic AP interval includes zero.
  The comparison does not establish a VQC advantage over the classical controls.
- Weighted KNN AP is **0.1206 with VMD** and **0.1383 with WST**. At the fixed
  threshold, sensitivity is only **1.31%** and **2.62%**, respectively. Its
  roughly 94% accuracy must be read alongside the **93.61% always-negative
  baseline accuracy** and the rare positive class.
- Both VQC arms exceed the constant-score AP baseline (0.0639), with positive
  paired AP intervals. These are fixed internal-validation results; they do not
  establish external or clinical validity or quantum advantage.

The full [statistical report](report.md) includes all grouped results and paired
primary-metric intervals. [metrics.csv](metrics.csv) retains every initialization
seed and classical control. [analysis.json](analysis.json) retains all reported
metrics and intervals. The Markdown report and analysis JSON match the original
report hashes exactly. The CSV export normalizes CRLF line endings to LF; all
parsed rows were verified identical. [artifact_check.json](artifact_check.json)
records artifact verification, each trial's runtime, and the completion state.

## Timing and saved artifacts

After extraction ended at 03:27:27 UTC, verification/bundling finished at 03:32:03,
model preparation at 03:32:23, all model fitting at 03:57:01, and reporting at
03:57:53. Stage timestamps indicate about 25 minutes for fitting and about
52 seconds for the final report. Each VQC fit recorded roughly 13.2–13.4 minutes
of training with three fits executing concurrently.

- Features/checkpoints: `results/omi_v1_gpu_retry256k/`
- Models, preprocessing, epochs, predictions and source manifest:
  `results/omi_models_v1_retry256k/`
- Workflow status and stage logs: `results/omi_development_workflow_v2/`

Large artifacts remain local and gitignored; the report copies do not replace
the saved models or feature archives. Keep completed runs and frozen protocols
intact. The next research step is to review the saved results and error patterns
before declaring a new experiment; no new tuning or official-test evaluation
was started during this completion review.
