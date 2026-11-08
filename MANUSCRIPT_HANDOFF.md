# Manuscript notebook handoff

The two notebooks contain saved outputs and can be opened directly for review.
Re-executing them requires the complete repository and the local artifacts below.
A Git clone alone omits data, caches, models and predictions. Keep the paths
unchanged when copying; the notebooks verify frozen inputs and sources.

| Notebook | Question and default behavior | Output |
|---|---|---|
| [ECGData LFCC–Swin](attention%20method/reproduce_swin/lfcc_swin_training.ipynb) | Three-class ARR/CHF/NSR classification; verifies and reuses completed Swin and logistic fits. | 11 code cells executed, including ECG-09 report. |
| [OMI LFCC–Swin and logistic](experiments/acs_omi_vmd_wst_vqc/notebooks/omi_lfcc_swin_logistic_training.ipynb) | Binary OMI classification; verifies and reuses completed undersampled fits and shows the comparison with oversampling. | Nine code cells executed, including validation thresholds, uncertainty and the audited improvement report. |

## Required local files to re-execute

Start with the whole repository checkout, then copy the following **gitignored**
files into the same relative locations:

**ECGData notebook**

- `ECGData.mat`.
- `results/patient_vmd_wst_vqc/` (frozen reference folds and alignment data).
- `attention method/features/ecgdata_lfcc_v1/` (feature cache).
- `attention method/results/ecgdata_lfcc_swin_v1/` (15 Swin and five logistic
  completed fits and predictions).
- `experiments/ecgdata/08_patient_learning_curve/results/completed.json` to
  display the completed ECG-08 report after re-execution. Without it the notebook
  reports that ECG-08 status is unavailable. ECG-09's report and audit are
  versioned, so they remain displayable without its model checkpoints.

**OMI notebook**

- `experiments/acs_omi_vmd_wst_vqc/omi_improvement_v2/results/manifest.json`
  and `experiments/acs_omi_vmd_wst_vqc/omi_improvement_v2/results/prepared/`.
- `experiments/acs_omi_vmd_wst_vqc/undersampled_v1/results/` (selected cohort,
  completed models, predictions and completion markers).

The OMI notebook's default execution does not read raw ACS archives or official
test waveforms. See its [README](experiments/acs_omi_vmd_wst_vqc/notebooks/README.md)
for environment and optional retraining details. The ECGData notebook's
[README](attention%20method/reproduce_swin/README.md) describes its dependencies.
Use Linux/macOS or WSL and a compatible scientific Python/Jupyter environment;
the notebooks report the recorded versions. A GPU is useful for retraining,
but the saved-output workflow reuses existing fits.

For a full research archive also preserve the gitignored model, cache, prediction,
and bootstrap-result folders in the corresponding experiment directories; the
small versioned reports alone do not reconstruct those files. Approximate current
local sizes are 67 MB for ECGData.mat, 31 MB for its reference result folder,
83 MB for its LFCC cache, 36 MB for its v1 fits, 642 MB for the OMI prepared
folder and 191 MB for OMI undersampled results. Sizes and filesystem allocation
can vary. Dataset sharing and local storage are separate from a Git commit.

## Reading the results

- [ECG-09 completion review](experiments/ecgdata/09_representation_comparison/reports/COMPLETION_REVIEW.md),
  [log](experiments/ecgdata/09_representation_comparison/EXPERIMENT_LOG.md) and
  [held-out class predictions](experiments/ecgdata/09_representation_comparison/reports/predictions.csv.gz).
- [OMI undersampling study](experiments/acs_omi_vmd_wst_vqc/undersampled_v1/README.md)
  [threshold comparison](experiments/acs_omi_vmd_wst_vqc/threshold_comparison_v1/REPORT.md),
  and the separately completed [OMI model improvement study](experiments/acs_omi_vmd_wst_vqc/omi_improvement_v2/reports/SUMMARY.md). The notebook also displays the audited improvement report.

ECGData is a small, repeatedly examined cohort with diagnosis/source confounding;
the reported internal improvements are not external clinical validation. OMI
precision and sensitivity depend on the cutoff and prevalence. Its threshold
analysis uses internal validation patients, and the official OMI test waveforms
were not used.
