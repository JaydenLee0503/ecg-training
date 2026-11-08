# ECGData LFCC–Swin manuscript notebook

[lfcc_swin_training.ipynb](lfcc_swin_training.ipynb) is the current executed
manuscript workflow. It covers patient-separated folds, LFCC features, the
original Swin and logistic controls, follow-up experiments and the audited
ECG-09 representation comparison. All 11 code cells have saved output. To
**view** those outputs, send the notebook itself; to **run** it, send the
repository and local artifacts described below.

## Running or sharing the current notebook

Use a complete checkout of this repository, preserving the relative paths, plus
these local files and directories, which Git does not provide:

- `ECGData.mat` at the repository root.
- `results/patient_vmd_wst_vqc/`: frozen patient folds, manifest and features.
- `attention method/features/ecgdata_lfcc_v1/`: verified LFCC feature cache.
- `attention method/results/ecgdata_lfcc_swin_v1/`: all 15 Swin fits and five
  logistic controls with their completion markers and predictions.
- `experiments/ecgdata/08_patient_learning_curve/results/completed.json` if the
  notebook should show the completed ECG-08 report on re-execution. Without this
  marker that section reports the status as unavailable; the saved output in the
  notebook remains visible without executing it.

The ECG-09 report and audit are versioned in
`experiments/ecgdata/09_representation_comparison/reports/`. The full ECG-09
feature cache, checkpoints and additional fit outputs are local/gitignored; they
are not required merely to display its report in this notebook. For a full
research archive, include those artifacts as well. The separate
[handoff guide](../../MANUSCRIPT_HANDOFF.md) covers the OMI notebook and
remaining files.

Start Jupyter with a working directory inside the repository, using an environment
with NumPy, SciPy, pandas, scikit-learn, joblib, PyTorch, Jupyter and IPython.
The recorded run used Python 3.12.3 and PyTorch 2.7.1+cu128; the environment
also used `attention method/.venv/torch-cu128` for its PyTorch installation.
Linux, macOS or WSL is needed for the code's `fcntl` file locks. The notebook
verifies its frozen sources and completed artifacts and normally reuses saved
fits. Set `ECG_RUN_FRESH=1` **before** starting the kernel only when a separate
full v1 retraining run is intentionally desired; it writes to a different output
directory. A different GPU or library version may produce different weights.

## Historical bundle

The `ecgvmd/` and `attention method/` copies inside this folder belonged to an
older self-contained training bundle. Its old Colab instructions applied to an
older notebook and must not be used to run the current manuscript notebook.
The current notebook searches for and imports the project checkout and reads the
saved artifacts listed above; copying this folder alone is insufficient for a
full execution. The historical bundle also omits the later controls and reports.
