# OMI LFCC–Swin and logistic training notebook

[omi_lfcc_swin_logistic_training.ipynb](omi_lfcc_swin_logistic_training.ipynb)
contains saved outputs for every code cell. It explains the patient split,
undersampling, LFCC features, Swin/logistic models, training, calibration,
validation results and patient-bootstrap threshold analysis in manuscript order.
By default it verifies and reads the completed study; it does not train again.

To execute the notebook, use the repository checkout, the existing Python
interpreter and `attention method/.venv/torch-cu128`, or install equivalent
NumPy, SciPy, pandas, scikit-learn, joblib, PyTorch and Jupyter dependencies in a
compatible environment. The working directory must be inside the repository.
Required local, gitignored files include:

- `experiments/acs_omi_vmd_wst_vqc/omi_improvement_v2/results/manifest.json`
  and its `results/prepared/` folder (cohort, LFCC cache and pooled features).
- `experiments/acs_omi_vmd_wst_vqc/undersampled_v1/results/` (completed model
  checkpoints, calibration and validation predictions, indices and markers).
- Versioned threshold-comparison and undersampling analyses and reports.

A Git clone alone does not contain these large outputs. The study loader/runner
will also verify its frozen source files. The default notebook does not read raw
ACS archives or official test waveforms. Python's `fcntl` dependency requires
Linux/macOS or Windows with WSL.

To intentionally train one new logistic model and all three new Swin seeds, set
`OMI_NOTEBOOK_TRAIN=1` **before** starting the kernel. This uses an isolated
`notebooks/results/notebook_retraining_v1/` directory, a context-hashed manifest,
a writer lock and the original selected patients. It fits on subfit patients,
chooses cutoffs on separate calibration patients, then writes validation
predictions. It does not overwrite the completed study. The results tables later
in the notebook still show the original completed comparison; fresh-fit metrics
are saved in the new output directory. GPU is optional but recommended. These
are exploratory internal results, not clinical validation.
