# OMI threshold comparison v1

Completed read-only patient-bootstrap comparison of existing undersampled and
oversampled LFCC logistic/Swin validation predictions at fixed 0.5 and the
previously frozen calibration targets 80/90/95%. See [PROTOCOL.md](PROTOCOL.md),
[REPORT.md](REPORT.md), [analysis.json](analysis.json), [audit.json](audit.json)
and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Bootstrap draws are in gitignored
`results/`. No model, score or threshold was fitted on validation or test.

Recompute from the repository root using the existing source artifacts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs_omi_vmd_wst_vqc/threshold_comparison_v1/analyze.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs_omi_vmd_wst_vqc/threshold_comparison_v1/audit.py
```
