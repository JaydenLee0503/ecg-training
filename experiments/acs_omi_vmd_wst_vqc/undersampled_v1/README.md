# OMI undersampled training v1

Patient-grouped, 1:1 positive/negative training comparison using fewer ECGs. See
[PROTOCOL.md](PROTOCOL.md) and, after completion, [REPORT.md](REPORT.md). Models,
selection indices, calibration thresholds, validation predictions and bootstrap
analysis are stored under gitignored `results/`. Existing sensitivity-study
controls and calibration/validation patients are reused read-only.

From the project root with host CUDA access:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH='attention method/.venv/torch-cu128' /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs_omi_vmd_wst_vqc/undersampled_v1/study.py --execute-training --device cuda
```

The fixed 40-epoch Swin uses seeds 0/1/2, and the logistic control uses C=1.
No official-test waveforms are decoded. A writer lock and pinned manifest protect
resumable fits. Do not launch a duplicate writer.
