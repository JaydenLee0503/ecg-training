# ECG-09 representation comparison

Compare 32 log-filterbank energies plus Swin with the verified ECG-07 LFCC–Swin
baseline; include matched filterbank logistic and a raw-waveform CNN control.
All use 21,222 standardized windows, the same five patient folds and seeds
0/1/2. Read [PROTOCOL.md](PROTOCOL.md), [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
and, after completion, `reports/REPORT.md` and `reports/analysis.json`. Files
under `features/` and `results/` are local and gitignored.

From the repository root, using the existing environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='attention method/.venv/torch-cu128' /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/ecgdata/09_representation_comparison/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='attention method/.venv/torch-cu128' /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/ecgdata/09_representation_comparison/study.py prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH='attention method/.venv/torch-cu128' /home/jaydenlee/venvs/test-ecg-training/bin/python -u -B experiments/ecgdata/09_representation_comparison/study.py train --execute-training --device cuda
```

The train action verifies and reuses saved features and completed fits, holds a
writer lock, saves epoch checkpoints and rejects source/manifest changes. Inspect
`results/status.json`, individual `completed.json` markers and `training.log`
before resuming. If training finished but reporting was interrupted, invoke
`study.py report` rather than refitting. The completed-run branch does not invoke
the report automatically. No official OMI test data are used by this ECGData study.
