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

Versioned `provenance/` copies preserve the feature/training manifests, feature
completion marker and extraction summary. They record hashes but do not replace
large gitignored feature caches, checkpoints, or predictions. CPU and optional
CUDA/recovery test logs are in `tests/`; setting `ECG09_TEST_CUDA=1` enables the
host GPU test.

A one-shot `monitor.py` waits for the launched training process to exit and then
runs `audit.py` on completed artifacts; inspect `monitor_status.json` and
`audit.log`. The monitor does not train and must not be duplicated. A passed
final audit is required before citing the newly generated results.

After the final audit passes, refresh the previously executed manuscript notebook
with `refresh_notebook.py` using the existing interpreter. Jupyter needs a local
kernel socket, which may require host execution in this environment. The helper
refuses to run without a passed audit, verifies output from every code cell, does
not train a model, and preserves any notebook edits made during execution.

## Completed results — 2026-11-07 EDT

All 35 new fits and the independent audit completed. Read the
[completion review](reports/COMPLETION_REVIEW.md), [results](reports/REPORT.md),
[analysis](reports/analysis.json) and [audit](reports/audit.json). A compact
[held-out prediction table](reports/predictions.csv.gz) is versioned; full model
outputs, probabilities and feature caches remain local and gitignored. The
executed ECGData manuscript notebook now includes the audited comparison. The
OMI analysis is separate and no official OMI test data were used. Do not retrain
completed fits merely because the session or computer restarted.
