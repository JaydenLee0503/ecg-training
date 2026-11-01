# ECG-08 patient-count learning curve

Implements the pre-existing [learning-curve protocol](../../../attention%20method/ecgdata_learning_curve_protocol.md)
using 25%, 50%, and 75% nested class-stratified training-patient subsets, three
subset draws paired with initialization seeds 0/1/2, the five fixed outer patient
folds, and matching logistic controls. The 100% point reuses verified original
models. See [REPORT.md](REPORT.md) after completion and `results/analysis.json`
for paired patient-bootstrap results. Per-fit checkpoints and predictions are
local and gitignored. Smaller subsets also mean fewer optimizer updates.

From the project root, with the existing environment and host CUDA access:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH='attention method/.venv/torch-cu128' /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/ecgdata/08_patient_learning_curve/study.py --execute-training --device cuda
```

Completed fits are checked and reused. Do not launch a second writer; the run
uses a lock. Source and protocol hashes are pinned in the results manifest.
