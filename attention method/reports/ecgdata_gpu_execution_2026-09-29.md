# ECGData LFCC + Swin GPU execution — 2026-09-29

The user requested training and confirmed the repository checkpoint. Commit
`eba3a78` contains the preceding ACS work; the working tree was clean before
this execution work. This request supersedes the older preparation-only status.

## Scope fixed before fitting

Execute the existing [ECGData protocol](../ecgdata_protocol.json) unchanged:
1,620 windows from 80 patients, the same five patient folds as the corrected
VMD/WST reference, LFCC (linear frequency), compact temporal Swin, seeds 0/1/2,
40 epochs per fit, and five pooled-LFCC logistic controls. There is no new model
search, early stopping or selection from outer-fold scores. No new exclusions
are introduced. Cepstral extraction is a fixed transform; its completed cache
is reused. The learned component is the classifier.

All new results go to `attention method/results/ecgdata_lfcc_swin_v1/`.
The ACS extraction remains paused. This work does not resume or modify it.
MFCC is an optional implementation and is outside this fixed LFCC experiment.

## Environment

- Python: `/home/jaydenlee/venvs/test-ecg-training/bin/python` (3.12.3).
- Existing numerical packages: NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0.
- Separate dependency target: `attention method/.venv/torch-cu128/`, gitignored.
- PyTorch: 2.7.1+cu128, selected for the RTX 5070. The prior CPU-only 2.6.0
  installation in `/tmp` no longer exists. [PyTorch's release notes](https://pytorch.org/blog/pytorch-2-7/)
  document Blackwell and CUDA 12.8 support; the [official version instructions](https://pytorch.org/get-started/previous-versions/)
  specify the 2.7.1 CUDA 12.8 wheel index.
- GPU observed before launch: NVIDIA GeForce RTX 5070, 12,227 MiB, driver
  610.57.01 (Windows KMD 610.88), approximately 1% utilization.
- Full float32, deterministic algorithms, TF32 disabled, one CPU thread,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. No mixed precision or compilation.

The dependency target is loaded through `PYTHONPATH`; the original Python
environment is not modified. Network downloads and GPU execution need access
outside the sandbox. The initial sandboxed download failed with DNS errors;
the approved external download is used instead. No model fit failed in that step.

## Verification and commands

The saved LFCC cache passed its complete checksum/source verification. The
[preflight inventory](ecgdata_preflight_integrity_2026-09-29.json) checked all 134
protected files and all 930 saved VQC improvement artifacts: zero changed or
missing files. There was no attention training output directory before this run.

Run from the repository root:

```bash
export PYTHONPATH="$PWD/attention method/.venv/torch-cu128"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s 'attention method/tests' -v
/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/scripts/validate_gpu.py'
/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/ecgdata.py' train --execute-training --device cuda
/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/ecgdata.py' report
```

The synthetic GPU check compares CPU/CUDA forward output and gradients, and
compares a two-epoch uninterrupted optimizer run with a run interrupted after
epoch one and resumed from disk. These checks are engineering verification,
not classifier evaluation. Their altered synthetic settings never enter the
real-data manifest. Completed real fits are reused and every epoch is saved by
the existing runner. A checksum/marker mismatch stops for inspection.

## Execution status

Completed at **2026-09-30 01:51:21 UTC** (2026-09-29 21:51:21 EDT): all 20 fits,
600 Swin epochs, no failed real-data trials and no logistic warnings. The training
routine took 407.645 seconds; dependency installation and synthetic checks are
excluded. [Full results and limitations](ecgdata_lfcc_swin_v1/results.md) include
all seeds, controls, uncertainty, runtime, predictions and learning curves.

All 21 CPU tests passed. The [GPU check](ecgdata_gpu_validation_2026-09-29.json)
passed, including bitwise-equivalent synthetic checkpoint recovery. All 20 saved
models were reloaded on CPU and reproduced their predicted classes; maximum
probability difference was 1.85e-6. The [postflight check](ecgdata_postflight_integrity_2026-09-29.json)
found all 134 protected files unchanged.

The actual launch used `scripts/run_ecgdata_gpu.py --execute-training`, which
runs the synthetic check, records the environment and invokes the existing
training runner. It prints fit/epoch progress every 30 seconds. Its
[execution record](ecgdata_execution_20260930T014422Z.json) and
[training log](ecgdata_gpu_training_2026-09-29.log) are retained. The
`scripts/audit_ecgdata_run.py` script then generated the report and verified
models through CPU inference. No scientific or original training-code changes
were needed. There is no remaining fit to resume for this protocol.
