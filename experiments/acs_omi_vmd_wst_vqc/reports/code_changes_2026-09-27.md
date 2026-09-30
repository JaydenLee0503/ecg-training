# ACS file changes — 2026-09-27 EDT

This inventory compares the working tree with Git HEAD. The tree was clean at the start of the ACS continuation. No commit was created.

## Existing files modified

Only one existing Python file was modified: `scripts/extract_gpu_retry.py`. Its changes add verification of the current-device pilot, include that validation script in the code hashes, and pin hardware through the new execution specification. The actual CUDA fingerprint equals the original run. Its VMD arithmetic and feature extraction calls were not changed.

| File | Change |
|---|---|
| `architects/SESSION_HANDOFF.md` | Progress, failure, restart and workflow documentation |
| `experiments/acs_omi_vmd_wst_vqc/EXPERIMENT_LOG.md` | Progress, failure, restart and workflow documentation |
| `experiments/acs_omi_vmd_wst_vqc/README.md` | Progress, failure, restart and workflow documentation |
| `experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md` | Progress, failure, restart and workflow documentation |
| `experiments/acs_omi_vmd_wst_vqc/scripts/extract_gpu_retry.py` | Current-device validation gate and execution hardware provenance |

## New files

| File | Purpose |
|---|---|
| `experiments/acs_omi_vmd_wst_vqc/protocols/gpu_extraction_retry128k_v1.json` | Separate execution/diagnostic policy; no overwrite of frozen scientific protocol |
| `experiments/acs_omi_vmd_wst_vqc/protocols/gpu_extraction_retry128k_v1.md` | Separate execution/diagnostic policy; no overwrite of frozen scientific protocol |
| `experiments/acs_omi_vmd_wst_vqc/protocols/vmd_04124_diagnostic_v1.json` | Separate execution/diagnostic policy; no overwrite of frozen scientific protocol |
| `experiments/acs_omi_vmd_wst_vqc/reporting.py` | Saved-prediction metrics and paired patient-cluster uncertainty |
| `experiments/acs_omi_vmd_wst_vqc/reports/code_changes_2026-09-27.md` | Evidence, failure report or file inventory |
| `experiments/acs_omi_vmd_wst_vqc/reports/retry_preparation_2026-09-27.json` | Evidence, failure report or file inventory |
| `experiments/acs_omi_vmd_wst_vqc/reports/retry_preparation_2026-09-27.md` | Evidence, failure report or file inventory |
| `experiments/acs_omi_vmd_wst_vqc/reports/retry_stop_2026-09-27.json` | Evidence, failure report or file inventory |
| `experiments/acs_omi_vmd_wst_vqc/scripts/bundle_gpu.py` | Verify complete feature coverage and assemble aligned matrices |
| `experiments/acs_omi_vmd_wst_vqc/scripts/diagnose_04124.py` | Diagnostic-only CPU cap reproduction and convergence trace for 04124/V5 |
| `experiments/acs_omi_vmd_wst_vqc/scripts/finish_development.py` | Wait for successful extraction, then run the fixed downstream comparison; stop on failure |
| `experiments/acs_omi_vmd_wst_vqc/scripts/report_models.py` | Saved-prediction metrics and paired patient-cluster uncertainty |
| `experiments/acs_omi_vmd_wst_vqc/scripts/train_models.py` | Fixed classical/VQC fitting with checkpoint recovery; has not fitted real ACS models |
| `experiments/acs_omi_vmd_wst_vqc/scripts/validate_retry_device.py` | 17-fit-record current-device comparison against saved references |
| `experiments/acs_omi_vmd_wst_vqc/tests/test_bundle.py` | Focused verification tests |
| `experiments/acs_omi_vmd_wst_vqc/tests/test_convergence_diagnostic.py` | Focused verification tests |
| `experiments/acs_omi_vmd_wst_vqc/tests/test_reporting.py` | Focused verification tests |
| `experiments/acs_omi_vmd_wst_vqc/tests/test_training.py` | Focused verification tests |
| `experiments/acs_omi_vmd_wst_vqc/tests/test_training_runner.py` | Focused verification tests |
| `experiments/acs_omi_vmd_wst_vqc/training.py` | Fixed classical/VQC fitting with checkpoint recovery; has not fitted real ACS models |
| `experiments/acs_omi_vmd_wst_vqc/vmd_convergence_diagnostic.py` | Diagnostic-only CPU cap reproduction and convergence trace for 04124/V5 |

## Production numerical code verified unchanged

Each file below still matches its SHA-256 recorded in the frozen extraction manifest:

- `ecgvmd/vmd.py`
- `ecgvmd/features.py`
- `ecgvmd/scatter.py`
- `ecgvmd/select.py`
- `ecgvmd/reupload.py`
- `experiments/acs_omi_vmd_wst_vqc/loader.py`
- `experiments/acs_omi_vmd_wst_vqc/pipeline.py`
- `experiments/acs_omi_vmd_wst_vqc/gpu_vmd.py`
- `experiments/acs_omi_vmd_wst_vqc/gpu_extraction.py`
- `experiments/acs_omi_vmd_wst_vqc/scripts/extract_gpu.py`

The pre-existing `acs_omi_protocol_v1_retry128k.json`, `gpu_retry.py`, and original scientific protocol were already present and were not edited. The new production run deliberately uses appended 64,000/128,000 retry limits while retaining tolerance 1e-7 and all other numerical settings.

## Generated files outside Git

- `results/retry_device_validation_20260927/`: new 17-record validation artifacts.
- `results/omi_v1_gpu_retry128k/`: new run, imported checkpoints, new features and retained stop logs; 3,700 completed records at its stop.
- `results/omi_development_workflow_v1/`: waiting/failure status; no real model fitting occurred.
- `results/vmd_04124_diagnostic_v1/`: new isolated CPU diagnostic artifacts.
- Temporary pinned dependencies restored under `/tmp/acs-gpu-deps-v1`; base Python environment unchanged.

The original `results/omi_v1_gpu/` artifacts and original ECGData/attention results were not overwritten.

## Additional files added after the initial inventory

The convergence diagnosis and 256k restart were prepared in the next work segment:

| File | Purpose |
|---|---|
| `protocols/acs_omi_protocol_v1_retry256k.json` | New protocol ID with only the appended 256,000 iteration limit |
| `protocols/gpu_extraction_retry256k_v1.json` | Parent/diagnostic hashes, hardware, reuse and failure policy |
| `protocols/gpu_extraction_retry256k_v1.md` | Human-readable bounded retry policy |
| `scripts/extract_gpu_retry256k.py` | Separate manifest, checkpoint reuse and resumable extraction runner |
| `scripts/diagnose_04124.py` | Fit-only CPU cap reproduction and stopping-statistic trace |
| `scripts/verify_04124_gpu.py` | GPU comparison against the converged CPU lead |
| `vmd_convergence_diagnostic.py` | Instrumented diagnostic copy; production arithmetic is not edited |
| `tests/test_convergence_diagnostic.py`, `tests/test_retry256k.py` | Observer parity and strict protocol-extension checks |
| `scripts/finish_development256k.py` | Wait for successful extraction, then call bundling, training and reporting with the 256k run paths |
| `reports/vmd_04124_diagnostic_2026-09-27.md` and `.json` | Diagnosis and CPU/GPU evidence |
| `reports/vmd_04124_convergence_2026-09-27.png` | Signal and stopping-statistic plot |
| `reports/resume_acs_2026-09-28.md` | Saved state and copyable resume commands |

The original code integrity JSON covers the numerical modules listed above and
records their matching frozen hashes. The diagnostic scripts and 256k runner are
additional new files; they do not overwrite or edit the frozen solver.
