# ACS-01 — ACS/OMI VMD versus WST with VQC and classical controls

Status: **complete**.

Completed original development comparison. CPU/GPU extraction attempts, convergence retries, benchmarks and the failed first workflow are supporting runs within this study. Official test patients remain reserved.

## Files

[Open the clickable file index](FILES.html), or use the canonical links below.

| Resource | Original location |
|---|---|
| `report` | [experiments/acs_omi_vmd_wst_vqc/reports/acs_omi_results_v1](../../reports/acs_omi_results_v1) |
| `artifacts` | [experiments/acs_omi_vmd_wst_vqc/results/omi_models_v1_retry256k](../../results/omi_models_v1_retry256k) |
| `protocol.md` | [experiments/acs_omi_vmd_wst_vqc/protocols/acs_omi_protocol_v1.md](../../protocols/acs_omi_protocol_v1.md) |
| `protocol_initial.json` | [experiments/acs_omi_vmd_wst_vqc/protocols/acs_omi_protocol_v1.json](../../protocols/acs_omi_protocol_v1.json) |
| `protocol_final.json` | [experiments/acs_omi_vmd_wst_vqc/protocols/acs_omi_protocol_v1_retry256k.json](../../protocols/acs_omi_protocol_v1_retry256k.json) |
| `protocols` | [experiments/acs_omi_vmd_wst_vqc/protocols](../../protocols) |
| `code/train_models.py` | [experiments/acs_omi_vmd_wst_vqc/scripts/train_models.py](../../scripts/train_models.py) |
| `code/report_models.py` | [experiments/acs_omi_vmd_wst_vqc/scripts/report_models.py](../../scripts/report_models.py) |
| `code/pipeline.py` | [experiments/acs_omi_vmd_wst_vqc/pipeline.py](../../pipeline.py) |
| `code/training.py` | [experiments/acs_omi_vmd_wst_vqc/training.py](../../training.py) |
| `extraction` | [experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu_retry256k](../../results/omi_v1_gpu_retry256k) |
| `workflow` | [experiments/acs_omi_vmd_wst_vqc/results/omi_development_workflow_v2](../../results/omi_development_workflow_v2) |
| `attempts/omi_v1` | [experiments/acs_omi_vmd_wst_vqc/results/omi_v1](../../results/omi_v1) |
| `attempts/omi_v1_parallel` | [experiments/acs_omi_vmd_wst_vqc/results/omi_v1_parallel](../../results/omi_v1_parallel) |
| `attempts/omi_v1_gpu` | [experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu](../../results/omi_v1_gpu) |
| `attempts/omi_v1_gpu_retry128k` | [experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu_retry128k](../../results/omi_v1_gpu_retry128k) |
| `attempts/omi_development_workflow_v1` | [experiments/acs_omi_vmd_wst_vqc/results/omi_development_workflow_v1](../../results/omi_development_workflow_v1) |
| `diagnostics/gpu_vmd_benchmark_v1` | [experiments/acs_omi_vmd_wst_vqc/results/gpu_vmd_benchmark_v1](../../results/gpu_vmd_benchmark_v1) |
| `diagnostics/loader` | [experiments/acs_omi_vmd_wst_vqc/results/loader](../../results/loader) |
| `diagnostics/retry_device_validation_20260927` | [experiments/acs_omi_vmd_wst_vqc/results/retry_device_validation_20260927](../../results/retry_device_validation_20260927) |
| `diagnostics/vmd_04124_diagnostic_v1` | [experiments/acs_omi_vmd_wst_vqc/results/vmd_04124_diagnostic_v1](../../results/vmd_04124_diagnostic_v1) |
| `diagnostics/vmd_04124_gpu_check_v1` | [experiments/acs_omi_vmd_wst_vqc/results/vmd_04124_gpu_check_v1](../../results/vmd_04124_gpu_check_v1) |
| `diagnostics/vmd_cap_01985_cpu_diagnostic_v1` | [experiments/acs_omi_vmd_wst_vqc/results/vmd_cap_01985_cpu_diagnostic_v1](../../results/vmd_cap_01985_cpu_diagnostic_v1) |
| `diagnostics/vmd_retry_extension_diagnostic_v1` | [experiments/acs_omi_vmd_wst_vqc/results/vmd_retry_extension_diagnostic_v1](../../results/vmd_retry_extension_diagnostic_v1) |

Keep commands relative to the repository root as documented in the original report. Scripts, frozen protocols, checkpoints and data have not moved. Shared code is intentionally linked rather than copied.

Large generated artifacts are gitignored and available only when the original local data are present. A fresh clone may therefore have missing artifact targets. The indexes use ordinary relative links and do not require symbolic-link support.

[All studies](../../../README.md).
