# ECG-01 — Patient-grouped VMD versus WST with VQC

Status: **complete**.

Corrected comparison on verified patient folds, including all three VQC seeds and classical controls. The earlier row-grouped comparisons are historical.

## Files

[Open the clickable file index](FILES.html), or use the canonical links below.

| Resource | Original location |
|---|---|
| `report.md` | [architects/patient_vmd_wst_vqc.md](../../../architects/patient_vmd_wst_vqc.md) |
| `artifacts` | [results/patient_vmd_wst_vqc](../../../results/patient_vmd_wst_vqc) |
| `code/matched_vqc.py` | [scripts/matched_vqc.py](../../../scripts/matched_vqc.py) |
| `code/summarize_patient_vqc.py` | [scripts/summarize_patient_vqc.py](../../../scripts/summarize_patient_vqc.py) |
| `code/ecgvmd` | [ecgvmd](../../../ecgvmd) |
| `feature_archive.npz` | [features/fixed_K8_a2000_L500_it2000_comparison.npz](../../../features/fixed_K8_a2000_L500_it2000_comparison.npz) |
| `attempts/smoke` | [results/patient_vqc_smoke](../../../results/patient_vqc_smoke) |
| `attempts/preparation.log` | [results/patient_vqc_prepare.log](../../../results/patient_vqc_prepare.log) |
| `attempts/training.log` | [results/patient_vqc_training.log](../../../results/patient_vqc_training.log) |
| `attempts/feature_rebuild.log` | [results/vmd_it2000_rebuild.log](../../../results/vmd_it2000_rebuild.log) |

Keep commands relative to the repository root as documented in the original report. Scripts, frozen protocols, checkpoints and data have not moved. Shared code is intentionally linked rather than copied.

Large generated artifacts are gitignored and available only when the original local data are present. A fresh clone may therefore have missing artifact targets. The indexes use ordinary relative links and do not require symbolic-link support.

[All studies](../../README.md).
