# H-03 — Historical WST/VMD comparison and temporal-pooling controls

Status: **historical**.

Original row-grouped comparison, bin-mean control and learning curves. Superseded for patient-independent interpretation by ECG-01; retains failed and interrupted logs.

## Files

[Open the clickable file index](FILES.html), or use the canonical links below.

| Resource | Original location |
|---|---|
| `report.md` | [architects/wst_vs_vmd_vqc.md](../../../architects/wst_vs_vmd_vqc.md) |
| `code/wst_vs_vmd_vqc.py` | [scripts/wst_vs_vmd_vqc.py](../../../scripts/wst_vs_vmd_vqc.py) |
| `artifacts/wst_vs_vmd_curve.log` | [results/wst_vs_vmd_curve.log](../../../results/wst_vs_vmd_curve.log) |
| `artifacts/wst_vs_vmd_curve_VMD_modes___rhythm.csv` | [results/wst_vs_vmd_curve_VMD_modes___rhythm.csv](../../../results/wst_vs_vmd_curve_VMD_modes___rhythm.csv) |
| `artifacts/wst_vs_vmd_curve_WST_log.csv` | [results/wst_vs_vmd_curve_WST_log.csv](../../../results/wst_vs_vmd_curve_WST_log.csv) |
| `artifacts/wst_vs_vmd_vqc.csv` | [results/wst_vs_vmd_vqc.csv](../../../results/wst_vs_vmd_vqc.csv) |
| `artifacts/wst_vs_vmd_vqc.log` | [results/wst_vs_vmd_vqc.log](../../../results/wst_vs_vmd_vqc.log) |
| `artifacts/wst_vs_vmd_vqc_preds.npz` | [results/wst_vs_vmd_vqc_preds.npz](../../../results/wst_vs_vmd_vqc_preds.npz) |
| `artifacts/wst_vqc.csv` | [results/wst_vqc.csv](../../../results/wst_vqc.csv) |
| `artifacts/wst_vqc.log` | [results/wst_vqc.log](../../../results/wst_vqc.log) |
| `artifacts/wst_vqc.killed-run.log` | [results/wst_vqc.killed-run.log](../../../results/wst_vqc.killed-run.log) |
| `artifacts/wst_vqc_preds.npz` | [results/wst_vqc_preds.npz](../../../results/wst_vqc_preds.npz) |
| `artifacts/wst_vqc_binmean.csv` | [results/wst_vqc_binmean.csv](../../../results/wst_vqc_binmean.csv) |
| `artifacts/wst_vqc_binmean.log` | [results/wst_vqc_binmean.log](../../../results/wst_vqc_binmean.log) |
| `artifacts/wst_vqc_binmean_preds.npz` | [results/wst_vqc_binmean_preds.npz](../../../results/wst_vqc_binmean_preds.npz) |

Keep commands relative to the repository root as documented in the original report. Scripts, frozen protocols, checkpoints and data have not moved. Shared code is intentionally linked rather than copied.

Large generated artifacts are gitignored and available only when the original local data are present. A fresh clone may therefore have missing artifact targets. The indexes use ordinary relative links and do not require symbolic-link support.

[All studies](../../README.md).
