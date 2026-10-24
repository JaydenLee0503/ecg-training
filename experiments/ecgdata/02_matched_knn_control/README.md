# ECG-02 — Matched KNN control and VQC training-fit diagnostic

Status: **complete**.

A separate fixed KNN diagnostic using the saved patient folds and VQC preprocessing. All 12 configurations are retained; previously fitted VQCs are reused, not counted as new fits.

## Files

[Open the clickable file index](FILES.html), or use the canonical links below.

| Resource | Original location |
|---|---|
| `report.md` | [architects/vqc_vs_spar_knn.md](../../../architects/vqc_vs_spar_knn.md) |
| `artifacts` | [results/patient_knn_diagnostic](../../../results/patient_knn_diagnostic) |
| `protocol.json` | [results/patient_knn_diagnostic/protocol.json](../../../results/patient_knn_diagnostic/protocol.json) |
| `code/vqc_knn_diagnostic.py` | [scripts/vqc_knn_diagnostic.py](../../../scripts/vqc_knn_diagnostic.py) |

Keep commands relative to the repository root as documented in the original report. Scripts, frozen protocols, checkpoints and data have not moved. Shared code is intentionally linked rather than copied.

Large generated artifacts are gitignored and available only when the original local data are present. A fresh clone may therefore have missing artifact targets. The indexes use ordinary relative links and do not require symbolic-link support.

[All studies](../../README.md).
