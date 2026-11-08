# OMI improvement v2 — completed

All neural seeds 0/1/2; three inner patient folds. Validation: 3,581 ECGs, 3,391 patients, 229 OMI ECGs.

Primary selected-Swin minus matched fixed-40 Swin AP: **+0.0066**, paired 95% interval **[-0.0115, +0.0243]**.
The interval includes zero; improvement is not established.

| Pipeline | AP | ROC AUC | Accuracy at 0.5 | Sensitivity at 0.5 | Specificity at 0.5 |
|---|---:|---:|---:|---:|---:|
| logistic_fixed | 0.1841 | 0.7635 | 78.44% | 59.39% | 79.74% |
| logistic_tuned | 0.2022 | 0.7778 | 78.25% | 62.88% | 79.30% |
| swin_fixed | 0.1723 | 0.7016 | 91.84% | 13.54% | 97.19% |
| swin0 | 0.1789 | 0.7509 | 89.14% | 28.38% | 93.29% |
| swin1 | 0.1688 | 0.7426 | 91.57% | 13.97% | 96.87% |
| raw | 0.2803 | 0.8286 | 92.67% | 24.75% | 97.32% |
| swin_selected | 0.1789 | 0.7509 | 89.14% | 28.38% | 93.29% |

| Pipeline | Calibration target | Achieved sensitivity | Specificity | Precision |
|---|---:|---:|---:|---:|
| logistic_fixed | 80% | 73.36% | 67.51% | 13.37% |
| logistic_fixed | 90% | 82.10% | 55.55% | 11.20% |
| logistic_fixed | 95% | 86.90% | 42.09% | 9.30% |
| logistic_tuned | 80% | 71.62% | 71.24% | 14.54% |
| logistic_tuned | 90% | 83.41% | 59.37% | 12.30% |
| logistic_tuned | 95% | 88.21% | 45.88% | 10.02% |
| swin_fixed | 80% | 79.18% | 40.97% | 8.97% |
| swin_fixed | 90% | 89.08% | 22.16% | 7.48% |
| swin_fixed | 95% | 93.60% | 14.40% | 7.08% |
| swin0 | 80% | 78.17% | 61.41% | 12.16% |
| swin0 | 90% | 86.46% | 40.80% | 9.11% |
| swin0 | 95% | 93.60% | 19.36% | 7.38% |
| swin1 | 80% | 72.93% | 63.71% | 12.24% |
| swin1 | 90% | 85.30% | 44.43% | 9.66% |
| swin1 | 95% | 94.32% | 19.84% | 7.50% |
| raw | 80% | 75.84% | 76.26% | 17.95% |
| raw | 90% | 90.10% | 50.47% | 11.30% |
| raw | 95% | 94.76% | 34.41% | 9.16% |
| swin_selected | 80% | 78.17% | 61.41% | 12.16% |
| swin_selected | 90% | 86.46% | 40.80% | 9.11% |
| swin_selected | 95% | 93.60% | 19.36% | 7.38% |

Model/epoch choices used only inner subfit predictions. Calibration cutoffs were fixed before new validation inference; no subsequent refitting.
The fixed logistic and fixed-40 Swin controls reuse verified sensitivity-study fits on exactly the same subfit/calibration patients. Swin0 reuses a final fit only when epoch40 was selected.
Neural rows are means of metrics across all seeds, not probability ensembles. The selected Swin reuses its chosen arm; it is not an extra fit.
New saved fits: 52; neural epoch records: 1350. Sum of recorded fit seconds: 7597.3.
Patient bootstrap: 2000/2000 valid draws, seed 20261023.
Intervals are pointwise and conditional on saved models, selected settings and calibration thresholds; no multiplicity adjustment, retraining or selection uncertainty. The primary was not replaced by the best observed secondary arm.
Validation has been repeatedly examined. Official test waveforms remain reserved. Accuracy alone is misleading at the natural low prevalence. Calibration targets need not transfer to validation. Raw CNN changes both representation and classifier; this is not evidence of transform superiority, clinical validity or quantum advantage.
All candidate scores, splits, sampling indices, fitted preprocessing, predictions, weights, optimizer/RNG checkpoints, warnings, failures and source snapshots are retained in this study. Large artifacts are local and gitignored.

[Full results and intervals](analysis.json) · [Every seed](metrics.csv) · [Thresholds and counts](operating_points.csv) · [Fits](fits.json) · [Histories](histories.json)
