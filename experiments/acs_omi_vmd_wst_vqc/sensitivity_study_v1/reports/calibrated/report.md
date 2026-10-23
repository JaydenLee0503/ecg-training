# ACS/OMI sensitivity study: calibration-selected thresholds

Each threshold was selected using only the held-out calibration patients within the original fit cohort. These tables evaluate those frozen thresholds on the unchanged validation patients. A calibration target is not a guarantee of validation sensitivity.

Validation: 3,581 ECGs, 3,391 patients, 229 OMI-positive ECGs. Natural prevalence retained; official test untouched.

VQC/Swin results average metrics across all initialization seeds 0/1/2, not probabilities. Every seed has its own cutoff. No seed is selected. All classical controls are included.

| Model | Target | Achieved sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | False alarms / 1,000 non-OMI |
|---|---:|---:|---:|---:|---:|
| new/vmd/logistic | 80% | 72.49% [66.24, 78.11] | 62.26% [60.60, 63.93] | 11.60% [9.95, 13.17] | 377.4 |
| new/vmd/logistic | 90% | 84.72% [79.62, 89.20] | 50.36% [48.63, 52.08] | 10.44% [9.09, 11.79] | 496.4 |
| new/vmd/logistic | 95% | 93.89% [90.50, 96.92] | 35.50% [33.90, 37.14] | 9.05% [7.95, 10.17] | 645.0 |
| new/vmd/weighted_knn | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| new/vmd/weighted_knn | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| new/vmd/weighted_knn | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| new/vmd/vqc | 80% | 71.47% [65.90, 76.91] | 57.48% [55.90, 59.03] | 10.30% [8.87, 11.69] | 425.2 |
| new/vmd/vqc | 90% | 82.97% [78.35, 87.42] | 45.89% [44.27, 47.48] | 9.48% [8.29, 10.71] | 541.1 |
| new/vmd/vqc | 95% | 92.14% [88.89, 95.19] | 30.81% [29.34, 32.31] | 8.34% [7.34, 9.39] | 691.9 |
| new/wst/logistic | 80% | 75.11% [69.44, 80.57] | 55.61% [53.95, 57.36] | 10.36% [8.92, 11.91] | 443.9 |
| new/wst/logistic | 90% | 81.22% [75.86, 86.10] | 45.20% [43.53, 46.98] | 9.19% [7.97, 10.47] | 548.0 |
| new/wst/logistic | 95% | 93.45% [89.91, 96.41] | 22.73% [21.42, 24.18] | 7.63% [6.68, 8.60] | 772.7 |
| new/wst/weighted_knn | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| new/wst/weighted_knn | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| new/wst/weighted_knn | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| new/wst/vqc | 80% | 78.75% [74.02, 83.19] | 47.72% [46.31, 49.22] | 9.36% [8.15, 10.65] | 522.8 |
| new/wst/vqc | 90% | 88.36% [84.75, 91.53] | 31.95% [30.56, 33.31] | 8.15% [7.15, 9.23] | 680.5 |
| new/wst/vqc | 95% | 91.85% [88.73, 94.37] | 23.78% [22.55, 24.93] | 7.61% [6.66, 8.61] | 762.2 |
| new/lfcc/logistic | 80% | 73.36% [67.38, 79.07] | 67.51% [65.96, 69.11] | 13.37% [11.54, 15.22] | 324.9 |
| new/lfcc/logistic | 90% | 82.10% [76.89, 87.08] | 55.55% [53.83, 57.20] | 11.20% [9.76, 12.74] | 444.5 |
| new/lfcc/logistic | 95% | 86.90% [82.19, 91.11] | 42.09% [40.46, 43.79] | 9.30% [8.12, 10.51] | 579.1 |
| new/lfcc/swin | 80% | 79.18% [75.77, 82.60] | 40.97% [40.07, 41.91] | 8.97% [7.86, 10.14] | 590.3 |
| new/lfcc/swin | 90% | 89.08% [86.51, 91.49] | 22.16% [21.42, 22.90] | 7.48% [6.58, 8.43] | 778.4 |
| new/lfcc/swin | 95% | 93.60% [91.45, 95.56] | 14.40% [13.76, 15.04] | 7.08% [6.22, 8.00] | 856.0 |
| constant_natural_prior | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| constant_natural_prior | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| constant_natural_prior | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |

## Fixed 0.5 cutoff and ranking metrics

| Model | AP | ROC AUC | Sensitivity | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| new/vmd/logistic | 0.1497 | 0.7472 | 66.38% | 70.41% | 13.29% | 70.15% |
| new/vmd/weighted_knn | 0.1041 | 0.6501 | 39.74% | 79.83% | 11.86% | 77.27% |
| new/vmd/vqc | 0.1243 | 0.7028 | 66.52% | 63.19% | 11.00% | 63.40% |
| new/wst/logistic | 0.1689 | 0.7244 | 66.38% | 64.98% | 11.46% | 65.07% |
| new/wst/weighted_knn | 0.0932 | 0.6031 | 40.17% | 76.31% | 10.38% | 74.00% |
| new/wst/vqc | 0.1347 | 0.7044 | 63.90% | 66.06% | 11.40% | 65.92% |
| new/lfcc/logistic | 0.1841 | 0.7635 | 59.39% | 79.74% | 16.69% | 78.44% |
| new/lfcc/swin | 0.1723 | 0.7016 | 13.54% | 97.19% | 24.92% | 91.84% |
| constant_natural_prior | 0.0639 | 0.5000 | 0.00% | 100.00% | 0.00% | 93.61% |

## Statistical interpretation

- 2,000/2,000 paired patient-bootstrap draws; seed 20261015; 0 single-class draws discarded.
- Thresholds use score >= cutoff, keep ties together, and choose the largest observed score meeting the target. No interpolated or randomized decisions. Discrete KNN scores can overshoot a target substantially.
- Bootstrap thresholds stay fixed. Intervals condition on the fitted models and calibration cutoffs; they exclude training, calibration-selection and split uncertainty.
- Intervals are pointwise, without multiplicity correction. ECG outcomes remain clustered by patient; mixed-label patients are retained without inventing a patient-level diagnosis.
- AP and ROC AUC are unchanged by moving a cutoff. Higher sensitivity can come with many false positives.
- The validation cohort has informed earlier research decisions. No external/clinical validation or quantum advantage is established.
- All model-minus-LFCC-logistic paired differences and intervals are retained in analysis.json. For new fits, achieved sensitivities may differ, so those differences are not a comparison at identical achieved sensitivity.

## Artifacts

- [All seed cutoffs, confusion counts and intervals](per_seed_metrics.csv)
- [All group metrics and intervals](group_metrics.csv)
- [Complete analysis and paired comparisons](analysis.json)
- [Operating-point figure](operating_points.png)
- Full prediction CSV and bootstrap draws are retained in the study results folder (gitignored).
