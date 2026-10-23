# ACS/OMI sensitivity study: saved-score exploration

Thresholds are chosen from the same validation labels used to describe these curves. These are exploratory operating points, not independently evaluated cutoffs.

Validation: 3,581 ECGs, 3,391 patients, 229 OMI-positive ECGs. Natural prevalence retained; official test untouched.

VQC/Swin results average metrics across all initialization seeds 0/1/2, not probabilities. Every seed has its own cutoff. No seed is selected. All classical controls are included.

| Model | Target | Achieved sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | False alarms / 1,000 non-OMI |
|---|---:|---:|---:|---:|---:|
| original/vmd/logistic | 80% | 80.35% [80.00, 81.31] | 51.37% [43.38, 58.10] | 10.14% [8.25, 12.00] | 486.3 |
| original/vmd/logistic | 90% | 90.39% [90.00, 91.25] | 35.65% [23.89, 43.40] | 8.76% [7.22, 10.47] | 643.5 |
| original/vmd/logistic | 95% | 95.20% [95.00, 96.31] | 18.32% [13.43, 33.94] | 7.37% [6.43, 9.12] | 816.8 |
| original/vmd/weighted_knn | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| original/vmd/weighted_knn | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| original/vmd/weighted_knn | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| original/vmd/vqc | 80% | 80.35% [80.00, 80.93] | 49.00% [39.95, 56.62] | 9.73% [7.98, 11.70] | 510.0 |
| original/vmd/vqc | 90% | 90.39% [90.00, 90.96] | 27.92% [21.19, 37.52] | 7.89% [6.80, 9.36] | 720.8 |
| original/vmd/vqc | 95% | 95.20% [95.02, 95.96] | 17.27% [11.80, 24.40] | 7.29% [6.36, 8.35] | 827.3 |
| original/wst/logistic | 80% | 80.35% [80.00, 81.30] | 50.98% [43.78, 59.00] | 10.07% [8.35, 12.15] | 490.2 |
| original/wst/logistic | 90% | 90.39% [90.00, 91.27] | 30.67% [19.47, 41.53] | 8.18% [6.88, 9.82] | 693.3 |
| original/wst/logistic | 95% | 95.20% [95.00, 96.33] | 13.87% [10.63, 28.35] | 7.02% [6.20, 8.52] | 861.3 |
| original/wst/weighted_knn | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| original/wst/weighted_knn | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| original/wst/weighted_knn | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| original/wst/vqc | 80% | 80.35% [80.00, 80.91] | 46.36% [37.88, 51.89] | 9.29% [7.76, 10.70] | 536.4 |
| original/wst/vqc | 90% | 90.39% [90.04, 90.96] | 26.45% [18.78, 33.78] | 7.75% [6.61, 9.01] | 735.5 |
| original/wst/vqc | 95% | 95.20% [95.02, 95.94] | 15.07% [8.20, 21.21] | 7.12% [6.14, 8.18] | 849.3 |
| balanced/vmd/logistic | 80% | 80.35% [80.00, 81.31] | 51.31% [42.62, 58.23] | 10.13% [8.21, 12.01] | 486.9 |
| balanced/vmd/logistic | 90% | 90.39% [90.00, 91.27] | 35.11% [25.23, 42.83] | 8.69% [7.29, 10.45] | 648.9 |
| balanced/vmd/logistic | 95% | 95.20% [95.00, 96.30] | 18.91% [13.36, 32.96] | 7.43% [6.46, 9.13] | 810.9 |
| balanced/vmd/weighted_knn | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| balanced/vmd/weighted_knn | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| balanced/vmd/weighted_knn | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| balanced/vmd/vqc | 80% | 80.35% [80.00, 80.93] | 47.09% [39.58, 55.27] | 9.40% [7.89, 11.27] | 529.1 |
| balanced/vmd/vqc | 90% | 90.39% [90.00, 90.99] | 27.64% [20.83, 36.50] | 7.87% [6.79, 9.36] | 723.6 |
| balanced/vmd/vqc | 95% | 95.20% [95.02, 96.00] | 16.47% [11.94, 24.66] | 7.22% [6.34, 8.40] | 835.3 |
| balanced/wst/logistic | 80% | 80.35% [80.00, 81.25] | 50.95% [42.61, 59.45] | 10.07% [8.28, 12.34] | 490.5 |
| balanced/wst/logistic | 90% | 90.39% [90.00, 91.28] | 29.59% [18.59, 41.28] | 8.06% [6.85, 9.79] | 704.1 |
| balanced/wst/logistic | 95% | 95.20% [95.00, 96.30] | 14.56% [10.86, 27.94] | 7.07% [6.21, 8.48] | 854.4 |
| balanced/wst/weighted_knn | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| balanced/wst/weighted_knn | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| balanced/wst/weighted_knn | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| balanced/wst/vqc | 80% | 80.35% [80.00, 80.93] | 40.58% [34.03, 48.32] | 8.47% [7.20, 10.07] | 594.2 |
| balanced/wst/vqc | 90% | 90.39% [90.04, 90.96] | 24.01% [17.30, 31.83] | 7.52% [6.51, 8.80] | 759.9 |
| balanced/wst/vqc | 95% | 95.20% [95.02, 95.95] | 13.15% [8.16, 19.50] | 6.97% [6.09, 8.01] | 868.5 |
| balanced/lfcc/logistic | 80% | 80.35% [80.00, 81.30] | 60.05% [50.30, 66.18] | 12.08% [9.87, 14.49] | 399.5 |
| balanced/lfcc/logistic | 90% | 90.39% [90.00, 91.37] | 35.68% [24.84, 48.33] | 8.76% [7.32, 10.87] | 643.2 |
| balanced/lfcc/logistic | 95% | 95.20% [95.00, 96.37] | 21.72% [17.02, 32.55] | 7.67% [6.69, 9.11] | 782.8 |
| balanced/lfcc/swin | 80% | 80.35% [80.00, 80.89] | 39.63% [32.06, 48.53] | 8.70% [7.36, 10.68] | 603.7 |
| balanced/lfcc/swin | 90% | 90.39% [90.04, 90.95] | 18.45% [14.54, 24.04] | 7.20% [6.33, 8.32] | 815.5 |
| balanced/lfcc/swin | 95% | 95.20% [95.02, 95.94] | 9.74% [6.20, 14.71] | 6.77% [5.94, 7.77] | 902.6 |
| constant_natural_prior | 80% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| constant_natural_prior | 90% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |
| constant_natural_prior | 95% | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 6.39% [5.63, 7.21] | 1000.0 |

## Fixed 0.5 cutoff and ranking metrics

| Model | AP | ROC AUC | Sensitivity | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| original/vmd/logistic | 0.1540 | 0.7262 | 61.57% | 71.51% | 12.86% | 70.87% |
| original/vmd/weighted_knn | 0.1206 | 0.6315 | 1.31% | 99.85% | 37.50% | 93.55% |
| original/vmd/vqc | 0.1442 | 0.7109 | 57.06% | 73.14% | 12.74% | 72.11% |
| original/wst/logistic | 0.1569 | 0.7277 | 66.81% | 66.17% | 11.89% | 66.21% |
| original/wst/weighted_knn | 0.1383 | 0.6406 | 2.62% | 99.94% | 75.00% | 93.72% |
| original/wst/vqc | 0.1352 | 0.6909 | 53.86% | 72.97% | 12.04% | 71.75% |
| balanced/vmd/logistic | 0.1548 | 0.7264 | 60.70% | 71.45% | 12.68% | 70.76% |
| balanced/vmd/weighted_knn | 0.0924 | 0.6206 | 36.68% | 79.47% | 10.88% | 76.74% |
| balanced/vmd/vqc | 0.1454 | 0.7071 | 63.32% | 66.79% | 11.52% | 66.56% |
| balanced/wst/logistic | 0.1553 | 0.7271 | 66.81% | 66.44% | 11.97% | 66.46% |
| balanced/wst/weighted_knn | 0.0999 | 0.6311 | 43.67% | 76.43% | 11.24% | 74.34% |
| balanced/wst/vqc | 0.1374 | 0.6852 | 62.45% | 65.29% | 10.95% | 65.11% |
| balanced/lfcc/logistic | 0.2091 | 0.7680 | 61.57% | 79.09% | 16.75% | 77.97% |
| balanced/lfcc/swin | 0.1680 | 0.7010 | 17.03% | 96.57% | 25.07% | 91.48% |
| constant_natural_prior | 0.0639 | 0.5000 | 0.00% | 100.00% | 0.00% | 93.61% |

## Statistical interpretation

- 2,000/2,000 paired patient-bootstrap draws; seed 20261015; 0 single-class draws discarded.
- Thresholds use score >= cutoff, keep ties together, and choose the largest observed score meeting the target. No interpolated or randomized decisions. Discrete KNN scores can overshoot a target substantially.
- Thresholds are reselected in each bootstrap draw: intervals describe the empirical curve at the requested sensitivity. This resampling does not remove selection optimism or independently validate a cutoff.
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
