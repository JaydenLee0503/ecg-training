# OMI fixed and calibration-selected threshold comparison

These are fixed previously selected thresholds; calibration targets do not guarantee validation sensitivity. All Swin values are means over seeds 0/1/2.

| Model | Cutoff | Sensitivity | Specificity | Precision | False positives |
|---|---|---:|---:|---:|---:|
| undersampled_logistic | fixed_0.5 | 61.57% | 67.36% | 11.42% | 1094.0 |
| undersampled_logistic | cal_target_80 | 72.93% | 55.76% | 10.12% | 1483.0 |
| undersampled_logistic | cal_target_90 | 86.46% | 42.27% | 9.28% | 1935.0 |
| undersampled_logistic | cal_target_95 | 90.83% | 30.19% | 8.16% | 2340.0 |
| undersampled_swin | fixed_0.5 | 70.74% | 69.32% | 13.66% | 1028.3 |
| undersampled_swin | cal_target_80 | 74.53% | 65.04% | 12.76% | 1172.0 |
| undersampled_swin | cal_target_90 | 86.90% | 44.06% | 9.71% | 1875.0 |
| undersampled_swin | cal_target_95 | 93.45% | 24.71% | 7.92% | 2523.7 |
| oversampled_logistic | fixed_0.5 | 59.39% | 79.74% | 16.69% | 679.0 |
| oversampled_logistic | cal_target_80 | 73.36% | 67.51% | 13.37% | 1089.0 |
| oversampled_logistic | cal_target_90 | 82.10% | 55.55% | 11.20% | 1490.0 |
| oversampled_logistic | cal_target_95 | 86.90% | 42.09% | 9.30% | 1941.0 |
| oversampled_swin | fixed_0.5 | 13.54% | 97.19% | 24.92% | 94.3 |
| oversampled_swin | cal_target_80 | 79.18% | 40.97% | 8.97% | 1978.7 |
| oversampled_swin | cal_target_90 | 89.08% | 22.16% | 7.48% | 2609.3 |
| oversampled_swin | cal_target_95 | 93.60% | 14.40% | 7.08% | 2869.3 |

2,000 paired patient-bootstrap draws; fixed previously selected thresholds. Intervals and paired differences are in [analysis.json](analysis.json). The same validation cohort has been repeatedly studied; comparison is internal and exploratory. Test ECGs untouched.
