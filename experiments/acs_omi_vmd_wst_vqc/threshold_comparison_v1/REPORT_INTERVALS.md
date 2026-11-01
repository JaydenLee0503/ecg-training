# OMI operating points with patient-bootstrap intervals

All values are validation estimates. Parentheses show pointwise 95% patient-bootstrap intervals; Swin metrics average seeds 0/1/2. Cutoffs were fixed at 0.5 or selected previously on calibration patients.

| Model | Threshold | Sensitivity (95% interval) | Specificity (95% interval) | Precision (95% interval) |
|---|---|---|---|---|
| undersampled_logistic | fixed_0.5 | 61.57% (55.22%–68.00%) | 67.36% (65.65%–68.95%) | 11.42% (9.67%–13.24%) |
| undersampled_logistic | cal_target_80 | 72.93% (66.81%–78.38%) | 55.76% (54.01%–57.36%) | 10.12% (8.72%–11.61%) |
| undersampled_logistic | cal_target_90 | 86.46% (81.78%–90.76%) | 42.27% (40.53%–43.96%) | 9.28% (8.05%–10.50%) |
| undersampled_logistic | cal_target_95 | 90.83% (86.78%–94.35%) | 30.19% (28.58%–31.72%) | 8.16% (7.13%–9.26%) |
| undersampled_swin | fixed_0.5 | 70.74% (66.52%–75.03%) | 69.32% (68.13%–70.59%) | 13.66% (11.90%–15.48%) |
| undersampled_swin | cal_target_80 | 74.53% (70.32%–78.96%) | 65.04% (63.75%–66.33%) | 12.76% (11.15%–14.47%) |
| undersampled_swin | cal_target_90 | 86.90% (83.78%–90.15%) | 44.06% (42.76%–45.33%) | 9.71% (8.53%–10.99%) |
| undersampled_swin | cal_target_95 | 93.45% (91.15%–95.62%) | 24.71% (23.75%–25.65%) | 7.92% (6.98%–8.94%) |
| oversampled_logistic | fixed_0.5 | 59.39% (53.05%–65.93%) | 79.74% (78.35%–81.11%) | 16.69% (14.23%–19.27%) |
| oversampled_logistic | cal_target_80 | 73.36% (67.23%–79.02%) | 67.51% (65.84%–69.14%) | 13.37% (11.56%–15.32%) |
| oversampled_logistic | cal_target_90 | 82.10% (76.75%–86.97%) | 55.55% (53.75%–57.27%) | 11.20% (9.71%–12.70%) |
| oversampled_logistic | cal_target_95 | 86.90% (82.33%–91.03%) | 42.09% (40.30%–43.78%) | 9.30% (8.13%–10.56%) |
| oversampled_swin | fixed_0.5 | 13.54% (10.70%–16.60%) | 97.19% (96.80%–97.56%) | 24.92% (19.75%–30.29%) |
| oversampled_swin | cal_target_80 | 79.18% (75.63%–82.51%) | 40.97% (40.03%–41.88%) | 8.97% (7.87%–10.17%) |
| oversampled_swin | cal_target_90 | 89.08% (86.43%–91.62%) | 22.16% (21.36%–22.96%) | 7.48% (6.58%–8.48%) |
| oversampled_swin | cal_target_95 | 93.60% (91.45%–95.60%) | 14.40% (13.74%–15.06%) | 7.08% (6.21%–8.02%) |

Intervals hold fitted models, selected training patients and cutoffs fixed; they omit retraining, threshold-selection and training-subset uncertainty. They are not adjusted for multiple comparisons. Target levels describe calibration, not guaranteed validation sensitivity. Existing validation patients have been repeatedly examined. The results do not establish clinical effectiveness.
