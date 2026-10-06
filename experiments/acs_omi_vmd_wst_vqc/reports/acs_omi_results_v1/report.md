# ACS OMI fixed internal-validation results

All eligible development records use the frozen patient split. Official test data remain reserved.
VQC rows average metrics over seeds 0/1/2. Intervals resample patients and retain their ECGs together.

| Model | Average precision (95% interval) | ROC AUC | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|
| vmd/logistic | 0.1540 [0.1261, 0.1903] | 0.7262 | 0.6157 | 0.7151 | 0.7087 |
| vmd/weighted_knn | 0.1206 [0.0949, 0.1572] | 0.6315 | 0.0131 | 0.9985 | 0.9355 |
| wst/logistic | 0.1569 [0.1308, 0.1922] | 0.7277 | 0.6681 | 0.6617 | 0.6621 |
| wst/weighted_knn | 0.1383 [0.1068, 0.1797] | 0.6406 | 0.0262 | 0.9994 | 0.9372 |
| vmd/vqc | 0.1442 [0.1196, 0.1763] | 0.7109 | 0.5706 | 0.7314 | 0.7211 |
| wst/vqc | 0.1352 [0.1132, 0.1668] | 0.6909 | 0.5386 | 0.7297 | 0.7175 |
| constant_prevalence | 0.0639 [0.0561, 0.0721] | 0.5000 | 0.0000 | 1.0000 | 0.9361 |

## Paired average-precision differences

- vmd_minus_wst/vqc: +0.0090, 95% interval [-0.0190, +0.0353].
- vmd_minus_wst/logistic: -0.0028, 95% interval [-0.0310, +0.0253].
- vmd_minus_wst/weighted_knn: -0.0177, 95% interval [-0.0526, +0.0218].
- vmd/vqc_minus_logistic: -0.0098, 95% interval [-0.0349, +0.0137].
- vmd/vqc_minus_weighted_knn: +0.0236, 95% interval [-0.0071, +0.0528].
- vmd/vqc_minus_constant_prevalence: +0.0802, 95% interval [+0.0591, +0.1100].
- wst/vqc_minus_logistic: -0.0216, 95% interval [-0.0468, +0.0046].
- wst/vqc_minus_weighted_knn: -0.0031, 95% interval [-0.0396, +0.0329].
- wst/vqc_minus_constant_prevalence: +0.0713, 95% interval [+0.0526, +0.0997].

These intervals condition on saved predictions. They do not include retraining uncertainty.
This fixed comparison does not establish clinical validity or quantum advantage.
