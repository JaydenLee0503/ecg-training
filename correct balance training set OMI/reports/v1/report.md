# Balanced OMI training: complete internal-validation comparison

Validation retains its natural prevalence and the original patient split. Official test ECGs remain reserved.
VQC/Swin rows average all seed 0/1/2 metrics, not probabilities. AP is primary.

| Model | AP (95% interval) | ROC AUC | Balanced accuracy | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| original/vmd/logistic | 0.1540 [0.1261, 0.1903] | 0.7262 | 0.6654 | 0.6157 | 0.7151 | 0.7087 |
| original/vmd/weighted_knn | 0.1206 [0.0949, 0.1572] | 0.6315 | 0.5058 | 0.0131 | 0.9985 | 0.9355 |
| original/wst/logistic | 0.1569 [0.1308, 0.1922] | 0.7277 | 0.6649 | 0.6681 | 0.6617 | 0.6621 |
| original/wst/weighted_knn | 0.1383 [0.1068, 0.1797] | 0.6406 | 0.5128 | 0.0262 | 0.9994 | 0.9372 |
| original/vmd/vqc | 0.1442 [0.1196, 0.1763] | 0.7109 | 0.6510 | 0.5706 | 0.7314 | 0.7211 |
| original/wst/vqc | 0.1352 [0.1132, 0.1668] | 0.6909 | 0.6341 | 0.5386 | 0.7297 | 0.7175 |
| balanced/vmd/logistic | 0.1548 [0.1267, 0.1911] | 0.7264 | 0.6607 | 0.6070 | 0.7145 | 0.7076 |
| balanced/vmd/weighted_knn | 0.0924 [0.0760, 0.1126] | 0.6206 | 0.5808 | 0.3668 | 0.7947 | 0.7674 |
| balanced/wst/logistic | 0.1553 [0.1292, 0.1899] | 0.7271 | 0.6663 | 0.6681 | 0.6644 | 0.6646 |
| balanced/wst/weighted_knn | 0.0999 [0.0820, 0.1240] | 0.6311 | 0.6005 | 0.4367 | 0.7643 | 0.7434 |
| balanced/vmd/vqc | 0.1454 [0.1215, 0.1789] | 0.7071 | 0.6505 | 0.6332 | 0.6679 | 0.6656 |
| balanced/wst/vqc | 0.1374 [0.1156, 0.1695] | 0.6852 | 0.6387 | 0.6245 | 0.6529 | 0.6511 |
| balanced/lfcc/logistic | 0.2091 [0.1726, 0.2619] | 0.7680 | 0.7033 | 0.6157 | 0.7909 | 0.7797 |
| balanced/lfcc/swin | 0.1680 [0.1418, 0.2030] | 0.7010 | 0.5680 | 0.1703 | 0.9657 | 0.9148 |
| constant_natural_prior | 0.0639 [0.0561, 0.0721] | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 0.9361 |

## Paired average-precision differences

- vmd/vqc: balanced minus original: +0.0013, 95% interval [-0.0105, +0.0156].
- vmd/logistic: balanced minus original: +0.0008, 95% interval [-0.0010, +0.0034].
- vmd/weighted_knn: balanced minus original: -0.0282, 95% interval [-0.0565, -0.0117].
- balanced/vmd: vqc minus logistic: -0.0094, 95% interval [-0.0342, +0.0178].
- balanced/vmd: vqc minus weighted_knn: +0.0531, 95% interval [+0.0317, +0.0830].
- swin minus balanced/vmd/vqc: +0.0225, 95% interval [-0.0116, +0.0550].
- wst/vqc: balanced minus original: +0.0021, 95% interval [-0.0117, +0.0154].
- wst/logistic: balanced minus original: -0.0016, 95% interval [-0.0041, +0.0002].
- wst/weighted_knn: balanced minus original: -0.0384, 95% interval [-0.0690, -0.0153].
- balanced/wst: vqc minus logistic: -0.0179, 95% interval [-0.0426, +0.0069].
- balanced/wst: vqc minus weighted_knn: +0.0375, 95% interval [+0.0185, +0.0645].
- swin minus balanced/wst/vqc: +0.0306, 95% interval [+0.0003, +0.0624].
- balanced: vmd minus wst/vqc: +0.0081, 95% interval [-0.0211, +0.0376].
- balanced: vmd minus wst/logistic: -0.0005, 95% interval [-0.0279, +0.0278].
- balanced: vmd minus wst/weighted_knn: -0.0075, 95% interval [-0.0266, +0.0109].
- lfcc: swin minus logistic: -0.0411, 95% interval [-0.0853, -0.0073].
- swin minus constant prior: +0.1040, 95% interval [+0.0814, +0.1361].

## Limits

- Exploratory follow-up on a previously examined validation set; no threshold tuning or early stopping.
- Repetition creates no independent patients. The fixed sampling seed is not a sampling-uncertainty analysis.
- Original VQC/logistic used balanced class weights. New fits use duplicated data and no class weights.
- VQC updates increase from 17,920 to 33,520; duplicated logistic rows also change effective regularization at C=1.
- LFCC + Swin changes the representation and classifier together; this is a complete-pipeline comparison.
- Patient-bootstrap intervals condition on saved predictions and exclude retraining uncertainty.
- No external validation, clinical validity, transform superiority or quantum advantage is established by this design.
