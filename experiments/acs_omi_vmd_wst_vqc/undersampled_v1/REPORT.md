# OMI undersampling v1 — internal validation

Training ECG counts: [733, 733]; patients: 1345.
Primary AP difference, undersampled minus oversampled logistic: -0.0535, 95% patient bootstrap [-0.0833, -0.0309].

| Model | AP | Sensitivity at 0.5 | Specificity at 0.5 |
|---|---:|---:|---:|
| undersampled_logistic | 0.1306 | 61.57% | 67.36% |
| undersampled_swin_seed0 | 0.1693 | 72.93% | 71.33% |
| undersampled_swin_seed1 | 0.1435 | 72.93% | 69.54% |
| undersampled_swin_seed2 | 0.1379 | 66.38% | 67.09% |
| oversampled_logistic | 0.1841 | 59.39% | 79.74% |
| oversampled_swin_seed0 | 0.1487 | 13.10% | 96.72% |
| oversampled_swin_seed1 | 0.1807 | 12.66% | 97.55% |
| oversampled_swin_seed2 | 0.1874 | 14.85% | 97.29% |

All three Swin seeds shown individually; natural-prevalence validation. Bootstrap is conditional on saved fits and selected patients; no external or clinical validation. The official test set was not processed.

Mean across all three Swin seeds: undersampled AP **0.1502**, sensitivity
**70.74%**, specificity **69.32%**; matched oversampled AP **0.1723**,
sensitivity **13.54%**, specificity **97.19%**. These threshold-0.5 estimates
are secondary. The comparison changes training size and normalization as well
as sampling, and covers one selected patient subset. It does not identify which
change caused the difference or quantify sensitivity to selecting another subset.

At the 95% calibration target, undersampled logistic achieved **90.83% sensitivity
and 30.19% specificity** on validation; the target is not guaranteed to transfer.
The paired patient-bootstrap interval conditions on fitted models and the fixed
subset; it excludes training and sampling uncertainty. The validation cohort has
been repeatedly examined. The three Swin fits took 72.20, 70.51 and 70.98 seconds;
logistic took 0.44 seconds, excluding global checks/reporting. Per-epoch histories,
all threshold points, predictions, warnings, selected patients and checkpoints are
retained in `results/`. [Independent audit](audit.json) verified saved-model
artifacts, predictions, calibration cutoffs and patient grouping. These numerical
checks are not proof of bug-free software.
