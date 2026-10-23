# ACS/OMI sensitivity study — complete and verified

All **14 new fits**, the saved-score comparison and the calibration-threshold
report completed at **2026-10-16 03:59:19 UTC** (October 15, 11:59 PM Toronto).
The workflow took **24 minutes 31 seconds**, excluding the earlier 113.679-second
preparation. Completion was independently verified on **2026-10-23**. No training,
feature extraction or new test evaluation was needed after resuming the chat.

## What changing the cutoff achieved

Higher sensitivity came with substantially more false positives. A calibration
target was not reliably achieved on the separate validation patients. For the
**same newly fitted LFCC + logistic model**:

| Cutoff selection | Validation sensitivity | Specificity | Precision | OMI caught / missed | False positives |
|---|---:|---:|---:|---:|---:|
| Fixed cutoff 0.5 | 59.39% | 79.74% | 16.69% | 136 / 93 | 679 |
| Calibration target 80% | 73.36% | 67.51% | 13.37% | 168 / 61 | 1,089 |
| Calibration target 90% | 82.10% | 55.55% | 11.20% | 188 / 41 | 1,490 |
| Calibration target 95% | 86.90% | 42.09% | 9.30% | 199 / 30 | 1,941 |

At the 90% calibration target, LFCC logistic sensitivity was **82.10%**
(pointwise 95% patient-bootstrap interval **76.89–87.08%**). It caught 52 more
positive ECGs than cutoff 0.5, at the cost of 811 additional false positives.
The 95% target still produced only 86.90% validation sensitivity. Do not label
these settings “90% sensitive” or “95% sensitive” on validation.

The full comparison at the **95% calibration target** is:

| Model | Achieved sensitivity (95% interval) | Specificity | Precision |
|---|---:|---:|---:|
| VMD + logistic | 93.89% [90.50, 96.92] | 35.50% | 9.05% |
| VMD + weighted KNN | 100.00% [100.00, 100.00] | 0.00% | 6.39% |
| VMD + VQC | 92.14% [88.89, 95.19] | 30.81% | 8.34% |
| WST + logistic | 93.45% [89.91, 96.41] | 22.73% | 7.63% |
| WST + weighted KNN | 100.00% [100.00, 100.00] | 0.00% | 6.39% |
| WST + VQC | 91.85% [88.73, 94.37] | 23.78% | 7.61% |
| LFCC + logistic | 86.90% [82.19, 91.11] | 42.09% | 9.30% |
| LFCC + Swin | 93.60% [91.45, 95.56] | 14.40% | 7.08% |

VQC and Swin rows average metrics over **all seeds 0/1/2**, with a separate
calibration cutoff per seed. KNN reaches the targets only by labeling every
validation ECG positive because the selected cutoff is zero. That negative
result is retained. Achieved sensitivities differ, so this table does not rank
models at identical achieved sensitivity or establish a superior pipeline.

LFCC logistic retained the highest AP point estimate in these new fits:
**0.1841**, versus Swin 0.1723, VMD VQC 0.1243 and WST VQC 0.1347.
Changing the decision cutoff leaves each fitted model's AP/ROC AUC unchanged.

## How thresholds were selected

The original fitting cohort was split by patient with StratifiedGroupKFold
(five folds, fold 0 held out, seed **20261015**). This was **one calibration
holdout**, not five sets of cross-validation fits. The other four folds supplied
11,458 ECGs from 10,859 patients (733 positives) for fitting; calibration used
2,866 ECGs from 2,717 patients (184 positives). Validation retained 3,581 ECGs
from 3,391 patients (229 positives). Patient overlap is zero. Mixed-label
patients retain their record-level labels; there were no new exclusions.

All learned feature selection/scaling used unique subfit records only.
Positive oversampling (seed **20261015**) retained every subfit ECG and produced
10,725 examples per class. Original classifier settings and 40-epoch budgets
were preserved. No hyperparameter/epoch search occurred, and no model was
refitted after choosing its cutoffs. The highest observed calibration cutoff
meeting each target was frozen before validation inference; all score ties
were retained. Calibration here selects a decision cutoff, not a probability
calibration model.

These new models use fewer fitting patients and optimizer updates than the
earlier balanced study. Differences between old and new models cannot be
attributed solely to threshold changes. The within-model comparison above
isolates the effect of changing the cutoff.

## Saved-score analysis and uncertainty

The [saved-score report](saved_scores/report.md) separately describes the
earlier original and balanced models. Its cutoffs use those same validation
labels and are **exploratory**, not independently validated. For example, the
old balanced LFCC logistic model reached 90.39% sensitivity, 35.68% specificity
and 8.76% precision at its descriptive 90% target. Those values must not replace
the new calibration-selected results.

Each report retained all **2,000 paired patient-bootstrap draws**, seed
**20261015**, without single-class discards. The saved-score analysis reselects
cutoffs in each draw to describe the curve. The new-fit analysis holds cutoffs
fixed and resamples validation patients. Its intervals exclude retraining,
calibration-selection and split uncertainty. They are pointwise, without
multiplicity correction. Existing validation patients have informed earlier
research; official test patients remain untouched. This is internal research,
not evidence of clinical validity, transform superiority or quantum advantage.

## Records and verification

- Eight focused checks passed before launch. An initial synthetic fixture
  omitted the required seed list; it was corrected before real-data preparation.
- All **360 neural epoch records** are saved: 240 VQC and 120 Swin.
- All 14 calibration and validation model-reload prediction checks were exact.
- No real fitting failure occurred. **42 recorded warnings** are
  the same NumPy 2.5 array-shape deprecation during serialization/reload. They
  are preserved in [fit records](fit_records.json); no warning was suppressed.
- The completion audit and October 23 restart audit verified **622 new artifact
  entries**, **1,088 protected earlier files**, **27 source snapshots** and
  **12 report exports**, with zero mismatches and no current source drift.
- Independent sklearn calculations reproduced **all 42 new operating points**,
  calibration cutoff selection, confusion counts and ranking metrics, and
  checked patient grouping and all 360 epochs. The earlier independent check
  reproduced all 75 saved-score operating points across 25 model/seed columns.
- Two October 23 audit export attempts hit a sandbox path-case mismatch
  (`/mnt/d/projects` versus resolved `/mnt/d/Projects`). Approved execution
  outside the sandbox saved the records; no model was retrained.

Files: [full calibrated report](calibrated/report.md),
[all per-seed cutoffs and counts](calibrated/per_seed_metrics.csv),
[analysis and paired intervals](calibrated/analysis.json),
[figure](calibrated/operating_points.png), [fit runtimes](fits.csv),
[epoch histories](epoch_histories.json), [independent verification](independent_check.json),
[restart audit](restart_check_2026-10-23.json).

The separate study folder also retains the protocol, split/sampling ledgers,
preprocessors, models/checkpoints, calibration/training/validation predictions,
bootstrap draws, source snapshots, and logs. Large artifacts in `results/`
remain local and gitignored; the committed reports do not replace them.
Implementation checkpoint: `5b74eb0`. No fit remains pending or needs restart.
