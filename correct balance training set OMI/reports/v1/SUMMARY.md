# Balanced OMI + Swin results — complete

All **14 new fits** and the final report completed at **2026-10-04T21:57:18Z**
(October 4, 5:57 PM Toronto time). The workflow exited 0. No fitting remains
pending. Do not retrain completed models when opening a new chat or restarting.

## Main findings

- **Oversampling increased VQC sensitivity but did not establish better average
  precision (AP).** VMD sensitivity increased from 57.06% to 63.32%, and WST from
  53.86% to 62.45%. Accuracy fell from 72.11% to 66.56% and from 71.75% to 65.11%.
- VQC AP changes were small: VMD +0.0013, paired 95% interval [-0.0105, +0.0156];
  WST +0.0021, interval [-0.0117, +0.0154]. Balanced-accuracy changes also had
  intervals including zero. This follow-up does not establish an overall VQC
  improvement over the original class-weighted models.
- **LFCC + logistic had the highest AP point estimate among the compared
  pipelines: 0.2091.** Its accuracy was 77.97%, sensitivity 61.57% and balanced
  accuracy 70.33%. Swin AP was 0.1680; the paired Swin-minus-LFCC-logistic interval
  was [-0.0853, -0.0073]. The simpler control outperformed Swin on the primary
  metric in this fixed internal-validation comparison.
- **Swin's 91.48% accuracy accompanies only 17.03% sensitivity.** Always-negative
  accuracy is already 93.61% on this validation set. Accuracy alone therefore
  gives an incomplete picture of OMI detection.
- Oversampled KNN gained sensitivity but lost AP: VMD 0.1206 → 0.0924; WST
  0.1383 → 0.0999. Both paired AP-difference intervals were below zero. The
  negative result is retained alongside the other controls.
- Swin overfit the training data: accuracy on unique original fit ECGs was
  98.82%, 100%, 100% for seeds 0/1/2, compared with validation accuracy
  90.59%, 92.01%, 91.85%. Validation sensitivities were 25.33%, 14.85%, 10.92%.
  These diagnostics did not trigger tuning or selection of a favorable seed.

## Validation results

VQC and Swin rows average metrics over all seeds 0/1/2. Validation remains at
3,581 ECGs from 3,391 patients, including 229 positive ECGs. The threshold is 0.5.

| Balanced training pipeline | AP | Accuracy | Balanced accuracy | OMI sensitivity |
|---|---:|---:|---:|---:|
| VMD + VQC | 0.1454 | 66.56% | 65.05% | 63.32% |
| WST + VQC | 0.1374 | 65.11% | 63.87% | 62.45% |
| VMD + logistic | 0.1548 | 70.76% | 66.07% | 60.70% |
| WST + logistic | 0.1553 | 66.46% | 66.63% | 66.81% |
| VMD + weighted KNN | 0.0924 | 76.74% | 58.08% | 36.68% |
| WST + weighted KNN | 0.0999 | 74.34% | 60.05% | 43.67% |
| LFCC + logistic | 0.2091 | 77.97% | 70.33% | 61.57% |
| LFCC + temporal Swin | 0.1680 | 91.48% | 56.80% | 17.03% |
| Constant natural fit prior | 0.0639 | 93.61% | 50.00% | 0.00% |

The [full report](report.md), [all-seed metrics](metrics.csv) and
[analysis JSON](analysis.json) retain every comparison, confusion count and
interval. The intervals use 2,000 valid paired patient-bootstrap draws with seed
20260925. They are pointwise, without multiplicity correction, and condition on
saved predictions; they exclude training and sampling uncertainty.

## Completed scope and preservation

- Balanced training retained all 14,324 original fit ECGs from 13,576 patients
  and appended 12,490 positive repetitions. Final counts are 13,407 per class,
  with sampling seed 20261003 shared by all new classifiers.
- Original VMD/WST features and fit-only preprocessing were reused unchanged.
  All 17,905 LFCC feature records were extracted and verified against the frozen
  waveform identities. LFCC normalization used unique original fit ECGs only.
- Six VQC fits and three Swin fits completed 40 epochs each: 240 VQC and 120
  Swin epoch-history entries. Five classical controls also completed. Every
  trial's recorded warning list is empty and saved-model predictions were checked.
- Swin used the RTX 5070 and took 783.632 seconds across its three fits. VQC used
  the original CPU simulator, three fits at a time. Model/extraction-stage launch
  to final report took 49 minutes 27 seconds, excluding earlier preparation and
  engineering checks. The feature-preparation routine took 175.194 seconds.
- Post-run verification checked 768 artifact entries through 21 completion/
  preprocessing markers, 27 current source files plus their snapshots, 529
  protected original artifacts and 4 report exports. **Zero missing or changed
  files.** See [artifact_check.json](artifact_check.json).
- Original ECG extraction and prior model results are preserved. Official test
  ECGs were not processed. Large data, models and checkpoints remain local and
  gitignored; source/report files have not been committed or backed up remotely.

## Interpretation and possible next work

This was an exploratory follow-up informed by the original validation results.
Oversampling adds no independent patients and increases VQC optimizer updates
from 17,920 to 33,520. At fixed logistic C=1, duplication also changes effective
regularization. LFCC + Swin changes the representation and classifier together.
No clinical validity, quantum advantage or VMD/WST superiority is established.

The strongest next candidate to investigate is LFCC + logistic. Any further
threshold calibration, Swin regularization or epoch selection should use a new
declared experiment and inner patient validation within the fit patients. Keep
the existing validation results in view as repeatedly examined internal evidence
and keep official test patients reserved until a final model/protocol is fixed.
No follow-up training or tuning has been started.
