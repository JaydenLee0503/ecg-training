# OMI improvement v2 — prospective protocol

The user approved implementation and execution of the proposed OMI comparison
on 23 October 2026. This is a separate study; no earlier protocol or result is
changed. Settings and executable source hashes freeze before real-data fitting.

## Cohort and separation

Reuse the sensitivity study's exact patient-separated partitions: 11,458 subfit
ECGs from 10,859 patients (733 positives), 2,866 calibration ECGs from 2,717
patients (184 positives), and 3,581 validation ECGs from 3,391 patients (229
positives). No new exclusions. Official test waveforms remain reserved; the
existing loader may check official-test metadata/ZIP member names while checking
the source archive, but does not decode test waveforms or expose test labels.

Use three StratifiedGroupKFold inner splits, seed 20261023, within subfit only.
All records of a patient stay together, including mixed-label patients. Fit
normalization and logistic scaling on unique inner training records, then
oversample positives to equal class counts. Preserve all original training
records. Inner duplication seeds are 20261023 + fold; final duplication uses
the exact sensitivity-study seed 20261015 and index sequence. Calibration and
validation retain natural prevalence. No class weighting is added to duplication.

## Candidates and selection

- LFCC logistic: C = 0.01, 0.1, 1, 10, 100; other solver settings unchanged.
  Temporal LFCC means/stds, training-only StandardScaler, 2,000 maximum iterations.
  Select C by pooled inner out-of-fold AP; exact ties favor smaller C.
- LFCC Swin: unchanged balanced-OMI architecture, with dropout 0 or 0.1 on
  attention probabilities/projection and both MLP outputs. For both arms retain
  epochs 10, 20, 40. Select each arm's epoch, and jointly select dropout/epoch,
  by mean pooled inner AP across seeds 0/1/2. Exact ties favor earlier epoch,
  then zero dropout. No favorable-seed selection or seed ensemble.
- Raw CNN: one fixed small residual 1-D network; native full 10-second, 12-lead,
  500 Hz waveforms in mV. No resampling, filtering, per-record z-scoring or
  augmentation. Training-only population mean/std per lead preserves differences
  in amplitude between ECGs. The architecture is fully specified in protocol.json;
  choose only epoch 10/20/40 by the same inner AP rule and all three seeds.

All neural fits use the balanced study's optimizer, batch size, learning rates,
weight decay and gradient clipping. The cosine horizon remains 40 epochs even
for earlier-stopped refits. Dropout zero is checked against the original Swin.
The raw CNN is an independently specified control, not a reproduction of a
published OMI network. Raw CNN changes representation and classifier together.
It runs regardless of Swin's results, preventing a result-dependent extension.

Budget: 27 neural inner fits (3 arms x 3 folds x 3 seeds), 15 logistic inner
fits, up to 9 neural final refits and one tuned logistic final refit. Reuse
verified fixed-C=1 and fixed-40 zero-dropout Swin controls from the sensitivity
study. Reuse a zero-dropout final model only if epoch40 is selected; reuse the
logistic control if C=1 is selected. The selected Swin procedure reuses its
chosen arm and has no extra fit. Preserve both Swin arms regardless of selection.

## Calibration and evaluation

Freeze all model/epoch choices before final calibration or validation scoring.
Fit selected settings on all subfit records only. On the separate calibration
patients choose the largest observed score meeting each 80/90/95% sensitivity
target, retaining ties with score >= cutoff. Save thresholds before validation
inference. Do not refit after calibration. Report achieved sensitivity rather
than treating the calibration target as guaranteed validation sensitivity.

Primary comparison: inner-selected Swin AP minus matched fixed-40 Swin AP.
Secondary: both Swin arms, tuned logistic, fixed logistic, raw CNN, comparisons
against tuned logistic and raw CNN versus selected Swin. Report AP, ROC AUC,
accuracy, balanced accuracy, sensitivity, specificity, precision, F1, confusion
counts at 0.5 and all calibrated operating points. All neural results average
metrics over seeds 0/1/2. Do not replace the primary with the best secondary arm.

Use 2,000 paired patient-cluster bootstrap draws, seed 20261023, averaging seed
metrics inside each draw and keeping fitted settings/cutoffs fixed. Preserve
draws; disclose any single-class draw exclusions. Intervals are pointwise without
multiplicity correction and omit retraining/selection/calibration uncertainty.
The existing validation cohort is repeatedly examined internal evidence, not
untouched external validation. No guaranteed improvement, clinical effectiveness,
quantum advantage, or transform superiority is claimed.

## Provenance, checks and recovery

Verify earlier completion markers and hashes before expensive work. Pin source
archives through the existing loader; verify each decoded development waveform
against the original split hash and record/patient/label identity. Preserve raw
quality decisions. Any unexpected exclusion fails the study instead of quietly
changing the cohort. Caches use recoverable batches; fits use two-slot epoch
checkpoints. Every retained candidate's predictions are reproduced after weight
reload. Test patient isolation, fit-only scaling, zero-dropout equivalence,
dropout recovery, raw-CNN gradients, GPU numerical agreement and threshold ties.

Retain source snapshots, splits, sampled indices, normalization, every candidate
score and choice, all seeds, histories, weights, predictions, uncertainty draws,
runtimes, environment, failures and warnings. Verify protected source artifacts
again at completion. Numerical tests do not prove software is bug-free. An
interrupted run resumes with the same frozen code; changed settings require a
new documented study rather than overwriting this one.
