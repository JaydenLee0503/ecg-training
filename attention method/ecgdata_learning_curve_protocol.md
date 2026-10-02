# ECGData LFCC + Swin learning curve — protocol v1 (draft)

**Status, 2026-09-29:** protocol written only. The runner is not implemented and
nothing has been fitted. Machine-readable settings: [ecgdata_learning_curve_protocol.json](ecgdata_learning_curve_protocol.json).

## Question

Does held-out LFCC + Swin performance still rise as training patients are added?
If it does at the full training size, more patients would likely help. If it
does not, the bottleneck is probably elsewhere (window length, features, source
confounding). The completed run showed 100% training accuracy and 83.35% held-out
window accuracy, which motivated this check.

## What changes and what stays fixed

Only the number of training patients changes. Everything else is the frozen
[base protocol](ecgdata_protocol.json): the verified 1,620-window LFCC cache, the
five outer patient folds, all test patients, the 33,607-parameter Swin, AdamW
settings, 40 epochs, batch 32, seeds 0/1/2 and the logistic control.

## Training subsets

| Fraction | ARR patients | CHF patients | NSR patients |
|---:|---:|---:|---:|
| 25% | 10 | 3 | 4 |
| 50% | 19 | 6 | 7–8 |
| 75% | 28–29 | 9 | 11–12 |
| 100% | 37–38 | 12 | 14–15 |

- Patients, not windows, are sampled. A selected patient contributes all their
  windows, leads and recordings; MIT-BIH 201/202 stay one patient.
- Sampling is stratified by class: `n = max(2, ceil(fraction × class count))`.
- Subsets are nested (25% ⊂ 50% ⊂ 75% ⊂ 100%), so each step only adds patients.
- Three subset draws per fraction (seed 20260929). Draw *r* is paired with
  initialization seed *r*, so the three fits vary in both patients and weights.
- The test fold is always the full held-out fold, identical at every fraction.
- Normalization, class weights and the logistic scaler are fitted on the subset only.

## Fits

| Fraction | Swin fits | Logistic fits |
|---|---:|---:|
| 25%, 50%, 75% | 45 new (3 × 3 draws × 5 folds) | 45 new |
| 100% | 15 reused, read-only | 5 reused, read-only |

The 100% point reuses the completed `results/ecgdata_lfcc_swin_v1` trials after
checking them against their saved checksum inventory. They are never refitted or
modified. Estimated GPU time is about 10–15 minutes (extrapolated, not measured).
No early stopping, tuning or selection uses test predictions.

## Evaluation

- **Primary:** window macro-F1 at each fraction, averaged over the three fits.
- **Secondary:** window accuracy, patient-vote accuracy and macro-F1, per-class
  recall (CHF especially), and training accuracy. Every fit is reported.
- **Uncertainty:** 2,000 paired patient-bootstrap draws (seed 20260925), with the
  same test patients resampled for all fractions and arms.
- **Contrasts:** 100% − 75% (primary), 100% − 50%, 100% − 25%, and Swin − logistic
  at each fraction.

## Interpretation, fixed before running

| Outcome for 100% − 75% macro-F1 | Conclusion |
|---|---|
| Difference > 0 and interval lower bound > 0 | Still improving; more patients likely help |
| Interval includes 0 and difference ≤ 0.01 | No detectable gain at this scale (not proof of a plateau) |
| Anything else | Inconclusive |

The curve will not be extrapolated to predict accuracy at a specific larger cohort size.

## Known limitations

- The primary step adds only about 16 patients per fold. The interval will likely
  be wide, and an inconclusive result is a realistic outcome.
- Fixed 40 epochs means smaller subsets get fewer optimizer steps. Training
  accuracy is recorded to show whether small subsets underfit.
- Three subset draws per fraction capture only part of the sampling variability.
- The cohort confounds diagnosis with source database. This curve can only say
  whether more patients *like these* help, not patients from new sources.

## Before running

1. Implement the runner, with an explicit training flag and no default action.
2. Tests: subset nesting, class minimums, no test-patient overlap, deterministic
   subset hashes, fit-only normalization and class weights, read-only reuse of the
   100% trials, and synthetic interrupted-fit recovery.
3. Hash protected files before and after the run.

Outputs go to `results/ecgdata_lfcc_swin_learning_curve_v1/` and
`reports/ecgdata_lfcc_swin_learning_curve_v1/`, both separate from the completed run.
