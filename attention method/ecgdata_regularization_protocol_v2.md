# ECGData LFCC–Swin epoch selection and dropout — v2

The user authorized execution on 15 October 2026 after discussing the training /
held-out accuracy gap. This first bounded study tests epoch selection and dropout.
Representation, longer-window, learning-curve and external-data studies remain
separate possible follow-ups; no improvement from them is presumed.

## Inputs and comparisons

Reuse the verified `features/ecgdata_lfcc_v1` cache: 1,620 windows, 80 patients,
unchanged five patient folds, 500 samples at 128 Hz, 13 × 1 × 16 LFCC values.
No patient/window exclusions or new extraction. Original code, cache, models,
predictions and protocols are read-only inputs. New artifacts reside in
`results/ecgdata_lfcc_swin_regularization_v2` and the matching reports directory.

R0 preserves the v1 architecture. R1 adds dropout 0.1 to attention probabilities,
attention projection and both MLP outputs (after GELU and after the second linear
layer). Both have the same 33,607 trainable parameters and matched initialization
for seeds 0, 1, 2. No DropPath, label smoothing, augmentation or oversampling.
Architecture, optimizer, class weighting and batch size otherwise follow v1.

## Selection and leakage controls

For each outer fold, construct three stratified inner folds at the patient level
using scikit-learn StratifiedKFold(shuffle=True, random_state=20261015 + outer_fold)
over sorted unique training patients. Each patient has one class. Save all patient
assignments and class counts. All leads/windows of a patient stay together.
Fit coefficient normalization and class weights separately in each inner fit.

Train each candidate / inner fold / seed for 40 epochs. Retain model weights and
validation probabilities at epochs 10, 20, 40. Save training loss and inner
validation metrics every epoch. For each candidate and epoch, pool the inner
out-of-fold predictions, calculate macro-F1 separately for each seed, and average
the three scores. Choose the epoch per candidate using that mean, then choose the
candidate/epoch jointly. Exact ties favor earlier epoch then R0 before R1.
Do not choose settings or seeds from outer predictions.

Refit **both** candidates on all outer-training patients at their own selected
epochs, for all three seeds. The combined selected procedure reuses the refit
chosen by the inner comparison; it does not train another model. This preserves
negative arms and distinguishes epoch-only from dropout results. Total: 90 inner
fits and 30 outer trials. R0 selections at epoch 40 reuse the corresponding
completed v1 fit after a fresh reload check; those models are not retrained.
All fits keep the cosine schedule horizon at 40 epochs,
including earlier-stopped refits. Outer predictions are generated only after all
inner selections are fixed and saved.

## Outputs and uncertainty

Primary: selected-procedure minus saved fixed-40-epoch v1 window macro-F1.
Also report R0, R1 and saved pooled-LFCC logistic controls, window accuracy,
per-class precision/recall/F1, confusion matrices, patient-vote metrics, every seed,
selected epochs, all inner candidates, histories, splits, predictions and runtimes.
Use the v1 patient-vote tie rule (lowest class index) and 2,000 paired patient
bootstrap draws, seed 20260925, with seed averaging inside each draw. Pointwise
95% intervals condition on saved predictions and omit retraining/selection
uncertainty. Retain failures and negative results. No model is declared superior
merely because its point estimate is larger; do not switch the primary outcome.

This cohort has already informed research decisions and confounds class with
source database. Nested selection does not make it untouched external validation.
Patient-vote and window accuracy remain different prediction units.

## Engineering and restart

Before real training: verify saved v1 artifacts/cache, test patient isolation,
normalization, model equivalence at zero dropout, retained candidate checkpoints,
deterministic dropout recovery, and CPU/GPU numerical agreement. Capture protected
file hashes and immutable source snapshots. Completed fits are verified and reused.
Checkpoints retain optimizer/scheduler and CPU/CUDA RNG state. Corrupt or changed
inputs fail closed; incomplete writes cannot be silently accepted. Reload candidate
and final checkpoints and reproduce their saved probabilities. Verify original
protected files again after execution. No package installation is planned.

Run with the existing interpreter and persistent PyTorch target, using
`scripts/run_ecgdata_regularization.py --execute-training --device cuda`.
This protocol and its JSON must be frozen before the real-data run.
