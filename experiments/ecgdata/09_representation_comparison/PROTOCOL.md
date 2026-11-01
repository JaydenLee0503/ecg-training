# ECG-09 — representation comparison, frozen protocol

## Question and design

Do 32 log linear-filterbank energies improve ECGData classification relative to
16 LFCCs? Does a small raw-waveform CNN provide a competitive control? Use the
completed ECG-07 all-window protocol and outputs read-only, same 21,222
nonoverlapping 500-sample standardized windows, 80 patients, five outer patient
folds, and seeds 0/1/2. No exclusions. ECGData is ARR/CHF/NSR; not OMI. No model
or threshold is selected using outer predictions. The cohort has informed earlier
research and confounds source and diagnosis; this is exploratory internal evidence.

The new filterbank input uses the same 128-sample periodic Hann, 32-sample hop,
256 FFT, one-sided PSD, 32 unit-sum linear bands 0–64 Hz, power floor 1e-12 and
right-reflection padding as ECG-07. Retain all 32 natural-log band energies
instead of the first 16 orthonormal DCT-II coefficients. Verify that the first
16 DCT coefficients reconstruct the existing LFCC cache within 2e-5. The Swin
uses the same architecture and 40-epoch training recipe; its input dimension
changes 16 to 32, adding projection parameters. Fit normalization on training
windows only. Include a balanced C=1 logistic control on per-band temporal mean
and standard deviation, fitting StandardScaler on training data only.

Raw control: fixed Conv1d(1,16,k7,stride2,pad3), GELU; Conv1d(16,32,k5,stride2,
pad2), GELU; Conv1d(32,64,k5,stride2,pad2), GELU; global average pool and linear
64-to-3 logits. Input is the same per-window standardized waveform: this does
not recover absolute amplitude. Train 40 epochs, batch 32, AdamW lr .001,
weight decay .01, cosine to .0001, class-weighted cross-entropy, gradient clip
1, shuffle seed offset 10000, and seeds 0/1/2; same as Swin. Different model
capacity and operations are disclosed. No tuning or augmentation. All arms use
the same windows and folds; absolute optimizer update counts match per fold.

## Outcomes

Primary: filterbank-Swin minus ECG-07 LFCC-Swin window macro-F1 on the same
1,620 v1 held-out windows, mean over seeds 0/1/2; 2,000 paired patient-cluster
bootstrap draws seed 20260925. Secondary: accuracy, per-class recall, patient
vote, all 21,222 held-out windows, filterbank logistic and raw CNN. Report all
seeds and both favorable and negative results. Bootstrap intervals condition on
the saved fits and predictions and do not capture model selection or repeated
use of this cohort. Save all patient splits, settings, seeds, exclusions, source
and data hashes, package/device versions, fit and epoch runtimes, warnings,
checkpoints, predictions, uncertainty and failures. No clinical, transform or
quantum advantage is claimed.
