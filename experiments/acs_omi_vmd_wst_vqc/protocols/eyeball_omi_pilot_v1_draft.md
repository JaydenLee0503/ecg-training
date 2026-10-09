# Rotational morphology for OMI — draft v1, 2026-10-05 EDT

Status: proposed research protocol, not frozen and not executed. The current
request is research into a possible next experiment. Implementation and
numerical details below must be made explicit before a scientific run. This
status records task scope; it adds no separate approval rule.

Read the [research review](../reports/eyeball_research_2026-10-05.md) for evidence,
source-access limitations, and existing results. All scientific work for this
adaptation belongs in this ACS workspace. Shared numerical utilities, if needed,
belong in `ecgvmd/`; preserve the frozen original implementations.

## Question and decision criteria

Does a compact EMD/Hilbert representation provide useful OMI discrimination on
our existing patient-separated cohort? Which gains, if any, depend on added
geometry, additional leads, or the classifier?

Engineering success means reproducible, numerically checked outputs with a
complete failure ledger and measured cost. It does not establish clinical or
predictive success. The scientific primary outcome is ECG-level average precision
(AP); report paired differences and intervals even when they favor a control or
include zero. Do not choose the most favorable seed, lead, threshold or duration
on the existing validation patients.

## Data and split

- Reuse the frozen 17,905-record eligible development cohort and its existing
  record/patient assignments: 14,324 fit ECGs from 13,576 patients; 3,581 validation
  ECGs from 3,391 patients. Positive counts are 917 and 229 respectively.
- Reuse split seed 20260924 and verify the copied assignment hash. Repeated ECGs
  from one patient remain in one partition. Preserve ECG-level labels; this
  cohort includes patients whose ECG labels differ across visits.
- Input: measured 10-second, 500 Hz signals in mV. Retain the same eligibility
  cohort even for a Lead-I-only comparison. Do not concatenate visits/leads or
  repeat a waveform to claim a longer recording.
- Official test patients stay reserved. The existing validation cohort has
  already informed research decisions, so its future scores remain exploratory.
- Default: unique fit records, no oversampling, and balanced training loss for
  logistic/VQC. KNN retains its declared inverse-distance weighting. Class
  weights, scalers and selectors are fitted within each training partition.
- If tuning is introduced, use three patient-grouped inner folds within fit
  patients, seed 20261005, with an explicit finite candidate list and AP-based
  selection rule written before scores are inspected. No tuning is currently
  specified. A later threshold policy must also use only inner fit predictions.

## Stage 1: numerical verification and fit-only pilot

First implement a separate extractor with no classifier training. Proposed
output: `results/eyeball_omi_pilot_v1/`; it does not exist as a completed run.
Before a real-data pilot, freeze an engineering manifest with the package
version, source hash, all settings and the following fixtures:

1. A periodic sinusoid with known analytic envelope and frequency; test Hz and
   cycles/minute conversion and the stated boundary treatment.
2. Amplitude-modulated and multi-frequency synthetic signals; verify finite
   outputs, reconstruction against the retained residual and deterministic
   repeat extraction. Do not assume an arbitrary EMD decomposition must recover
   a particular set of modes exactly.
3. Flat/short/nonfinite inputs and fewer-than-four-IMF outputs. Return explicit
   failure records, not zeros masquerading as valid features.
4. Circles and a deliberately nonuniformly sampled trajectory, verifying the
   chosen centroid definition, interpolation convergence and unit scaling.
5. Near-zero envelopes, phase discontinuities and degenerate paths. Preserve
   diagnostics for masked points and every numerical safeguard.

Proposed real-data sample: 32 distinct fit patients, chosen by sorting SHA-256
of `eyeball_omi_pilot_v1|patient_id`, then choosing one ECG per patient by the
equivalent record hash. Resolve hash ties by identifier. Do not use outcomes
in this selection and do not replace a failing case with an easier example.
Save the selection ledger before feature extraction.

Run Lead I first, then all 12 leads on the same selected records after numerical
checks pass. Save per-lead runtime, memory where measurable, number of IMFs,
sifting counts, termination reasons, reconstruction error, endpoint diagnostics,
raw features and waveform hashes. Save readable geometry plots with common
physical axes; any magnified plot must also label its changed scale.

Assess 5-second halves against the measured 10-second record and synthetic noise,
amplitude scaling, polarity and endpoint perturbations, with perturbation seed
20261005. Declare perturbation strengths and feature-change tolerances in the
engineering manifest before execution. These checks describe 5/10-second
sensitivity; they do not validate 30-second recordings or wearable conditions.
Record every perturbation. Do not choose processing settings by diagnosis
separation in the pilot.

## Extraction decisions that remain unresolved

| Item to freeze | Requirement |
|---|---|
| Source fidelity | Obtain readable equations or document each independent implementation choice. Name the result an adaptation unless equivalence is verified. |
| EMD | Pin distribution/version, spline/extrema method, boundary extension, stopping thresholds, iteration cap, output ordering, and residual separation. |
| Signal preparation | State filtering, resampling, units and amplitude policy explicitly. Proposed starting point is the existing native input; any different preparation is a new declared pipeline. |
| Hilbert and frequency | Specify extension/cropping, phase unwrapping, derivative scheme, timestamps, low-amplitude policy and whether signed frequencies are retained. Do not silently inherit legacy clipping. |
| Global summaries | Verify weighting, averaging order, normalization and zero-denominator behavior. An average of ratios and a ratio of averages need not agree. |
| Geometry | Specify which trajectory is used, arc-length interpolation spacing, duplicate-point handling, open/closed path treatment and centroid definition. |
| Failure coverage | Cap exhaustion, insufficient IMFs and nonfinite features must stop the main run and be reported. Any retry or fallback requires a new documented version; no silent row deletion. |
| Resources | Use measured pilot time and storage to estimate a full run. No extraction/training duration is promised from the literature. |

This table intentionally prevents a draft with unverified numerical settings
from being presented as an executable reproduction.

## Stage 2: proposed fixed classifier comparison

After the engineering review, create a separate frozen protocol and output
directory, proposed `results/eyeball_omi_models_v1/`. Do not overwrite the pilot
or any existing completed result.

| Proposed representation | Purpose | Classifiers |
|---|---|---|
| Lead I, proposed 12 descriptors | Single-lead feasibility; no feature selection required | Logistic, weighted KNN, VQC |
| All 12 leads, proposed 144 descriptors | Compare with existing all-lead inputs | Train-only mRMR-12 then the same three classifiers |
| All-lead descriptors, geometry removed | Within-method test of geometry's contribution | Same selection budget and classifiers |
| All-lead descriptors, all features standardized | Detect whether reduction obscures classical performance | Logistic control |
| Frozen original VMD/WST predictions | Historical matched-cohort reference | Existing completed classifiers; no retraining |
| Existing LFCC logistic/Swin predictions | Stronger historical complete-pipeline reference | Existing completed classifiers; sampling differs |
| Constant fit-prevalence scores | Prevalence/AP and always-negative accuracy reference | No fitted classifier |

For the primary all-lead 12-input comparison, copy the original fit-only mRMR and
angle-scaling procedure, fitting it to the new representation. All three
classifiers in that arm receive exactly the same transformed inputs. The
all-feature logistic diagnostic uses its own fit-only StandardScaler and is
clearly labeled as a separate input condition.

Proposed fixed classifier settings: logistic C=1, lbfgs, max_iter=2000, balanced
weights; KNN k=10, Euclidean distance and inverse-distance weights; original
six-qubit two-layer VQC, no reupload, 40 epochs, batch 32, learning rate
0.02→0.002 and balanced loss. VQC initialization seeds: 0/1/2, all reported.
Record deterministic-estimator random_state=0 where the API accepts it.
Copy the exact optimizer schedule and seed behavior from the original protocol
into the eventual manifest rather than relying on this shorthand.

No temporal-Swin or image-network fit is proposed in the initial experiment.
A compact fixed feature vector has no declared temporal sequence for the
existing Swin model; a trajectory-sequence or image input would require an
additional architecture, information-budget comparison and protocol. The
previous Swin result remains in the report as context.

A further EMD-versus-VMD experiment using identical rotational summaries could
help separate decomposition effects. Its mode count/order and selection would
need a separate specification. The original K=8 VMD archive is not automatically
equivalent to a four-component EMD representation.

## Evaluation and complete recording

- Save every ECG's patient ID, partition, label, score, predicted label, model,
  seed and preprocessing identity. Evaluation is ECG-level; cluster uncertainty
  by patient. Do not apply ECGData-style patient voting to differing visit labels.
- Primary AP; also ROC AUC, sensitivity, specificity, precision, F1, balanced
  accuracy, accuracy and confusion counts. Fixed threshold 0.5 for the initial
  comparison. Any later calibrated threshold is a distinct analysis selected
  within training patients.
- Use 2,000 paired patient-bootstrap draws, seed 20260925, shared across arms.
  Average stochastic-model metrics across seeds within each draw. Report every
  seed separately and the mean. Document invalid draws and the interval scope:
  conditional on saved predictions, excluding retraining/selection uncertainty.
- Nominate full-geometry versus geometry-removed all-lead logistic AP as the
  primary geometry contrast before execution. Other comparisons are secondary
  and pointwise; do not select the largest observed contrast as the main result.
- Preserve losses, epoch/checkpoint histories, training diagnostics, runtime,
  environment, source/protocol/data hashes, all warnings and errors. Save
  predictions for every successful fit, including weak or failed-to-converge
  candidates, with their validity status clearly labeled.
- `attempts.jsonl` records each start, completion, interruption, error and retry,
  including UTC timestamps, settings, patient/record/lead identifiers and output
  paths. Save a cohort/exclusion ledger even if no additional rows are excluded.
- Maintain an artifact inventory with SHA-256 and byte sizes and content-hashed
  completion markers. Inspect existing markers before resume; use new output
  directories for changed settings. Never erase a negative result during retry.
- Export final tables/report under `reports/` and update the experiment log and
  both handoffs. Record pending work and failures explicitly. Git documentation
  does not back up gitignored data, caches or models.

## Execution record

As of 2026-10-05 EDT: only this draft and its research review are complete.
No pilot records have been selected or processed; no synthetic numerical tests,
classifier fits, or model performance evaluations have run for this method.
