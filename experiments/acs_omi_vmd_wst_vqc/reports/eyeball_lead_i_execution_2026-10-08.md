# Lead-I feasibility execution — 2026-10-08 EDT

The user authorized a larger Lead-I-only extraction check followed, if reliable,
by a simple class-weighted logistic experiment with patient separation and
noise/cropping evaluation. This is a separate study; it does not resume or alter
the completed all-lead pilot.

## Declared design

The [frozen-at-launch protocol](../protocols/eyeball_lead_i_feasibility_v1.json)
selects 2,048 distinct patients solely from the original fit partition, excluding
the 32 earlier pilot patients. Patient and record selection and the new split
use fixed SHA-256 identifiers without labels. There is one ECG per patient:
1,536 model-fit patients and 512 internal-evaluation patients. The first 512
separately ranked model-fit patients form the expanded engineering check.

The unchanged extractor uses four EMD/Hilbert components, Lead I, 10 seconds at
500 Hz in mV, the 64k cap, and the previous descriptors. No solver criterion,
filter, endpoint policy or numerical rejection check is relaxed.

The new feasibility gate is **at least 99% successful engineering extractions**
(at most five failures among 512), then at least 99% coverage of the model-fit
partition. This was selected prospectively as an engineering usability limit,
not a clinical threshold. It differs explicitly from the prior all-lead
zero-failure gate. Failed cases are never replaced. If either gate fails, the
study completes its failure report before evaluation extraction or fitting.

Both logistic models use C=1, balanced class weights, seed 0, lbfgs, 2,000 maximum
iterations, threshold 0.5 and StandardScaler fitted only on successful model-fit
rows. They compare all 12 descriptors with the first 10 (geometry removed).
No tuning, oversampling, VQC or Swin fit is included. The protocol fixes minimum
class counts without allowing resampling to attain them.

Failed fit records are listed as omitted from fitting. Every evaluation patient
remains in the primary metrics: failed extraction receives the original selected
model-fit prevalence as an explicitly flagged fallback. This is not a valid
feature vector or an ECG-derived model score. Covered-only metrics and coverage
by class are also reported. Clean scores are compared with fixed-model scores
under 30 dB noise, 20 dB noise and removing 0.1 seconds at each end. Noise seeds
start at 20261009 in record-ID-sorted evaluation order. Paired patient bootstrap:
2,000 draws, seed 20261009, preserving the same draws for all conditions.

These are exploratory internal-holdout results. They are not directly comparable
to the historical LFCC/Swin numbers evaluated on the original validation cohort.
Original validation and official test patients remain unused.

## Checks and recording

Three focused checks passed in 0.146 seconds: outcome-independent patient
selection and separation, training-only scaling and explicit fallback behavior
with exact model reload parity, and paired-bootstrap equality on identical
predictions. [Test log](eyeball_lead_i_tests_2026-10-08.log).
Joblib emitted a NumPy 2.5 shape-assignment deprecation warning during the reload
fixture; the saved/reloaded predictions matched exactly. The warning is retained.
These checks do not prove absence of bugs.

Source, protocol and cohort are frozen by the run manifest. Every extraction has
a terminal record, including failures, and every stage is restartable. Results
go to `results/eyeball_lead_i_feasibility_v1/`; readable exports go to
`reports/eyeball_lead_i_feasibility_v1/`. Existing completed work is reused or
verified, never retrained because a session was interrupted. The earlier pilot
and the commit workflow preference were committed as `8b69d35`.

Status at preparation: code and protocol prepared; no new waveform extraction,
classifier fit or prediction has started. The continuation outcome will be
appended after execution.
