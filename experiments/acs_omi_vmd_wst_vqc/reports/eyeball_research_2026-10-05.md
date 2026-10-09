# ECG rotational morphology: research review — 2026-10-05 EDT

Status: literature review and proposed experiment only. No new feature extraction,
model fitting, predictions, or accuracy result. The user asked to research the
“eyeball” approach as a possible next experiment and keep recording every result.
This review is not an instruction to restart any completed experiment.

## Evidence located

The matching source is Alavi et al., [Revealing Hidden Myocardial Infarction
Signatures from Brief Single-Lead Electrocardiograms: A Novel Framework for Smart
Wearable Applications](https://www.medrxiv.org/content/10.64898/2026.07.08.26357521v1.full),
medRxiv, 2026, DOI 10.64898/2026.07.08.26357521, version 1. A subsequent
peer-reviewed version was not verified in this search.

The study used 170 healthy and 80 AMI subjects, with selected resting 30-second
Lead-I-equivalent Holter segments. AMI recordings were obtained 24–48 hours after
the index event. Individual-feature ROC AUCs ranged approximately 0.61–0.78;
these are not held-out OMI classification accuracies. The representation uses
four EMD components and twelve descriptors: component mean frequencies and
envelopes, two global summaries, and two centroid coordinates. Geometry uses
arc-length resampling before centroid calculation. Duration comparisons covered
30 seconds to five minutes. The accessible text did not establish an independent
OMI classifier evaluation or real wearable deployment performance.

Access limitation: the search service exposed substantial indexed primary-source
text, including methods and results, but direct HTML/PDF/supplement requests
failed (the full-text HTML request returned HTTP 403). Equation images were not
readable in the extracted text. Exact preprocessing, EMD settings, supplementary
details, and an author implementation remain unverified. Do not infer that these
details are absent from the complete paper. No full paper or author code was
downloaded. A dataset-provider page also timed out.

## Relevance to this project

Our recommendation is to test whether the proposed compact representation adds
predictive information on the existing ACS task. It is a candidate, not evidence
that our accuracy or OMI sensitivity will improve.

| Question | Project evidence and implication |
|---|---|
| Can we reproduce the acquisition setting directly? | Our eligible ACS records contain 10 seconds at 500 Hz, in mV. A new run must be described as a 10-second adaptation. Repetition or padding cannot supply additional observed heartbeats. |
| Are the target labels interchangeable? | Our frozen target is OMI versus non-OMI within ACS data. Do not relabel non-OMI patients as healthy or substitute a different MI label. |
| Is Hilbert analysis new to our code? | No. `ecgvmd/features.py` already derives envelope and instantaneous-frequency statistics from VMD modes. New value would need to come from the decomposition, aggregation, geometry, or their interaction. |
| Can we reuse every old feature formula? | No. Existing VMD instantaneous frequency is clipped to [0, fs/2]. That is a frozen legacy choice, not a verified requirement of the proposed method. |
| Is ECGData.mat an equivalent replication? | No. Its ARR/CHF/NSR task and repeatedly examined 80-patient cohort differ. The current Swin cache uses standardized 500-sample windows; it cannot be treated as calibrated 30-second input. Any ECGData extension needs its own protocol and outputs. |

These dataset and implementation facts come from the [frozen ACS protocol](../protocols/acs_omi_protocol_v1_retry256k.json),
[loader report](acs_loader_2026-09-24.md), [shared descriptor implementation](../../../ecgvmd/features.py),
and [ECGData Swin report](../../../attention%20method/reports/ecgdata_lfcc_swin_v1/results.md).

Our methodological assessment: a nonsignificant difference between duration
groups would not by itself establish individual repeatability or equivalence.
We should directly measure feature changes under duration, endpoint, amplitude,
and noise perturbations on fit patients. Visual separation should be demonstrated
with a declared sampling rule and common axis scales, including failures, rather
than choosing attractive examples after inspecting disease labels. A geometric
display alone does not establish a physiological mechanism or diagnostic value.

## Feasibility and implementation cautions

[SciPy's Hilbert documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html)
defines the analytic signal and the envelope/phase-derivative construction.
For a real component `u(t)`, use `z(t) = u(t) + i H[u](t)`, envelope `|z(t)|`, and
frequency `(1 / 2π) d unwrap(arg z(t)) / dt`. Hz and cycles/minute must be labeled
explicitly; an IMF's rotation rate is not automatically the patient's heart rate.

Our implementation checks should cover phase unwrapping, sample-time alignment,
near-zero amplitude, boundary effects, and units before any disease scoring.
The time-sample mean of a complex trajectory, its arc-length-weighted centroid,
and a filled-region centroid are different quantities; do not substitute one
without declaring the change. Preserve physical amplitude through feature
extraction. A classifier scaler fitted on training patients is a separate step
from normalizing each raw ECG to unit variance.

[PyEMD's EMD documentation](https://pyemd.readthedocs.io/en/latest/emd.html)
describes sifting thresholds, extrema interpolation, boundary mirroring and
iteration limits. Its decomposition return can include a residual; the separate
IMF/residual accessor must be used or checked so a trend is not counted as a
fourth IMF. Pin an implementation and every setting; package defaults alone are
not a reproducible specification or proof of equivalence with another solver.

The existing interpreter was checked: Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1.
Distributions `EMD-signal`, `emd`, and `pyemd` were not installed in that
environment. No packages were installed or upgraded during this review.

## Recommended next experiment

The [draft pilot and comparison protocol](../protocols/eyeball_omi_pilot_v1_draft.md)
starts with synthetic verification and a small, label-independent sample of fit
patients. It then proposes Lead I and all-lead feature comparisons using logistic
regression, weighted KNN, and the existing VQC configuration with all seeds.
Extraction settings must be resolved and frozen before model scoring.

The initial comparison should use the original unique training ECGs and balanced
class weights where applicable. A second resampling experiment would answer a
different question. Keep accuracy, sensitivity, specificity, balanced accuracy,
AP and uncertainty together; the primary metric remains AP.

Current references, already completed:

| Pipeline | AP | Accuracy | OMI sensitivity |
|---|---:|---:|---:|
| Original VMD + VQC, mean seeds 0/1/2 | 0.1442 | 72.11% | 57.06% |
| Original WST + VQC, mean seeds 0/1/2 | 0.1352 | 71.75% | 53.86% |
| Balanced LFCC + logistic | 0.2091 | 77.97% | 61.57% |
| Balanced LFCC + Swin, mean seeds 0/1/2 | 0.1680 | 91.48% | 17.03% |
| Always-negative baseline | 0.0639 | 93.61% | 0.00% |

Sources: [original comparison](acs_omi_results_v1/report.md) and
[balanced comparison](../../../correct%20balance%20training%20set%20OMI/reports/v1/report.md).
The balanced LFCC results are historical pipeline references; their sampling and
training differ from the proposed weighted fits, so that contrast alone cannot
isolate a transform effect. Reusing a cache does not justify reusing a scaler
fitted outside a new inner training fold.

## Record of this work

- Completed: primary-source search, local protocol/code inspection, environment
  inspection, completion-status inspection, this review and a draft protocol.
- Existing original ACS, balanced OMI, and ECGData Swin status files all report
  complete. They were read, not rewritten. This was not a full artifact rehash.
- New experimental runs, training seeds executed, measured extraction runtimes,
  patient predictions, and new performance estimates: none.
- Pending: readable equations/source implementation review; explicit numerical
  settings; engineering implementation and checks; measured pilot cost; frozen
  scientific comparison and eventual execution.
- Failures/limitations retained: primary-source fetch restrictions and missing
  EMD distribution. Neither is an experimental model failure.

Every eventual attempted run must retain its status, failures, exclusions,
patient split, seeds, settings, runtime, predictions and uncertainty. The draft
specifies an artifact ledger and prevents a failed attempt from disappearing
when the method or retry settings change. No research outcome should be recorded
as a completed experiment until the corresponding artifacts exist.
