# OMI undersampling experiment log

## 2026-10-31 — implementation and launch

The user requested consideration of balancing using fewer ECGs. Declared the
patient-grouped 1:1 undersampling protocol, retaining all positive subfit patients
and every record for each selected patient. The actual cohort contains 733
positive and 733 negative ECGs from 1,345 patients. Calibration and validation
remain untouched at natural prevalence. A synthetic mixed-label-patient test
passed, including deterministic selection and held-out exclusion. Started fixed
logistic plus Swin seeds 0/1/2 on the host GPU, with same-cohort oversampled
controls. Check completion markers and REPORT.md for the outcome. No official
test waveform is processed.

## Completed

All four new fits finished, with calibration thresholds frozen before validation
scoring. Primary undersampled-minus-oversampled logistic AP was -0.0535, paired
95% patient-bootstrap interval [-0.0833, -0.0309]. Three-seed Swin means and
threshold tradeoffs are in REPORT.md. Independent audit initially used an
incorrect `indices` key for the older baseline prediction file; after switching
to the saved record IDs it passed, reproducing per-model metrics and all 12
calibration thresholds and checking patient grouping and saved hashes. No fit
was repeated for the audit. Test waveforms were not processed.
