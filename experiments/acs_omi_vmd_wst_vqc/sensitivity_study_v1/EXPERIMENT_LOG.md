# Sensitivity study log

## 2026-10-15 — Authorization and implementation

User approved the saved-score sensitivity comparison and new patient-separated
threshold-selection fits, with a separate folder and complete records. Protocol
fixes targets 80/90/95%, all 14 balanced fits, seeds 0/1/2, one calibration fold
inside the original fit cohort, subfit-only learned preprocessing/oversampling,
and unchanged natural validation. Old results are protected by hashes.

First verification: six tests passed; the seventh found a missing `seeds` entry
in the synthetic fixture, corrected before real-data preparation. No production
numerical changes were required. Initial in-sandbox NVIDIA probe failed because
GPU access is blocked; approved outside-sandbox probe confirmed RTX 5070.
No new dependency installation or waveform extraction.

Final focused suite: all eight tests passed in 63.357 seconds. Synthetic checks
include the real Swin fit adapter, calibration/validation separation, exact model
reload, immutable completion reuse, patient grouping, tied cutoffs, weighted
bootstrap equivalence and validation-label isolation. Source verification and
subfit-only preprocessing were then prepared. Six single-thread CPU workers are
allowed on the 16-logical-CPU machine, alongside one GPU worker. Existing study
code remains unchanged; generated inputs and source snapshots stay local.

## 2026-10-15 23:35 EDT — Saved scores complete; new fits running

Preparation completed at 02:50:05Z (113.679 seconds), with no warnings or new
exclusions and zero patient overlap. Subfit/calibration/validation ECG counts:
11,458 / 2,866 / 3,581; patient counts: 10,859 / 2,717 / 3,391; positives:
733 / 184 / 229. Oversampling produced 10,725 examples per class. Source audit
verified 1,088 protected artifacts. Implementation committed as `5b74eb0` using
the configured user identity. The sandbox denied the initial Git index write;
approved escalation completed the commit. Unrelated existing changes were excluded.

Approved GPU workflow started at 03:34:48Z. Saved-score analysis completed all
2,000 patient-bootstrap draws, with no single-class draw discarded. Figures
were inspected. Balanced LFCC logistic at the exploratory 90% target had
90.39% sensitivity, 35.68% specificity, 8.76% precision, 207 TP / 22 FN /
2,156 FP. Cutoffs were selected on those same validation scores, so this is
descriptive. Independent calibration fits are now running; no new fit result
is selected or tuned based on the descriptive analysis.

## 2026-10-23 — Completed run recovered, independently checked and documented

The workflow had finished on 2026-10-16 at 03:59:19Z in 1470.974 seconds.
All 14 fits and both reports were reused. No training, inference, waveform
extraction, protocol changes or official-test processing occurred on resumption.
All 360 epoch entries and all 42 calibrated operating points were independently
verified. Restart audit: 622 new artifact entries, 1,088 protected earlier files,
27 source snapshots and 12 report exports; zero mismatches or source drift.
The original completion markers and reports remain unchanged.

Results, all eight pipelines and uncertainty are summarized in
[reports/SUMMARY.md](reports/SUMMARY.md). LFCC logistic's 90% calibration target
yielded 82.10% validation sensitivity, 55.55% specificity, 11.20% precision,
188 TP / 41 FN / 1,490 FP; cutoff 0.5 yielded 136 TP / 93 FN / 679 FP.
VMD logistic at the 95% target yielded 93.89% sensitivity and 35.50% specificity.
VMD/WST VQC at that target averaged 92.14%/91.85% sensitivity over all seeds.
KNN required a zero cutoff and labeled all validation ECGs positive. Targets
did not reliably transfer; no cutoff was retuned on validation.

No real-data fitting failures. All 42 serialization/reload warnings
concern the NumPy 2.5 shape deprecation; exact model reload checks passed.
Fit records, runtime CSV and all epoch histories are now exported for review.
The two initial audit exports failed on the sandbox's path-case mismatch;
approved outside-sandbox execution completed them. Independent check script
was prepared during the original session and run on October 23. The pre-existing
Swin audit modification and unrelated regularization outputs remain untouched.
