# ECG-08 experiment log

## 2026-10-31 — implementation and launch

Implemented the previously drafted nested class-stratified patient learning
curve using the saved baseline LFCC cache, unchanged outer folds, original Swin
and logistic settings, and seeds 0/1/2. The 100% point verifies/reuses original
fits. The new runner uses a separate output directory and content-hashed source
manifest. A synthetic test passed for deterministic, nested patient selection,
class coverage and held-out isolation. Host GPU was available. Training launched
with an explicit flag and a writer lock; check `results/completed.json`, per-fit
markers and REPORT.md for current state. No outcome is claimed at launch.

## Completed

All 45 new Swin and 45 logistic fits completed; 100% baseline fits were reused.
Primary macro-F1 difference 100% minus 75% was +0.0207, paired 95% interval
[+0.0024,+0.0373]. Fewer training patients also produced fewer optimizer updates.
The independent audit verified 90 completion markers, patient isolation, nested
subsets, training-only normalization and class weights, and pooled metrics for
all eight arms. Sum of fit durations was 888.2 s. The launch-time manifest did
not directly hash every transitive imported source (e.g. `regularized_swin.py`);
this provenance limitation is retained in the report.
