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
