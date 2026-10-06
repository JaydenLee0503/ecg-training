# Balanced OMI experiment log

## 2026-10-04 EDT — all models/report complete and verified

The workflow finished at 21:57:18 UTC and exited 0: all 14 fits, 240 VQC and
120 Swin epoch-history entries, and 2,000 valid patient-bootstrap draws. A
post-run check verified 768 artifact entries, 27 current source files and their
snapshots, 529 original artifacts and 4 report exports; no missing/changed files.
All recorded trial warning lists are empty. Original models/features remain
intact; official test ECGs remain reserved. Total workflow time from the model/
extraction launch was 49m 27s; Swin fits took 783.632 seconds combined on CUDA.

Oversampling increased VQC sensitivity to 63.32%/62.45% (VMD/WST), while accuracy
fell to 66.56%/65.11%. Primary AP was 0.1454/0.1374; paired changes from the
original models had intervals including zero. KNN AP decreased with oversampling.
Swin mean AP was 0.1680, accuracy 91.48% and sensitivity 17.03%. LFCC + logistic
had the highest AP point estimate, 0.2091, accuracy 77.97% and sensitivity 61.57%.
The paired Swin-minus-LFCC-logistic AP interval was [-0.0853, -0.0073].

Swin unique-fit accuracy was 98.82%, 100%, 100%, versus validation 90.59%, 92.01%,
91.85%, documenting overfitting rather than selecting a seed. Saved the summary
and artifact check alongside all-seed metrics and full intervals. This is
internal exploratory evidence with pointwise intervals, not external validation
or a quantum-advantage finding. No further tuning/training was started.

## 2026-10-04 17:16 EDT — LFCC complete; first real Swin fit complete

All 17,905 LFCC records and unique-fit normalization are saved; the preparation
record reports 175.194 seconds. The matched LFCC logistic control completed in
2.388 seconds with no fit warnings and exact reload predictions. Swin seed 0
completed 40 epochs at 21:15:57 UTC, recorded 266.934 seconds, and passed exact
saved-model prediction checks. It saved validation and original-training
predictions; no validation selection/tuning was performed.

At 21:16:42 UTC Swin seed 1 had reached 7/40; all three VMD VQC seeds had
14/40. Both VMD classical controls were complete. WST fits, the remaining Swin
work and final paired report remain in the active automatic workflow. This is
a progress snapshot, not a final accuracy or model-superiority finding.

## 2026-10-04 17:08 EDT — preparation verified and full workflow launched

The user resumed work. Restored the exact missing temporary dependencies; the
GPU was idle and no Python training job was running. Improved coordinated child
shutdown and persistent logging before freezing the manifest. Added and passed
four integration checks; full CPU discovery passed 11 cases, with 2 CUDA skips.
The earlier explicit CUDA checks and existing numerical regression checks passed.

Real preparation verified 13,407 fit examples per class, every original retained,
unchanged validation and byte-identical reused preprocessing. It checked 529
protected original artifacts and froze 27 implementation/protocol/test files.
The two real LFCC engineering ECGs passed waveform identity, expected dimensions,
finite coefficients and exact repeat checks. The model/extraction stages started
at 21:07:51 UTC; at 21:08:39 UTC extraction had saved 6,144/17,905 ECGs while
balanced VMD/WST training ran concurrently. Swin and the final report are queued
in the same workflow. See the handoff and preparation review for actual status.

## 2026-10-03/04 EDT — implemented and tested; paused before real-data work

The user authorized the separate folder and the balanced VMD/WST comparison,
then added a Swin Transformer comparison. Created the protocol, preparation,
LFCC extraction, Swin training, reporting and workflow code without changing
original numerical implementations or saved results.

Declared static positive oversampling with seed 20261003, common to all new
classifiers, while preserving the original patient split and natural validation
prevalence. VQC/logistic class weights are removed only in the new experiment.
The LFCC + Swin arm uses the existing temporal model, all initialization seeds
0/1/2, fixed 40 epochs, and a matched pooled-LFCC logistic control. The protocol
documents the increased VQC update budget and the full-pipeline nature of the
Swin comparison.

Seven new CPU tests passed (two CUDA tests skipped in that command); all four
explicit GPU-file tests passed, including recovery and a full-shaped batch.
All five relevant existing ACS regression tests passed. Joblib emitted array-
shape deprecation warnings; no assertion failed. Temporary exact ACS dependencies
were restored after the initial sandbox-network installation attempt failed.

The user paused work with “let's do it tomorrow.” All test commands have exited.
No real-data preparation, LFCC extraction, balanced model fitting or report has
run. Full integration review/checks and the eventual experiment remain pending.
See [SESSION_HANDOFF.md](SESSION_HANDOFF.md) for exact resume steps and limitations.
