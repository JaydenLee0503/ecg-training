# Balanced OMI experiment handoff

## Complete and verified — 2026-10-04 EDT

**All 14 fits and the final report completed at 21:57:18 UTC (5:57 PM Toronto).**
The workflow exited 0; session 33149 is closed. No work remains running or pending
for this protocol. Do not resume/retrain this completed experiment.

Read the [results summary](reports/v1/SUMMARY.md), [full report](reports/v1/report.md),
[all-seed metrics](reports/v1/metrics.csv) and [artifact check](reports/v1/artifact_check.json).

- Balanced VMD/WST VQC accuracy is **66.56% / 65.11%**, sensitivity **63.32% /
  62.45%**, and primary AP **0.1454 / 0.1374**, averaging all seeds. Oversampling
  increased sensitivity but did not establish AP or balanced-accuracy improvement
  over the original class-weighted models; overall accuracy decreased.
- LFCC + Swin achieved **91.48% accuracy**, **17.03% sensitivity**, AP **0.1680**.
  LFCC + logistic has the highest AP point estimate, **0.2091**, with **77.97%
  accuracy** and **61.57% sensitivity**. Its paired AP comparison favored the
  logistic control over Swin. Always-negative accuracy is already 93.61%.
- All six VQC and three Swin fits reached 40 epochs; all five classical controls
  finished. The report used all 2,000 paired patient-bootstrap draws.
- Post-run checks verified **768 artifact entries**, **27 current source files
  plus snapshots**, **529 protected baseline artifacts**, and **4 exact report
  exports**, with zero missing/changed files. Recorded trial warning lists are
  empty. Original numerical code/results remain intact and official test ECGs
  remain reserved.
- Total workflow from model/extraction-stage launch to report: **49m 27s**.
  Swin fits used the RTX 5070 for about **13m 4s combined**; VQC used the original
  CPU simulator. Earlier progress estimates and running-state entries below are
  historical.

Models/checkpoints/features remain in `results/v1/` (gitignored); report copies
are in `reports/v1/`. Source and report files are saved locally but uncommitted.
Any further experiment needs a separate documented protocol/output. No follow-up
tuning or official-test evaluation has been started.

## Live checkpoint — 2026-10-04 17:16 EDT

The full workflow remains active in session **33149**, workflow PID **27742**,
balanced child PID **28430** at launch. Check live status before starting another
process; do not restart completed work.

- All **17,905 LFCC records** are complete and packaged, with fit-only
  normalization. Extraction/preparation recorded 175.194 seconds.
- **Swin seed 0 completed all 40 epochs**, saved validation and original-training
  predictions, and passed exact model-reload prediction checks at
  **21:15:57 UTC**. It took 266.934 seconds; model has 2,175,718 parameters and
  peak CUDA allocation was 261,116,928 bytes. No training warnings were recorded.
- At **21:16:42 UTC**, **Swin seed 1 was at 7/40**; seed 2 remained pending.
  Both VMD classical controls and the LFCC logistic control are complete.
- At that snapshot, all three **VMD VQC seeds were at 14/40**. WST fits remain
  queued. VQC epochs take about 37 seconds; roughly 40–45 minutes of the entire
  workflow remained at the snapshot, including a reporting allowance.
- The workflow automatically runs the remaining fits and final paired report.
  No performance conclusion has been drawn from the partially completed run.

Saved outputs and code remain local. The current source/protocol is frozen; only
documentation is being updated. Original ECG extraction and completed ACS
experiments remain intact, and official test patients remain reserved.

## Resumed and full workflow running — 2026-10-04 17:08 EDT

The user resumed work. The full workflow started its model/extraction stages at
**2026-10-04T21:07:51Z**, after successful preparation and real-data LFCC checks.
The earlier paused-state entries below are historical.

- Balanced training inputs are complete: **13,407 examples per class**, all
  14,324 unique original fit ECGs retained; unchanged natural validation contains
  3,581 ECGs / 229 positives. Preprocessors were copied byte for byte.
- The full current CPU suite passed 11 tests with 2 explicit CUDA skips; the
  earlier explicit CUDA tests passed. Four new integration cases cover reuse,
  recovery, reporting identity/prior checks and coordinated worker shutdown.
- LFCC engineering checks passed on fit ECGs 00001 (negative) and 00004
  (positive), each finite with shape 37 × 12 × 20 and exact repeat extraction.
- At 21:08:39Z, LFCC extraction had saved 6,144/17,905 ECGs. Balanced VMD/WST
  training runs concurrently; GPU Swin follows LFCC completion. No new model
  performance is being reported at this snapshot.
- Unified execution session: **33149**. Workflow PID **27742**, balanced child
  PID **28430** at launch. Verify current liveness/status; do not treat old PIDs
  or this dated progress count as authoritative after a restart.
- State/logs: `results/v1/status.json`, `run.log`, `balanced.log`; per-fit
  checkpoints live under `balanced/` and subsequently `lfcc_models/`.
- New manifest SHA-256:
  `8217704d106df9c8b772cb61f5a21c091296ad531ee142789557d354810481bc`.
  **Pinned code/protocol is frozen.** Documentation can be updated as results
  arrive. Preserve this run and use a new version for scientific changes.

Read the [implementation/preparation review](reports/implementation_review_2026-10-04.md).
Preparation checked 529 original artifact files and saved 27 source/protocol/test
snapshots. Original results remain intact; official test ECGs remain reserved.
The workflow now writes `run.log` automatically and stops its owned child process
group on interruption/failure. Those changes supersede the older pending-review
and missing-log notes below. Unfinished epochs/batches may be repeated on resume.

## Earlier paused state — 2026-10-04 EDT (superseded by the resume above)

The user said **“let's do it tomorrow.”** Work is paused. Do not launch training
until the user resumes it. The preceding request authorized a separate balanced
ACS comparison in this exact folder, including a Swin Transformer experiment.

## Actual state

- Protocol, implementation and tests are saved here. The new files are untracked
  in Git; they have not been committed or backed up remotely.
- **No real-data preparation, oversampled dataset, LFCC extraction, classifier
  fit or statistical report has run for this experiment.** There is no
  `results/v1/` directory yet and no frozen new-run manifest.
- All three launched test commands exited 0. No workflow or training job was
  launched; there is no new experiment computation left running.
- Original ECG extraction, model implementations, protocols and saved results
  were not edited. Existing changes to ACS documentation/reports predate this
  balancing implementation. Keep those changes intact.

## What is implemented

- `common.py`: paths, atomic artifacts, checksums, locks, run provenance,
  deterministic oversampling and patient/record validation.
- `balanced.py`: reuse original VMD/WST mRMR/angle preprocessing; retain all
  original fit ECGs and duplicate positive fit examples to 50:50; use the
  unchanged frozen trainer with class weights removed in the separate run.
- `lfcc.py`: extraction batches with recovery and waveform/record verification;
  normalization fitted on unique original fit ECGs only.
- `swin.py`: existing temporal Swin architecture, unweighted loss, seeds 0/1/2,
  two-slot checkpoint commits, optimizer/scheduler/RNG recovery, exact saved-model
  prediction checks and a pooled-LFCC logistic control.
- `report.py`: all seeds and controls, comparison with original predictions,
  natural original fit-prevalence baseline, paired patient-bootstrap intervals.
- `run.py`: explicit commands and an `all` workflow that runs CPU VQC training
  alongside LFCC extraction and subsequent GPU Swin fits.

Read [README.md](README.md) and [protocol.json](protocol.json). Sampling seed is
**20261003**; VQC/Swin initialization seeds are **0/1/2**. Original fit counts are
13,407 negative / 917 positive; planned balanced counts are 13,407 each, including
12,490 repeated positives. Validation stays at 3,581 ECGs / 229 positives.
There are 14 proposed new fits: 6 VQC, 4 VMD/WST classical, 3 Swin and 1 LFCC
logistic. Swin uses 40 epochs, batch 128, AdamW and the previously implemented
64/128/256-width temporal architecture. VQC retains batch 32 and 40 epochs.

## Verification completed before pausing

- New CPU discovery: 9 tests discovered, 7 passed and 2 explicit CUDA tests
  skipped, 39.944 seconds, exit 0.
- Explicit GPU test file: all 4 tests passed, 44.703 seconds, exit 0. Includes
  CPU and CUDA checkpoint-resume equivalence, an interrupted checkpoint commit,
  a full ACS-shaped batch of 128 on CUDA, and LFCC control normalization/reload.
- Existing ACS training/training-runner/reporting regression checks: all 5
  passed, 1.886 seconds, exit 0.
- These cover 14 distinct test cases across the new and existing suites.
  Test fitting used synthetic data and temporary directories only.
- Joblib emitted NumPy 2.5 array-shape deprecation warnings during reload;
  assertions passed. They were warnings, not training failures.
- The RTX 5070 was available outside the sandbox. GPU access is blocked inside
  the sandbox; the initial in-sandbox probe reported that restriction.
- Restored exact temporary dependencies to `/tmp/acs-gpu-deps-v1` outside the
  sandbox: CuPy 14.2.0, cuda-pathfinder 1.8.2 and NumPy 2.5.2. The first pip
  attempt failed due to sandbox DNS restrictions; the permitted retry succeeded.

## Resume sequence

1. Read this handoff and the original ACS handoff. Check actual process state
   and any newly created completion markers before starting work.
2. Check the existing interpreter and dependency targets; do not reinstall
   packages blindly. Temporary `/tmp` packages may be removed after a reboot.
3. Finish the implementation review and integration checks before freezing the
   new manifest. In particular, exercise prepare/reuse and LFCC/report identity
   checks. Synthetic unit tests do not yet validate the full real-data workflow.
4. Review workflow interruption behavior: currently `run.py` records a surviving
   balanced child PID after a parent failure. Inspect/stop or await that child
   before restarting; do not start duplicate training. Improve coordinated
   shutdown if needed before the first real run.
5. Run `prepare`, inspect the sampling ledger/counts, unchanged validation,
   identical reused preprocessing, and protected baseline inventory. A small
   real-data LFCC engineering check is still pending.
6. Once checks pass, run the authorized full experiment and monitor actual
   timings. Do not promise an extraction or training duration from synthetic
   timings. Keep all seeds, warnings and negative results.
7. Finish the paired report, verify protected original artifacts, and update
   this handoff, the experiment log and repository/ACS handoffs.

The README contains the eventual full-run command. `run.py` writes
`results/v1/balanced.log`; its own stdout is not automatically saved to
`run.log`, so capture that output when launching. The new run has not been
launched or scheduled. Official test patients remain reserved.
