# Balanced OMI implementation review — 2026-10-04

The user resumed the previously paused experiment. The real balanced inputs are
prepared, and the full workflow has been launched. This document records the
preparation checks; it contains no new model-performance results.

## Verified preparation

- Original training: 14,324 ECGs from 13,576 patients, with 13,407 negative and
  917 positive ECGs. Original validation: 3,581 ECGs from 3,391 patients, with
  229 positives. Patient overlap is zero.
- Sampling seed 20261003 retains every original training row once and appends
  12,490 positive draws with replacement. Final counts are 13,407 per class.
  The same saved index list is used for every new classifier.
- Both VMD/WST preprocessor files were copied byte for byte. Every resampled
  input array was checked against its expected original-row indexing. All
  validation arrays, labels and identities remain exactly equal to the originals.
- VQC and logistic class weights are `null` in the new classifier protocol;
  the original protocol still uses `balanced`. Other classifier settings are
  inherited from the original manifest.
- Preparation verified 529 protected original artifact files. It pinned 27
  implementation/protocol/test files in a separate manifest and saved source
  snapshots. Large source data and original model directories were not rewritten.
- New manifest SHA-256:
  `8217704d106df9c8b772cb61f5a21c091296ad531ee142789557d354810481bc`.

## Verification

- Current full CPU suite: 13 cases discovered, 11 passed, 2 explicit CUDA cases
  skipped; 50.515 seconds, exit 0. The earlier CUDA suite passed all 4 cases,
  including full ACS batch dimensions and exact checkpoint recovery.
- Added integration checks cover preparation/reuse, changed-artifact rejection,
  interrupted LFCC batch recovery, fit-only normalization, the original natural
  class-prior baseline, mismatched prediction identities, report reuse/export
  recovery, and stopping the workflow's own child process group.
- The first integration-test attempt had 3 setup errors because its mocked
  context omitted the run directory that production context creates. Fixed the
  fixture; all four integration cases then passed in 1.041 seconds. This was
  not a model or data failure.
- Existing regression checks previously passed all five training/reporting
  cases. No original numerical source was changed afterward.
- Joblib array-shape deprecation warnings were observed; checks passed.

## Operational changes before freezing the run

The workflow now captures its output in `run.log`, starts the balanced-training
child in its own process group, and stops that group if the parent is interrupted
or fails. VQC resumes committed epochs, Swin uses two checkpoint slots, and LFCC
extraction resumes verified batches. An unfinished epoch/batch may be repeated.
Report exports can be repaired from a completed report without recomputation.

Added a real-data LFCC engineering check on the first eligible fit record of each
class. It requires finite 37 × 12 × 20 features, exact repeat extraction and the
frozen waveform identity before the full workflow starts model fits.

The GPU was idle before launch, and no Python training process was running.
Restored the exact temporary ACS packages after `/tmp/acs-gpu-deps-v1` disappeared:
CuPy 14.2.0, cuda-pathfinder 1.8.2 and NumPy 2.5.2. The separate existing PyTorch
2.7.1+cu128 target is retained. Preparation initially failed because the sandbox
mounted the resolved project path read-only; the permitted outside-sandbox retry
passed. These operational failures did not change scientific settings.

## Remaining work at this snapshot

Observe the real-data LFCC check and full workflow, retain all 14 fits and their
histories/predictions, finish paired patient-bootstrap reporting, recheck original
artifact integrity, and record the actual result and runtime. The manifest is
frozen: do not edit pinned code or protocol to accommodate a running experiment.
