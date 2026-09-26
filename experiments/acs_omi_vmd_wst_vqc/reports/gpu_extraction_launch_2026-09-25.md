# GPU extraction launch — 2026-09-25 EDT

Full extraction **started at 2026-09-26T00:57:48.594680+00:00** (20:57 EDT on September
25) on the RTX 5070. At this report's 2026-09-26T01:01:08.559236+00:00 snapshot, the runner had saved
**291 / 17,905 ECGs**, including all 131 checkpoints from preparation
and the recovery test, and was processing the next batch. This is a running
extraction, not completed features or trained-model performance.

The run is `results/omi_v1_gpu/`; its manifest SHA-256 is
`a2f0288767177d1d801744659a79517c1b6af0f4123dab5d6d631bcfb48fc685`. The [JSON report](gpu_extraction_launch_2026-09-25.json)
contains the frozen manifest, code/environment/hardware hashes, split/source
identities, all pilot diagnostics, checkpoint recovery inventory, batch timings,
and the dated launch-status snapshot. The
[execution protocol](../protocols/gpu_extraction_v1.md) and original scientific
protocol remain unchanged. No official-test waveform or classifier is processed.

## Validation and reuse

- All four device tests passed outside the sandbox in 1.441 seconds. Together
  with the 37 earlier passing CPU-side tests, all 41 ACS tests have now passed
  across those invocations. Earlier sandbox CUDA access errors are retained in
  the integration report; they occurred before numerical comparisons.
- Preparation completed at 00:08:04 UTC on September 26. It preserved 115 CPU
  NPZ archives byte-for-byte, recorded their origin and source hashes, and kept
  the original CPU run unchanged.
- The integrated 16-record pilot completed at 00:10:33 UTC. Both feature arms
  were exactly equal to the CPU results; names, physical-input hashes, per-lead
  iterations/retry limits and all 288 retry attempts agreed. Kymatio coefficient
  errors on records 11678 and 17045 were zero. The batch took
  19.024 seconds and its timed session took 26.341 seconds.
- Repeating the completed pilot verified/reused all 16 records without source
  decoding or CUDA computation. A sandbox attempt failed to write the lock at
  the resolved Windows path; the outside-sandbox retry passed.
- A controlled extraction session imported 115 checkpoints, computed 16 new
  records, and stopped after one batch at 00:18:14 UTC. All 131 checkpoints were
  verified, the writer lock was released, and original/imported CPU artifacts
  remained unchanged. The first new batch took 21.451 seconds.
- Full extraction resumed all 131 records. After it began, the recovery
  inventory was checked again with no changes. New checkpoints are being saved.

## Early timing and limits

The first 160 newly extracted ECGs in the full session took
176.825 seconds across 10 batches, about
3,257 ECGs/hour. Extrapolation at this early pace gives roughly
5.4 hours remaining at the snapshot. This is a small-sample estimate:
convergence counts, machine load and filesystem costs can change it. It includes
both feature arms and checkpoint writes, but excludes model training and the
one-time preflight/startup checks. The earlier 3.63-hour estimate covered only
VMD/descriptors and must not be presented as complete extraction time.

## Monitoring, interruptions and remaining work

`results/omi_v1_gpu/extraction_status.json` is the live progress record.
`extract_batches.jsonl` records each completed batch's IDs and wall time;
`features/` contains per-record features, completion metadata and attempt logs.
`recovery_verification.json` preserves the successful 131-record recovery check.
The launch PID was 192389; a PID or saved `running` marker alone is not proof of
liveness after a restart. Confirm status timestamps/counters are advancing and
respect the exclusive run lock before attempting a resume.

Keep the PC awake for continuous progress. SIGINT/SIGTERM requests a stop after
the current batch. An abrupt shutdown may repeat unfinished records in that
batch; completed, verified records are reused. Raw data and generated results
are gitignored and require a separate backup. The original CPU run stays stopped.

No ACS classifier training has started. After complete extraction, verify
coverage/convergence and implement bundling, fit-only preprocessing, all three
VQC seeds per feature arm, weighted-KNN/logistic controls, predictions, histories,
metrics and paired patient-cluster uncertainty. There is no model-accuracy or
quantum-advantage claim from these engineering checks.
