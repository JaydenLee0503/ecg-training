# ACS retry preparation — 2026-09-27

## Subsequent launch update

The one-batch recovery check completed 16 new ECGs in 18.74 batch seconds and
stopped cleanly at 1,812. All 1,812 checkpoints passed verification; imported
feature bytes remained unchanged. Full extraction resumed from those checkpoints.
At 2026-09-28T02:36:47 UTC, 1,956/17,905 records were complete and status was running.
This is a launch snapshot, not a completion claim or fixed ETA.

The downstream workflow is active and waiting for successful full extraction. It
will verify/bundle all records, fit the fixed classical/VQC models and generate
patient-cluster reports. Its state is `results/omi_development_workflow_v1/status.json`.
Real ACS fitting and final metrics remain pending. Two additional focused tests
passed for complete bundle integrity and classical artifact recovery, bringing
the CPU checks run in this session to 46 passing tests; four opt-in device tests
were skipped, separately from the successful 17-real-record GPU validation.

## Preparation record

The prior handoff stopped before the saved higher-limit diagnostic. Inspection
found that both CPU and GPU diagnostics had subsequently completed: 01985/V5
converged at iteration 67,827 under a 128,000 limit, without relaxing tolerance.
The production run itself had not resumed.

The user requested continuation of ACS work. The temporary GPU dependency folder
was missing after restart; the exact pinned packages were restored separately.
An initial sandbox package download failed due to network isolation; the authorized
outside-sandbox install succeeded. GPU access also requires outside-sandbox execution.

## Completed

- 40 CPU-side ACS tests passed; four device tests were skipped by the opt-in gate.
- A new 17-fit-record GPU check completed in 127.658 seconds. Saved VMD and WST
  features, physical inputs and all convergence diagnostics agreed exactly.
  All 17 WST checks against Kymatio passed. See the accompanying JSON evidence.
- The current CUDA fingerprint equals the original run's recorded fingerprint.
- A separate retry protocol and run were prepared, preserving original files.
  The new run imported 1,795 parent records and one diagnostic ECG.
- ACS-local model fitting, epoch recovery, bundling and statistical reporting
  implementations were added. Four focused tests passed, covering exact VQC
  recovery/parity, tamper rejection, clustered resampling and metric equivalence.

## Pending at this preparation snapshot

One-batch recovery verification, full extraction, full feature bundling, real ACS
model fitting and results reporting. No predictive performance is available.
Read `results/omi_v1_gpu_retry128k/extraction_status.json` for subsequent progress;
this document does not imply the extraction or training has completed.

The original scientific protocol and numerical modules remain frozen. A changed
retry policy has its own output directory and manifest. Official test data remain
reserved. No attention-method training was started.
