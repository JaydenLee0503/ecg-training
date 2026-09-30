# Full 256k extraction resumed — 2026-09-28 EDT

The user requested the full run. Extraction resumed from **3,716 verified ECGs**
at 2026-09-29T01:19:03 UTC in `results/omi_v1_gpu_retry256k/`, with no batch limit.
The manifest remains
`fc080f2481d1cb1c671f99f78876391d1d1af5cba7bd74a3c2040182cb224474`.
No numerical code, scientific settings, seeds, splits or exclusions changed.

## Checks completed before continuation

- A restart initially failed because the temporary GPU packages were missing.
  Restored CuPy 14.2.0, cuda-pathfinder 1.8.2 and NumPy 2.5.2 under
  `/tmp/acs-gpu-deps-v1`; the base Python environment was unchanged.
- All 3,700 parent checkpoints passed verification before the recovery batch.
- The controlled 16-ECG batch stopped cleanly at 3,716 records at 00:50:12 UTC.
  Verification passed at 00:59:27 UTC, including unchanged imported feature bytes.
- The full runner checked its pinned context, validated pilot and completed
  checkpoints again before continuing. It reuses the 3,716 saved records.

## Observed progress and runtime

At **01:25:38 UTC**, the saved snapshot recorded **4,052/17,905 ECGs**, including
336 new records in this full session. Its first 21 completed batches took
13.57–29.29 seconds each (median 19.07 seconds). Throughput including session
startup was **2,881 ECGs/hour**, suggesting **4.81 hours remaining for extraction**
at that rate. Later convergence retries can change this estimate; it excludes
feature bundling, model fitting and reporting.

The earlier estimate of days incorrectly extrapolated the exceptional recovery
batch. That batch included 04124/V5, requiring 128,234 iterations; its shared VMD
and descriptor time was 153.497 seconds, within 155.967 seconds total. Its
297 ECG/hour session rate was unsuitable as the normal production estimate.
The earlier [GPU benchmark](gpu_vmd_benchmark_2026-09-25.md) estimates 3.63 hours
for VMD and descriptors alone and does not include the full extraction pipeline.

## Automated continuation and saved state

The already authorized `scripts/finish_development256k.py` workflow is running
and waiting for successful extraction. It then verifies and bundles all 17,905
feature records, runs the fixed four classical fits and six VQC fits, and writes
predictions and patient-cluster uncertainty. It stops on extraction failure,
interruption, changed implementation or a downstream error. No model had been
trained at this snapshot; official test records remain reserved.

Use the live markers rather than this dated snapshot:

- Extraction: `results/omi_v1_gpu_retry256k/extraction_status.json`
- Completed batch timing: `results/omi_v1_gpu_retry256k/extract_batches.jsonl`
- Workflow: `results/omi_development_workflow_v2/status.json`
- Later models/report: `results/omi_models_v1_retry256k/`

Exact commands and restart guidance are in the [resume guide](resume_acs_2026-09-28.md).
Do not start a duplicate writer or rerun completed recovery checks. An additional
convergence failure requires investigation and a documented amendment before
changing the frozen run. Keep the computer awake while processing.

Machine-readable evidence: [launch snapshot](retry256k_launch_2026-09-28.json).
Only documentation/report files were edited during this launch. A sandbox attempt
to save the JSON report returned a read-only-filesystem error; the same report
write succeeded outside the sandbox. Extraction continued during that operation.
