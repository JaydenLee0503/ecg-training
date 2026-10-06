# ACS extraction resumed — 2026-10-01 EDT

The user requested continuation of the paused ACS run. Restored the missing
temporary packages to `/tmp/acs-gpu-deps-v1`: CuPy 14.2.0, cuda-pathfinder 1.8.2
and NumPy 2.5.2. The existing base environment was retained.

The runner validated its frozen context and pilot, acquired the lock at
2026-10-02T01:28:54 UTC and scanned the saved checkpoints. Extraction resumed at
**01:35:15 UTC**, reusing all **12,916 completed ECGs**, without a batch limit.
The run is `results/omi_v1_gpu_retry256k/`; its manifest remains
`fc080f2481d1cb1c671f99f78876391d1d1af5cba7bd74a3c2040182cb224474`.
No numerical code, protocol, patient split, seed or exclusion changed. The
completed one-batch recovery experiment was not repeated.

## Progress snapshot

At **02:39:29 UTC**, extraction was running with **15,780/17,905 ECGs saved**,
including **2,864 new ECGs** this session. The first 179 completed batches had
a median duration of 20.26 seconds (range 14.79–91.56 seconds). The session rate
was 2,508 ECGs/hour. The remaining 2,125 ECGs would take about **51 minutes** at
that rate. Later retries can change throughput; verification, model fitting and
reporting take additional time.

The existing `scripts/finish_development256k.py` workflow was restarted under
its saved manifest and is waiting for successful extraction. It will verify
and bundle every feature record before the authorized fixed four classical fits,
six VQC fits and patient-cluster reporting. No ACS model had been fitted at this
snapshot; official test records remain reserved.

## Live status and recovery

- Extraction: `results/omi_v1_gpu_retry256k/extraction_status.json`
- Batch timing: `results/omi_v1_gpu_retry256k/extract_batches.jsonl`
- Workflow: `results/omi_development_workflow_v2/status.json`
- Later model outputs: `results/omi_models_v1_retry256k/`

Check current markers and process liveness before resuming; do not start duplicate
extractors or workflows. The old workflow failure file records the September 29
user-requested pause. Investigate any new convergence failure before changing the
frozen protocol. Only documentation/report files changed during this continuation.

Evidence: [JSON snapshot](acs_resume_2026-10-01.json).
Commands and dependency recovery: [resume guide](resume_acs_2026-09-28.md).
