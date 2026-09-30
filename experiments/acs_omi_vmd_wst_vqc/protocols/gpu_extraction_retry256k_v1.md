# ACS bounded 256k retry amendment — 2026-09-27 EDT

Parent: `results/omi_v1_gpu_retry128k/`, stopped with 3,700 completed ECGs at
04124/V5. The exact parent manifest and stop hashes are in
`gpu_extraction_retry256k_v1.json`; scientific settings are in
`acs_omi_protocol_v1_retry256k.json`.

## Sole scientific change

Append a deterministic restart at **256,000** iterations after 128,000. Keep
tolerance 1e-7, initialization, all VMD/WST settings, input units and duration,
record eligibility, patient split, preprocessing and classifier budgets unchanged.
The runner rejects any other settings change. Another final nonconvergence stops
the run, preserving logs and completed checkpoints; no silent exclusion is allowed.

The fit lead 04124/V5 was separately checked: unchanged CPU and GPU solvers both
converged at 128,234 iterations, with mode differences below 6e-13. The CPU diagnostic
had a bounded 512,000 allowance; it stopped at the actual tolerance crossing. The
GPU check used the proposed 256,000 bound. Neither diagnostic modified production.
See the [diagnosis report](../reports/vmd_04124_diagnostic_2026-09-27.md).

## Preservation and validation

- New output: `results/omi_v1_gpu_retry256k/`; keep both previous runs intact.
- Reuse all 3,700 converged parent checkpoints byte for byte for feature files and
  logs. Rebind only copied metadata to the new manifest and record parent hashes.
- Retain the 17-record validated pilot and pin the new single-lead CPU/GPU evidence.
- Reuse the unchanged 16-ECG production batching and numerical implementations.
- Verify a controlled one-batch stop/recovery before full continuation.
- The official test remains reserved. No incomplete feature bundle can train models.

## Downstream work

After full extraction, `scripts/finish_development256k.py` runs complete feature
verification/bundling and the already-declared fixed comparison. It writes workflow
state to `results/omi_development_workflow_v2/` and model outputs to
`results/omi_models_v1_retry256k/`. This workflow uses explicit source/output arguments;
the stopped previous workflow is retained. Bundling recognizes only the two declared
retry protocol IDs and delegates to their strict manifest readers.

The six VQC fits, four classical fits, all seeds, epoch recovery, prediction saving,
fixed threshold and paired patient-cluster reporting are unchanged. No training
budget or tuning decision is based on validation performance.
