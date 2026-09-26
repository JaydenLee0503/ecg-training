# GPU VMD / CPU WST extraction protocol v1

This execution protocol implements the unchanged [ACS OMI scientific protocol](acs_omi_protocol_v1.md).
Its machine-readable settings are in [gpu_extraction_v1.json](gpu_extraction_v1.json).
It does not change patient assignments, input preprocessing, transform settings,
feature definitions, convergence criteria, classifier budgets or evaluation rules.

The output directory is `results/omi_v1_gpu/`, separate from both CPU runs and
the completed speed benchmark. Preparation verifies the frozen parent manifests,
the benchmark's numerical pass and every benchmark artifact checksum. It records
the current code, Python packages, CUDA runtime/driver/device, original source
archive identities, split/ledger hashes, and all imported checkpoint hashes.

VMD uses the benchmark's unchanged FP64/complex128 kernel on the RTX 5070.
Each batch contains at most 16 ECGs / 192 leads. NumPy FFT/reconstruction and
four-lead descriptor groups stay on the CPU. Capped leads restart from the
original initialization at 2,000 / 4,000 / 8,000 / 16,000 / 32,000 iterations.
Every attempt is appended and flushed before retrying or reporting failure.
Final nonconvergence stops the run; it never creates an unplanned exclusion.
WST uses the frozen CPU implementation, with one CPU library thread.

All 115 completed CPU feature archives are eligible for reuse. Preparation
copies their NPZ bytes without recomputation and preserves their original
metadata/attempt-log hashes in the new manifest. Each copied checkpoint records
its CPU origin. Newly computed checkpoints record GPU VMD / CPU WST origin.
The resulting feature archive deliberately documents both execution origins;
no accuracy or label-based decision determines which backend supplied a record.

Before full extraction, the integrated runner must recompute the same 16
label-independent fit ECGs as the completed benchmark. VMD features must satisfy
atol=1e-6, rtol=1e-5; WST and names must match exactly. Physical-input hashes,
final iterations, retry limits and all attempt diagnostics must agree exactly
(attempt order is normalized because GPU batches schedule leads differently).
The original two engineering records also repeat the Kymatio coefficient check.
Pilot checkpoints are separate from the reused CPU features. Rerunning a
completed pilot verifies its checkpoints and returns without GPU/source work.

Feature archives and completion metadata use temporary files, fsync, and atomic
replacement. Only a valid completion JSON with matching identity, feature hash
and attempt-log hash can be skipped. Orphaned NPZ files are recomputed. A lock
prevents concurrent writers. SIGINT/SIGTERM requests a stop after the current
batch; an abrupt process/PC shutdown can require repeating unfinished records
in that batch. These mechanisms do not constitute a backup of the local data.

The `--max-batches 1` extraction option provides a controlled recovery check:
save one new batch, stop, verify all saved records, then resume the same manifest.
Failures retain their traceback, active IDs, attempts and completed checkpoints.
Changed numerical code, settings or GPU runtime require a separate documented run.

Only the 17,905 eligible fit/validation ECGs are processed. Official-test
waveforms, model fitting, feature selection, validation predictions and accuracy
comparisons are outside this extraction step. No new scientific seed is used;
the GPU solver retains deterministic initialization. Timing includes source
loading, WST, checkpoint writes and validation overhead, with batch and session
times recorded separately. The earlier 3.63-hour estimate excludes these costs.
