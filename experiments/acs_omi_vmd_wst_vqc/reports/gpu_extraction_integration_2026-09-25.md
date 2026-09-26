# GPU extraction integration and descriptive rename — 2026-09-25

Status at 23:48 UTC: **implementation and CPU-side tests complete; integrated
GPU pilot and full extraction have not run**. No ACS classifier has been trained.

The study directory is now `experiments/acs_omi_vmd_wst_vqc/`. The enclosing
`experiments/` directory was not renamed. Imports, active commands, AGENTS.md
and repository navigation were updated. No symlink or junction is required.
SHA-256 checks before and after moving verified **476 protected files with zero
changes**, including raw data, existing results, scientific/execution protocols,
historical reports and source snapshots. See the
[move inventory](descriptive_rename_2026-09-25.json).

The three saved serial, parallel and GPU benchmark manifests remain byte-identical.
The new [migration record](../provenance/descriptive_rename_v1.json) accepts only
their specific hashes and original code maps, and verifies both original source
snapshots and relocated code. Shared numerical methods and the benchmarked CUDA
kernel are unchanged. Re-running the completed benchmark successfully verified
all 48 saved artifacts without computation after the move.

The new [runner](../scripts/extract_gpu.py) and
[orchestration module](../gpu_extraction.py) implement a separate resumable
GPU VMD / CPU WST extraction under the
[execution protocol](../protocols/gpu_extraction_v1.md). They preserve the frozen
scientific protocol and patient split. Preparation will import the 115 verified
CPU checkpoints, recording their origin and original hashes, into a new run.
The new run directory has **not yet been created**.

Implemented checks include: per-record atomic feature/completion files;
identity/feature/attempt-log checksums on resume; exclusive writer lock; bounded
16-ECG batches; per-lead retry logging before failures; no silent numerical
exclusions; clean SIGINT/SIGTERM stop at batch boundaries; and a separate
16-record CPU/reference comparison gate before full extraction. These are
implementation properties with unit coverage, not yet a completed real-data
GPU recovery test.

Validation actually performed:

- CPU-side ACS suite: 41 discovered tests in 5.129 seconds; **37 passed and four
  opt-in GPU tests skipped**. Six new tests cover final nonconvergence logging,
  invalid/test-input rejection, orphan recovery and corruption rejection,
  strict migration provenance, SIGTERM handling, and completed-run reuse.
- An explicit opt-in GPU test invocation inside the sandbox produced **four
  environment errors**, all `cudaErrorInsufficientDriver`, before GPU numerical
  comparisons could execute. `/dev/dxg` is absent inside this sandbox. This does
  not overturn the earlier successful device benchmark outside the sandbox.
- The outside-sandbox GPU test permission request was interrupted before
  execution. No successful new device test, integrated pilot, runtime estimate
  or full-run launch is claimed.
- Existing CPU checkpoints were checked after relocation, including their
  frozen identities, feature dimensions/precision and artifact hashes.

Remaining order: run the four device tests in a GPU-visible environment;
prepare the new run; execute and verify the integrated pilot; repeat the pilot
command to check reuse; run one full-extraction batch and verify recovery;
then resume full extraction. Record actual timings and outcomes before updating
the current handoff. Feature bundling, model fitting and statistical reporting
remain later work.

Historical reports and protocols retain their original paths as provenance;
replace the old workspace prefix with the new prefix when following an old
command. Current commands are in [SESSION_HANDOFF.md](../SESSION_HANDOFF.md).
