# ACS session handoff — 2026-09-25

## Latest status: extraction stopped at VMD convergence limit

At **2026-09-26 01:29:28 UTC (21:29 EDT, September 25)**, the full GPU run stopped
with 1,795/17,905 completed ECGs. Record **01985, lead V5**, exhausted the frozen
32,000-iteration cap after all declared retries. The process exited with code 1.
All 1,795 checkpoints passed full identity/feature/attempt-log verification after
the stop; 16,110 records remain pending. A separate CPU reference attempt at the
same settings also reached the 32,000 cap with finite outputs (16.466 seconds).
No record was excluded, no production setting changed and extraction has not
resumed. See the [stop report](reports/gpu_extraction_stop_2026-09-25.md), its JSON
and [CPU diagnostic](reports/vmd_cap_01985_cpu_diagnostic_2026-09-25.json).

Next: investigate bounded higher-cap convergence on the single fit lead while
keeping tolerance fixed. Any production retry-policy amendment needs a new
recorded protocol/output manifest, explicitly reusing verified checkpoints.
Do not restart the full frozen command unchanged or silently raise its limits.
The earlier runtime estimate is not an active ETA. No ACS model has been trained.

## Launch history (before the stop)

Full GPU VMD / CPU WST extraction started at **2026-09-26T00:57:48.594680+00:00**
(20:57 EDT, September 25), using `results/omi_v1_gpu/`. At the 2026-09-26T01:01:08.559236+00:00
report snapshot, 291/17,905 ECGs were complete. The run reused all 131
saved checkpoints: 115 imported CPU records plus 16 from the controlled GPU
recovery test. The launch PID was 192389; check current progress, not this
dated count or PID, before resuming. No ACS classifier training has started.

Read the [launch report](reports/gpu_extraction_launch_2026-09-25.md) and its JSON.
All four device tests passed in 1.441 seconds, supplementing the 37 passing
CPU-side tests. The integrated 16-ECG pilot had exactly equal CPU/GPU VMD and WST
features, input hashes, iterations/retry limits and attempt diagnostics; Kymatio
errors were zero. Its rerun reused all 16 checkpoints without computation.
The one-batch recovery test stopped cleanly, all 131 records passed verification,
and full extraction resumed them without changing their files.

Manifest SHA-256: `a2f0288767177d1d801744659a79517c1b6af0f4123dab5d6d631bcfb48fc685`.
The scientific protocol, split, GPU execution protocol, code and environment
are now pinned. Do not edit their numerical implementation during this run or
rewrite a manifest to accommodate changes. Changes require another documented
run. Documentation may be updated as results arrive.

The canonical workspace is `experiments/acs_omi_vmd_wst_vqc/`; the parent
`experiments/` folder was not renamed. The move verified 476 protected files
unchanged, including the data and saved runs. Historical paths are retained as
provenance; [descriptive_rename_v1.json](provenance/descriptive_rename_v1.json)
provides narrow hash-checked migration for the three older run manifests.
The original CPU run remains interrupted with 115 records. Do not restart it
while GPU extraction is active.

The earlier [GPU benchmark](reports/gpu_vmd_benchmark_2026-09-25.md) measured
19.06× speedup for VMD/descriptors over eight CPU workers. Its 3.63-hour estimate
excludes WST/checkpoint overhead and training. Use current batch timing for
extraction estimates; the initial launch snapshot suggested about 5.4 hours
remaining, based on only 160 new ECGs. VQC training duration remains unmeasured.

CuPy 14.2.0, cuda-pathfinder 1.8.2 and matching NumPy 2.5.2 live separately in
`/tmp/acs-gpu-deps-v1`; the base environment is unchanged. Temporary dependencies
may disappear after cleanup. CUDA is unavailable inside the current sandbox
(`/dev/dxg` absent); GPU commands ran with actual outside-sandbox permission.
An attempted sandbox pilot restart also failed to write the lock at the resolved
Windows path, then passed outside. These access failures are documented, not
numerical failures. Read [PLAN.md](PLAN.md) and [LOGS.md](LOGS.md).

## Scope and location

The active task is ACS **OMI versus non-OMI** classification. The user selected
OMI and requested that this study have its own folder. All ACS-specific work now
lives under `experiments/acs_omi_vmd_wst_vqc/`; the root `ecgvmd/` package supplies shared
numerical methods. Original ECGData experiments remain in place and are described
in the [repository handoff](../../architects/SESSION_HANDOFF.md).

Read the [workspace guide](README.md), [OMI protocol](protocols/acs_omi_protocol_v1.md),
[preparation report](reports/acs_omi_preparation_2026-09-24.md), and
[ACS log](EXPERIMENT_LOG.md) before changing or running the study. The project's
[AGENTS.md](../../AGENTS.md) applies here too.

## Completed

- Verified ACS Figshare v1 downloads, now `data/CSV.zip` and `data/ECG_row_data.zip`
  relative to this folder. Both official identities and all ZIP CRCs passed.
- Loader handles the source's extrema-in-header convention, preserves native
  500 Hz signals in mV, keeps patient IDs and hides official test labels.
- The all-12-lead policy accepts 17,905 official-training ECGs from 16,967
  patients; 55 training records are excluded. There are 1,995 reserved official
  test ECGs, including 2 flagged for exclusion. No test performance is available.
- OMI split seed 20260924: 14,324 fit ECGs / 13,576 patients (917 positive), and
  3,581 validation ECGs / 3,391 patients (229 positive). No patient overlap.
  All 114 patients with different OMI labels across their ECGs keep their
  record-level labels and stay together in one partition.
- Fit records 11678 and 17045 passed the engineering check: 2,688 VMD descriptors
  and 2,808 WST features per ECG. All VMD leads converged after recorded retries;
  WST matched Kymatio exactly. The check took 60.106 seconds. Both happened to be
  OMI-negative; this was not a predictive evaluation.
- Folder reorganization retained all original dataset/result/protocol bytes.
  Active imports and commands use this workspace. The original run manifest
  stays unchanged; its narrow hash-checked migration is recorded in `provenance/`.
- All 23 ACS tests passed after relocation, including two new migration checks.
- The separate eight-worker extraction runner passed all 31 ACS tests and a
  16-record pilot in 234.593 seconds. All VMD leads converged; both serial
  reference comparisons and repeated Kymatio checks were exact. A pilot rerun
  verified and reused all 16 checkpoints without recomputing.

## Monitoring and eventual resume commands

From the repository root, inspect current progress:

```bash
cat experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu/extraction_status.json
tail -n 3 experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu/extract_batches.jsonl
```

Do not launch a second coordinator while timestamps/counters are advancing.
Monitoring note from 2026-09-26 01:10 UTC: a read-only sandbox `flock` probe
reported an available lock while the outside-sandbox extraction process and its
checkpoint counts continued advancing. Lock visibility across these execution
environments is therefore not a reliable liveness check. Use the existing job
session and advancing status timestamps; do not infer that resuming is safe
from a sandbox lock probe alone. No second writer was launched.
The command below is for recovery from an operational interruption. The current
numerical stop must be resolved through the diagnostic/amendment described above
before further full extraction. Repeating this frozen command would encounter
the same failing lead. A later validated run needs its own resume instructions:


```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu extract
```

The runner verifies the frozen manifest/code/environment, pilot and all completed
checkpoints before resuming. A complete run verifies and returns without GPU or
waveform computation. The exclusive lock rejects competing coordinators; a
stale `.run.lock` file is not itself an active lock. Do not delete checkpoints or
rerun preparation merely because a chat or PC restarted.

Keep the PC awake for uninterrupted extraction. SIGINT/SIGTERM finishes the
current batch before stopping. An abrupt shutdown may repeat unfinished records
in one 16-ECG batch. Completed feature files require their matching completion
metadata and checksums for reuse. This is resumability, not a data backup.

The [execution protocol](protocols/gpu_extraction_v1.md) describes the validated
pilot, CPU checkpoint reuse, numerical thresholds, retries, timing and recovery.
Detailed status/attempt/checkpoint locations are in [LOGS.md](LOGS.md).

## After extraction completes

Verify all 17,905 record identities, both feature arms, finite values, convergence
and split coverage. Implement bundling for the GPU manifest; the old serial
bundle command cannot read it. Model fitting and statistical-reporting runners
remain pending. The scientific plan uses fit-only mRMR-12 and angle scaling,
six qubits, two blocks, encoding once, 40 epochs and seeds 0/1/2, plus matched
weighted KNN and logistic controls. Average precision is primary. Always-negative
validation accuracy is 93.605%; accuracy alone is misleading. Save every seed,
history, prediction, runtime and paired patient-cluster uncertainty. Keep official
test patients reserved. Extraction does not automatically start model training.

No result establishes clinical validity, research novelty, transform superiority
or quantum advantage. ACS scores are not directly comparable to the original
ARR/CHF/NSR task or the unverified paper's validation accuracy.
