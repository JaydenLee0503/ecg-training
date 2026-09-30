# ACS session handoff — 2026-09-25

## Paused for sleep — 2026-09-29 00:41 EDT

The user asked to wrap up and continue tomorrow. Extraction accepted a graceful
SIGTERM, finished and saved its current batch, and exited with code 0 at
2026-09-29T04:41:06 UTC: **12,916/17,905 ECGs complete (72.1%)**, **4,989 pending**.
This full session saved 9,200 new ECGs after resuming 3,716. The final status is
`interrupted`, with `stop_signal: 15` and no active records. Both extraction and
the downstream workflow have exited. No ACS model has been fitted.

The full post-stop integrity check passed at **04:57:05 UTC**: all **12,916
checkpoints verified**, 4,989 pending, and reused parent feature bytes unchanged.
The verifier also exited successfully. Snapshot: [pause record](reports/pause_acs_2026-09-29.json).

The workflow recorded `stage: stopped`, `status: failed` at 04:41:19 UTC with
`Extraction stopped: interrupted`. This is its expected response to the requested
pause, not a numerical/model failure. Preserve this record. Restart the same
workflow after the resumed extraction status becomes `running`.

The run directory remains `results/omi_v1_gpu_retry256k/`, with manifest
`fc080f2481d1cb1c671f99f78876391d1d1af5cba7bd74a3c2040182cb224474`.
The final extraction rate was 2,662 ECGs/hour, suggesting about 1 hour 52 minutes
for the pending extraction at that rate, plus restart checks and later model work.
This is a historical estimate, not an active ETA. Keep the current code/protocol.

Use the [resume guide](reports/resume_acs_2026-09-28.md), which records the exact
commands and temporary dependency location. Start the full command without a
batch limit; do not repeat the completed controlled recovery experiment. Earlier
running-state entries below are historical.

## Full continuation — 2026-09-28 21:25 EDT

The user requested the full run. `results/omi_v1_gpu_retry256k/` resumed at
2026-09-29T01:19:03 UTC from **3,716 verified ECGs**, without a batch limit.
At 01:25:38 UTC it had reached **4,052/17,905**, with 336 new records saved.
The first 21 batches took 13.57–29.29 seconds (median 19.07); the session rate
was 2,881 ECGs/hour. This gives about 4.81 hours remaining for extraction at that
snapshot, excluding model work and subject to further convergence retries.
The earlier estimate of days from the exceptional recovery batch is superseded.

The controlled recovery and its full verification already passed. Do not repeat
them solely because a chat restarts. The 3,700 imported feature files remain
unchanged. Temporary GPU packages were restored after reboot. This launch made
documentation changes only; the code, protocol, split and run manifest are fixed.

`scripts/finish_development256k.py` is running and waiting for extraction to
complete. It then verifies/bundles the full feature archive, fits the authorized
fixed comparison and reports patient-cluster uncertainty. It stops on extraction
failure or interruption. No model had been trained at this snapshot. Official
test records remain reserved.

Live markers (relative to this workspace):

- `results/omi_v1_gpu_retry256k/extraction_status.json`
- `results/omi_v1_gpu_retry256k/extract_batches.jsonl`
- `results/omi_development_workflow_v2/status.json`

Read the [launch report](reports/retry256k_launch_2026-09-28.md), its JSON evidence
and the [resume guide](reports/resume_acs_2026-09-28.md). Check current timestamps
and job liveness before resuming; do not launch duplicate writers. Earlier
sleep/stopped-state sections below are historical.

## Sleep/restart checkpoint — 2026-09-28 EDT

No extraction or training process is currently running. The **new 256k run is
prepared and verified**, not started: manifest
`fc080f2481d1cb1c671f99f78876391d1d1af5cba7bd74a3c2040182cb224474`, 3,700
verified feature checkpoints, 14,205 pending. Its 3,700 reused feature files are
unchanged from the verified 128k parent. Start with one batch and run `verify`, then
continue without a batch limit. Exact commands, status check, dependencies and
downstream workflow command are in [resume guide](reports/resume_acs_2026-09-28.md).

The failing fit ECG 04124/V5 now has a separate CPU trace and matching GPU check:
both solvers converge at 128,234; maximum mode difference is 5.7421e-13. The new
run appends a 256k limit only. The 128k run remains frozen at its failure. Read the
[diagnosis](reports/vmd_04124_diagnostic_2026-09-27.md) and
[256k retry protocol](protocols/gpu_extraction_retry256k_v1.md).

The original production numerical code still matches the frozen run manifest.
The file-by-file change list is in
`reports/code_changes_2026-09-27.md`; no source commit was made. Exact GPU package
install remains isolated under `/tmp/acs-gpu-deps-v1`; restore the same pinned
versions there if a full reboot clears `/tmp`.

## Latest status — stopped again, 2026-09-27 23:13 EDT

The amended extraction stopped at 2026-09-28T03:13:02 UTC with **3,700/17,905**
ECGs complete (14,205 pending). Fit ECG **04124, lead V5** exhausted every retry,
including 128,000 iterations. The downstream workflow detected the failure and
stopped at 03:13:20 UTC. No ACS model was trained and official test data remain
reserved. The earlier running-status paragraphs are historical.

All 3,700 completion metadata files are present; full checksum verification was
not repeated during this status check. Preserve these checkpoints and the frozen
run. Next investigate the single failing fit lead and convergence behavior before
any further production amendment. Do not restart the unchanged command or silently
raise limits/exclude the record. Evidence: `reports/retry_stop_2026-09-27.json`.

## Update — 2026-09-27 EDT: amended extraction and downstream workflow

The user authorized continuing ACS extraction and the planned fixed model
comparison. The old stop described below is historical. The new run is
`results/omi_v1_gpu_retry128k/`, manifest
`c5f0a49a223328bdde73aa4d5c382dc3de49d0f1208671e646a0eddf54d408b7`.
The original `omi_v1_gpu` directory remains frozen and unchanged.

The saved higher-limit CPU/GPU diagnostic resolved 01985/V5 at iteration 67,827.
The new policy appends 64,000/128,000 retries, keeping tolerance and every other
scientific setting fixed. Exact temporary GPU packages were restored after restart.
A fresh 17-fit-record GPU validation passed all feature/input/convergence comparisons
and Kymatio checks in 127.658 s. The CUDA fingerprint matches the original run.

Preparation reused 1,795 parent checkpoints plus the diagnostic ECG. A one-batch
recovery check reached 1,812; all 1,812 checkpoints were verified, with imported
feature bytes unchanged. Full continuation started at 2026-09-28T02:33:16 UTC.
Check `results/omi_v1_gpu_retry128k/extraction_status.json` for current progress;
do not start a second extraction writer. Another numerical cap must stop the run.

`scripts/finish_development.py` is waiting for successful extraction, then will run
complete verification/bundling, train-only preprocessing, four classical fits, six
VQC fits and reporting. State and stage logs: `results/omi_development_workflow_v1/`.
It stops on extraction interruption, failure or changed code. Model outputs:
`results/omi_models_v1_retry128k/`. Do not launch duplicate model jobs. No real ACS
model had been fitted at launch. No attention training was started.

New ACS-local implementations include `scripts/bundle_gpu.py`, `training.py`,
`scripts/train_models.py`, `reporting.py`, and `scripts/report_models.py`.
Synthetic tests verified exact original-VQC parity after epoch/optimizer/RNG recovery,
paired patient resampling, metric equivalence, bundle integrity, and classical model
artifact reuse. Shared numerical modules and the original scientific protocol remain
unchanged. Official test data remain reserved.

See [retry protocol](protocols/gpu_extraction_retry128k_v1.md) and
[preparation report](reports/retry_preparation_2026-09-27.md), with JSON evidence.

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
