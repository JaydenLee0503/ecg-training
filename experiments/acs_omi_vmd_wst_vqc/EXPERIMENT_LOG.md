# ACS experiment log

## 2026-09-23 — ACS download and source-format verification (no training)

Verified the user's `experiments/acs/data/CSV.zip` and `experiments/acs/data/ECG_row_data.zip` against the official
Figshare v1 sizes and MD5 hashes. Both match; all 39,912 archive members pass CRC.
Decoded all 19,955 raw ECG payloads, matched every metadata record to its waveform,
and verified 18,909 patients with zero overlap between the official train/test
partitions. Test disease labels are withheld. No identical raw waveform files
were found. The archives are ignored by Git and were not modified or extracted.

The official source has systematic WFDB header errors: all 19,955 records store
channel maxima/minima in fields intended for initial values/checksums. Two
training records contain 7 seconds of samples despite declaring 10 seconds;
52 records have a constant lead, and 3 training records contain missing-value
markers. These are source issues, not download corruption. No data exclusions,
header corrections, feature extraction, or model training were performed.

Saved a reproducible verifier, official source snapshot, detailed JSON flags,
and the [ACS verification report](reports/acs_download_verification_2026-09-23.md).
That report also records the initial sandbox write failure, approved reruns, and
the final 37.497-second scan. Download integrity passes; the verifier deliberately
returns exit 1 for the source-format failures. A task/loader/exclusion protocol is
still needed. OMI is only 6.41% of training records, so accuracy alone is unsuitable
for judging a future model. Existing experiments and frozen inputs were unchanged.

## 2026-09-24 — ACS loader implemented and fully validated (no training)

Added `experiments/acs/loader.py` to read the verified ZIP archives directly, with explicit
patient IDs, optional diagnosis targets, hidden test labels, native 500 Hz signals
in mV, and named lead selection. Source hashes and metadata/partition consistency
are checked on opening. The source's extrema-in-header convention is validated
against each waveform and logged; original headers and archives are unchanged.
The default policy requests all 12 leads and excludes short records, flat
requested leads, and missing samples in requested leads, without imputation.

The completed 43.703-second audit checked all 19,955 ECGs and accepted **17,905
training records / 16,967 patients** and **1,993 test records / 1,889 patients**.
It logged **57 exclusions**: 52 flat-lead, 3 missing-marker, and 2 short records.
Eligible OMI training labels are 1,146 positive and 16,759 negative; test labels
remain absent. OMI was a loader validation choice, not a frozen scientific task.
All **11 unit tests passed**. Independent ledger checks verified identities,
patient separation, label availability, counts, hashes, and exact agreement of
the excluded set with the previous source audit.

The initial shell launch encountered a read-only mount before auditing; it was
rerun with approved escalation. The first actual run failed because the initial
implementation incorrectly required vessel annotations to be binary. Category 2
is present in those fields. Restricted binary validation to the six diagnosis
targets and added a regression test. Retained the failed run status and used
`experiments/acs/results/loader/2026-09-24_all_leads_omi_v2/` for the completed rerun, with a
19,955-row decision ledger, source/code hashes, summary, and completion marker.

API, commands, full counts, exclusions, failure history, and remaining protocol
decisions are in the [loader report](reports/acs_loader_2026-09-24.md) and its
version-controlled JSON summary. No ACS features or models were computed, and
the existing ECGData runners remain unchanged. A future official test evaluation
must declare how the two excluded test records affect coverage or submission.

## 2026-09-24 — ACS OMI protocol, patient split and matched feature check

The user selected OMI detection. Declared a separate fixed-budget ACS protocol
before model scoring: all 12 leads, full 10-second native 500 Hz inputs, VMD
descriptors versus pooled log-WST, fit-only mRMR-12 and angle scaling, a compact
six-qubit VQC with seeds 0/1/2, and matched weighted-KNN/logistic controls.
Average precision is primary; full metrics and paired patient-cluster intervals
are planned. The earlier ECGData protocols and completed models are unchanged.

Saved all 17,905 eligible development assignments: 14,324 fit ECGs / 13,576
patients (917 OMI-positive) and 3,581 validation ECGs / 3,391 patients (229
positive), using split seed 20260924. Verified zero patient overlap and retained
record-level labels for 114 patients with different OMI labels across their ECGs.
Official test ECGs remain reserved. Always-negative validation accuracy would
be 93.605%; this label-count calculation is not a trained-model result.

Added resumable prepare/smoke/extract/bundle commands and shared preprocessing /
classifier interfaces. All 21 focused tests passed in 1.936 seconds. The real
two-ECG check finished at 23:35:11 EDT in 60.106 seconds. Both records produced
2,688 VMD and 2,808 WST features; WST matched Kymatio exactly on all 24 leads.
VMD needed retries on 17 lead instances, with 20 capped attempts and no final
capped leads; maximum observed count 4,988. VMD took 36.611 / 16.662 seconds
per ECG versus WST 0.096 / 0.081 seconds. A repeated smoke command verified and
reused both completed checkpoints without recomputation.

The initial preparation write failed on the read-only sandbox mount and was
rerun with approved escalation; the feature check used the required write
escalation too. An initial Matplotlib cache warning was resolved for the final
test invocation by using a temporary cache directory. Full extraction was not
launched; serial VMD cost is substantial, and two records are not a robust
cohort runtime benchmark. No ACS model was fitted or scored, and no test signal
was processed by this experiment. Details, exact hashes, all attempt records,
artifact paths, and remaining work:
[preparation report](reports/acs_omi_preparation_2026-09-24.md),
[protocol](protocols/acs_omi_protocol_v1.md), and the linked machine-readable files.

## 2026-09-25 — ACS study moved into its own workspace

At the user's request, moved all ACS-specific source, scripts, tests, protocols,
reports, both data ZIPs and saved run artifacts into `experiments/acs/`. Moved the
ACS chronology from the root log into this file and added an ACS README and
session handoff. The shared repository handoff and agent instructions now route
ACS work here; original ECGData work retains its existing locations.

All 36 moved files matched their original hashes immediately after relocation.
Historical JSON, the scientific protocol JSON, dataset archives and saved result
files remain byte-identical. Active code changed only to use the new locations,
imports and strict compatibility checks; the verifier also now requires a new
output filename so that a rerun cannot overwrite the frozen audit. Five original
source snapshots preserve the code corresponding to saved audit/run hashes.

All 23 ACS tests passed in 1.977 seconds, including two new tests that reject
unrecognized manifests and changed migrated/shared sources. The relocated CLI
was invoked from `/tmp` and verified/reused both completed feature checkpoints
without recomputation. Root ignore rules cover the new data and result paths.
The final integrity check rechecked the original 930 VQC artifacts and 68
original source/report files, as well as the relocated immutable artifacts.

A documentation-update attempt exceeded the tool's output limit before applying
edits. The replacement script initially hit the sandbox's read-only mount and
then completed with approved escalation. No model training, new data download,
experiment resampling or Git commit was performed. The move inventory and exact
manifest migration are recorded under `provenance/`; final validation is in
[the organization record](reports/organization_2026-09-25.json).

## 2026-09-25 — Parallel feature pilot passed; full extraction started

Following the user's request to continue, implemented a separate bounded
eight-process extraction runner and declared an execution protocol. The
scientific protocol, patient split, source archives and existing numerical code
remain unchanged. The new run copies the verified split and records its own
manifest in `results/omi_v1_parallel/`. All 31 ACS tests passed in 5.332 seconds,
including eight new tests of process execution, failures, checkpoints, identity,
reference agreement and writer locking. Initial preparation encountered the
sandbox's read-only mount; the approved rerun completed successfully.

The label-independent 16-fit-record pilot finished in 234.593 seconds at
20:18:27 UTC. Both original reference records matched the serial vectors and
diagnostics exactly, and their repeated Kymatio errors were zero. All 192 VMD
leads converged after recorded retries, with 96 capped intermediate attempts
and a maximum iteration count of 7,583. A repeated pilot command verified and
reused all 16 saved records. A later independent check verified the manifests,
split/ledger hashes, patient separation and all 26 then-completed checkpoints.

Full extraction started at 21:03:57 UTC with eight workers and reused the pilot
records. The archived snapshot at 21:06:57 UTC has 28/17,905 complete; consult
the saved stage status for current progress. No official test ECGs are included.
The pilot rate of 245.53 ECGs/hour extrapolates to 72.92 hours for feature
extraction, with substantial small-sample uncertainty. This is not VQC training
time: no ACS model has been fitted or scored, and training has not been timed.
The planned six VQC fits each use 40 epochs; parallel-manifest bundling, fitting
and reporting code remain pending. The extractor does not start training.

Exact settings, environment, seeds, timings, per-record artifact hashes,
convergence diagnostics, reference errors, tests, failure history and remaining
work are in the [parallel extraction report](reports/parallel_extraction_2026-09-25.md)
and its linked JSON summary. Checkpoints/attempt logs are local and gitignored.
No predictive comparison or quantum-advantage conclusion is available.

Hardware follow-up to the user's runtime question confirmed an RTX 5070
(12,227 MiB VRAM, driver 610.88) and Ryzen 7 8700F (8 physical cores). The initial
GPU query was sandbox-blocked; approved read-only detection succeeded. Current
VMD is NumPy/CPU-only and represents 99.77% of summed pilot transform-worker
time (1,496.997 seconds VMD; 3.466 seconds WST). Recommended a separate,
numerically validated acceleration benchmark before changing execution.
No GPU timing, implementation change or model training occurred; the existing
CPU extraction continues. Hardware details and reference guidance are recorded
in the parallel report.

## 2026-09-25 — Plan/log guide added; descriptive folder rename unfinished

The user asked for the experiment plan, a more descriptive folder name and log
locations. Added PLAN.md and LOGS.md. Selected `acs_omi_vmd_wst_vqc` as the
descriptive name, but the physical rename has not completed.

The first helper could not inspect the host extraction PID from the sandbox.
An approved live-move attempt temporarily stopped the coordinator/workers,
failed to rename the open directory with PermissionError 13, and resumed all
stopped processes in its finally block. Extraction continued successfully.
Next, an intentional SIGINT to the verified coordinator stopped extraction
for maintenance. Pool shutdown finished, the runner retained a KeyboardInterrupt
trace and an `interrupted` status, and exited 130 with 115 completed checkpoints
at 21:26:58 UTC. Up to eight in-flight ECGs may need recomputation on resume.

Two closed-directory moves rolled back after Linux symlink creation failed
with PermissionError 1, including the approved outside-sandbox attempt. A
Windows directory-junction fallback also returned nonzero and rolled back.
The user interrupted a further junction diagnostic. Subsequent read-only
inspection confirmed `experiments/acs/` still exists, the requested new folder
does not, and extraction remains interrupted at 115 completed records.

No frozen code, scientific settings or saved manifests were rewritten, and no
GPU experiment or model training was started. The live-state guides now state
the interruption explicitly. The rename remains pending; repeated compatibility
link attempts did not solve it. Existing checkpoints can be resumed using the
original command and original directory.

## 2026-09-25 — GPU VMD benchmark requested and started

The user explicitly requested a GPU speed test. Added a separate FP64 CUDA VMD
prototype and declared the numerical/timing protocol before real-ECG timings.
The frozen production sources, patient split and all existing checkpoints remain
unchanged; CPU extraction stays interrupted at 115 records during timing.

CuPy was absent. Its first isolated install failed on sandbox network access;
the approved retry installed CuPy 14.2.0, cuda-pathfinder 1.8.2 and matching
NumPy 2.5.2 under `/tmp/acs-gpu-deps-v1`, without changing the base environment.
The local CUDA/NVRTC installation is 12.8; CuPy reports build/linked runtime
12.9, with driver API 13.3. The device is an RTX 5070 with 12,227 MiB VRAM.
Four device tests passed in 1.802 seconds; all 31 existing ACS tests passed in
4.872 seconds.

The benchmark uses the same 16 label-independent fit ECGs as the CPU pilot.
On the first two records, GPU features and iteration counts match exactly;
maximum mode error is 6.0004e-15 mV and centre-frequency error 1.1013e-13 Hz.
Measured serial CPU/GPU times are 54.412/16.814 seconds. The eight-worker CPU
baseline and three 192-lead GPU trials are still running/pending, so this is
not yet a final throughput conclusion. Stage events, failure handling,
environment, input hashes and numerical outputs are being saved in
`results/gpu_vmd_benchmark_v1/`. See the
[benchmark report](reports/gpu_vmd_benchmark_2026-09-25.md) and
[protocol](protocols/gpu_vmd_benchmark_v1.md). No model training or full GPU
extraction has been launched.

## 2026-09-25 — GPU benchmark complete: 19.06× measured VMD speedup

The 16-record benchmark completed at 22:12:38 UTC. The fresh eight-worker CPU
baseline took 222.416 seconds; GPU repetitions took 11.998, 11.670 and 11.656
seconds (median 11.670). All 48 record-level GPU comparisons have exactly equal
stored float32 VMD features and exact names/iteration counts/retry limits.
Independent verification matched all 288 VMD attempts per trial, including 96
capped intermediate attempts and zero final capped leads. Maximum final count
was 7,583. First-two-record mode/frequency errors remain at floating-point scale.

Median speedup is 19.06× over eight CPU workers, or 4,936 ECGs/hour. The 3.63-hour
cohort extrapolation covers VMD and CPU descriptors only; WST, source loading,
checkpoint writes and all classifier training are excluded. CPU was timed
once and GPU three times on just 16 fit ECGs; no full-run duration is established.
Context initialization, warmup and source loading are recorded separately.

Re-invoking the benchmark verified all 48 saved artifacts without recomputation.
Independent checks reverified all 115 production checkpoints and their frozen
manifest/settings, with no problems. The detailed [report](reports/gpu_vmd_benchmark_2026-09-25.md)
and [JSON record](reports/gpu_vmd_benchmark_2026-09-25.json) contain all settings,
hardware, code hashes, per-record checks, timing repetitions and artifact hashes.
The CPU run remains interrupted. A resumable GPU production runner with a new
execution manifest is the next implementation step; neither full GPU extraction
nor ACS model fitting has been launched by this benchmark.

## 2026-09-25 — Descriptive rename complete; resumable GPU integration implemented

Renamed the actual study directory to `experiments/acs_omi_vmd_wst_vqc/` without
symlinks/junctions. Updated imports, commands, root navigation and AGENTS.md.
Before/after SHA-256 verification found zero changes in 476 protected files.
The old manifests and protocols remain byte-identical; an explicit narrow
migration verifies original snapshots and current sources for all three known
runs. The completed benchmark reverified its 48 artifacts without recomputation.
All 115 saved CPU feature checkpoints also passed identity/hash/dimension checks.

Implemented a separate GPU VMD / CPU WST runner and execution protocol. It is
designed to reuse CPU checkpoints, validate a 16-record integrated pilot,
checkpoint each ECG, log retries before failures, stop at batch boundaries and
resume only pending records. Numerical kernels and scientific settings are
unchanged. New run preparation, integrated pilot, recovery check and full launch
have not occurred; `results/omi_v1_gpu/` does not yet exist. No models trained.

The CPU-side suite passed 37 tests, with four opt-in GPU tests skipped, in
5.129 seconds. The explicit GPU test invocation inside the sandbox failed with
four `cudaErrorInsufficientDriver` environment errors before numerical checks.
The GPU device `/dev/dxg` is absent in the sandbox. The outside-sandbox device
permission request was interrupted before execution. This is an execution-access
block, not evidence of a numerical test failure. Earlier successful benchmark
device tests remain recorded separately. See the
[integration report](reports/gpu_extraction_integration_2026-09-25.md),
[new protocol](protocols/gpu_extraction_v1.md) and updated handoff for next steps.

## 2026-09-25 EDT — GPU pilot/recovery passed; full extraction started

The user explicitly requested full GPU extraction and reiterated continuation.
All four GPU device tests passed in 1.441 seconds outside the sandbox. Preparation
imported 115 existing CPU checkpoints unchanged into `results/omi_v1_gpu/`.
The integrated 16-ECG pilot matched both feature vectors, physical inputs,
convergence counts/retry limits and all 288 attempt diagnostics exactly. Both
Kymatio reference errors were zero. Pilot batch/session timings were
19.024/26.341 seconds. Its rerun reused all 16 records without
computation. The initial sandbox rerun hit read-only `.run.lock` access at the
resolved Windows path; its outside-sandbox retry succeeded.

The controlled one-batch extraction stopped at 131 complete ECGs (115 reused,
16 new) at 00:18:14 UTC on September 26. All 131 checkpoints and original/imported
CPU hashes were verified; the released writer lock and hashes are saved in
`recovery_verification.json`. Full extraction started at 2026-09-26T00:57:48.594680+00:00
and reused all 131. Initial snapshot: 291/17,905 complete at 2026-09-26T01:01:08.559236+00:00.
Recovery checkpoint hashes remained unchanged after resume. The first
160 new ECGs took 176.825 batch seconds; early remaining-time extrapolation
was 5.4 hours, subject to convergence/machine-load variation. No training or
official-test waveform processing occurred. See the
[launch report](reports/gpu_extraction_launch_2026-09-25.md) and JSON for all
settings, identities, validation checks, timing and pending stages.

## 2026-09-25 EDT — Dataset size and VQC capacity discussion

The user asked whether the larger cohort requires a larger VQC. The frozen
pipeline still supplies 12 selected features per ECG to the same six-qubit,
two-block circuit, in batches of 32. Additional ECGs increase training work;
they do not increase the per-record encoding dimension. No ACS VQC has been
trained, so there is no measured evidence yet that this circuit lacks capacity.
The fit partition has 14,324 ECGs, including 917 OMI-positive records; validation
has 3,581 ECGs, including 229 positives. Whether fewer training patients suffice
requires a measured learning curve, not an assumption about the number of qubits.

The current extraction and frozen baseline settings remain unchanged. A separate,
prospectively documented follow-up could compare training-patient subset sizes
and then investigate capacity/optimization if indicated, preserving patient
separation and keeping the official test reserved. This discussion did not
implement or launch a new model experiment. Extracted features can be reused
for such follow-ups without repeating VMD/WST extraction.

Background: larger randomly initialized circuit families can suffer vanishing
gradients under the conditions studied by McClean et al.,
https://arxiv.org/abs/1803.11173. This is not a diagnosis of the present VQC.
Also corrected a stale PLAN.md introduction that still described the now-complete
folder rename as pending. No frozen numerical code, protocol or manifest changed.

## 2026-09-25 EDT — Full extraction stopped; saved checkpoints verified; CPU cap reproduced

The user's progress check found a numerical stop at 01:29:28 UTC on September
26 (21:29 EDT). The full extraction process exited 1 after record 01985/V5
exhausted the final 32,000-iteration limit, following capped attempts at 2,000,
4,000, 8,000 and 16,000. It is a fit-partition record. The other 191 leads in the
active batch had final uncapped attempts, but no final feature checkpoint was
written for that batch because VMD failed before descriptors/WST were yielded.
The exception and every attempt are preserved. No silent exclusion occurred.

All 1,795 saved checkpoints passed verification of identity, hashes, dimensions,
precision, finite values and attempt logs. They comprise 115 imported CPU and
1,680 GPU VMD / CPU WST records; 16,110 ECGs remain pending. The frozen manifests,
scientific settings and production code remain unchanged.

A separately declared diagnostic reran the unchanged CPU reference solver on
01985/V5 at the same final cap, initialization, tolerance and other settings.
It also returned 32,000 iterations with capped=true in 16.466 seconds, with finite
modes and centre frequencies. This reproduces the cap symptom on CPU; its cause
and response to higher caps remain to be investigated. The diagnostic protocol,
probe source snapshot, result arrays and summary are saved locally, with tracked
JSON evidence and a stop report. No production restart or classifier training
was performed. Current handoffs now state the stop; the earlier ETA is inactive.

See [stop report](reports/gpu_extraction_stop_2026-09-25.md),
[integrity check](reports/gpu_extraction_stop_2026-09-25.json) and
[CPU reference diagnostic](reports/vmd_cap_01985_cpu_diagnostic_2026-09-25.json).

## 2026-09-27 EDT — bounded retry run and fixed downstream comparison

The user authorized continuation of ACS work. Saved CPU/GPU higher-limit diagnostics
resolved 01985/V5 at iteration 67,827. Restored the exact temporary GPU packages;
the first sandbox download failed on network isolation, then the authorized install
succeeded. Revalidated 17 fit ECGs on GPU: all saved feature/input/convergence
comparisons and all Kymatio checks passed in 127.658 seconds. CUDA reports the same
hardware/runtime/driver fingerprint as the original run.

Prepared a separate 128,000-cap retry protocol/run, preserving old results. Reused
1,795 parent checkpoints and the diagnostic ECG, ran one new 16-record batch, then
verified all 1,812 checkpoints and byte-identical reused features. Full extraction
resumed in `results/omi_v1_gpu_retry128k/`; use its live status for progress.

Added ACS-local complete-bundle verification, fixed classifier fitting, exact VQC
epoch/optimizer/RNG recovery and patient-cluster statistical reporting. The 44 CPU
checks in the expanded suite passed (four device tests skipped); two additional
focused tests passed for classical artifact recovery and complete bundle integrity.
No real ACS model has yet been trained. A downstream process waits for successful
extraction, then runs the declared model comparison and reporting; it stops on
failure or interruption. Official test and attention training remain untouched.

Evidence and commands: [retry protocol](protocols/gpu_extraction_retry128k_v1.md),
[preparation report](reports/retry_preparation_2026-09-27.md), and JSON evidence.

## 2026-09-27 23:13 EDT — amended extraction hit another convergence cap

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


## 2026-09-28 EDT — convergence diagnosis and saved resume state

The user asked to diagnose 04124/V5 and clarify changed files. The frozen CPU
solver reproduces the production cap at 128,000 (finite outputs), then converges
at 128,234 with the unchanged 1e-7 tolerance. The frozen GPU solver also converges
at 128,234; maximum CPU/GPU mode difference is 5.7421e-13. The stopping statistic
is nonmonotonic and only 1.091e-7 at the previous cap. A separate 256,000-limit
protocol/run was prepared; all 3,700 imported parent checkpoints verified unchanged.
No new extraction started before sleep; 14,205 records remain pending.

Original production VMD/GPU/WST/loader/extractor hashes match the frozen manifest.
The only existing Python file edited during continuation is the retry runner's
manifest/device-validation gate. New files add diagnostics, the separate 256k
retry, bundling, and the already-planned downstream workflow. Full inventory:
[code-change report](reports/code_changes_2026-09-27.md). One unrelated off-by-one
in the unused scalar VMD reference counter is separately documented; no shared
numerical source was changed.

[Resume guide](reports/resume_acs_2026-09-28.md) records status, one-batch recovery,
full extraction, dependency restoration and downstream workflow commands.

## 2026-09-28 EDT — verified recovery and full 256k continuation

Restored the exact temporary GPU dependencies after the initial restart failed
with a missing CuPy distribution; the base environment was unchanged. Verified
all 3,700 saved parent checkpoints, then completed the controlled 16-record batch
at 2026-09-29T00:50:12 UTC. Full verification passed for all 3,716 records at
00:59:27 UTC, including unchanged imported feature bytes.

The user explicitly requested the full run. It resumed at 01:19:03 UTC with no
batch limit and reached 4,052/17,905 by the 01:25:38 UTC snapshot. Its first 21
batches took 13.57–29.29 seconds (median 19.07); the rate including session startup
was 2,881 ECGs/hour. This suggests 4.81 hours remaining for extraction at the
snapshot, subject to later retries. The earlier days estimate incorrectly used
the exceptional recovery batch containing 04124/V5 as a normal throughput sample.

The downstream workflow is waiting for successful extraction, then verifies and
bundles every feature checkpoint before the already authorized fixed model fits
and statistical report. No model had been fitted at the snapshot; official test
records remain reserved. No numerical code, settings, seeds or split changed.
A sandbox JSON report write failed with a read-only-filesystem error; the same
write succeeded outside the sandbox while extraction continued.

See the [launch report](reports/retry256k_launch_2026-09-28.md), its JSON evidence,
the updated handoffs and the [resume guide](reports/resume_acs_2026-09-28.md).

## 2026-09-29 00:41 EDT — paused cleanly for sleep

The user requested that work stop for the night and resume tomorrow. Verified
the extractor PID against its command before sending SIGTERM. It handled the
signal, finished/checkpointed the active batch, and exited with code 0 at
04:41:06 UTC with **12,916/17,905 records complete**, **4,989 pending**, no active
records, and `stop_signal: 15`. The full session saved 9,200 new ECGs after reusing
3,716. Its final measured rate was 2,661.68 ECGs/hour.

The downstream workflow detected `interrupted` and exited 1 at 04:41:19 UTC.
Its stopped/failed status and traceback record the designed response to this
user-requested pause. No new convergence failure occurred, no record was excluded,
and no model fitting or official-test evaluation took place. Saved protocols,
numerical code and feature artifacts remain in their existing run directories.

Updated both handoffs, the ACS README and the resume guide. Tomorrow, resume the
same extraction without a batch limit and restart the same downstream workflow
only after extraction reports `running`. Do not repeat completed recovery work.

Post-stop verification finished successfully at **04:57:05 UTC**: all **12,916
saved checkpoints passed**; 4,989 remain pending; reused parent feature files
are unchanged. The verifier exited 0. The report snapshot is
[pause_acs_2026-09-29.json](reports/pause_acs_2026-09-29.json). No extraction,
training or verification process from this session is left running.

## 2026-10-01 EDT — extraction and downstream workflow resumed

The user requested continuation. Restored the missing pinned GPU packages to
`/tmp/acs-gpu-deps-v1` while retaining the base environment. The full runner
validated its frozen context/pilot and all 12,916 saved checkpoints, then resumed
at 2026-10-02T01:35:15 UTC without a batch limit. The completed one-batch recovery
experiment was not repeated. Numerical code, settings, patient split and seeds
remain unchanged.

At 02:39:29 UTC, extraction was running with **15,780/17,905 ECGs saved**, including
2,864 new records this session. Its first 179 completed batches took a median
20.26 seconds (range 14.79–91.56 seconds); the session rate was 2,508 ECGs/hour.
The remaining 2,125 ECGs suggest about 51 minutes of extraction at that rate,
excluding verification, models and reporting and subject to later retries.

Restarted the existing downstream workflow under its saved manifest. It is
waiting for successful extraction before full verification/bundling and the
authorized fixed comparison. No ACS model had been fitted at this snapshot,
and official test records remain reserved. Updated the handoffs and resume guide.
See the [resume report](reports/acs_resume_2026-10-01.md) and its JSON snapshot.

## 2026-10-01 23:27 EDT — all ACS extraction checkpoints completed

The extractor reached **17,905/17,905 ECGs** at 2026-10-02T03:27:27 UTC and exited
0. The final session reused 12,916 records and saved the remaining 4,989; its
recorded elapsed time was 7,230.605 seconds and average rate 2,483.94 ECGs/hour.
No final active records remain. Preserve the completed extraction and its frozen
protocol; no recomputation is needed after a new chat or restart.

At 03:27:52 UTC the existing downstream workflow automatically started `bundle`.
Full archive verification/bundling was running at the snapshot; model preparation,
fixed classifier fits and reporting remain downstream work. No ACS model had yet
been fitted and official test data remain reserved. Updated handoffs and resume
guidance distinguish completed extraction from the ongoing workflow.
Evidence: [completion record](reports/extraction_complete_2026-10-01.json).

## 2026-10-01 23:43 EDT — full feature verification passed; model fitting active

The complete feature bundle passed verification at 2026-10-02T03:32:03 UTC:
17,905 ECGs, zero final capped leads, maximum final VMD iteration count 146,271.
VMD/WST shapes are (17905, 2688) and (17905, 2808); the bundle SHA-256 is
`cf7cd681f1d336ea791eddf124e55d2dae73460374fe4bb91053db0b05c41c8b`.
Verification/bundling took about 4 minutes 11 seconds. Model preparation completed
at 03:32:23 UTC using the fixed fit/validation partitions.

The workflow entered `train_models` automatically. At 03:43:58 UTC, both VMD
classical controls were complete and all three VMD VQC seeds had epoch 37/40
checkpoints. WST fits and the final patient-cluster report remained pending.
No numerical settings changed and official test data remain reserved. These are
progress observations, not a completed performance comparison.

## 2026-10-02 EDT — ACS model comparison and reporting completed and checked

All ten fits completed at 03:57:01 UTC and the workflow/report completed at
03:57:53 UTC (October 1, 23:57 EDT). The workflow exited 0. Six VQC fits completed
all 40 epochs using seeds 0/1/2 for each front end; all four classical controls
also completed. All 240 epoch records are saved and all trial warning lists are
empty. The final report used all 2,000 valid paired patient-bootstrap draws,
seed 20260925, on 3,581 validation ECGs from 3,391 patients.

Post-run verification checked the training manifest hash, all ten completion
markers, all saved model/prediction/epoch artifacts named by those markers, and
the report artifacts: **516 entries**, zero missing files or hash mismatches.
Copied the generated report, full analysis and all-seed metrics into
`reports/acs_omi_results_v1/`. The report and analysis retain exact source bytes
and hashes; the CSV only normalizes CRLF to LF and its parsed rows match exactly.
The artifact check records the export hashes, normalization, each fit's runtime
and completion marker.

Primary AP is 0.1442 for VMD + VQC and 0.1352 for WST + VQC (means over seeds).
Their paired difference is +0.0090, 95% interval [-0.0190, +0.0353]. Logistic
controls have AP 0.1540 / 0.1569; VQC-minus-logistic AP intervals include zero.
There is no established transform superiority or quantum advantage. KNN's high
accuracy accompanies very low positive-class sensitivity; the always-negative
baseline is already 93.61% accurate. Official test records remain reserved.

No further experiment was started. Updated handoffs and recovery guidance to
mark the entire fixed comparison complete. Read the
[results summary](reports/acs_omi_results_v1/SUMMARY.md) and linked full report
before choosing further work. Completed artifacts should be reused after restart.

## 2026-10-04 EDT — separate balanced OMI + Swin follow-up paused

At the user's explicit request, the new implementation lives in the repository
root at `correct balance training set OMI/`. It plans balanced fit-only positive
oversampling for VMD/WST classifiers and the existing LFCC temporal Swin model,
with the original validation patients and all initialization seeds retained.
Original numerical code, protocols and saved results were not edited.

New CPU and GPU synthetic checks and five existing ACS regression tests passed.
No real-data preparation, LFCC extraction or new fitting ran. The user said
“let's do it tomorrow”; all test processes have exited, and the work is paused.
Read the [new handoff](../../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
for the pending integration review and exact continuation state.

## 2026-10-04 17:08 EDT — separate balanced comparison resumed

The user resumed `correct balance training set OMI/`. New integration checks and
real-data preparation passed: all original fit ECGs retained, 13,407 training
examples per class, unchanged validation and byte-identical VMD/WST preprocessing.
Two fit-only LFCC engineering checks passed. At 21:07:51 UTC, the full workflow
started balanced VMD/WST fits alongside LFCC extraction, with GPU Swin and final
paired reporting following automatically. The new manifest is frozen; original
ACS numerical sources and saved results remain intact. See the separate handoff
for live status and avoid duplicate jobs. No new accuracy result was reported
at this launch snapshot.

## 2026-10-04 EDT — separate balanced OMI + Swin comparison completed

The root-folder balanced experiment finished all 14 fits and the paired report
at 21:57:18 UTC, exiting 0. It saved 240 VQC and 120 Swin epoch-history entries
and used all 2,000 patient-bootstrap draws. Post-run verification checked 768
artifact entries, 27 current source files plus snapshots, 529 protected original
artifacts and 4 report exports, with zero missing/changed files.

Balancing increased VQC sensitivity but did not establish better AP. Balanced
VMD/WST VQC accuracy is 66.56%/65.11%, sensitivity 63.32%/62.45%, AP 0.1454/0.1374.
LFCC + logistic has AP 0.2091 and outperformed Swin (AP 0.1680) in their paired
primary-metric comparison. Swin accuracy 91.48% accompanies sensitivity 17.03%.
The original saved ACS models and features are intact; official test patients
remain reserved. No additional tuning was started. Read the
[full summary](../../correct%20balance%20training%20set%20OMI/reports/v1/SUMMARY.md)
before interpreting accuracy or choosing another experiment.

## 2026-10-05 EDT — rotational morphology literature review; no experiment run

Researched the user's suggested ECG “eyeball” method and documented a proposed
EMD/Hilbert pilot and fixed classifier comparison. Read the
[research review](reports/eyeball_research_2026-10-05.md) and
[draft protocol](protocols/eyeball_omi_pilot_v1_draft.md). These are research
artifacts, not numerical results. Exact source/implementation details remain
pending; no pilot records, features, model scores or new initialization seeds
were executed. Existing VMD descriptors already include Hilbert-derived features,
so the proposed comparison explicitly investigates additional representation
and geometry choices.

Primary-source indexed text was accessible, but direct full-text/PDF/supplement
fetches failed and equation images were unreadable in extraction. A provider
page timed out. Those limitations are retained in the review; a complete author
implementation and preprocessing specification were not verified. Checked the
existing Python environment; an EMD distribution was absent and nothing was
installed. Original ACS, balanced OMI and ECGData Swin status files report
complete. No full artifact checksum audit was run for this documentation task.

The draft requires immutable run versions, fit-only numerical checks, patient
separation, classical controls, all VQC seeds, paired uncertainty, and an attempt
ledger including negative results and failures. Official test ECGs remain
reserved. Updated both handoffs; no completed protocol, source, data, feature
archive or saved model was edited.

Documentation verification initially detected a formula rendered as an unintended
Markdown link. Converted the formula to code notation before repeating the link
and whitespace checks; all eight local links and the whitespace checks then
passed. This was a documentation issue, not an experiment failure.

## 2026-10-05 EDT — EMD setup, synthetic verification, failed pilot and retry

The user authorized execution and package download. Installed EMD-signal 1.6.4
in an isolated results directory, retaining existing numerical dependencies.
The first sandbox request failed DNS; an escalation was interrupted before a
later authorized installation succeeded. Eight focused tests passed in 1.145s.
Preflight checked 1,542 existing artifact entries with zero mismatches.

The label-independent 32-fit-patient pilot stopped at 02:32:27 UTC on its first
ECG, 17086 / I: component three exhausted the 1,000 iteration bound. Rejected
the capped decomposition and retained all diagnostics. A separate bounded
diagnostic converged with 4,000 allowed iterations (actual sifts 60/506/1715/53),
with exact uninstrumented PyEMD parity and zero reconstruction error. A separate
16,000-bound pilot reuses the unchanged sample and stopping criteria. It was
running in session 1157 at the last snapshot; no model training had started.
Read [the execution report](reports/eyeball_execution_2026-10-05.md), current
status files and saved logs. These engineering outcomes are not accuracy results.

## 2026-10-08 EDT — rotational morphology resumed; bounded assessment launched

Restart audit verified 1,424 protected files, 14 Eyeball completion markers and
76 source entries, with zero mismatches. No download or reinstall was needed.
Recovered the previous session's final outcomes: the 16k pilot failed after nine
Lead-I completions; the default-64k diagnostic for 15689 / I converged at 46,081
sifts with exact package parity, while the alternative FIXE_H=5 / 1k diagnostic
failed. All earlier negative results remain saved.

Created a separate 64k protocol and assessment runner. Three new reuse/failure/
tamper checks passed in 0.747 seconds. The unchanged 32-patient pilot started at
2026-10-09T02:29:34Z, importing ten verified prior outputs and using four CPU
workers to collect all baseline and perturbation outcomes. Any baseline failure
prevents full extraction and fitting. Read the
[continuation record](reports/eyeball_continuation_2026-10-08.md) for the eventual
outcome and current saved status. This is an engineering assessment, not a new
OMI accuracy estimate; official test patients remain reserved.

## 2026-10-08 EDT — rotational assessment completed; negative engineering outcome

Assessment and report finished at 2026-10-09T02:46:22Z and 02:46:46Z, both
exiting 0. All 384 baseline outcomes were collected: 355 successes, 29 iteration
caps across 16/32 patients. Lead I passed in 32/32; 32 exact repeats matched.
Ten verified prior outputs were reused. All 224 perturbations returned valid
features, but cropping 0.1 seconds per end triggered descriptor warnings in
25/32 ECGs; both noise levels triggered warnings in 32/32. Amplitude and polarity
had zero threshold warnings after undoing their expected physical effects.

The full-run gate failed. No full extraction, classifier fit, new accuracy or
held-out evaluation was launched. The result applies to this adaptation and
iteration ceiling, not to rotational morphology in general. The sample and
every failure remain intact. Runtime, all seeds/settings, patient-bootstrap
draws, outcomes, comparisons and the gallery are saved in the
[complete report](reports/eyeball_assessment_2026-10-08/report.md); read the
[engineering review](reports/eyeball_continuation_2026-10-08.md) for interpretation.
The report verified 3,184 assessment artifact entries. Preservation verification
rechecked all 1,424 protected earlier files with zero mismatches. Both handoffs
now identify this run as complete, with further research requiring a separate
documented experiment rather than a resume.

## 2026-10-08 EDT — Lead-I-only feasibility study authorized and prepared

After clarification that all 32 Lead-I baselines passed, the user authorized a
larger fixed training-patient sample and conditional class-weighted logistic
comparison. The new protocol declares 2,048 unique original-fit patients,
1,536/512 internal fit/evaluation split, a 512-patient engineering stage,
prospective 99% coverage gates, explicit failure/fallback handling and 2,000
paired patient-bootstrap draws. Prior pilot patients are excluded prospectively.
No solver settings change and no original-validation or official-test ECG is used.
All three new checks passed; the joblib/NumPy deprecation warning is retained.
Read [the execution record](reports/eyeball_lead_i_execution_2026-10-08.md).
No real-data extraction or model fitting had started at this preparation snapshot.
The user requested incremental commits under their own configured Git identity;
the completed all-lead pilot and this preference were committed as `8b69d35`.
