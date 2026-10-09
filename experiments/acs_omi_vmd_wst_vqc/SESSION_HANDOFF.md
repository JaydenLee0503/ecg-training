# ACS session handoff — 2026-09-25

## Separate Lead-I follow-up authorized and prepared — 2026-10-08 EDT

The user authorized the larger Lead-I extraction check and conditional logistic
study. Read [the execution record](reports/eyeball_lead_i_execution_2026-10-08.md)
and `protocols/eyeball_lead_i_feasibility_v1.json`. Code:
`lead_i_study.py`, `scripts/eyeball_lead_i.py`; output:
`results/eyeball_lead_i_feasibility_v1/`. The first 512 new fit patients must have
at least 99% extraction coverage before extending to the fixed 2,048-patient
study. This new, prospective engineering threshold permits explicit failures;
it is not the prior all-lead zero-failure gate. Model-fit coverage is checked
again before evaluation extraction or fitting. Failure handling and all class
count requirements are fixed in the protocol. Official test and original
validation patients remain untouched.

Three focused checks passed (one retained joblib/NumPy deprecation warning;
exact model reload parity passed). No real-data work has started at this
preparation snapshot. Check status/completion markers before launch or resume;
do not edit frozen numerical code. User preference: commit verified increments
as work proceeds, under their configured identity without AI co-author trailers.
Prior completed pilot and preference checkpoint: `8b69d35`.

## Eyeball assessment complete; engineering gate failed — 2026-10-08 EDT

The 64k assessment completed at **2026-10-09T02:46:22Z** and its report at
**02:46:46Z**. Both processes exited 0; nothing remains running. It preserved
the same 32 fit patients and every failure. **355/384 baseline leads succeeded;
29 exhausted the limit, affecting 16 patients.** Lead I succeeded for all 32;
all 32 exact repeat checks matched. Ten earlier completed outputs were imported
with verified provenance. All 224 perturbations produced valid features.

Validity did not establish stability: removing 0.1 seconds at each end triggered
descriptor warnings in 25/32 ECGs, and noise at both levels triggered warnings
in 32/32. Amplitude/polarity changes stayed within warning thresholds after
undoing their expected effects. Read the
[full report](reports/eyeball_assessment_2026-10-08/report.md) and
[engineering review](reports/eyeball_continuation_2026-10-08.md), including the
descriptive patient-bootstrap uncertainty and fixed-sample gallery.

**Do not scale or fit the proposed classifiers from this run.** It failed the
declared baseline gate, and perturbation sensitivity warrants further work.
No new accuracy, model predictions or classifier seeds were produced. A revised
extractor or Lead-I-only study would need a separate protocol and output, not
an automatic retry. Official test patients remain reserved.

Saved assessment: `results/eyeball_omi_pilot_retry64k_v1/`; saved report:
`reports/eyeball_assessment_2026-10-08/`. Both have hashed `completed.json`
markers. Report verification checked 3,184 assessment artifact entries; a
separate preservation audit rechecked all 1,424 protected earlier files without
mismatches. Three new reuse/failure tests passed; eight prior numerical checks
remain unchanged. The isolated EMD installation is intact; do not reinstall or
repeat completed work. Earlier failed pilots/diagnostics remain saved. Their
old running snapshots and session IDs below are historical.

## Eyeball pilot running with bounded retry — 2026-10-05 EDT

Execution is authorized. EMD-signal 1.6.4 is installed only under
`results/eyeball_dependencies_v1/`. All eight focused tests passed; 1,542 old
artifact entries checked with zero mismatches (1,424 protected unique files).

The first 1,000-bound pilot failed on 17086 / P18802 / I, third component capped.
The preserved `results/eyeball_cap_17086_v1/` diagnostic resolved it after 1,715
sifts with unchanged stopping criteria and exact uninstrumented-package parity.
The separate `results/eyeball_omi_pilot_retry16k_v1/` pilot now uses a 16,000 bound
on the same 32 fit patients; at 02:36:21 UTC, nine Lead-I ECGs were complete.
The initial pilot, its source snapshots and every failure remain intact.

Read the [execution report](reports/eyeball_execution_2026-10-05.md) and
`results/eyeball_omi_pilot_retry16k_v1/status.json` for state before restarting.
Active session at launch: 1157; do not assume session IDs survive restart.
Log: `reports/eyeball_pilot_retry16k_2026-10-05.log`. Numerical source and
engineering manifests are frozen; do not edit them in place. Planned model
comparison follows engineering review; no new classifier or official-test
evaluation has started.

## Rotational morphology review and draft complete — 2026-10-05 EDT

The user proposed researching the ECG “eyeball” method for a possible next
experiment, with all outcomes retained. Saved the [research review](reports/eyeball_research_2026-10-05.md)
and [draft pilot/comparison protocol](protocols/eyeball_omi_pilot_v1_draft.md).
Only research/documentation finished: no extractor implementation, pilot,
synthetic checks, new predictions or fitting. The EMD package is not installed
in the checked existing environment; no environment changes were made.

The draft proposes fit-only numerical checks followed by a separate declared
classifier comparison. Source equations and exact processing choices still need
verification or explicit adaptation. Keep the existing 10-second OMI cohort,
patient assignments and official-test reservation. Existing completed results
remain the references; their completion statuses were read, not recomputed.
No new experiment is active. Record every later attempt, failure, exclusion,
seed, setting, runtime and prediction as specified in the draft.

## Separate balanced OMI + Swin comparison complete — 2026-10-04 EDT

All **14 new fits and the final report** completed at **21:57:18 UTC** and the
workflow exited 0. Nothing remains running or pending in that protocol. Read the
[separate results summary](../../correct%20balance%20training%20set%20OMI/reports/v1/SUMMARY.md)
and [handoff](../../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md).

Oversampling increased VQC sensitivity to 63.32%/62.45% (VMD/WST), while accuracy
fell to 66.56%/65.11%. AP 0.1454/0.1374 did not establish an improvement over the
original weighted VQC models. LFCC + logistic led the AP point estimates at
0.2091; LFCC + Swin had AP 0.1680, accuracy 91.48% and sensitivity 17.03%.
The paired AP comparison favored LFCC logistic over Swin. See the linked report
for all seeds, controls, intervals and limits.

Verification found zero mismatches across 768 new artifact entries, 27 current
source files and snapshots, 529 protected original artifacts and 4 report exports.
All 360 neural-model epoch-history entries are saved. Original ACS extraction,
protocols and saved results remain intact; official test patients remain reserved.
Do not restart completed experiments. Prior running/paused entries below are
historical. No follow-up tuning or final-test evaluation has been started.

## Separate balanced workflow live checkpoint — 2026-10-04 17:16 EDT

The full balanced experiment remains running. All LFCC features/normalization
are complete; Swin seed 0 finished 40 epochs and passed exact reload checks.
At 21:16:42 UTC seed 1 had 7/40 epochs, while all VMD VQC seeds had 14/40.
Three classical controls are complete. Remaining fits and the paired report
run automatically. Read the [active handoff](../../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
before starting any job. Original ACS code/results and reserved official test
patients remain intact; no new final performance result is reported yet.

## Separate balanced OMI + Swin full workflow started — 2026-10-04 17:08 EDT

The user resumed the new root-folder experiment. Its preparation retained all
original fit ECGs, balanced training to 13,407 examples per class and preserved
the original validation/preprocessors. CPU integration and real-data LFCC checks
passed. At 21:07:51 UTC the full workflow began balanced VMD/WST fitting alongside
LFCC extraction; Swin training and paired reporting follow automatically.
Read the [active handoff](../../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
for actual progress before launching anything. Original ACS numerical code,
protocols and completed results remain unchanged. The paused entries below are
historical; official test ECGs remain reserved.

## Separate balanced OMI + Swin follow-up paused — 2026-10-04 EDT

The user explicitly placed the new experiment in the repository-root folder
`correct balance training set OMI/`, with balanced training and a Swin comparison.
Implementation and synthetic tests are saved, but **no real-data preparation,
LFCC extraction or new model training has run**. The user paused work until
tomorrow, and all launched test commands have exited.

Read its [handoff](../../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
and [experiment log](../../correct%20balance%20training%20set%20OMI/EXPERIMENT_LOG.md)
before resuming. Original extraction/model code, protocols and completed results
were not edited. Do not restart the original completed ACS experiment.

## Entire development comparison complete — verified 2026-10-02 EDT

The workflow completed at **2026-10-02T03:57:53 UTC** and exited 0. All **17,905
ECGs**, **six VQC fits**, **four classical fits**, and the **final report** are
complete. All VQC fits saved 40 epochs with seeds 0/1/2 for each front end,
giving 240 epoch records. A post-run check verified **516 model/report artifact
entries**, with zero missing/changed files. No ACS job remains running.

Read the [results summary](reports/acs_omi_results_v1/SUMMARY.md),
[full statistical report](reports/acs_omi_results_v1/report.md),
[all-seed metrics](reports/acs_omi_results_v1/metrics.csv) and
[artifact check](reports/acs_omi_results_v1/artifact_check.json).
The copied report and analysis match the generated report hashes. The CSV export
only normalizes line endings; all rows were verified identical.

Primary average precision: VMD + VQC **0.1442**, WST + VQC **0.1352**, each
averaged across seeds. Paired difference **+0.0090**, 95% interval
**[-0.0190, +0.0353]**. Logistic controls have AP **0.1540 / 0.1569**. The primary
intervals do not establish transform superiority or a VQC advantage over these
classical controls. KNN's roughly 94% accuracy accompanies 1.31% / 2.62%
sensitivity; always-negative accuracy is already 93.61%. Official test records
remain reserved. No new tuning or final-test evaluation has been started.

Saved local runs: `results/omi_v1_gpu_retry256k/`,
`results/omi_models_v1_retry256k/`, `results/omi_development_workflow_v2/`.
Completion markers: extraction `extraction_status.json`, model `completed.json`
and `report_completed.json`, workflow `status.json`. All are complete.
Do not restart extraction or retrain completed fits after a new chat/reboot.
Older active-stage entries below are historical. Next: review these results
before declaring any new experiment.

## Full verification passed; model fits running — 2026-10-01 23:43 EDT

`results/omi_v1_gpu_retry256k/bundle.json` reports complete for all **17,905 ECGs**
at 2026-10-02T03:32:03 UTC, with zero final capped leads and maximum final VMD
iteration count 146,271. Feature dimensions are 17,905 × 2,688 for VMD and
17,905 × 2,808 for WST. Full verification/bundling took about 4 minutes 11 seconds.

Model preparation finished at 03:32:23 UTC, and the existing workflow is now in
`train_models`. At the 03:43:58 UTC snapshot, VMD logistic and weighted KNN fits
were complete and all three VMD VQC seeds had saved epoch 37/40. WST fits were
still pending. There are three worker processes; the declared six VQC fits and
four classical controls retain their fixed settings. Final reporting is pending;
no result comparison has yet been reviewed. Official test records remain reserved.

Check `results/omi_development_workflow_v2/status.json`, its `train_models.log`,
and each trial's status/epoch markers under `results/omi_models_v1_retry256k/`.
The extraction status's `models_trained: 0` field is its historical completion
snapshot and is not a live training counter. Do not launch duplicate model jobs.

## Extraction complete; verification/bundling running — 2026-10-01 23:27 EDT

`results/omi_v1_gpu_retry256k/extraction_status.json` is **complete** for all
**17,905/17,905 ECGs** at **2026-10-02T03:27:27 UTC**. The extractor exited 0.
The final session reused 12,916 saved records and added all 4,989 remaining ECGs;
its recorded elapsed time was 7,230.605 seconds. No active records remain.
Do not restart extraction or regenerate its saved features.

The existing downstream workflow automatically entered `bundle` at
**03:27:52 UTC**. It is performing full verification/bundling, then will prepare
the models, run the fixed comparison and report. No ACS model had been fitted
at this snapshot; full bundle verification was still pending. Official test data
remain reserved. The extraction completion marker does not describe later model
progress; use `results/omi_development_workflow_v2/status.json` and its stage logs.
Do not launch a second workflow while this one is running.

Evidence: [completion record](reports/extraction_complete_2026-10-01.json).
Earlier extraction progress snapshots and estimates below are historical.

## Extraction resumed — 2026-10-01 EDT

The user requested continuation. Restored the exact temporary packages to
`/tmp/acs-gpu-deps-v1`: CuPy 14.2.0, cuda-pathfinder 1.8.2, NumPy 2.5.2.
The base environment was unchanged. The full extractor validated its context,
pilot and all **12,916 saved checkpoints**, then resumed at
**2026-10-02T01:35:15 UTC** under the same 256k run/manifest. No numerical code
or settings changed; the completed one-batch recovery experiment was not repeated.

At **02:39:29 UTC**, extraction had reached **15,780/17,905 ECGs**, including
2,864 new records this session. Its average rate was 2,508 ECGs/hour, suggesting
about 51 minutes for the remaining 2,125 records at that rate. This excludes
later verification/model work and can change with convergence retries.

The existing downstream workflow has restarted and reports
`waiting_for_extraction`. It will verify/bundle the complete features before the
authorized fixed model comparison and reporting. No ACS model had been fitted
at this snapshot; official test records remain reserved. Do not start duplicate
extractors or workflows. Read the [resume report](reports/acs_resume_2026-10-01.md)
and its JSON snapshot; the [resume guide](reports/resume_acs_2026-09-28.md) contains
commands and the temporary dependency location. The prior pause sections below
describe the historical September 29 stop.

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
