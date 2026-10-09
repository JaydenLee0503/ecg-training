# Session handoff — 2026-09-25

## Lead-I feasibility study ended at its engineering gate — 2026-10-08 23:39 EDT

The frozen run completed at **2026-10-09T03:39:02Z** and exited normally. The
engineering check covered **500/512 Lead-I ECGs (97.66%)**, below the
prospective 99% gate. All 12 failures were EMD iteration-cap failures, all in class 0
(29/29 positives succeeded; too few to interpret). As declared, the study
stopped before any further extraction or fitting: **no classifier was trained
and no performance result exists.** Validation and test patients were untouched.
Completion marker verified by re-invocation without recomputation. Read the
[execution record outcome](../experiments/acs_omi_vmd_wst_vqc/reports/eyeball_lead_i_execution_2026-10-08.md) and `experiments/acs_omi_vmd_wst_vqc/reports/eyeball_lead_i_feasibility_v1/report.md`.
Nothing is running. A continuation needs a new protocol and output directory;
none has been proposed or approved. The running entry below is historical.

## Lead-I engineering check running — 2026-10-08 23:35 EDT

Codex launched the frozen Lead-I study at **2026-10-09T03:31:53Z**
(`python -B -m experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_lead_i`, log
`experiments/acs_omi_vmd_wst_vqc/reports/eyeball_lead_i_run_2026-10-08.log`).
Codex usage then ran out and Claude Code took over monitoring; no code, protocol
or setting was changed. At 03:35:16Z the engineering stage had **275/512
terminal extractions with 5 numerical failures**, the maximum the prospective
99% gate allows. One further failure fails the gate, which stops the study
after its failure report, before evaluation extraction or any fit. This is a
live snapshot, not a result: read `results/eyeball_lead_i_feasibility_v1/status.json`
and `attempts.jsonl` for the current state. Extractions are saved per record,
so an interrupted run resumes with the same command; do not launch a duplicate.

## Lead-I-only follow-up authorized and prepared — 2026-10-08 EDT

The user authorized a larger Lead-I check and conditional logistic comparison.
The separate [execution record](../experiments/acs_omi_vmd_wst_vqc/reports/eyeball_lead_i_execution_2026-10-08.md)
describes the fixed 2,048-patient fit-only cohort, 512-patient engineering check,
prospective 99% extraction-coverage gate, patient-separated internal evaluation,
explicit fallback for failed evaluation inputs, and all recording requirements.
Three focused checks passed. No new ECG has been processed at this preparation
snapshot; consult run status before launching. Earlier all-lead results remain
unchanged. The user also requested committing completed work as we go, using
their configured identity without AI co-author trailers; this is in AGENTS.md.
Completed all-lead pilot checkpoint: `8b69d35`.

## Rotational morphology pilot complete; training gate failed — 2026-10-08 EDT

The separate 64k assessment completed at **2026-10-09T02:46:22Z**; its report
completed at 02:46:46Z. Both processes exited 0. **355/384 baseline leads passed;
29 hit the iteration limit across 16/32 patients.** Lead I passed for all 32
patients and all 32 exact repeats matched. All 224 perturbations produced valid
features, but cropping 0.1 seconds per end caused descriptor warnings in 25/32
ECGs; both noise levels caused warnings in 32/32. Expected amplitude/polarity
effects were distinguished from instability.

Read the [full engineering report](../experiments/acs_omi_vmd_wst_vqc/reports/eyeball_assessment_2026-10-08/report.md)
and [continuation/review](../experiments/acs_omi_vmd_wst_vqc/reports/eyeball_continuation_2026-10-08.md).
All 3,184 assessment artifact entries verified; all 1,424 protected earlier files
remain unchanged. No classifier was trained and no new accuracy is available.
Do not resume or scale this completed pilot: a revised extraction or single-lead
study needs a new documented protocol. No job is running; official test patients
remain reserved. Older running-state entries below are historical.

## Rotational morphology pilot active — 2026-10-05 EDT

The user authorized execution and the isolated EMD package download. Eight
focused checks passed and 1,542 protected artifact entries verified unchanged.
The first pilot stopped on an EMD iteration cap; its failed run is preserved.
A fit-only diagnostic resolved that ECG without relaxing convergence criteria.
The same 32-patient pilot is running in a separate 16,000-bound directory.
Read the [execution record](../experiments/acs_omi_vmd_wst_vqc/reports/eyeball_execution_2026-10-05.md)
and [ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) before any
restart. No classifier has been trained for this representation yet. Earlier
research-only entries are historical.

## Rotational morphology researched; pilot draft only — 2026-10-05 EDT

The user asked to research the ECG “eyeball” approach as a possible next
experiment and retain every result. The [research review](../experiments/acs_omi_vmd_wst_vqc/reports/eyeball_research_2026-10-05.md)
and [draft protocol](../experiments/acs_omi_vmd_wst_vqc/protocols/eyeball_omi_pilot_v1_draft.md)
are saved in the ACS workspace. The recommendation is a fit-only numerical pilot,
then a fixed EMD/Hilbert feature comparison with classical controls and VQC.
Exact numerical settings/source fidelity remain unresolved; the draft is not
frozen. No new feature extraction, package installation, training or performance
result occurred. Existing status files for original ACS, balanced OMI and
ECGData Swin report complete; they were inspected without retraining. Read the
review for the acquisition/task mismatch, source-access limits and recording
requirements. No new experiment is running or waiting to resume.

## Balanced OMI + Swin complete and verified — 2026-10-04 EDT

The separate `correct balance training set OMI/` workflow completed all **14
fits and the paired report** at **21:57:18 UTC (5:57 PM Toronto)** and exited 0.
No experiment job remains running. Do not retrain completed fits after restart.
Read the [results summary](../correct%20balance%20training%20set%20OMI/reports/v1/SUMMARY.md)
and [handoff](../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md).

Balanced VMD/WST VQC AP is **0.1454/0.1374**, accuracy **66.56%/65.11%**, and
sensitivity **63.32%/62.45%**, averaging all seeds. Oversampling increased
sensitivity but did not establish an AP improvement over the original weighted
models. LFCC + Swin has **91.48% accuracy**, **17.03% sensitivity**, AP **0.1680**.
LFCC + logistic has the highest AP point estimate, **0.2091**, with **77.97%
accuracy** and **61.57% sensitivity**; its paired AP comparison favored the
logistic control over Swin. Always-negative accuracy is already **93.61%**.

All 240 VQC and 120 Swin epoch-history entries are saved. Post-run verification
checked 768 artifact entries, 27 current source files and snapshots, 529 protected
original artifacts and 4 report exports, with zero mismatches. Official test
patients remain reserved. These are exploratory internal-validation results with
pointwise patient-bootstrap intervals; no quantum advantage or external validity
is established. Prior live/paused entries below are historical. Further research
needs a separate protocol; no follow-up tuning has started.

## Balanced OMI + Swin live checkpoint — 2026-10-04 17:16 EDT

The separate full workflow is running. LFCC extraction and normalization are
complete for all 17,905 ECGs. Swin seed 0 finished 40 epochs and passed exact
saved-model reload checks; seed 1 reached 7/40. All three VMD VQC seeds reached
14/40. Three classical controls are complete; WST fits, the last Swin seed and
the final paired report remain in the automatic workflow. Original completed
results remain intact. Read the [active handoff](../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
for live logs/session details. Do not start a duplicate run or treat this dated
snapshot as the current counter. No new final accuracy result is available yet.

## Balanced OMI + Swin full workflow started — 2026-10-04 17:08 EDT

The user resumed the separate `correct balance training set OMI/` experiment.
Preparation passed: 13,407 training examples per class, all original fit ECGs
retained, unchanged validation and reused VMD/WST preprocessing. Integration
checks passed, and real LFCC checks passed on one fit ECG from each class.
The full workflow entered fitting/extraction at **21:07:51 UTC**; at 21:08:39 UTC,
LFCC extraction had reached 6,144/17,905 ECGs while balanced VMD/WST training ran.
Swin GPU training follows LFCC completion, then paired patient-bootstrap reporting.

Read the [active experiment handoff](../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
for live status/log locations, pinned manifest and continuation details. Do not
launch duplicate jobs or edit frozen code. Original completed ACS results remain
intact and official test patients remain reserved. The paused entries below are
historical; no new performance result is available at this launch snapshot.

## Balanced OMI + Swin follow-up paused — 2026-10-04 EDT

The user requested a separate experiment in `correct balance training set OMI/`,
including positive oversampling and an LFCC + temporal Swin comparison, then
said “let's do it tomorrow.” Code, protocol and synthetic verification are saved.
**No real-data preparation or training has started for this follow-up.** All test
processes have exited; do not launch work until the user resumes it.

Read the [new experiment handoff](../correct%20balance%20training%20set%20OMI/SESSION_HANDOFF.md)
for completed tests, pending integration checks, environment and restart steps.
Its location follows the user's explicit request. Original ACS numerical code,
protocols and completed results remain intact. The files are saved locally and
uncommitted; they have not been backed up remotely.

## ACS comparison complete and checked — 2026-10-02 EDT

The entire ACS workflow completed at **2026-10-02T03:57:53 UTC**: all 17,905
feature records, six VQC fits (seeds 0/1/2 for each front end), four classical
controls and the final patient-cluster report. All 240 VQC epoch records are
saved. A post-run check verified 516 model/report artifact entries with no
missing or changed files. No ACS job remains running; do not retrain completed
fits after a new chat or restart. Official test records remain reserved.

VMD/WST VQC average precision is **0.1442 / 0.1352**, averaged over all three
seeds. The paired difference is +0.0090, 95% interval [-0.0190, +0.0353].
Logistic controls have AP 0.1540 / 0.1569. This run does not establish transform
superiority or quantum advantage. Read the
[results summary](../experiments/acs_omi_vmd_wst_vqc/reports/acs_omi_results_v1/SUMMARY.md)
and its linked full report before interpreting results. Earlier progress entries
below are historical. Next work requires a new research decision, not a resume.

## ACS features verified; models training — 2026-10-01 23:43 EDT

All 17,905 ACS feature records passed full verification and bundling at
2026-10-02T03:32:03 UTC, with zero final capped leads. Model preparation finished
at 03:32:23 UTC and the existing workflow entered `train_models`. At the
03:43:58 UTC check, both VMD classical controls were complete and the three VMD
VQC seeds had saved epoch 37/40; the WST fits and final report were still pending.
Use `results/omi_development_workflow_v2/status.json` inside the ACS workspace
for current stage and per-trial markers for fitting progress. Do not start
duplicate jobs. Official test records remain reserved.

## ACS extraction complete — 2026-10-01 23:27 EDT

All **17,905/17,905 eligible development ECGs** finished extraction at
2026-10-02T03:27:27 UTC. The extractor exited 0; do not restart extraction.
The existing downstream workflow automatically entered `bundle` at 03:27:52 UTC
and is checking the complete feature archive before model preparation/training.
No ACS model had been trained at this snapshot. Official test records remain
reserved. See the [ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md)
and [completion record](../experiments/acs_omi_vmd_wst_vqc/reports/extraction_complete_2026-10-01.json).
Earlier extraction counts/ETAs below are historical; use the workflow status for
current progress and avoid duplicate workflow processes.

## ACS resumed — 2026-10-01 EDT

The user requested continuation of the paused ACS run. Restored the exact
temporary GPU packages (CuPy 14.2.0, cuda-pathfinder 1.8.2, NumPy 2.5.2) to
`/tmp/acs-gpu-deps-v1`. The full 256k extractor rechecked all 12,916 saved
checkpoints and resumed at 2026-10-02T01:35:15 UTC. At 02:39:29 UTC it had
reached **15,780/17,905**, with 2,125 pending. No numerical code or protocol changed.

The existing downstream workflow has restarted and is waiting for successful
extraction before verification, the fixed model comparison and reporting. No ACS
model had been trained at this snapshot. The rate of 2,508 ECGs/hour suggests
about 51 minutes of extraction remaining at the snapshot, excluding subsequent
model work. Check live status before starting another process.
See the [ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) and
[resume report](../experiments/acs_omi_vmd_wst_vqc/reports/acs_resume_2026-10-01.md).
Earlier paused-state entries below are historical.

## ECGData learning-curve protocol drafted — 2026-09-29 EDT

Draft protocol only: [attention method/ecgdata_learning_curve_protocol.md](../attention%20method/ecgdata_learning_curve_protocol.md).
Its runner is not implemented and nothing has been trained.

## ECGData LFCC + Swin completed — 2026-09-29 EDT

The user authorized training after checkpoint commit `eba3a78`. All **15 Swin
fits and five logistic controls** completed on the unchanged ECGData protocol,
with 600 saved epochs, seeds 0/1/2 and the existing five patient folds. Swin ran
on the RTX 5070; the training routine took **407.645 seconds**, excluding setup.
There are no failed real-data trials and no remaining attention fits to resume.

Three-seed mean LFCC + Swin window accuracy is **83.35%**, macro-F1 **0.8006**,
and patient-vote accuracy **94.17%**. The pooled LFCC logistic control has 76.48%
window accuracy and 0.7342 macro-F1. The paired patient-bootstrap interval for
the primary macro-F1 difference is [0.0224, 0.1143]. These are exploratory results
on a repeatedly studied 80-patient cohort with diagnosis/source confounding.
All Swin fits attained 100% training accuracy; held-out accuracy remains lower.
Do not interpret the result as external validation or isolate transform effects
from this comparison of complete pipelines.

All 21 CPU tests, GPU forward/gradient and synthetic checkpoint-recovery checks,
and saved-model reload checks passed. All 134 protected files remained unchanged;
all 930 older VQC improvement artifacts passed the preflight integrity check.
Read the [full results](../attention%20method/reports/ecgdata_lfcc_swin_v1/results.md)
and [attention handoff](../attention%20method/SESSION_HANDOFF.md) before reporting
or restarting. The original cache, scientific protocol and training implementation
were unchanged. CUDA dependencies are local and gitignored.

The older attention preparation-only entries below are historical. ACS remains
paused; this work did not resume its extraction or train ACS models.

## ACS paused for sleep — 2026-09-29 00:41 EDT

The user asked to wrap up and continue tomorrow. Extraction finished its active
batch and stopped cleanly at **12,916/17,905 ECGs (72.1%)**, with **4,989 pending**,
at 2026-09-29T04:41:06 UTC. The extractor exited successfully after SIGTERM.
The downstream workflow stopped at 04:41:19 UTC because extraction was interrupted;
its `failed` marker records this expected pause, not a new convergence failure.
No ACS model was trained. Both extraction and the downstream workflow have exited.
The final full integrity check passed at 04:57:05 UTC for all **12,916 saved
records**, with the 3,700 reused parent feature files unchanged.

Resume the same 256k run tomorrow, reusing the saved records; do not restart from
scratch or repeat the completed one-batch recovery experiment. Then restart the
same downstream workflow once extraction reports `running`. Read the updated
[ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) and
[resume guide](../experiments/acs_omi_vmd_wst_vqc/reports/resume_acs_2026-09-28.md).
The prior running-status snapshots below are historical.

## ACS full continuation — 2026-09-28 21:25 EDT

The user requested the full run. GPU extraction resumed from 3,716 verified ECGs
and reached **4,052/17,905** at 2026-09-29T01:25:38 UTC. Its early production rate
was about 2,881 ECGs/hour, suggesting 4.81 hours remaining for extraction at that
snapshot. The earlier estimate of days came from an exceptional recovery batch
and is superseded. Check the live status rather than treating this as a fixed ETA.

The downstream workflow is waiting for successful extraction, then will verify
and bundle the features, run the already authorized fixed model comparison and
write the report. No ACS model had been trained at this snapshot. See the
[launch report](../experiments/acs_omi_vmd_wst_vqc/reports/retry256k_launch_2026-09-28.md),
[ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) and
[resume guide](../experiments/acs_omi_vmd_wst_vqc/reports/resume_acs_2026-09-28.md).
Earlier sleep/stopped-state entries below are historical. Do not launch duplicate
extraction or workflow processes.

## ACS sleep/restart state — 2026-09-28 EDT

No ACS process is active. The new 256k run is prepared and verified with 3,700
reusable checkpoints and 14,205 pending. Exact restart commands are in the
[ACS resume guide](../experiments/acs_omi_vmd_wst_vqc/reports/resume_acs_2026-09-28.md).
04124/V5 converged at iteration 128,234 on both frozen CPU and GPU solvers; the
new run appends a 256,000 bound only. Original production code checksums pass.
See the [diagnosis](../experiments/acs_omi_vmd_wst_vqc/reports/vmd_04124_diagnostic_2026-09-27.md)
and [file-change inventory](../experiments/acs_omi_vmd_wst_vqc/reports/code_changes_2026-09-27.md).

## ACS stop — 2026-09-27 23:13 EDT

The amended ACS extraction stopped at **3,700/17,905 ECGs**: fit record 04124/V5
reached the 128,000-iteration cap. Its downstream workflow also stopped; no ACS
model training occurred. Preserve checkpoints and investigate convergence before
a further amendment. See the [ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md)
and [failure evidence](../experiments/acs_omi_vmd_wst_vqc/reports/retry_stop_2026-09-27.json).
Earlier running-status updates below are historical.

## ACS continuation — 2026-09-27 EDT

The user authorized ACS extraction and its planned fixed development comparison.
The separate `experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu_retry128k/` run
reuses 1,795 production checkpoints plus the higher-limit diagnostic ECG. The
diagnostic resolved 01985/V5 at iteration 67,827 without relaxing tolerance.
A fresh 17-record GPU check passed. One-batch recovery reached 1,812 verified ECGs.

Full extraction resumed; its status file is authoritative. A downstream workflow
waits for successful completion before bundling, all six VQC fits and four classical
fits, then patient-cluster reporting. State is in `results/omi_development_workflow_v1/`
inside the ACS workspace. No real ACS model had been fitted at launch.
Read the [updated ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md)
and [retry report](../experiments/acs_omi_vmd_wst_vqc/reports/retry_preparation_2026-09-27.md).
The earlier stopped-status paragraphs below are historical. No attention training
was started, and official ACS test data remain reserved.

## New attention method — implementation only

Latest: the user approved **ECGData preparation, without training**, while keeping
ACS intact. The separate [ECGData architecture record](ecgdata_attention_preparation.md)
documents exact baseline window/fold reuse, the completed CPU LFCC cache, compact
three-class Swin and prepared training/evaluation runner. Original ACS settings
and data remain unchanged. No GPU work or classifier fitting was started.

The consolidated [architecture record](attention_method_architecture.md) documents
both implemented components, exact dimensions/settings, data safeguards, completed
verification, environment setup and pending work. Added at the user's request;
this documentation update did not start extraction or training.

The user requested a separate `attention method` folder for cepstral features and
a Swin Transformer, explicitly **without training**. See its
[handoff](../attention%20method/SESSION_HANDOFF.md) and
[implementation report](../attention%20method/reports/implementation.md).
LFCC/MFCC extraction, a temporal Swin adaptation, read-only ACS integration and
fit-only normalization are implemented. All 11 CPU checks passed; synthetic
forward inference produced finite logits without changing weights. The existing
ACS patient split hash/counts were checked read-only. No real-data extraction,
training or performance evaluation has run for this new method. Original VMD/WST
code/results remain unchanged. Temporary CPU PyTorch lives in
`/tmp/ecg-attention-deps`; the original environment was not modified.

## Current state

The completed VMD/WST experiments are saved and intact. After the user reported an accidental PC shutdown, all **930 files** listed in `results/vqc_improvement/artifact_inventory.csv` were checked against their saved SHA-256 hashes and sizes: **zero missing or changed files**. This check was repeated while creating this handoff. It describes that point in time; recheck if later changes or another interruption make integrity uncertain.

The improvement experiment finished at **2026-09-22 04:19:52 UTC** (00:19:52 EDT). All **90 inner fits and 30 outer fits** completed, with **8,280 epoch records**, **360 inner checkpoints**, and no failed training trials. All 13 regression tests passed before launch. Post-run validation checked patient separation, saved models/checkpoints, selection scores, preprocessing, and prediction coverage.

There is no unfinished training job from that experiment to resume. Process/session IDs from earlier chats are not durable checkpoints.

The active ACS / OMI study has its own workspace at `experiments/acs_omi_vmd_wst_vqc/`.
Read its [session handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) and
[workspace guide](../experiments/acs_omi_vmd_wst_vqc/README.md) before ACS work. It contains
its own code, tests, data, results, protocols, reports, and experiment log.
The original ECGData experiments described below remain in their existing locations.

Latest ACS update: GPU extraction **stopped at 2026-09-26 01:29:28 UTC** with
1,795/17,905 ECGs saved. Record 01985/V5 exhausted the frozen 32,000-iteration
limit. All 1,795 checkpoints passed integrity checks; a separate CPU reference
attempt reproduced the capped result. No production settings changed and the
run has not resumed. Read the [stop report](../experiments/acs_omi_vmd_wst_vqc/reports/gpu_extraction_stop_2026-09-25.md)
before proceeding. Investigate convergence, then document any amended retry
policy in a separate run that reuses verified checkpoints. No ACS model training
has started; earlier running-status/ETA reports are historical.

## Read these reports first

1. [VQC improvement results](vqc_improvement_results.md): final results, all candidate choices, uncertainty, learning curves, saved artifacts, and reproduction commands.
2. [Frozen improvement protocol](vqc_improvement_protocol.md): candidate family, optimizer, nested validation, seeds, and selection rules.
3. [Corrected VMD/WST comparison](patient_vmd_wst_vqc.md): verified patient mapping and original matched VQC baseline.
4. [KNN comparison and training-fit diagnostic](vqc_vs_spar_knn.md): classical controls on the same inputs and limits of the published-paper comparison.
5. [WST numerical audit](scattering_audit_2026-09-21.md): reference checks, source matching, and earlier corrections.
6. [Experiment log](../EXPERIMENT_LOG.md): research chronology. Older entries include superseded results; respect their corrections.

## Results to carry forward

Window metrics below are mean scores across three initialization seeds for each VQC. KNN is deterministic. These are patient-held-out results on the existing ECGData task.

| Front end | Classifier | Window accuracy | Window macro-F1 | Patient-vote accuracy |
|---|---|---:|---:|---:|
| VMD | Original VQC | 67.98% | 0.6489 | 80.00% |
| WST | Original VQC | 64.79% | 0.6192 | 78.33% |
| VMD | Nested compact/reupload VQC | 67.94% | 0.6552 | 77.50% |
| WST | Nested compact/reupload VQC | 63.05% | 0.6011 | 75.83% |
| VMD | Weighted KNN, k=10, inverse distance | 74.38% | 0.7008 | 78.75% |
| WST | Weighted KNN, k=10, inverse distance | 70.86% | 0.6494 | 82.50% |

**The attempted VQC improvement did not establish a better model.** All paired patient-bootstrap intervals for changes from the original VQC include zero. Keep the originals as reference models and retain the unsuccessful experiments. Do not promote the best single seed as the result.

The KNN rows use exactly the same selected 12 angle-scaled features as the VQC. Other KNN variants, including inverse-square weighting and full feature sets, are preserved in the diagnostic report. KNN does not dominate every per-class or patient metric.

The paired intervals for VMD versus WST also include zero. Neither the original fixed-classifier comparison nor the new tuned-pipeline comparison establishes transform superiority. There is no demonstrated quantum advantage.

## Dataset and correctness findings

- ECGData contains 162 lead rows from **81 recordings and 80 patients**, not 162 independent patients. MIT-BIH records 201 and 202 belong to the same patient.
- The source mapping in `ecgvmd/ecgdata_subjects.csv` / `.json` was verified against source ECG excerpts; the loader checks waveform hashes.
- The old row-grouped evaluation leaked patients. Older README/QUANTUM_STAGE numbers are historical exploratory results, not the current reference.
- The corrected experiments use 1,620 windows, five patient folds, 224 VMD descriptors versus 882 standard log-WST features, and training-only mRMR-12 plus angle scaling.
- Standard WST coefficients matched Kymatio exactly on the current 1,620-window configuration. Other reference/edge checks had errors around floating-point precision. A minor time-bin metadata calculation was corrected; the current T=64 coefficients were unaffected. This is numerical evidence, not proof of absence of all bugs.
- VMD convergence retries left zero iteration-capped windows in the corrected feature archive.
- Source snippets were previously downloaded to verify patient identities. They were not a replacement training dataset. The improvement run reused the existing feature archive.
- Class and source-database identity remain confounded in the original ARR/CHF/NSR cohort. The cohort has informed repeated research decisions; it is not untouched external validation.

The paper discussed with the user is **3-D Attractor Reconstruction for Enhanced ECG Classification of Arrhythmia and Congestive Heart Failure**, DOI **10.1109/JSEN.2025.3572080**. Its available abstract reports 94% validation and 93.2% test accuracy using weighted KNN with attractor-derived features. Its full patient split and feature protocol were not verified. Do not claim a replication, directly equate its score with ours, or accuse its authors of leakage.

## What was tested in the improvement run

The original baseline used a 12-qubit, two-layer VQC, 40 epochs, batch size 32, learning rate 0.05, balanced loss, and seeds 0/1/2.

The follow-up encoded all 12 features on six qubits with RY/RZ angles. It tested two blocks with one encoding, two blocks with repeated encoding, and three blocks with repeated encoding. The readout had 24 quantum observables and a linear three-class head, without a raw-feature bypass. Training used Adam with cosine learning-rate decay from 0.02 to 0.002 over 80 epochs.

For each front end and outer fold, three inner patient folds selected architecture and epoch from 20/40/60/80 using pooled inner macro-F1. Selected settings were refitted with three seeds. Both front ends had the same search budget. Outer fits never used test labels for tuning.

Six selections chose one encoding; four chose repeated encoding. Five chose 20 epochs, three 40, one 60, and one 80. More layers or longer training did not consistently help.

## Repository cleanup — 2026-09-22

The root `scratch/` folder was archived to `legacy/scratch/`, preserving both files
byte for byte. The script is historical interactive work; its window cache remains
local and gitignored. Removed 33 rebuildable Python bytecode files and the stale
`ecgvmd_bundle.zip` (34 files, 621,947 bytes total). Rebuild the ZIP with
`make_colab_bundle.py` when needed.

Dataset files, features, saved results, source-verification excerpts, reference
material, and distinct notebook backups were retained. The active package/scripts
and frozen experiment inputs were unchanged. All 930 saved improvement artifacts
were checked after cleanup. Details and per-file hashes are in the
[cleanup record](repository_cleanup_2026-09-22.md) and
[cleanup manifest](repository_cleanup_2026-09-22.json).

## Files and environment

| Location | Purpose |
|---|---|
| `results/patient_vmd_wst_vqc/` | Corrected baseline features, 30 original VQC fits, predictions, and summaries |
| `results/patient_knn_diagnostic/` | Matched KNN predictions and original VQC training-fit diagnostics |
| `results/vqc_improvement/` | All 120 new fits, checkpoints, histories, predictions, environment/source snapshots, and checksums |
| `architects/vqc_improvement_tables/` | Version-controlled result tables, all epoch histories, split assignments, predictions, and artifact inventory |
| `architects/vqc_improvement_figures/` | Comparison and learning-curve PNG/SVG files |
| `ecgvmd/reupload.py` | New experimental compact/reupload classifier |
| `scripts/improve_vqc.py` | Resumable, content-hashed nested experiment runner |
| `scripts/report_vqc_improvement.py` | Saved-model validation, metrics, and paired patient-bootstrap reporting |
| `scripts/archive_vqc_improvement.py` | Histories, predictions, figures, and artifact inventory export |
| `tests/test_reupload_vqc.py` | Circuit, derivative, checkpoint, optimizer, and patient-split tests |

Use the existing interpreter if available:

```bash
/home/jaydenlee/venvs/test-ecg-training/bin/python --version
```

Training used `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and ten worker processes. Versions and code hashes are in `results/vqc_improvement/environment.json` and `manifest.json`.

Large results/features are gitignored. A fresh clone may lack them even when the reports are present. Report missing artifacts honestly; do not silently regenerate a different experiment. Completed fits resume only under the same content-hashed manifest. Use a new output directory for changed code or settings. Documentation/reporting scripts can write files and should not be confused with a read-only integrity check.

## Read-only restart check

From the repository root, this uses only the Python standard library. It verifies completion and every file in the saved artifact inventory without retraining or modifying results:

```bash
python3 -B - <<'PY'
from pathlib import Path
import csv
import hashlib
import json

run = Path("results/vqc_improvement")
status = json.loads((run / "completed.json").read_text())
assert status["status"] == "complete", status
with (run / "artifact_inventory.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

problems = []
for row in rows:
    path = run / row["path"]
    if not path.is_file():
        problems.append((str(path), "missing"))
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
        problems.append((str(path), "changed"))

print(status)
print({"files_checked": len(rows), "problems": problems})
raise SystemExit(bool(problems))
PY
```

If files differ, inspect why before overwriting anything: an intentional later edit and corruption are different explanations. A missing completion marker requires inspection of trial sidecars and logs; the existence of some checkpoints alone does not mean a run completed.

## Current ACS work

Continue in [experiments/acs](../experiments/acs_omi_vmd_wst_vqc/README.md). The
[ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) records completed patient
splits, the two-record feature check, and the completed 16-record parallel pilot.
Full eight-worker extraction started on 2026-09-25; check the ACS status files
before resuming. The pilot suggests about three days for feature extraction
alone; ACS VQC training time remains unmeasured and no ACS model is fitted.
No completed experiment needs retraining because its files were reorganized.
The relocation preserved saved ACS manifests and all result bytes; the new runner
verifies the recorded import/path migration before resuming the original run.

## Earlier PTB-XL proposal — superseded by ACS preparation

The recommendation was **PTB-XL v1.0.3**, with 21,799 ten-second ECGs from 18,869 patients, using its official patient-separated folds: 1–8 training, 9 validation, 10 final test.

A proposed first task is **atrial fibrillation (AFIB) versus sinus rhythm (SR)**, initially lead II at 100 Hz. Label exclusions, handling of conflicting rhythm statements, and the exact protocol still need to be established. Sinus rhythm is a rhythm label, not a guarantee of overall cardiac health. This task changes the original ARR/CHF/NSR problem; scores would not be directly comparable. Retraining the method on it would test a new task, not externally validate the old three-class fitted model.

Both VMD and WST should receive the same records, leads, durations, patient splits, selection/scaling rules, and declared tuning budget. Update sampling-rate and duration assumptions explicitly; do not apply the old 128 Hz configuration or ECGData-specific source map blindly. Include matched classical controls and multiple VQC initialization seeds. Use training/validation data to establish the protocol and keep the test set untouched until final evaluation.

Chapman–Shaoxing was suggested as an alternative rhythm dataset; the original release covers 10,646 patients. If retaining CHF as a target is essential, reconsider dataset suitability rather than inventing an equivalent CHF label.

Sources checked on 2026-09-22:

- [PTB-XL v1.0.3 and recommended splits](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL label definitions](https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv)
- [Original Chapman–Shaoxing paper](https://www.nature.com/articles/s41597-020-0386-x)

These were earlier options. Continue from the verified ACS preparation described
above and the user's current request, without automatically starting training.
