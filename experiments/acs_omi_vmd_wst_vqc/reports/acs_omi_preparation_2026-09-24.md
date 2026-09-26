# ACS OMI preparation and engineering check — 2026-09-24

Paths and commands updated for the ACS workspace reorganization; scientific
results and machine-readable historical records are unchanged. See
[the workspace guide](../README.md).

## Completed work

The user selected **OMI detection** for the first ACS experiment. Declared
[protocol v1](../protocols/acs_omi_protocol_v1.md), saved a patient-separated fit/validation
split, and connected the ACS loader to both feature front ends. Added shared
training-only mRMR/scaling and unfitted VQC/classical-model interfaces. All code
is separate from the completed ECGData experiments.

Implemented commands in [`experiments/acs/scripts/experiment.py`](../scripts/experiment.py)
for `prepare`, `smoke`, `extract`, and `bundle`. Their shared input, split,
transform, preprocessing, and classifier code is in
[`experiments/acs/pipeline.py`](../pipeline.py).
Only preparation and the two-record engineering check have run. **Full feature
extraction and ACS classifier fitting have not started.** No official test
waveform was read by this experiment and no prediction was submitted externally.

## Frozen cohort and split

| Partition | ECGs | Patients | OMI positive ECGs | OMI negative ECGs |
|---|---:|---:|---:|---:|
| Fit | 14,324 | 13,576 | 917 | 13,407 |
| Internal validation | 3,581 | 3,391 | 229 | 3,352 |
| Total eligible development | 17,905 | 16,967 | 1,146 | 16,759 |

All 17,905 eligible official-training ECGs appear exactly once. There is **zero
patient overlap** across fit, validation, and the reserved official test set.
The split uses seed 20260924 and the declared five-fold grouped stratifier, with
fold 0 as validation and the other folds combined for fitting. It is one held-out
validation split, not a completed five-fold evaluation. The 55 source training
exclusions remain recorded in `source_records.csv`.

There are **114 patients with different OMI labels across their retained ECGs**.
The record labels are preserved; no majority-vote patient label is invented.
All repeated ECGs from a patient remain in the same partition.

The official 1,995 test ECGs, including the 2 quality exclusions, remain reserved
with labels absent. An always-negative prediction would score **93.605% accuracy**
on this internal validation split while detecting zero OMI cases. This is a
deterministic label-count baseline, not a trained model result, and explains why
average precision is primary in the new protocol.

## Actual transform checks

The two fit records were selected reproducibly by hashing protocol ID and record
ID, without using labels. Both happen to be OMI-negative. This is an engineering
check with no estimate of class-specific performance.

| Record | Patient | VMD seconds | WST seconds | Kymatio seconds | Maximum WST coefficient difference |
|---|---|---:|---:|---:|---:|
| 11678 | P17577 | 36.611 | 0.096 | 0.125 | 0.0 |
| 17045 | P04328 | 16.662 | 0.081 | 0.081 | 0.0 |

Both arms received the exact same full 10-second, 12-lead mV samples at 500 Hz.
Each record produced **2,688 VMD descriptors** and **2,808 pooled WST features**,
with finite values and lead-prefixed column identities. At these ACS settings,
WST has 234 paths per lead; its path metadata and unpooled coefficients matched
Kymatio. Features concatenate leads rather than increasing the sample count.

VMD required retries for 17 of the 24 record/lead combinations. There were
20 capped lead-attempts across the retry stages; every lead ultimately converged.
The largest observed count was **4,988 iterations**, under an 8,000-iteration retry
limit. The saved summary's `retried_leads` field counts capped attempts (20),
not unique lead instances (17); per-record attempt logs give the full distinction.
No ECG was removed because of transform convergence.

The check completed at **2026-09-24 23:35:11 EDT**
(`2026-09-25T03:35:11.719180+00:00`) in **60.106 seconds**, including source
verification, reference checks, and artifact writes. Summed transform times were
53.273 seconds for VMD and 0.177 seconds for WST. This is a two-ECG timing check
with one process and one BLAS/OpenMP thread, not a representative cohort benchmark.
Naively extending the mean VMD time to 17,905 ECGs gives about 132.5 hours
(5.5 days) serially; signal variability, retries, hardware, and parallelism make
that a rough planning estimate. Full extraction was not launched in this setup
task. A broader timing sample or a separately validated parallel runner would
help plan that computation without changing the scientific settings.

## Validation and failures

All **21 tests passed** in 1.936 seconds: 11 loader tests plus 10 new experiment
tests. They cover deterministic patient grouping, exclusion of official test
inputs, label-independent engineering selection, fitting selection/scaling only
on fit patients, class/feature budgets, the binary VQC interface, convergence
failure handling, identical physical lead inputs, Kymatio agreement, feature
checkpoint identity/checksums, and restart without waveform rereading.
The binary VQC test trains one synthetic 16-row, one-epoch fixture; it is not
an ACS fit or a predictive result.

Independent real-data checks verified full cohort coverage, all frozen
manifest/code/environment/split/ledger hashes, patient separation, hidden test
labels, and the 114 mixed-label patients. Repeating the real `smoke` command
reported **“Already complete: 2 verified records”** and reused the completed
feature checkpoints without rerunning transforms.

The first `prepare` command encountered the workspace's read-only sandbox mount
when creating its output directory. It stopped before saving a prepared run and
was rerun with approved escalation. Saving the engineering artifacts used the
same required file-write escalation. No data replacement or download occurred.
The initial test invocation printed a Matplotlib cache-directory warning; the
final test invocation set `MPLCONFIGDIR=/tmp/acs-matplotlib` and passed without
that warning. No scientific computation failed in the completed engineering
check; expected capped attempts are retained as convergence diagnostics.

## Artifacts and continuation

`experiments/acs/results/omi_v1/` contains:

- `protocol.json`, `manifest.json`, `manifest.sha256`: exact settings, code/source
  identities, environment, and creation time.
- `splits.csv`: all 17,905 record/patient/label/partition assignments and waveform
  hashes; `source_records.csv`: the prior all-record eligibility ledger.
- `preparation.json` and `prepare_status.json`: completed preparation counts.
- `smoke/status.json`: completed engineering check summary; `smoke/*.npz`:
  both feature vectors and names; `smoke/*.json`: per-record provenance,
  checksums and timing; `smoke/*_attempts.jsonl`: every VMD lead attempt.

The [version-controlled summary](acs_omi_preparation_2026-09-24.json) preserves
the complete manifest, split counts, engineering summaries, per-record results,
and verification outcomes. Large data/results remain gitignored.

Protocol digest:
`37ebedf1fa4774f3a7efd2a832fdba6a5a79c7ef8d95bd518b796173822b5eaa`.
Split CSV SHA-256:
`d036f1eff7c350638e1e7e738d1ded6d017b4c7b62380a8debf89e89a3c5faba`.
Manifest SHA-256:
`ac3886997718235601183e537d7478b04dbdcd1970c9b0441b44777a28a51e16`.

The protocol contains reproduction/resume commands. Use `extract` only for a
deliberate full run, then `bundle` to create aligned matrices; the engineering
subset cannot stand in for the full archive. Afterwards the remaining work is
the model-fit/report runner: fit each arm's shared 12-feature preprocessing on
fit patients, run all three VQC seeds and declared controls, save predictions
and histories, and calculate the specified patient-cluster uncertainty. Those
results do not exist yet. The current evidence verifies setup and sampled
numerics, not bug-free software, clinical validity, or quantum advantage.
