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
