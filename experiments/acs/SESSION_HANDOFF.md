# ACS session handoff — 2026-09-25

## Scope and location

The active task is ACS **OMI versus non-OMI** classification. The user selected
OMI and requested that this study have its own folder. All ACS-specific work now
lives under `experiments/acs/`; the root `ecgvmd/` package supplies shared
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

## Next work

Full ACS feature extraction and classifier training **have not started**. There
is no long-running job to resume. Completed engineering artifacts are in
`results/omi_v1/smoke/`; `results/omi_v1/` holds the protocol, manifest, splits and
source ledger. Loader audits are under `results/loader/`, including the retained
failed first loader run and its successful `_v2` replacement.

Use the guide's `smoke` command for checkpoint verification without recomputing
completed transforms. Do not rerun `prepare` on the existing directory. New
code/settings require a new output directory; the import/path migration does
not grant a general exemption from that rule.

Plan/execute full feature extraction, possibly with a separately validated
parallel runner: observed VMD time was 16.662–36.611 seconds per ECG, giving a
rough serial extrapolation of 5.5 days. The current CLI provides
prepare/smoke/extract/bundle only. Model fitting and statistical reporting still
need a runner after full features are available.

The protocol specifies fit-only mRMR-12 and angle scaling, a six-qubit/two-block
VQC with encoding once, 40 epochs and seeds 0/1/2, and matched weighted-KNN and
logistic controls. Average precision is primary; save full metrics, predictions,
histories and paired patient-cluster uncertainty. Always-negative internal
validation accuracy is 93.605%, so accuracy alone is misleading. Keep the official
test patients reserved, and declare test coverage/submission handling separately.

No current result establishes model quality, clinical validity, research novelty,
transform superiority or quantum advantage. Do not compare ACS scores directly
with the original ARR/CHF/NSR task or an unverified paper's validation accuracy.
