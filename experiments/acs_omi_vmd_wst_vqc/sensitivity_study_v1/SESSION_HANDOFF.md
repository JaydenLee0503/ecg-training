# Sensitivity study handoff — 2026-10-15

The user authorized both the saved-score comparison and new patient-separated
threshold-selection fits, then explicitly requested a separate folder and full
recording. This folder contains the entire new study. Earlier source runs remain
read-only. See [README.md](README.md) and [protocol.json](protocol.json).

Preparation/verification in progress. No real fit has started at this entry.
Check `results/status.json`, completion markers and logs before launching.
The proposed run has 14 new fits and targets 80%, 90%, and 95% sensitivity.
One held-out patient calibration fold selects thresholds before validation
inference; no model refit follows calibration. All learning is confined to the
other four inner folds. Official test patients remain reserved.

All eight focused tests passed in 63.357 seconds, including a two-epoch synthetic
Swin adapter fit with separate calibration/validation IDs and completed-fit reuse.
The full test output is retained in `reports_tests.log`. Source integrity and
subfit-only preprocessing preparation are running; consult `preparation.log`.
The machine has 16 logical CPUs and 23 GiB RAM; use six single-thread CPU workers
alongside one GPU worker. This changes concurrency only, not model settings.

The first seven-test run had six passes and one fixture error: the synthetic
protocol omitted the VQC seed list required by the shared classifier factory.
The fixture was corrected before any real preparation. This was not a real-data
training failure. The sandbox GPU probe was denied; an approved external probe
confirmed the RTX 5070. Existing PyTorch is reused; no installation is planned.

An unrelated pre-existing modification to
`attention method/reports/swin_research_audit_2026-10-12.json` must be preserved
and excluded from study commits.
