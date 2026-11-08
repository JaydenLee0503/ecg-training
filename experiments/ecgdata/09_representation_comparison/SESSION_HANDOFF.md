# ECG-09 session handoff — 2026-11-01 UTC

The user authorized the existing-cohort 32-log-band versus 16-LFCC Swin
comparison, matched logistic control and raw-waveform CNN control, with a full
logbook. No external dataset was selected. Protocol is frozen in [PROTOCOL.md](PROTOCOL.md).
Original ECG-07 data and models are read-only sources; the new study has separate
`features/`, `results/`, logs and eventual `reports/`.

At the launch snapshot 2026-11-01T03:46Z, host GPU training began with 35 planned
new fits and seeds 0/1/2. At 04:02Z, filterbank logistic and all three Swin fits
for fold 0 had completion markers; raw-CNN fitting followed. These are historical
snapshots, not current progress or final outcomes. `results/status.json` updates
only after each fit; the currently active epoch is in its `checkpoint.json`.
Read `training.log`, status, per-fit markers and process liveness on the host.
The launch PID was 90076 and the separate monitor's status is in
`monitor_status.json`; PIDs may be stale or reused after restart.

The separate `monitor.py` waits for the training writer to exit, checks the run
and report, and runs `audit.py` once. It never starts training; its output and
failures are in `monitor_status.json`, `monitor.log`, and eventual `audit.log`.
Do not infer audit success from its launch. It has passed temporary-fixture tests,
but the full saved-result audit cannot run until the experiment completes.

Before any restart, inspect host process liveness, lock, status and current
checkpoint together. Never start a second writer while one is active. If fitting
has been interrupted and the writer has exited, the identical train command in
[README.md](README.md) verifies manifests, reuses completed fits and resumes from
the current saved epoch. Do not change `study.py`, `PROTOCOL.md` or pinned source
files: a source/settings change requires a new run folder. A corrupt or incomplete
checkpoint must be investigated rather than silently overwritten. If the run
completed but the report did not, use the separate `report` action; if reporting
finished, run the audit if no monitor is active and no successful audit exists.

`EXPERIMENT_LOG.md`, `preparation.log`, `training.log`, `tests/` and versioned
`provenance/` record the work and source/input fingerprints. Feature and model
artifacts and full predictions are local and gitignored; versioned metadata is
not a backup. After the final audit, append actual results, all failures and
remaining caveats to the study log and both relevant handoffs, and commit the
completed report/analysis with the user's configured identity. Do not cite
incomplete or unaudited outcomes. Bootstrap intervals are patient-cluster and
conditional on saved predictions; the repeatedly used 80-patient cohort has
diagnosis/source confounding and is not external validation.

## Resumed 2026-11-07 EDT

The old Nov-1 status was stale after restart. Verified 23 completed fits (117
listed artifacts), 14 frozen sources, all three feature-cache files and the
interrupted `band_swin_seed1_fold3` epoch-6 checkpoint with zero mismatches. The
existing interpreter and RTX 5070/CUDA fingerprint matched the saved manifest.
The unchanged host GPU runner was resumed at approximately 2026-11-08T00:04Z,
reusing all completed fits. Launch writer PID 430581; do not trust PIDs across
restarts. A one-time monitor was also restarted. Read live `results/status.json`,
`training_resume_2026-11-07.log`, the active checkpoint and
`monitor_status.json` before any action. No final report/audit existed at launch;
the resume is not permission to run a second writer. Full records and the
initial inspection-helper key error are in EXPERIMENT_LOG.md.

After the full audit passes, run `refresh_notebook.py` using the existing Python
environment on the host (Jupyter local kernel sockets are sandbox-restricted).
It refuses unaudited results, performs no training, checks all cell outputs and
preserves concurrent notebook edits. Its two safety tests passed; the real refresh
was not run at the resume snapshot. Commit the refreshed notebook and completed
reports after the remaining saved-artifact checks.

Post-restart GPU driver/software inventory: `restart_environment_2026-11-07.json`.
Current NVIDIA driver 610.88; the original ECG-09 launch manifest did not pin
its driver version, so equality across the restart cannot be established. Other
recorded numerical-library versions and GPU model match the original records.
