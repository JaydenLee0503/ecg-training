# OMI improvement v2 handoff — 23 October 2026

## Completed and independently checked — 2026-11-07 19:53 EST

The resumed workflow completed at 2026-11-08T00:53:34Z. Do not retrain: the
completion marker, [summary](reports/SUMMARY.md), and all 52 saved fit records
are present. The primary selected-Swin minus fixed-Swin AP difference is +0.0066
with a paired patient-bootstrap 95% interval [-0.0115, +0.0243], so improvement
is not established. Raw CNN AP was 0.2803 versus tuned logistic 0.2022; their
pointwise paired AP difference was +0.0781 [0.0294, 0.1288]. At the 90%
calibration target, raw CNN achieved 90.10% sensitivity and 50.47% specificity.
This repeatedly examined validation cohort is not external or clinical evidence.

The completion audit verified 1,274 output entries and 1,732 protected files.
An independent sklearn check reproduced AP/AUC for all 14 model/seed columns,
all 42 calibration thresholds, and patient separation; see
[independent check](reports/independent_check.json). All seeds 0/1/2 and the
2,000 patient-bootstrap draws are retained. Large artifacts remain gitignored.
Official test waveforms remain reserved. Earlier running entries are historical.

## Resumed — 2026-11-07 19:12 EST

The earlier process was no longer running and the writer lock was free. All 27
neural inner fits and model selections had finished; the tuned logistic, fixed
controls and Swin0 seed 0 final fit had completion markers. The code manifest
matched all 39 pinned source files. The host GPU was idle. The authorized `all`
workflow resumed at 2026-11-08T00:12:13Z (PID 439380 at launch); it verifies
and reuses completed work. Check status and the writer lock before any restart.
No new final comparison is available at this snapshot.

## Full workflow launched — 21:32 EDT

The authorized all command started at **2026-10-24T01:32:20Z**, workflow PID
102537, execution session 24142 at launch. These IDs are not durable after a
restart. Read results/status.json, workflow.log and per-fit status/checkpoints.
Do not launch a duplicate writer. The workflow automatically advances through
raw caching, 27 neural and 15 logistic inner fits, final refits, calibration,
reporting and integrity checks. No new validation result exists at this entry.

Implementation checkpoint: **b018f78**. Preparation completed in 141.244 seconds
including imports, verified **1,732 protected files**, and retained all 17,905
development ECGs with zero new exclusions and zero patient overlap. Manifest:
`a9f655bad8be1f95e1c9aa1d77f7acc73100122afe25896c03e43eb200eaf2ec`.
All 9 CPU/GPU preflight checks passed. Earlier preparation entries below are
historical; the status files are authoritative.

The user authorized the three-part OMI improvement comparison. Source and
prospective protocol are implemented in this separate directory. All 9 CPU/GPU
checks passed (24.950 seconds on the host), including numerical agreement and
exact dropout recovery. Preparation is running; no new real-data fit has started
at this entry. Read preparation.log and results/status.json for actual status.

Read README.md and PROTOCOL.md. Check results/status.json and completed.json
before any launch. The all command automatically advances through preparation,
raw cache, inner fitting, selection, final fitting/calibration and reporting.
Do not duplicate a running writer or restart a completed fit.

Existing interpreter: /home/jaydenlee/venvs/test-ecg-training/bin/python. Existing
PyTorch 2.7.1+cu128 is reused from attention method/.venv/torch-cu128. Sandbox
probe reports CUDA unavailable; approved host GPU preflight passed. The first
preparation failed before creating outputs due to the Projects/projects sandbox
path-case mismatch; approved host preparation is underway. No dependencies were
installed.

Original sensitivity, balanced OMI and VMD/WST results remain read-only sources.
Official test waveforms remain reserved. Report missing artifacts or failed
checks explicitly; do not silently regenerate an earlier experiment.
