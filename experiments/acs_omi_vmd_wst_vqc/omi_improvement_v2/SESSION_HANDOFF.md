# OMI improvement v2 handoff — 23 October 2026

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
