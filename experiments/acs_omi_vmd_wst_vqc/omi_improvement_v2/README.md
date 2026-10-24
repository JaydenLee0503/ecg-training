# OMI improvement v2

Authorized on 23 October 2026: compare LFCC Swin dropout/epoch selection,
tuned LFCC logistic regression, and a small raw 12-lead CNN. Everything new
belongs in this folder; previous experiments are read-only sources.

Read [the protocol](PROTOCOL.md), [machine-readable settings](protocol.json),
[current handoff](SESSION_HANDOFF.md), and [experiment log](EXPERIMENT_LOG.md).
Completion markers and hashes, rather than this dated prose, determine whether
work needs resuming. Completed fits are verified and reused.

Commands from the repository root with the existing interpreter:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/acs_omi_vmd_wst_vqc/omi_improvement_v2/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMI_IMPROVEMENT_CUDA=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/acs_omi_vmd_wst_vqc/omi_improvement_v2/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.omi_improvement_v2.run all --device cuda
```

The all command prepares and verifies inputs, caches raw development waveforms,
trains all inner candidates, freezes selection, refits selected candidates,
selects calibration thresholds, evaluates validation, reports uncertainty and
audits outputs. One writer lock prevents duplicate workflows. Epoch checkpoints
retain model, optimizer, scheduler and CPU/CUDA RNG state. The same command
resumes partial work; source/context drift is rejected. GPU access may require
execution outside the workspace sandbox on this WSL host.

Large features, raw caches, weights, predictions and bootstrap draws live under
`results/` and are gitignored. Committed documentation does not back them up.

When complete: [summary](reports/SUMMARY.md), [all metrics](reports/analysis.json),
[every seed](reports/metrics.csv), [operating points](reports/operating_points.csv),
[fit records](reports/fits.json), [epoch histories](reports/histories.json),
[verification](reports/audit.json). Until their completion marker exists, these
links are planned outputs, not performance claims.
