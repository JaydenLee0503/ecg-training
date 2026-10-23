# ACS/OMI sensitivity study v1

Separate study authorized on 2026-10-15. Existing experiments and frozen results
are read-only sources. All code, tests, protocol, reports, logs and generated
artifacts for this study live in this folder. Large `results/` artifacts are
gitignored and are not included in a fresh clone.

Two distinct analyses are required:

1. Describe saved original and balanced model scores at 80%, 90%, and 95%
   sensitivity, with paired patient-bootstrap intervals. Cutoffs chosen on these
   same validation labels are exploratory curve summaries, not validated cutoffs.
2. Train all 14 balanced pipelines on a new subfit partition and select each
   cutoff on a separate calibration partition of the original fit patients.
   Evaluate the frozen cutoffs on the unchanged validation cohort. Validation
   sensitivity may miss its calibration target. No official test data are used.

The [protocol](protocol.json) fixes one inner patient-separated calibration fold
(five-fold splitter, fold 0, seed 20261015). This is one held-out calibration
split, not five-fold training or cross-validation. Four folds fit the models;
one chooses the cutoffs. This avoids transferring thresholds to a refitted model
whose score distribution could change. All learned preprocessing uses unique
subfit ECGs, followed by shared seeded oversampling within subfit only. Neural
seeds 0/1/2 and the original 40-epoch training budgets are retained. No epoch,
architecture or class-weight search is performed in this threshold experiment.

Models: VMD/WST logistic, distance-weighted KNN and three VQC seeds each; LFCC
logistic and three Swin seeds. Constant-score controls are also reported. No new
waveform extraction or exclusions. All splits, scores, checkpoints, histories,
warnings, failures, cutoffs, confusion counts and durations are saved.

Run from repository root using the existing interpreter:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omi-sensitivity-matplotlib /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/acs_omi_vmd_wst_vqc/sensitivity_study_v1/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omi-sensitivity-matplotlib /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.run prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/omi-sensitivity-matplotlib /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.run all
```

`all` verifies and reuses completed stages. GPU access may require execution
outside the sandbox. No dependencies are installed by this runner. The source
snapshot and manifest freeze numerical code/settings before preparation. Do not
edit frozen code or launch duplicate workflows. Epoch checkpoints permit recovery
of unfinished VQC/Swin fits. A single Swin process runs alongside up to six CPU
fits. Logs are in `results/logs/`; stage state is `results/status.json`.

Reports are exported to `reports/saved_scores/` and `reports/calibrated/`.
`results/manifest.json` pins sources; `reports/audit.json` records completion and
checksums. The final calibration comparison conditions on fitted models/cutoffs;
bootstrap uncertainty excludes retraining, split and calibration uncertainty.
The validation cohort has been examined previously, so it is internal evidence.
No clinical or quantum advantage is claimed.

## Completed results

All 14 fits and both analyses are complete. Read the [results summary](reports/SUMMARY.md) and [latest handoff](SESSION_HANDOFF.md). Completion and artifacts were independently verified on 2026-10-23. Do not retrain completed fits.
