# OMI threshold comparison log

## 2026-10-31 EDT / 2026-11-01 UTC — completed saved-score analysis

At the user's request, compared the fixed 0.5 cutoff and previously frozen 80%,
90%, and 95% calibration targets for the completed undersampled and matched
oversampled LFCC logistic and Swin models. No classifier or threshold was fitted
and no validation-selected cutoff was used. All 8 per-seed models and 16 grouped
operating points were reconstructed; 2,000 paired patient-cluster bootstrap
resamples use seed 20261101. Neural metrics average all seeds 0/1/2. The official
test waveforms were not processed. [REPORT.md](REPORT.md), [analysis.json](analysis.json)
and `results/bootstrap.npz` retain outcomes and uncertainty. Independent audit
reconstructed all 16 grouped operating points and seven metrics each from the
parent analysis and reproduced bootstrap quantiles from all 2,000 retained draws;
see [audit.json](audit.json). The single repeatedly examined internal validation
cohort and fixed trained models limit generalization and uncertainty scope.

The command completed successfully; elapsed execution time was not separately
captured. The run and audit logs are retained (`run.log`, `audit.json`). The
executed OMI notebook displays this comparison and can explicitly initiate a
separate LFCC/Swin plus logistic retraining workflow; default notebook execution
verifies/reuses completed models and does not train or decode test waveforms.

Generated the supplemental `REPORT_INTERVALS.md` from the audited analysis JSON
so all 16 operating points show pointwise 95% intervals for sensitivity,
specificity and precision. No predictions, thresholds, models or draws changed.
