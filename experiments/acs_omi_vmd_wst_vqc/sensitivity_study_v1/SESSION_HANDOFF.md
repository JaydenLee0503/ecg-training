# Sensitivity study handoff — 2026-10-15

## Sensitivity study complete; independently verified — 2026-10-23 EDT

All 14 fits and both sensitivity reports completed at **2026-10-16T03:59:19Z**
(October 15, 11:59 PM Toronto), taking 24m31s excluding preparation. No fit is
pending. On October 23, independent verification reproduced all 42 calibrated
operating points and all 360 epochs; restart checks verified 622 new artifact
entries, 1,088 protected source files, 27 snapshots and 12 exports with zero
mismatches. No source drift, retraining or official-test processing occurred.

Read [the completed summary](reports/SUMMARY.md). At the 95% calibration target,
VMD logistic achieved **93.89% sensitivity / 35.50% specificity**; VMD/WST VQC
achieved **92.14% / 91.85% sensitivity** and **30.81% / 23.78% specificity**, means
over seeds 0/1/2. LFCC logistic achieved only **82.10% sensitivity** at its 90%
calibration target (55.55% specificity), demonstrating that targets do not
guarantee validation performance. Higher sensitivity came with many false
positives. KNN labeled every validation ECG positive. All warnings, negative
results and intervals are retained; no clinical or quantum advantage is claimed.

Earlier running/preparation entries are historical. Preserve completed outputs;
do not restart the workflow. Large checkpoints/predictions remain gitignored.

## Workflow launched; saved-score analysis complete — 23:35 EDT

Implementation checkpoint: `5b74eb0`. Preparation completed at
2026-10-16T02:50:05Z in 113.679 seconds: 11,458 subfit ECGs / 10,859 patients
(733 positives), 2,866 calibration ECGs / 2,717 patients (184 positives), and
unchanged 3,581 validation ECGs / 3,391 patients (229 positives). Zero patient
overlap, zero added exclusions, no warnings. Balanced subfit has 10,725 examples
per class. All 1,088 protected earlier artifacts passed verification.

After GPU execution approval, `run all` started at 2026-10-16T03:34:48Z.
Execution session at launch: 13855; workflow PID 70431. These IDs are historical
after restart; inspect actual status and markers. Logs: `workflow.log` and
`results/logs/`. All six VQC seeds and Swin seed 0 started; classical controls
complete as workers become available. Do not launch a duplicate workflow.

Saved-score analysis finished all 2,000 bootstrap draws. Read
[its report](reports/saved_scores/report.md). At its exploratory 90% target,
balanced LFCC logistic achieved 90.39% sensitivity, 35.68% specificity and 8.76%
precision (207 TP, 22 FN, 2,156 FP). These are same-validation-label curve points,
not independently validated thresholds. New calibration-based results are pending.
The earlier preparation paragraphs below describe the prelaunch snapshot.

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
