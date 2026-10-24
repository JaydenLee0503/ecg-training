# OMI improvement v2 experiment log

## 2026-10-23 — authorization and implementation

The user approved OMI Swin dropout/epoch selection, logistic tuning and a raw
12-lead CNN control, retaining the separate-folder/full-recording requirement.
Read project/ACS handoffs and completed reports. Existing fits are complete and
will be reused where settings match. The prospective protocol isolates subfit
inner model selection, calibration thresholds and validation evaluation.

Implemented 3-fold inner selection, seeds 0/1/2, candidate retention, resumable
fits, native raw-waveform extraction, independent threshold reconstruction and
paired patient bootstrap. CPU tests are running. The sandbox GPU probe reports
CUDA unavailable; real GPU execution remains pending the environment check.
No real-data training has started at this entry.

## 2026-10-23 — preflight passed; preparation started

CPU verification: 8 passed, GPU check skipped, 23.318 seconds. Approved host
verification: all 9 passed, 24.950 seconds, including GPU forward/gradient
agreement and exact interrupted-training recovery. Full output is in
preflight.log. Two NumPy 2.5/joblib array-shape deprecation warnings occurred
during the synthetic logistic reload; they are retained, not suppressed.

The first preparation invocation failed before acquiring a writer lock because
the sandbox permits lowercase /mnt/d/projects while Path.resolve returns
/mnt/d/Projects. No study data or fit was created by that failed invocation.
An approved host rerun is preparing the study; preparation.log retains output.
