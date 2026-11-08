# OMI improvement v2 experiment log

## 2026-11-07 — completed, audited and independently checked

The resumed workflow completed at 2026-11-08T00:53:34Z, taking 2,546.596 seconds
this session, with 52 saved fit records and 1,350 neural epoch records. The
predeclared Swin AP difference was +0.0066 [-0.0115, +0.0243]; improvement was
not established. Raw CNN AP 0.2803 versus tuned logistic 0.2022 is an exploratory
secondary result, with paired AP difference +0.0781 [0.0294, 0.1288]. At the 90%
calibration target raw CNN sensitivity was 90.10% and specificity 50.47%.

The completion audit verified 1,274 output entries and 1,732 protected files.
Independent sklearn recomputation matched AP/AUC for all 14 model/seed columns,
reconstructed all 42 threshold choices and checked patient separation. All 2,000
paired patient-bootstrap draws were valid. Intervals are pointwise and condition
on fitted models and frozen selections; validation was previously examined and
official test waveforms remain reserved. See reports/SUMMARY.md and
reports/independent_check.json for the completed evidence.

## 2026-11-07 — resume after interruption

On return, the status still said running but the old host PID was absent and
the writer lock was available. Source manifest a9f655bad8be1f95e1c9aa1d77f7acc73100122afe25896c03e43eb200eaf2ec
verified all 39 pinned source files. The saved log recorded all inner candidates
and frozen selections, plus completion of the first final Swin fit. The host GPU
had no active compute processes. Sandbox GPU inspection failed; approved host
inspection succeeded. The same authorized workflow resumed at 00:12:13 UTC on
November 8, PID 439380, appending to workflow.log. It verifies completion markers
before reuse. The previous interruption cause was not established.

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

## 2026-10-23 21:32 EDT — full workflow launched

Preparation completed at 2026-10-24T01:18:36Z, 141.244 seconds including imports,
verifying 1,732 protected files. All 17,905 development ECGs were retained with
zero patient overlap and zero added exclusions. Manifest SHA-256 is
a9f655bad8be1f95e1c9aa1d77f7acc73100122afe25896c03e43eb200eaf2ec.

Committed verified implementation/protocol as b018f78 using the configured user
identity. The first Git staging attempt was blocked by the read-only sandbox
index; the approved host commit succeeded. The full GPU workflow started at
2026-10-24T01:32:20Z, PID 102537 (launch execution session 24142). Its output is
retained in workflow.log. Existing preparation was verified and reused. No new
validation result exists at this launch entry; check actual completion markers.
