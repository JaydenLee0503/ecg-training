# Rotational morphology continuation — 2026-10-08 EDT

The user requested continuation. The existing environment and isolated
EMD-signal 1.6.4 installation are intact; no download or installation was needed.
The [restart audit](eyeball_restart_2026-10-08.json) checked all 1,424 protected
files, 14 saved Eyeball completion markers and 76 source entries with zero
mismatches. No completed classifier was retrained.

## Recovered outcomes from the previous session

The 16,000-bound pilot **failed**, rather than remaining active as its earlier
handoff snapshot said. Nine Lead-I ECGs completed before record 15689 / patient
P16909 / Lead I reached the limit on component three. Sift counts were
392 / 742 / 15,999 (capped) / 1,830. All outputs and failure records remain saved.

The completed stopping diagnostic resolved that same fit ECG with the original
stopping criteria and a 64,000 bound: **392 / 742 / 46,081 / 1,288** sifts.
Decomposition took 26.7454 seconds. A separate uninstrumented PyEMD call returned
exactly equal IMFs and residual; reconstruction error was zero. The separately
declared FIXE_H=5 / 1,000-bound alternative failed on components three and four
(385 / 939 / 999 capped / 999 capped), taking 1.8394 seconds. That alternative
was not adopted. Its negative result remains preserved alongside the successful
default-criterion diagnostic in `results/eyeball_stopping_diagnostic_v1/`.

## Declared assessment

The new [64k specification](../protocols/eyeball_omi_engineering_retry64k_v1.json)
keeps the same 32 fit patients, signals, convergence criteria and descriptors.
Only the ceiling changes scientifically. The runner now collects all independent
numerical outcomes, instead of stopping at the first failed lead, using four CPU
workers. It attempts all 384 baseline leads and 224 Lead-I perturbations. Exact
repeats apply to successful Lead-I baselines; a failed baseline remains visible
and cannot be replaced by another patient. Perturbation comparisons require
both outputs to succeed, with missing comparisons explicitly recorded.

Ten verified prior baseline outputs are eligible for import: nine uncapped
16k outputs and the default-64k diagnostic. Reuse checks input identity,
environment, numerical settings, reconstruction, analytic descriptors and
source/artifact hashes. Imported timing values retain their original scope;
the diagnostic timing covers decomposition only. Full extraction and fitting
require every baseline to pass and a documented perturbation review. No further
ceiling increase is scheduled for this assessment.

The separate output directory is `results/eyeball_omi_pilot_retry64k_v1/`.
Each terminal success or numerical failure is recorded and reused after restart;
changed inputs or artifacts are rejected. Assessment completion is distinct from
passing the engineering gate. No classifier accuracy is available from this work.

## Execution status

All three new checkpoint/reuse tests passed in 0.747 seconds; the
[test record](eyeball_assessment_tests_2026-10-08.log) is saved. The eight earlier
numerical checks remain unchanged and their saved results were verified.

The assessment started at 2026-10-09T02:29:34Z (October 8 EDT) and imported all
ten eligible completed outputs. The [execution log](eyeball_pilot_retry64k_2026-10-08.log)
and saved status track current progress. Final outcomes will be appended here
after completion. Official test patients remain reserved.

## Completed outcome and engineering review

The assessment completed at **2026-10-09T02:46:22Z**, followed by its verified
report at **02:46:46Z** (October 8 EDT). Both processes exited 0; no Eyeball
process remains running. Read the [full report](eyeball_assessment_2026-10-08/report.md)
and [trajectory gallery](eyeball_assessment_2026-10-08/pilot_geometry.png).
Assessment completion does **not** mean the engineering gate passed.

- **355/384 baseline leads succeeded; 29 reached the 64k cap**, affecting 16/32
  patients. There were no other numerical failure types. The descriptive
  patient-bootstrap interval for the lead failure fraction is 3.65%–12.51%
  (2,000 saved draws, seed 20261008); this is a small fit-only engineering sample.
- **Lead I succeeded for all 32 patients**, and all 32 repeated extractions
  exactly matched features, modes, residual and trajectory. Ten earlier outputs
  were imported with verified provenance. Maximum converged sifts: 61,945;
  maximum accepted reconstruction error: zero mV.
- **All 224 perturbations produced valid features**, but validity did not imply
  stability. Removing 0.1 seconds from each end produced at least one descriptor
  warning in **25/32** ECGs and a centroid warning in **26/32**. The first and last
  five-second halves produced descriptor warnings in 31/32 and 32/32; Gaussian
  noise at both 30 dB and 20 dB produced warnings in **32/32** each. Warnings use
  the predeclared thresholds, not disease labels or model scores.
- After undoing the expected amplitude scaling or coordinate sign change,
  amplitude doubling and polarity inversion produced **zero threshold warnings**.
  Amplitude-adjusted first-ten-feature differences were exactly zero; polarity's
  maximum symmetric relative difference was approximately 6.8e-6. Their raw
  coordinate/envelope warnings must not be interpreted as unexpected instability.

**Decision:** this declared all-lead adaptation does not pass the full-run gate.
Lead I is numerically feasible on this sample, but endpoint/noise sensitivity
also needs investigation. Do not launch the proposed full extraction or any of
the 16 classifier fits from this run. A revised extraction or Lead-I-only study
would be a separate documented experiment, not a resume of this completed pilot.
No Eyeball accuracy, AP, model prediction or model initialization seed was
produced. Existing classification results remain the references. This outcome
does not establish that rotational morphology in general is ineffective.

The assessment measured 978.3 seconds before gallery/final manifest writing.
Four workers ran concurrently; imported timing scopes differ, so this is not a
serial benchmark or a reliable full-cohort runtime forecast. The sample contains
two OMI-positive ECGs, chosen without outcome use. No patient was removed or
replaced; no validation or official-test ECG was processed.

The report verified **3,184 assessment artifact entries**, saved a SHA-256/byte
inventory, all baseline outcomes, raw perturbation comparisons, uncertainty
draws and the full 32-patient gallery. The separate
[preservation check](eyeball_preservation_2026-10-08.json) rechecked all **1,424
protected earlier files**, with zero mismatches. The three new tests and eight
previous numerical tests are engineering evidence, not proof of bug-free code.
