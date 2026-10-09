# Eyeball / rotational morphology engineering assessment

**Assessment complete; baseline gate failed.**
355/384 baseline leads succeeded across the unchanged 32 fit patients. 29 failed; 16/32 patients had at least one failed lead.

Lead I succeeded for 32/32 patients. All 32 attempted repeat checks exactly matched saved features, modes, residual and trajectory. 10 baseline outputs were imported with verified provenance.

224/224 perturbations succeeded; 224 could be compared with successful baselines.

**Decision:** Do not scale or train: baseline engineering gate failed. No Eyeball classifier was fitted; there is no new accuracy estimate.

## Protocol and saved evidence

Independent 10-second, 500 Hz, physical-mV adaptation of the proposed EMD/Hilbert representation. Four IMFs, unchanged PyEMD 1.6.4 stopping criteria, 64,000 iteration ceiling, 0.25-second endpoint trim, signed frequency, and open-trajectory arc-length centroid. This is not a verified reproduction of the preprint. The cap was amended using fit-only convergence diagnostics, with prior failures retained.

The fixed sample contains 2 OMI-positive ECGs; labels did not select patients or numerical settings. No validation or official-test ECG was processed. Perturbation seeds are 20261005 through 20261036 in the saved cohort order. No model initialization seeds or predictions exist because fitting did not run.

The raw run saves every task outcome, input hash, convergence trace, runtime, source snapshot and attempt ledger. Completed outputs include four-component reconstruction checks; capped results are not accepted as features. Missing perturbation comparisons remain missing and are not silently counted as passes.

## Convergence

| Lead | Failed / 32 |
|---|---:|
| I | 0 |
| II | 1 |
| III | 1 |
| aVR | 0 |
| aVL | 0 |
| aVF | 1 |
| V1 | 4 |
| V2 | 5 |
| V3 | 8 |
| V4 | 5 |
| V5 | 2 |
| V6 | 2 |

Maximum converged baseline sifts: 61945; maximum reconstruction error: 0.0 mV.

Baseline lead failure fraction: 7.55%; descriptive 95% patient-bootstrap interval [3.65%, 12.51%] (2,000 draws, seed 20261008). This small, repeatedly examined engineering sample does not establish a population failure rate or clinical validity.

## Perturbations

| Perturbation | Succeeded / 32 | Comparable | Any descriptor warning | Centroid warning |
|---|---:|---:|---:|---:|
| amplitude_x2 | 32 | 32 | 32 | 30 |
| crop_100ms | 32 | 32 | 25 | 26 |
| first_half | 32 | 32 | 31 | 30 |
| last_half | 32 | 32 | 32 | 29 |
| noise_20dB | 32 | 32 | 32 | 30 |
| noise_30dB | 32 | 32 | 32 | 28 |
| polarity | 32 | 32 | 0 | 32 |

Warnings use the predeclared raw-feature thresholds: symmetric relative change >0.25 in any of the first ten descriptors, or centroid displacement / baseline envelope >0.1. Warnings are descriptive, never exclusion criteria. **Amplitude doubling is expected to change envelope and coordinate features; polarity inversion is expected to negate coordinates.** Their raw warnings alone do not establish instability. The JSON includes comparisons after undoing those expected transformations. Shortening, noise and endpoint cropping also change the input; these checks do not measure diagnostic accuracy.

## Runtime and limits

Assessment session: 978.3 seconds before gallery/final manifest writing. Summed baseline task durations: 3198.7 seconds. Four workers ran concurrently; imported timings were measured earlier and the diagnostic import timing covers decomposition only. These durations are not a controlled serial benchmark and should not be extrapolated as a full-dataset runtime.

The numerical and checkpoint tests do not prove the absence of bugs. The result applies to this declared adaptation and iteration bound; it does not reject rotational morphology generally, establish transform superiority, or establish quantum advantage. A changed stopping algorithm, preprocessing scheme or feature definition would require a separate documented experiment.

[All baseline outcomes](baseline_outcomes.csv) · [Full numerical analysis](analysis.json) · [Fixed-sample trajectory gallery](pilot_geometry.png) · [Continuation record](../eyeball_continuation_2026-10-08.md)
