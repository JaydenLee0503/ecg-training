The paper's 94% result is not a matched benchmark for the present VQC experiment.
The current VQC also has a measurable window-accuracy deficit against a fixed
weighted KNN control on identical inputs and patient folds.

**Paper evidence.** Gupta et al., DOI
[10.1109/JSEN.2025.3572080](https://doi.org/10.1109/JSEN.2025.3572080), describe SPAR,
radial/angular density features combined across sliding windows, and weighted KNN.
The [available abstract](https://www.researchgate.net/publication/392176131_3-D_Attractor_Reconstruction_for_Enhanced_ECG_Classification_of_Arrhythmia_and_Congestive_Heart_Failure)
reports 94% validation and 93.2% test accuracy on 162 ECG signals. The full methods
remain unavailable here: patient separation, window lengths, feature dimension,
fusion details, and KNN settings are unverified. No leakage finding is attributed
to these authors.

For background, [the authors' earlier SPAR method paper](https://www.cinc.org/archives/2019/pdf/CinC2019-073.pdf)
describes placing delayed copies of an ECG on separate coordinate axes to form a
trajectory, projecting it to a plane, and measuring the density and shape of that
projection. This explains the underlying representation; it does not supply the
missing settings of the 2025 classifier experiment.

**Matched diagnostic, run 2026-09-21.** Reused the completed comparison's 1,620
windows, 80 verified patients, five outer folds, and the exact saved in-fold feature
selections/scalers. Every seed's selections and scalers were checked to be identical.
KNN uses k=10 and Euclidean distance, with two declared weighting rules: 1/d and
1/d². No tuning or selection of a winning configuration was performed. The latter
rule gives only exact matches nonzero weight when any distance is zero.

The angle-input condition supplies exactly the same 12 numbers to KNN and VQC.
VQC numbers are the previously completed three-seed means; KNN is deterministic.
These are diagnostic point estimates, not a proof of population-level superiority.

| Pipeline | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 |
|---|---:|---:|---:|---:|
| VMD / VQC | 67.98% | 0.6489 | 80.00% | 0.7791 |
| VMD / 10-NN, 1/d, same 12 angle inputs | 74.38% | 0.7008 | 78.75% | 0.7415 |
| VMD / 10-NN, 1/d², same 12 angle inputs | 74.32% | 0.6987 | 78.75% | 0.7461 |
| WST / VQC | 64.79% | 0.6192 | 78.33% | 0.7627 |
| WST / 10-NN, 1/d, same 12 angle inputs | 70.86% | 0.6494 | 82.50% | 0.7888 |
| WST / 10-NN, 1/d², same 12 angle inputs | 70.00% | 0.6395 | 82.50% | 0.7846 |

The VMD patient-vote result does not improve with this KNN substitution, despite
higher window accuracy. Report the prediction unit rather than treating the metrics
as interchangeable.

With all candidate features and training-fitted standard scaling, inverse-distance
KNN gives 74.75% for VMD and 69.07% for WST. With 12 selected, standard-scaled features
it gives 73.21% and 70.37%. Thus the current diagnostic does not support a blanket
claim that reducing to 12 features caused the deficit. All 12 fixed configurations
are saved in `results/patient_knn_diagnostic/metrics.csv`.

**Training fit.** Inference with all 30 saved VQCs on their own training folds gives
mean training accuracy 77.72% for VMD and 74.22% for WST (mean over folds and seeds),
with training macro-F1 0.7586 and 0.7264. Mean first/final class-weighted training
loss changes from 0.9576 to 0.5585 for VMD and from 0.9338 to 0.5790 for WST.
Training occurred and the models learned, but their current fit and held-out
generalization are both limited. These checks do not distinguish circuit
expressivity, optimization, objective weighting, and remaining feature limitations.
They do not diagnose a barren plateau or demonstrate that more epochs will help.

**Interpretation.** The observed KNN window-accuracy gain is 6.40 percentage points
on VMD inputs and 6.07 on WST inputs. The classifier/training choice therefore
contributes to the observed deficit under our fixed protocol. The rest of the gap
to 94% cannot be assigned to a cause without matching the SPAR representation,
temporal aggregation, evaluation unit, and patient split. The earlier random-window
audit demonstrated that splitting choices alone can produce scores near 94%; it
does not establish which choices this paper made.

Retain patient separation and tune any next VQC within inner patient folds. Useful
next comparisons are alternate feature reductions, encoding/readout choices, and
optimization schedules with a declared search budget shared across front ends.
Compare those against classical controls on identical inputs; do not optimize
against the outer-fold results or aim to reproduce an unmatched published number.

To reproduce the diagnostic without retraining the VQC:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ecg-mpl-cache python scripts/vqc_knn_diagnostic.py --train-scores --n-jobs 5
```

Outputs are `results/patient_knn_diagnostic/{metrics.csv,predictions.npz,protocol.json,vqc_training_fit.csv}`.
