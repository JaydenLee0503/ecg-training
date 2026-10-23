# ECGData LFCC–Swin regularization v2 — completed

Completed 2026-10-16T04:32:07.126574+00:00. All 90 inner fits and 30 outer trials completed; 0 outer trials reused saved v1 fits. Seeds: 0, 1, 2.

Primary selected-procedure minus v1 window macro-F1: **+0.0093**, paired 95% interval **[-0.0023, +0.0212]**.

This experiment does not establish an improvement over v1.

| Pipeline | Window accuracy | Window macro-F1 | Patient-vote accuracy |
|---|---:|---:|---:|
| Inner-selected procedure | 84.12% | 0.8099 | 94.58% |
| R0: epoch selection only | 83.46% | 0.8012 | 93.75% |
| R1: dropout 0.1 + epoch selection | 84.49% | 0.8144 | 95.83% |
| Saved v1: fixed epoch 40 | 83.35% | 0.8006 | 94.17% |
| Saved pooled LFCC logistic | 76.48% | 0.7342 | 87.50% |

Stochastic rows average all three seeds; they are not seed ensembles.

## Selected settings

| Outer fold | R0 epoch | R1 epoch | Joint selection |
|---|---:|---:|---|
| 0 | 20 | 40 | R1, epoch 40 |
| 1 | 10 | 10 | R0, epoch 10 |
| 2 | 20 | 40 | R1, epoch 40 |
| 3 | 20 | 40 | R1, epoch 40 |
| 4 | 20 | 40 | R0, epoch 20 |

## Every seed

| Pipeline | Seed | Window accuracy | Macro-F1 | Patient-vote accuracy |
|---|---:|---:|---:|---:|
| v1 | 0 | 81.98% | 0.7836 | 92.50% |
| v1 | 1 | 84.75% | 0.8201 | 96.25% |
| v1 | 2 | 83.33% | 0.7981 | 93.75% |
| logistic |  | 76.48% | 0.7342 | 87.50% |
| R0 | 0 | 82.04% | 0.7852 | 92.50% |
| R0 | 1 | 85.06% | 0.8235 | 95.00% |
| R0 | 2 | 83.27% | 0.7949 | 93.75% |
| R1 | 0 | 83.27% | 0.8027 | 95.00% |
| R1 | 1 | 85.31% | 0.8251 | 97.50% |
| R1 | 2 | 84.88% | 0.8155 | 95.00% |
| selected | 0 | 83.02% | 0.7994 | 95.00% |
| selected | 1 | 85.19% | 0.8255 | 95.00% |
| selected | 2 | 84.14% | 0.8048 | 93.75% |

## Evaluation and limitations

Same 1,620 windows and 80 patients as v1; no exclusions. Five outer patient folds, three stratified inner patient folds; fit-only normalization and class weights. All candidate selections were saved before outer evaluation. Earlier refits retain the 40-epoch cosine horizon.

Primary outcome is window macro-F1. All intervals use 2,000 paired patient draws (seed 20260925), average initialization seeds inside draws, and are pointwise without multiplicity adjustment. They condition on predictions and exclude retraining/selection uncertainty.

Repeatedly studied 80-patient cohort; diagnosis/source confounding remains. This is exploratory internal evidence, not external or clinical validation. Patient voting aggregates multiple windows/leads and is a different prediction unit. No advantage of a particular transform or quantum model is established.

Sum of fit runtimes: 3017.977 seconds, including fit-level checkpointing and reload checks; excludes global preflight/reporting. Failed invocation records: 0. All retained candidate and final weights were reloaded on the training device with saved probabilities and predicted classes checked.

## Artifacts

- [Full analysis, per-class metrics, confusion matrices and intervals](analysis.json)
- [Every seed](metrics.csv), [outer predictions](predictions.csv), [fits and runtimes](fits.csv)
- [Training and inner-validation histories](histories.csv), [all candidate scores](selection.csv), [patient splits](splits.csv)
- [Protocol](../../ecgdata_regularization_protocol_v2.md)

Per-fit inner validation probabilities at epochs 10/20/40, retained weights, exact window indices, optimizer/RNG checkpoints, source snapshots and integrity markers are under the matching results directory (gitignored). Saved v1 models and reports are unchanged.
