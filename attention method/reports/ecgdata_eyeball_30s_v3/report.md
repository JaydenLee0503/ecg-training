# ECGData 30-second Eyeball experiment (v3)

## Outcome

The frozen experiment completed all five patient-separated folds on 80 patients. The balanced RBF SVC reached **81.25% patient-vote accuracy** (65/80) and **0.7621 patient-vote macro-F1**. The balanced logistic control reached **67.50%** patient-vote accuracy and **0.6766** macro-F1. These are exploratory results on the dataset's three labels (ARR, CHF, NSR), not myocardial infarction or OMI detection.

| Model | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 | Patient-vote macro-AP |
|---|---:|---:|---:|---:|---:|
| Balanced logistic | 61.53% | 0.6104 | 67.50% | 0.6766 | 0.8243 |
| Balanced RBF SVC | 70.35% | 0.6434 | **81.25%** | **0.7621** | 0.8312 |

Patient-bootstrap 95% intervals for the RBF SVC were 72.5–90.0% for patient-vote accuracy, 0.6409–0.8597 for patient-vote macro-F1, and 0.7394–0.9139 for patient-vote macro-AP. For the logistic control, patient-vote accuracy was 57.5–77.5% and macro-F1 was 0.5674–0.7701. Intervals are conditional on these saved out-of-fold predictions; they do not represent external validation.

RBF SVC window-level recall was 0.8002 for ARR, **0.4020 for CHF**, and 0.7068 for NSR. The comparatively low CHF recall is material; the headline patient-vote accuracy does not imply uniformly strong class performance.

## Fixed protocol and execution

The experiment used the frozen [protocol](ecgdata_eyeball_30s_v3/protocol.json): 10 fixed 30-second clips per lead row (1,620 intended clips), the existing five patient folds, per-window z-scoring, four EMD modes using `EMD-signal` 1.6.4 with a 64,000-iteration cap, and 12 analytic-signal/Hilbert/centroid features. Evaluation used balanced logistic regression and a balanced RBF SVC (`C=1`, `gamma=scale`); no hyperparameter search was performed. Feature extraction ran with four processes. Both estimators were fitted on each outer training partition and generated out-of-fold predictions for held-out patients.

Extraction produced 1,531 valid clips (94.51%) and 89 EMD iteration-cap failures. Failures were ARR 49, CHF 4, NSR 36. The prespecified minimum overall coverage and per-fold class safeguards passed. Every one of the 80 patients retained at least one valid clip; 30 patients had at least one failed clip. Fit time across the 10 model fits was approximately 1.0 second, excluding feature extraction.

## Interpretation and limits

The RBF SVC's patient-vote accuracy is below the previously reported 94.17% for the ECGData LFCC temporal Swin experiment. That Swin result uses a different representation, model, and 3.9-second window pipeline, so the figures are not a controlled head-to-head comparison. The cohort is small (80 patients), has source/diagnosis confounding documented in the existing ECGData reports, and does not establish wearable robustness, external validity, clinical utility, or MI/OMI performance. EMD failures also remove some intended clips, with failures concentrated in ARR. The analysis retained the failed records in the prediction export and reported coverage explicitly.

## Reproducibility artifacts

The versioned machine-readable exports in this directory include the protocol, extraction summary, failure list, artifact inventory, completion marker, manifest, training summary, metrics, and clip-level predictions. Large resumable checkpoints and extraction artifacts are in the gitignored `attention method/results/ecgdata_eyeball_30s_v3/` directory. See the attention-method [session handoff](../../SESSION_HANDOFF.md) for restart and audit status. No training remains pending.
