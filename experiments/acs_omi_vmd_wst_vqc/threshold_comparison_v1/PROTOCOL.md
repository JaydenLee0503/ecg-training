# OMI threshold comparison on completed models

Read only the completed undersampled and oversampled LFCC logistic/Swin models
from `undersampled_v1`. Use their previously frozen thresholds for 80/90/95%
calibration sensitivity and the fixed 0.5 threshold; no threshold is selected
using validation. Recompute per-seed confusion counts, precision, sensitivity,
specificity and false positives on the unchanged 3,581 validation ECGs grouped
by 3,391 patients. Neural metrics are averaged over seeds 0/1/2, not ensembles.

2,000 paired patient-cluster bootstrap resamples (seed 20261101), resampling
patients and all their ECGs together; thresholds and predictions remain fixed.
Intervals exclude refitting, threshold-selection and training-subset uncertainty.
The internal validation cohort has been examined repeatedly; calibration targets
need not transfer. No official test ECG is processed, no classifier fitted and
no clinical validity is claimed. Save inputs, hashes, all draws, tables and
warnings, including negative findings. The parent is read-only.
