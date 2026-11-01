# ECG-08 patient-count learning curve

Three patient-subset draws paired with seeds 0/1/2; five patient folds; 40 epochs. Logistic uses the same subsets.

| Fraction | Swin window accuracy | Swin macro-F1 | Swin patient-vote accuracy | Logistic window accuracy |
|---|---:|---:|---:|---:|
| 25% | 75.95% | 0.7089 | 83.33% | 74.30% |
| 50% | 79.34% | 0.7490 | 89.17% | 74.96% |
| 75% | 82.18% | 0.7799 | 91.25% | 75.99% |
| 100% | 83.35% | 0.8006 | 94.17% | 76.48% |

Primary 100% minus 75% macro-F1: +0.0207, 95% patient-bootstrap interval [+0.0024, +0.0373].

Patient bootstrap conditional on predictions; subset and initialization vary together; fewer patients also means fewer optimizer steps.
The 80-patient cohort is repeatedly studied and confounds diagnosis with source. No external validation.

The primary interval is above zero under the protocol's interpretation rule,
consistent with improvement between these two training sizes on this cohort. It
is not evidence of a general learning curve on new populations or a prediction
of the accuracy of a larger dataset. Each smaller training subset also received
fewer optimizer updates, and the three replicates pair subset draw with model
initialization. The interval is conditional on the saved predictions, without
retraining or a new patient split.

All 45 new Swin fits and 45 matching logistic fits completed; summed fit time
was approximately 888.2 seconds. Logistic fits recorded no warnings. A separate
[audit](audit.json) verified all 90 new fit markers, nested subsets, patient
separation, all 45 Swin normalization/class-weight calculations and pooled
window accuracy/macro-F1. The original 100% models were checksum-verified and
reused. Per-fit predictions, checkpoints, histories and patient subsets are in
the local gitignored `results/` directory; [analysis.json](analysis.json) copies
the final metric and interval summary. Numerical checks do not prove software
is bug-free.

The run manifest hashes the study runner, original draft, base cache and several
training/report helpers, but does not directly hash every transitive import such
as `model.py` and `regularized_swin.py`. The original baseline cache and trial
markers remain verified, and the audit checked the saved artifacts, but that
incomplete launch-time source fingerprint limits exact provenance of this run.
