The corrected comparison trains both pipelines on the existing `ECGData.mat` and
keeps all leads and repeat recordings from each patient in the same validation fold.
All 30 fold fits completed on 2026-09-21: two pipelines, five patient folds, and
three initialization seeds. The training data remained unchanged.

**Completed results.** Values are mean ± sample standard deviation across the three
initialization seeds. Each seed has one out-of-fold prediction for every window.

| Pipeline | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 |
|---|---:|---:|---:|---:|
| VMD descriptors + VQC | 67.98% ± 0.70 pp | 0.6489 ± 0.0089 | 80.00% ± 3.31 pp | 0.7791 ± 0.0361 |
| Standard WST + VQC | 64.79% ± 0.94 pp | 0.6192 ± 0.0068 | 78.33% ± 2.89 pp | 0.7627 ± 0.0301 |

Patient voting combines all sampled windows from both leads and any repeat recording
of a held-out patient. Its score answers a different question from predicting one
3.91-second window.

VMD has the higher observed mean. The paired intervals below include zero, so the
performance difference remains uncertain; these results do not establish equivalence.
The comparison concerns these feature pipelines under the declared VQC budget.

| Difference: WST minus VMD | Observed difference | Paired 95% interval |
|---|---:|---:|
| Window accuracy | −3.19 pp | [−9.01, +2.74] pp |
| Window macro-F1 | −0.0297 | [−0.0856, +0.0265] |
| Patient-vote accuracy | −1.67 pp | [−10.42, +6.67] pp |
| Patient-vote macro-F1 | −0.0164 | [−0.1106, +0.0700] |

These are 5,000 paired bootstrap resamples of the 80 patients, applying the same
patient multiplicities to both pipelines and all three seeds, then averaging each
metric across seeds. They condition on the saved predictions and do not include
retraining or tuning uncertainty. Seeds are not treated as independent patient samples.

All 30 saved models passed checks of patient separation, training-only scaler means,
12 selected features, 40 epochs, finite losses/parameters, class weights, saved-model
prediction spot checks, and exact once-per-window prediction coverage. All VMD windows
met the solver's stopping tolerance. Eight regression tests passed, including the WST
reference and metadata checks. See `results/patient_vmd_wst_vqc/validation.json` and
`paired_comparison.json` for the saved checks and intervals.

**Identity verification.** The 162 ECG rows were matched individually to the original
PhysioNet source channels using two separate signal excerpts, at 0–12 seconds and
256–268 seconds. After resampling, the interior ten seconds from each excerpt
independently identified the same row. The weakest correlation was 0.99999238;
every source channel and every local row matched exactly once. These excerpts are
identity evidence, not additional training examples.

The cached record lists, headers, and short excerpts total 1,695,906 bytes (about
1.7 MB). The original `ECGData.mat` is unchanged; no replacement dataset is used.

The mapping contains 81 two-lead recordings and 80 patients. Both MIT-BIH records
201 and 202 are assigned to one patient, following the [original database
documentation](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm).
The mapping, hashes, and matching evidence are in `ecgvmd/ecgdata_subjects.csv` and
`ecgvmd/ecgdata_subjects.json`. `scripts/verify_ecg_sources.py` reproduces verification
from the cached public reference excerpts. The loader checks the SHA256 of every
local waveform before applying this mapping; it rejects reordered or altered rows.

Under the previous row-grouped folds, 72 of these 80 patients spanned multiple folds;
1,460 of 1,620 validation windows had another lead/recording from the same patient
in training. The corrected folds have zero such crossings.

**Fixed experiment.** Both arms use the same 1,620 nonoverlapping 500-sample windows
(10 sampled per ECG row, sample seed 0), z-scored independently. Five-fold
`StratifiedGroupKFold`, seed 0, holds out patients. Every patient appears in exactly
one test fold, and every train and test fold includes all three classes.

| Setting | VMD + VQC | WST + VQC |
|---|---|---|
| Input windows and patient folds | Identical | Identical |
| Representation | 8 VMD modes × 28 descriptors = 224 features | Standard scattering, J=6, Q=(8,1), T=64, orders 0/1/2 = 882 features |
| Transform parameters | alpha=2000, tau=0, DC mode, uniform initialization, tol=1e-7 | Orders 1/2 logged; signed order 0 linear; all 7 time bins retained |
| Feature selection | mRMR to 12, fitted within training fold | Same |
| Scaling | Training-fitted StandardScaler, tanh, pi/2 | Same |
| VQC | 12 qubits, depth 2, 40 epochs, batch 32, learning rate 0.05 | Same |
| Class loss weights | Balanced, computed from training labels | Same |
| Initialization/shuffle seeds | 0, 1, 2 | Same |
| Early stopping / test-fold tuning | None | None |

VMD starts with a 2,000-iteration limit. Ten windows initially reached this limit;
retrying them with limits of 4,000 and 8,000 left **zero capped windows**. The largest
observed iteration count was 4,483. No windows were discarded. Standard WST was
rechecked against Kymatio on real signals before training.

The audit also found a metadata error for non-dyadic scattering scales:
`bin_seconds` now reports the actual dyadic output stride, while
`invariance_seconds` reports the averaging scale. This changes no coefficients or
features and leaves the current T=64 spacing and scale equal to 0.5 seconds.

The VMD arm excludes rhythm and global decomposition descriptors. The alternative
Morlet-envelope descriptor pipeline is outside this two-arm experiment. The
candidate dimensions differ before selection, so conclusions concern these specified
feature pipelines under a common 12-feature VQC budget.

**Outputs and reproduction.** See the commands at the top of `README.md`.
`scripts/matched_vqc.py` saves a protocol manifest, feature matrices, per-window fold
assignments, each fitted pipeline, each set of circuit weights, selected feature names,
and out-of-fold predictions. Repeating the same command resumes completed folds.
A changed manifest requires a separate output directory. The main tables are
`results/patient_vmd_wst_vqc/metrics.csv` and `summary.csv`.

Report window accuracy and macro-F1, plus accuracy and macro-F1 after patient-level
majority voting. The three seed runs share the same patients and folds; their spread
measures initialization variability, not sampling uncertainty from new patients.
The common 40-epoch budget is inherited from earlier exploratory work, so this rerun
does not constitute an untouched external test. Class and source database also remain
confounded in this dataset.
