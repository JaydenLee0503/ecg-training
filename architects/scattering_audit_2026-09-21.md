The scattering arithmetic passes the checks below at the current experimental settings. The existing comparison is useful as an exploratory comparison of feature pipelines, but it is not yet a validated comparison on independent patients. No production code was changed during this audit.

Follow-up: the user subsequently authorized implementation. The patient mapping has
now been verified against original source signals; the fixes and rerun are documented
in [the corrected comparison](patient_vmd_wst_vqc.md). The findings below describe
the state before those changes.

**Numerical verification, rerun on 2026-09-21.** The environment used NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, and Kymatio 0.3.0. The feature archive was `features/fixed_K8_a2000_L500_it500_eeb2053b.npz`, whose configuration produces 1,620 standardized windows, 500 samples each at 128 Hz.

- All 1,620 windows were transformed using both `scatter_batch` and `scatter_kymatio` at J=6, Q=(8,1), T=64, order=2. Maximum absolute coefficient difference: **0.0**. Shape: 1,620 × 126 paths × 7 bins.
- Path orders, center frequencies, bandwidth metadata, feature names, and flattened feature ordering passed checks. The 882-column log feature matrices matched exactly; order zero remained signed and linear. Coefficients and features were finite. Changing batch partitioning preserved results.
- The six existing reference configurations passed, with maximum difference **1.388e-16**.
- Sixteen additional configurations passed across lengths 127, 500, 501, and 1,024, including non-dyadic T, first-order-only transforms, and zero, constant, impulse, sinusoidal, and random inputs. Maximum difference: **1.110e-16**. Kymatio emitted its border-support warning for some configurations; agreement does not eliminate boundary effects.
- Labels and row groups matched the saved VMD archive. Recomputing nine VMD windows spanning all three classes reproduced their saved float32 modes exactly. This spot check is stronger than comparing labels/groups alone, which cannot detect within-record window permutations.

These checks provide strong evidence of agreement with Kymatio for the tested inputs, not a proof that every possible parameter combination is bug-free. The Kymatio definition of the cascade is documented [here](https://www.kymat.io/codereference.html).

**The most consequential finding is the grouping variable.** `ecgvmd/data.py:98` assigns `record_ids=np.arange(data.shape[0])`. However, the original MathWorks attribution says that each two-channel source file was separated into two rows: 48 ARR source files, 15 CHF source files, and 18 NSR source files, yielding 162 rows from **81 source recordings**. The provenance text was extracted from the [official MathWorks archive](https://github.com/mathworks/physionet_ECG_data/blob/main/ECGData.zip), and the local `ECGData.mat` CRC32 matches that archive's entry exactly: `9657c593`.

Therefore, row-grouped CV prevents windows of the same row crossing folds, but does not establish source-record or patient independence. Supporting signal checks found 56 mutual-nearest row pairs with QRS-energy correlation above 0.8; 49 of these pairs lie in different current folds. For example, zero-based rows 0 and 43 have correlation 0.955 and lie in folds 3 and 0. These are candidate relationships, not an authoritative patient mapping. Do not infer the mapping from row adjacency or simple half-block offsets: those arrangements failed the signal check on this file.

A proper rerun needs verified source/patient IDs, with both leads and every window belonging to one patient kept together. Source grouping alone may still be insufficient: the [MIT-BIH database](https://physionet.org/content/mitdb/1.0.0/) includes 48 records from 47 subjects. This affects both VMD and WST, and their ranking could change after correction. Neither the existing scores nor this audit's row-grouped KNN scores should be labeled patient-independent.

**What the current feature comparisons measure.** `scripts/wst_vs_vmd_vqc.py` compares VMD mode descriptors plus nine rhythm features against standard log scattering coefficients. The same rows, folds, in-fold mRMR, scaler, and classifier budget are good controls. Different feature construction is legitimate when the claim concerns these complete pipelines; it cannot isolate an inherent advantage of VMD over scattering. A transform-focused ablation should also address the asymmetric rhythm features, candidate feature dimensions, and selection/tuning budget.

`WST env + descriptors` requires a separate label, such as **Morlet-envelope descriptors**. `order1_envelopes` returns the unaveraged first-order modulus, followed by handcrafted descriptors. It omits the standard scattering averaging and second-order coefficients. Those nonnegative envelopes also differ from signed VMD modes: their zero crossings and spectral/instantaneous-frequency descriptors have different interpretations. Giving both arrays the same descriptor function does not make the representations identical except for adaptivity. The VMD mode block additionally contains three global descriptors. Scores for this alternative should not be presented as scores for standard second-order WST.

The VMD archive contains **554/1,620 windows (34.20%)** that reached `max_iter=500`. That does not invalidate a declared finite-iteration classifier pipeline, but claims about converged VMD require a larger iteration budget and rechecking convergence. Both arms should receive comparable tuning resources within training folds. Repeated CV seeds on the same dataset do not create independent patient samples: the paired t-test p-values in `wst_descriptor_probe.py` are not sufficient evidence of population-level superiority, and failure to reject a difference is not proof of equivalence.

**The identified paper.** Gupta et al., DOI [10.1109/JSEN.2025.3572080](https://doi.org/10.1109/JSEN.2025.3572080), reports 94% validation and 93.2% test accuracy. Its [public abstract](https://www.researchgate.net/publication/392176131_3-D_Attractor_Reconstruction_for_Enhanced_ECG_Classification_of_Arrhythmia_and_Congestive_Heart_Failure) describes SPAR attractor reconstruction, radial/angular density features fused across sliding windows, and weighted KNN on the same 162-signal dataset. This is a different feature pipeline from WST. Full methods were not accessible: window lengths, split unit, overlap, feature fitting, and patient separation remain unverified. Leakage cannot be attributed to this paper from the abstract.

**A measured explanation to investigate.** The following results were rerun on the same 882 WST features. Every scaler was fitted inside its training fold, all runs used five folds with seed 0, and the classifier was the only other component. These numbers reproduce the corresponding existing `results/knn_leakage.csv` entries.

| Model | Random-window accuracy | ECGData-row-grouped accuracy |
|---|---:|---:|
| 1-nearest neighbor | 94.69% | 74.38% |
| 5-NN, inverse-distance weights | 93.02% | 74.94% |
| 10-NN, inverse-distance weights | 92.53% | 75.74% |
| 10-NN, inverse-squared-distance weights | 92.90% | 75.93% |

The experiment shows that accuracy can approach 94% without changing the scattering implementation: changing the split allows the classifier to use nearby examples from already-seen recordings. It does **not** establish that Gupta et al. made this choice. Their full methods/code are needed before attributing the result to leakage. “Weighted KNN” also needs an exact definition: MATLAB supports [inverse and squared-inverse distance weights](https://www.mathworks.com/help/stats/choose-a-classifier.html).

Other credible explanations are longer input recordings, a different averaging scale, more features than the 12-feature quantum budget, classifier tuning, record voting versus individual-window prediction, and accuracy versus macro-F1. For a concrete protocol difference, the [MathWorks WST ECG example](https://www.mathworks.com/help/wavelet/ug/ecg-signal-classification-using-wavelet-time-scattering.html) transforms 512-second recordings with a 150-second invariance scale. This project uses 3.90625-second inputs and T/fs=0.5 seconds. Those experiments supply very different temporal information to the classifier even though they use the same dataset.

**Minor metadata issue.** `ScatterResult.bin_seconds` describes output bin spacing but returns T/fs. For non-dyadic T, the actual spacing is `2**floor(log2(T))/fs`. At T=50 it reports 0.390625 seconds whereas the output stride is 0.25 seconds. T/fs still describes the chosen low-pass scale. This naming/documentation issue does not affect the current T=64 coefficients or scores.

Audit outputs are in `results/scattering_audit/`: `summary.json`, `reference_configs.csv`, `edge_configs.csv`, `knn_splits.csv`, `candidate_related_rows.csv`, `Modified_physionet_data.txt`, and `run.log`. `reproduce.py` reruns the numerical and classifier checks from the project root using `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python results/scattering_audit/reproduce.py`. The provenance extraction and official-archive CRC comparison were separate read-only checks; rerunning the script does not repeat the network check. The results directory is gitignored. The audit leaves the production implementation and previous experiment results unchanged.
