# ECGData Swin: literature review and improvement priorities

Research date: **12 October 2026, America/Toronto**. This is a targeted research
review and experiment proposal, not a systematic review or a new trained result.
Recommendations combine the inspected implementation, saved results and primary
research sources. Proposed parameter values are hypotheses, not proven optima.

**Recommendation:** first test modest regularization with inner patient-level
epoch selection. Then investigate whether preserving waveform morphology helps
more than changing the attention mechanism. External ECG pretraining is the most
interesting larger extension, but requires substantially more adaptation.

## What is actually in this project

The main project compares VMD and standard WST representations with VQC and
classical controls. The separate `attention method/` workflow adds LFCC and a
small temporal Swin. ECGData is the ARR/CHF/NSR task; ACS/OMI is a separate task
and is outside this proposed experiment. The completed VQC improvement attempt
did not establish an improvement; its negative result remains part of the
[project record](../../architects/vqc_improvement_results.md).

The [frozen Swin results](ecgdata_lfcc_swin_v1/results.md) use 1,620 windows from
162 lead rows, **81 recordings and 80 patients**, with five patient-separated
folds and initialization seeds **0, 1, 2**. Each fit trains on 64 patients and
tests on 16. Records 201 and 202 remain grouped as one patient.

| Saved pipeline | Window accuracy | Window macro-F1 | Patient-vote accuracy |
|---|---:|---:|---:|
| LFCC + temporal Swin, mean of three seeds | 83.35% | 0.8006 | 94.17% |
| Pooled LFCC + logistic regression | 76.48% | 0.7342 | 87.50% |
| Original VMD + VQC, mean of three seeds | 67.98% | 0.6489 | 80.00% |
| Original WST + VQC, mean of three seeds | 64.79% | 0.6192 | 78.33% |

The Swin primary macro-F1 interval is [0.7453, 0.8432]; its paired difference
from logistic is +0.0664, interval [0.0224, 0.1143]. These saved patient-bootstrap
intervals condition on predictions and omit model-selection/retraining uncertainty.
The 94.17% is an average of patient-vote scores, **not a seed ensemble**. All
15 Swin fits attained 100% training accuracy. A large training/held-out gap
motivates regularization, but does not identify its cause or guarantee a remedy.

A fresh read-only audit verified **all 129 entries** in the saved Swin artifact
inventory by size and SHA-256, with no missing or changed files. It checked the
completion marker, 15 fit summaries, 600 history rows, and 1,620 predictions for
80 patients per seed. Window and patient-vote accuracies recomputed from the
export match the report. This was not a model reload or a full audit of other
experiments. See [audit JSON](swin_research_audit_2026-10-12.json) and
[reproducible PowerShell audit](../scripts/research_swin_audit_2026-10-12.ps1).

The saved class results help identify what improvement must mean:

| Class | Patients | Mean window recall | Mean window F1 | Mean patient-vote recall |
|---|---:|---:|---:|---:|
| ARR | 47 | 88.82% | 0.8807 | 99.29% |
| CHF | 15 | **72.00%** | **0.7313** | 84.44% |
| NSR | 18 | 78.24% | 0.7897 | 88.89% |

These are descriptive averages over all seeds. CHF is the weakest window class,
but has only 15 patients. Do not tune on individual held-out mistakes or exclude
hard patients. One additional correct patient changes one seed's patient accuracy
by 1.25 percentage points, illustrating the coarse resolution of this cohort.

## Implementation findings that change the recommendations

The inspected [protocol](../ecgdata_protocol.json),
[model](../attention_ecg/model.py), [extractor](../attention_ecg/cepstral.py) and
[training loop](../attention_ecg/ecgdata_training.py) establish this path:

```mermaid
flowchart LR
    A[500 ECG samples at 128 Hz] --> B[Per-window z-score]
    B --> C[1-second Hann frames, 0.25-second hop]
    C --> D[32 log linear-filter energies]
    D --> E[DCT, keep 16 coefficients]
    E --> F[13 temporal tokens]
    F --> G[Swin widths 16 and 32, pair merge to 7 tokens]
    G --> H[Mean pooling and 3-class head]
```

1. **This is already small.** It has 33,607 parameters and four attention blocks,
   with window size 4 and shift 2. It is randomly initialized and is not the
   pretrained 2-D image Swin used in many papers. Enlarging it is a low priority.
2. **There is no dropout or stochastic depth.** Attention, MLP and residual paths
   lack these operations. AdamW weight decay is 0.01. The official Swin code
   supports attention/projection/MLP dropout and DropPath; that gives an
   implementation precedent, not evidence that any rate improves this ECG task.
   [Official implementation](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py).
3. **The runner always takes epoch 40.** There is no inner validation or epoch
   selection. Existing histories contain training loss, not a validation curve.
   The saved run retains final weights, so earlier models cannot be recovered
   simply by reading those loss histories.
4. **The representation is a potential bottleneck.** The 500-sample input spans
   3.90625 seconds, but becomes 13 strongly overlapping spectral frames. The
   magnitude-squared FFT discards phase and is invariant to sign reversal;
   retaining 16 of 32 DCT coefficients additionally discards spectral detail.
   Local magnitude spectra can still encode useful morphology indirectly, so
   this is not a claim that LFCC removes all morphology.
5. **Several common augmentations would be ineffective or inappropriate here.**
   For nonconstant inputs, positive global gain cancels under per-window z-score:
   `z(a*x) = z(x)` for `a > 0`. Polarity reversal leaves LFCC power unchanged.
   Thus gain before normalization and sign flips do not create materially new
   LFCC examples, apart from numerical effects. Time reversal, arbitrary warping
   and image rotations are not justified default ECG augmentations.
6. **Window labels are inherited cohort labels.** An ARR window is not guaranteed
   to contain an annotated abnormal beat. The patient/window metric gap is
   compatible with a benefit from aggregation; it does not prove a labeling bug.
   Pooling many windows also changes the information available at prediction.

## Recent papers worth reading

The table separates published work from the specific preprint versions checked.
Publication status refers to those versions; this review does not establish that
no later journal version exists. None is a verified replication of this exact
ECGData LFCC protocol. Scores below explain the source result, not a leaderboard
against the local 83.35%/94.17%.

| Paper and status | Verified contribution | Relevance and limits for this project |
|---|---|---|
| **Chen et al., 2024**, *A novel method of swin transformer with time-frequency characteristics for ECG-based arrhythmia detection*. Frontiers in Cardiovascular Medicine. [Paper](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1401143/full), DOI 10.3389/fcvm.2024.1401143. | Complex-Morlet CWT maps with Swin; reports 99.34% intra-patient and 98.37% inter-patient accuracy on MIT-BIH. | Closest direct architectural precedent. Its AAMI heartbeat categories and 2-D maps differ from ARR/CHF/NSR LFCC windows. Those numbers are not targets for this pipeline; exact patient-ID equivalence was not independently audited. |
| **Aman et al., 2026**, *A Systematic Comparison of Data Representations for Transformer-Based ECG Arrhythmia Classification*. AAAI Bridge Program, PMLR 317:37–45. [Proceedings](https://proceedings.mlr.press/v317/aman26a.html). | The abstract reports 96.5% accuracy/F1 for raw 1-D ViT versus 92.6% accuracy for STFT 2-D ViT on CPSC2018, with comparable parameter counts; fusion did not exceed raw 1-D. | Supports testing raw input and representation controls. Only the proceedings abstract was accessible; the full PDF failed retrieval, so split details and claimed causal explanations remain unverified. AUROC is not a calibration measure. |
| **Jiang et al., 2025**, *ECG-Mamba: Cardiac Abnormality Classification With Non-Uniform-Mix Augmentation on 12-Lead ECGs*. IEEE JTEHM 13:461–470. [Paper record](https://pubmed.ncbi.nlm.nih.gov/41220791/), DOI 10.1109/JTEHM.2025.3613609. | Uses a bidirectional state-space model and applies MixUp conservatively to a subset of data across epochs. The authors report that standard mixing can be harmful. | Useful evidence against treating augmentation as universally beneficial. A long-sequence Mamba replacement has little computational motivation for only 13 tokens. Its 12-lead multilabel benchmark differs from this task. |
| **Na et al., 2024**, *Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram* (**ST-MEM**). ICLR. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/412fb8623bf8b6d56fb6285ea295447e-Abstract-Conference.html), [official code](https://github.com/vuno/ST-MEM). | Reconstructs masked 12-lead ECGs while modeling temporal and lead structure; studies adaptation to different lead combinations. | A strong methodological reference for ECG-specific masked pretraining. The current temporal Swin input is single-lead cepstra, so its checkpoint is not a direct weight replacement. |
| **Shu et al., December 2025**, **CLEF**, *Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models*. arXiv preprint. [Paper](https://arxiv.org/abs/2512.02180), [official repository](https://github.com/Nokia-Bell-Labs/ecg-foundation-model). | Single-lead encoders pretrained using ECGs from 161K MIMIC-IV patients, with clinical metadata guiding contrastive negatives; evaluated on 18 tasks across seven held-out datasets. Repository advertises code and checkpoints in three sizes. | Most directly relevant transfer-learning lead because each current input contains one lead. First try frozen embeddings with a simple head. Lead identity, duration, sampling, normalization, licenses and pretraining overlap require checking; metadata-guided training cannot be reproduced from ECGData alone. |
| **Lin et al., April 2026**, **ECG-LFM**, *A self-supervised electrocardiogram foundation model for empowering cardiovascular disease prediction and genetic factor discovery*. Nature Communications 17:5771. [Paper](https://www.nature.com/articles/s41467-026-72436-2), DOI 10.1038/s41467-026-72436-2; [author code](https://github.com/biomed-AI/ECG-LFM). | Combines contrastive learning with masked modeling after pretraining on over ten million 12-lead ECGs; studies disease prediction and biological associations. | A recent published example of large-scale ECG representation learning. Its scale and 12-lead assumptions make it background for transfer, not a sensible scratch-training recipe for 80 patients. Code link verified from the article; checkpoint usability was not tested. |
| **Gazeran et al., May 2026**, **ECG-NAT**, *A Self-supervised Neighborhood Attention Transformer for Multi-lead Electrocardiogram Classification*. arXiv preprint. [Paper](https://arxiv.org/abs/2605.13194). | Proposes hierarchical neighborhood attention, masked-autoencoder pretraining and supervised-contrastive plus cross-entropy fine-tuning. | Interesting alternative to shifted windows, especially with longer sequences. The checked abstract says code will be available on acceptance; implementation availability and detailed results were not validated here. Low priority for a 13-token model. |
| **Wan et al., March 2025**, **OpenECG**, *Benchmarking ECG Foundation Models with Public 1.2 Million Records*. arXiv preprint. [Paper](https://arxiv.org/abs/2503.00711). | Compares SimCLR, BYOL and MAE on a multi-source collection, including target-dataset exclusion and data-scaling analyses. | Useful motivation for dataset diversity and overlap checks. Excluding a target dataset from pretraining does not automatically prove untouched downstream patient evaluation. Its reported saturation cannot be extrapolated to an 80-patient learning curve. |
| **Nolin-Lapalme et al., March 2025**, *Foundation models for generalizable electrocardiogram interpretation: comparison of supervised and self-supervised electrocardiogram foundation models*. **medRxiv preprint** in the checked PubMed record. [Record](https://pubmed.ncbi.nlm.nih.gov/40093248/), DOI 10.1101/2025.03.02.25322575. | Compares DeepECG-SSL and DeepECG-SL trained on over one million ECGs, including external evaluation and low-label downstream tasks. | Provides a useful supervised-pretraining control for SSL. A PubMed entry does not itself mean peer review. Its diagnostic and biomarker tasks do not establish transfer gains on ECGData. |

The 2025 Scientific Reports article [*Automated heart disease detection using
Swin Transformer and ECG signal processing: a high-accuracy approach*](https://www.nature.com/articles/s41598-025-23097-6)
was also screened. Search excerpts report 99.8% accuracy, but direct article/PDF
retrieval failed in this session. It is retained as a reading lead, not used to
justify a performance target or a split-quality claim. The search did not verify
an exact match to the current LFCC + temporal Swin implementation; that is not
evidence of novelty or priority.

## Ranked methods to improve the current Swin

### 1. Regularization and patient-level epoch selection — first experiment

Keep the 33,607-parameter architecture and existing windows/folds. Compare a
small number of separately identifiable changes: dropout 0.1, stochastic depth
with maximum rate 0.1, or label smoothing 0.05. These are proposed starting
values. Do not combine everything in the first arm, since a successful bundle
would not identify the useful component. Keep existing class weighting initially;
adding oversampling on top changes the effective objective and needs its own
comparison. Weighted label-smoothing semantics must be explicit and tested.

Select among epochs 10, 20 and 40 using three inner patient folds inside each
outer training partition. Fit normalization and class weights separately inside
every inner fit. Select one configuration/epoch using pooled inner macro-F1
averaged over seeds 0/1/2, then refit on all 64 outer-training patients. The outer
16 patients must not decide dropout, stopping, calibration or aggregation.

This is more informative than increasing epochs: the existing model already
memorizes its training windows. A smaller model (e.g. widths 8/16) and a modest
weight-decay change are later alternatives if the bounded first test is inconclusive.

### 2. Test information loss before adding a larger network

Use the same 500-sample inputs to compare the current 16 LFCCs with all 32
log-filterbank energies. That removes DCT truncation while keeping the framing
and bandwidth constant; match or disclose the small input-projection parameter
change. Then test a shorter 64-sample frame/hop 16 representation separately.
That gives 29 frames with the current right-padding convention, trading spectral
resolution for finer timing. Increasing FFT size alone does not restore temporal
information or true spectral resolution lost to short frames.

A stronger next step is a compact raw-waveform Conv1D branch. It can preserve
signed waveform information that the LFCC branch cannot recover. Compare
**raw-only**, **LFCC-only**, and **raw + LFCC** under matched training budgets.
Start with feature concatenation and a small linear head; complex cross-attention
adds capacity before complementary information has been demonstrated. The Aman
study above motivates this question but also gives a negative fusion result.

This proposed branch starts from the same standardized 500 samples. It therefore
tests retained shape, not restoration of original absolute mV amplitude. Studying
amplitude requires a separate preprocessing protocol and data-unit verification.

### 3. Use augmentations that change the effective input

Evaluate mild additive noise on the waveform before recomputing LFCC, or mask
one temporal feature token after training-fitted normalization. Start separately,
with a predeclared probability and strength selected only in inner folds. A
zeroed normalized token represents the fit mean; it is feature masking, not a
physiological ECG segment. Use immutable base caches plus deterministically
seeded training augmentation, and save augmentation settings/RNG state.

Do not apply a conventional frequency mask to the LFCC coefficient axis and
describe it as frequency masking: DCT coefficients are not frequency bins.
Frequency-band masking belongs before the DCT, on log-band energies. With only
13 tokens, aggressive masking can remove too much evidence. MixUp is a secondary
ablation, not an assumed improvement; the ECG-Mamba paper supplies a relevant
counterexample to that assumption.

### 4. Test whether shifted windows are needed at this sequence length

Compare the LFCC Swin with a compact temporal CNN and a small global-attention
encoder using the same feature sequences. Global attention over 13 tokens needs
only 169 pair scores per head before padding, so a large-scale efficiency
argument is weak here. Keep widths/parameter counts and tuning budgets as close
as practical; report residual differences. A no-merging Swin ablation tests
whether compressing 13 frames to seven prematurely loses useful detail.

The current logistic control discards temporal order. Its lower score therefore
does not isolate the value of attention. A temporal CNN is an essential control
for any claim that Swin itself provides the gain.

### 5. Align aggregation with the intended prediction unit

If the desired output is a patient diagnosis from multiple clips, compare fixed
majority vote against probability averaging and, later, multiple-instance
learning (MIL). MIL would treat a patient's clips as a bag rather than force
each clip to contain diagnostic evidence. Use simple mean pooling as the first
bag control before learned attention pooling, since only 80 bags exist.

Do not concatenate noncontiguous random clips and treat them as a continuous
rhythm trace. Any learned bag pooling, threshold or calibration belongs inside
training/inner validation. Patient aggregation is not a window-accuracy gain and
uses more input at inference. Keep the original window macro-F1 primary unless
a new patient-level protocol explicitly changes the target. No alternate
aggregation scores were computed or selected in this review.

### 6. External pretraining — most interesting larger extension

Start with an existing single-lead encoder such as CLEF, first frozen with
logistic regression, then limited fine-tuning. This tests whether reusable ECG
features help before undertaking custom Swin pretraining. Verify the actual
checkpoint input contract; never create twelve supposed independent leads by
duplicating one lead. Upsampling 128 Hz inputs does not recreate missing high
frequencies, and padding 3.9 seconds does not create ten seconds of physiology.

To retain the present architecture, a separate masked-LFCC pretraining experiment
on external ECGs is possible, using the same transform and masks that preserve
temporal positions. It cannot reconstruct information already discarded by
LFCC. Larger datasets, data access, patient de-duplication and a scratch control
are prerequisites for a credible comparison.

Keep **all outer-test patients out of pretraining**, even without labels, for an
inductive claim. Full MIT-BIH/CHF/NSR source recordings can contain the same
patients as ECGData: using unlabelled extra segments from those held-out patients
would contaminate the comparison. A deliberately transductive study would need
different labeling and must not be compared as if patient-independent.

## A bounded first experiment, not yet implemented or frozen

Suggested new identifier: `ecgdata_lfcc_swin_regularization_v2`. Store its future
protocol, runner and outputs separately; do not edit the v1 protocol or shared
model behavior in ways that break the saved experiment.

| Candidate | Change from the v1 architecture | Epoch candidates |
|---|---|---|
| R0 | Original architecture; inner-selected epoch | 10, 20, 40 |
| R1 | Dropout 0.1 in attention probabilities, projection and MLP | 10, 20, 40 |
| R2 | Residual DropPath, linearly increasing to 0.1 over four blocks | 10, 20, 40 |
| R3 | Label smoothing 0.05, otherwise original architecture | 10, 20, 40 |

Use three inner patient folds and all three initialization seeds. Four candidates
times five outer folds times three inner folds times three seeds requires
**180 inner fits**, plus **15 selected outer refits**. Saving candidate epochs
within each 40-epoch fit avoids fitting every epoch choice from scratch. This
is a proposed budget, not work already launched or a runtime guarantee.

For consistency, retain the **40-epoch cosine schedule horizon** when evaluating
and refitting an earlier chosen epoch; stop the refit at that epoch. Changing
the horizon to the selected epoch would silently change the optimizer trajectory.
Predeclare ties (earlier epoch, then candidate order R0–R3), patient stratification
seed (proposed 20261012), and aggregation. Three inner folds have very few CHF
patients, so preserve and report their exact class counts and all failures.

Compare the selected procedure with saved fixed-40-epoch v1 and pooled LFCC
logistic predictions on unchanged outer windows. Primary outcome: difference
in window macro-F1, with paired patient-bootstrap intervals over seed means.
Report every seed, per-class recall/F1, patient voting, confusion matrices,
training/inner curves, selected epochs, fit runtimes and model sizes. NLL/Brier
scores may be added as calibration diagnostics; any temperature fitting requires
inner predictions. Do not select or highlight a winning outer seed.

Before execution, implement a separate resumable runner; test inner/outer
patient isolation, fit-only statistics, deterministic augmentation/DropPath
recovery if used, candidate-epoch retention, and saved-model reloads. Hash all
inputs and preserve negative arms. If the selected procedure does not improve
the paired result convincingly, retain v1 and report the attempted improvement.

The existing [learning-curve draft](../ecgdata_learning_curve_protocol.md) is
also useful, but answers a different question: whether adding training patients
helps at fixed settings. It remains unimplemented and was not run here. Adding
more windows from existing patients is not the same experiment.

## Research direction beyond another accuracy increment

The strongest scientific question is **which information and inductive bias
produce patient-level generalization**. A staged representation/classifier study
can compare raw waveform, LFCC and temporal WST under matched windows and patient
splits, with logistic/MLP, temporal CNN and Swin controls. Existing static VMD or
WST descriptors must not be relabeled as a temporal sequence; retain actual
coefficient-time axes or use an appropriate vector control.

Diagnosis and acquisition database are confounded in ECGData. A source classifier
cannot disentangle the issue because source predicts class. Leaving an entire
source out also removes its associated class from training, so it is not a
valid ordinary three-class generalization test. No regularizer can establish
the missing cross-source evidence. An additional cohort containing overlapping
diagnoses across acquisition sources is needed for that claim. Retraining on a
new AF/sinus task would test a new task, not externally validate ARR/CHF/NSR.

All existing outer folds have already informed research choices, including this
review. Proper inner folds prevent direct new tuning leakage, but cannot turn
this repeatedly studied cohort into an untouched confirmatory test set.

## Work completed and limits

Completed: project/report/code inspection, targeted primary-source searches,
saved-artifact checks, descriptive class analysis and this ranked proposal.
Search scope included ECG Swin, raw/time-frequency representations, ECG-specific
augmentation, masked modeling, and foundation models, emphasizing 2024–2026.
This is a selective reading list, not an exhaustive or publication-quality
systematic search. Access limitations are recorded beside the affected papers.

No packages, datasets or checkpoints were downloaded; no feature extraction,
training, new model inference, ACS work or frozen-protocol changes occurred.
The local audit uses PowerShell and requires no Linux Python environment.
The initial shell sandbox could not access the workspace; approved read access
resolved it. Git ownership checks required a command-scoped `safe.directory`
setting; no global Git configuration was changed. These checks establish the
reported evidence, not absence of software bugs or a promised accuracy gain.
