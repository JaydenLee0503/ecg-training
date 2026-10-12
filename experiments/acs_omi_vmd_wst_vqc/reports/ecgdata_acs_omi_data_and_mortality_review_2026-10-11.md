# ECG classification from conventional machine learning to multiresolution, transformer and quantum approaches: datasets, reported errors and research gaps

**Article type:** Narrative literature review and proposed research framework

**Literature and source access cutoff:** 11 October 2026

**Scope:** ECGData.mat, acute coronary syndrome and occlusion myocardial infarction; conventional machine learning, deep learning, VMD, WST, LFCC, Swin Transformers and variational quantum classifiers.

## Abstract

Electrocardiographic classification research spans substantially different clinical tasks, including rhythm recognition, heart-failure cohort classification and identification of coronary occlusion. This review examines the structure and provenance of ECGData.mat and a public acute coronary syndrome dataset, places these resources within global cardiovascular mortality, and compares conventional machine learning with recent deep learning and signal-processing approaches. Published performance is interpreted through its target, evaluation unit, patient separation and decision threshold. False-negative and false-positive rates are distinguished from overall classification error. The proposed research direction is a controlled comparison of adaptive decomposition, wavelet scattering, cepstral representations, attention and quantum classification. Its scientific value would depend on demonstrating reproducible gains under matched conditions, preservation of clinically relevant information and generalization beyond the development cohort. The review identifies patient dependence, source–diagnosis confounding, inconsistent OMI definitions, information loss during preprocessing, inadequate classical controls and absent clinical-outcome evidence as central gaps. It reports published findings and develops hypotheses; it presents no new experimental results.

**Keywords:** electrocardiogram; occlusion myocardial infarction; machine learning; wavelet scattering; variational mode decomposition; linear frequency cepstral coefficients; Swin Transformer; quantum machine learning.

## 1. Review scope and clinical motivation

This is a targeted narrative review, not a systematic review or meta-analysis. Sources comprise official dataset documentation, original methodological papers, primary ECG studies and WHO mortality estimates. Recent research from 2023–2026 is considered alongside foundational methods. Searches combined ECG, OMI, machine learning, deep learning, VMD, wavelet scattering, LFCC, Swin Transformer and quantum classification. Publisher or author-hosted material was prioritized. Where only an abstract could be verified, conclusions are restricted accordingly. Numerical results from different tasks are not pooled.

### 1.1 Worldwide mortality

| Mortality measure | Observation year | Estimated deaths worldwide | Interpretation |
|---|---:|---:|---|
| All causes | 2023 | **61 million** | Total deaths from all causes |
| Ischaemic heart disease (IHD) | 2023 | **9.5 million** | Approximately 16% of all deaths |
| Stroke | 2023 | **6.8 million** | Approximately 11% of all deaths |
| Cardiovascular diseases (CVDs), collectively | 2022 | **19.8 million** | Approximately 32% of deaths in that year |

The first three estimates come from the WHO fact sheet updated on 2 October 2026; the CVD estimate comes from the fact sheet updated on 31 July 2025. Publication year and mortality observation year must remain distinct. [WHO causes of death, ref. 1](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death); [WHO cardiovascular diseases, ref. 2](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29).

IHD and stroke belong within the broader CVD category; their counts should not be added to the CVD total. Nor should the 2022 CVD count be divided by the 2023 all-cause total to reconstruct a same-year percentage. IHD deaths cannot be relabelled as OMI deaths. The sources above do not provide a separate, globally comparable OMI mortality estimate.

Neither ECG dataset considered here provides the population coverage and mortality follow-up required to estimate worldwide deaths. Classification performance is also distinct from mortality prediction or prevention. A mortality study needs a defined population, vital status, follow-up duration and censoring; evidence that a diagnostic system prevents deaths requires evaluation of its use in clinical care.

## 2. Dataset structure, provenance and interpretation

### 2.1 ECGData.mat: a processed three-cohort dataset

ECGData.mat is a MathWorks distribution derived from three PhysioNet databases. Its preprocessing includes physical scaling, resampling to 128 Hz, separating the two source channels and truncating each signal to 65,536 samples. It is a processed subset rather than the complete original recordings. [MathWorks distribution, ref. 3](https://github.com/mathworks/physionet_ECG_data).

| MATLAB element or property | Structure or value | Meaning |
|---|---|---|
| ECGData | Scalar structure | Container for signals and labels |
| Data | 162 × 65,536 double-precision values | Rows are individual lead signals; columns are time samples |
| Labels | 162 × 1 cell array | ARR, CHF or NSR |
| Sampling frequency | 128 Hz | Supplied by the distribution documentation |
| Duration per row | 512 seconds | Calculated as 65,536 / 128 |
| Embedded patient and survival fields | Absent | Patient grouping requires original-source identities |

The schema and class counts are described by the MathWorks example. [MathWorks dataset documentation, ref. 4](https://www.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html).

| Class | Original database | Derived signal rows | Source recordings | Source patients |
|---|---|---:|---:|---:|
| ARR | MIT-BIH Arrhythmia | 96 | 48 | 47 |
| CHF | BIDMC Congestive Heart Failure | 30 | 15 | 15 |
| NSR | MIT-BIH Normal Sinus Rhythm | 36 | 18 | 18 |
| **Total** | Three cohorts | **162** | **81** | **80** |

Patient totals follow the original database populations; the row counts reflect separation of two channels per recording. MIT-BIH records 201 and 202 belong to the same individual. Thus, channels, repeated recordings and derived windows cannot be treated as independent patients. [MIT-BIH documentation, ref. 5](https://physionet.org/content/mitdb/1.0.0/); [BIDMC CHF documentation, ref. 6](https://physionet.org/content/chfdb/1.0.0/); [NSR documentation, ref. 7](https://physionet.org/content/nsrdb/1.0.0/).

An important limitation follows from this construction: each class corresponds to a different source database. Disease label and source are therefore confounded. Patient separation prevents reuse of a person's signals across partitions, but it cannot, by itself, establish that the classifier learned disease rather than acquisition or cohort characteristics. Additional windows increase the number of observations without increasing the number of independent patients.

ARR is a broad cohort label, not a specific rhythm diagnosis for every beat. CHF represents a selected severe-heart-failure cohort, while NSR indicates the source cohort's rhythm status rather than comprehensive absence of disease. These labels do not define myocardial infarction or coronary occlusion. [ARR, ref. 5](https://physionet.org/content/mitdb/1.0.0/); [CHF, ref. 6](https://physionet.org/content/chfdb/1.0.0/); [NSR, ref. 7](https://physionet.org/content/nsrdb/1.0.0/).

### 2.2 ACS/OMI: a clinically distinct classification problem

Du et al. collected 19,955 ECGs from 18,909 patients in Chongqing, China, during 2015–2024, within seven days before angiography. [ACS dataset paper, ref. 8](https://doi.org/10.1038/s41597-026-07278-0).

| Property | Published representation |
|---|---|
| Waveforms | Twelve leads, nominally 10 seconds, 500 Hz; 5,000 samples per lead |
| Storage | WFDB .dat/.hea pairs; optional median waveforms in .med format |
| Metadata | CSV, including patient identifiers and clinical information |
| Labels | AMI, OMI, CTO, NSTEMI, STEMI and UA; overlapping binary indicators |
| Official division | 90% training, 10% testing; public test diagnoses withheld |

The release defines OMI as acute culprit-vessel occlusion with pre-PCI TIMI flow grade 0 or 1. Non-OMI does not mean healthy. These labels are not six mutually exclusive classes. The versioned data citation is Figshare v1. [Dataset definition, ref. 8](https://doi.org/10.1038/s41597-026-07278-0); [data release, ref. 9](https://doi.org/10.6084/m9.figshare.29925314.v1).

This sampling frame is a selected hospital cohort, not community screening. Its outcome is anatomically and clinically different from ARR/CHF/NSR classification. Comparing the two datasets is useful for studying how representations behave across tasks, but combining their accuracies would not estimate a common clinical capability.

### 2.3 Citation practice

A study using ECGData.mat should credit the MathWorks derivative, identify its preprocessing, and cite all three original PhysioNet datasets (references 3–7). Associated publications include the [MIT-BIH database paper, ref. 27](https://doi.org/10.1109/51.932724), the [CHF study, ref. 28](https://doi.org/10.1016/S0735-1097%2886%2980478-8), and the [original, ref. 29](https://doi.org/10.1161/01.CIR.101.23.e215) and [current, ref. 30](https://doi.org/10.1038/s44360-026-00096-z) PhysioNet platform publications. A study using ACS/OMI should cite both the versioned dataset and its descriptor (references 8–9). A filename alone is not a dataset citation.

Any subsequent filtering, signal transformation, relabelling or patient partitioning should be reported as the new study's methods, rather than attributed to the original data creators.

## 3. From conventional machine learning to deep learning

### 3.1 Conventional machine learning

Conventional ECG systems commonly separate representation from classification: extract a feature vector, then estimate a decision rule. Features may encode heartbeat intervals, waveform shape or transform coefficients; classifiers can include linear discriminants, logistic regression, support vector machines and tree ensembles. This separation allows researchers to test whether an improvement comes from the representation or the classifier.

An influential example is de Chazal et al. (2004), who combined morphology and interval features with linear discriminant classifiers using separate development and assessment recordings. Their study demonstrates that structured feature engineering and independent assessment predate deep learning. It also exposes substantial class-specific errors that aggregate accuracy could obscure. [de Chazal et al., ref. 10](https://doi.org/10.1109/TBME.2004.827359).

Conventional learning remains relevant to OMI. Al-Zaiti et al. (2023) developed and externally validated models using 7,313 consecutive patients from multiple clinical sites. Their work connected discriminative ECG features with clinical interpretation and investigated OMI risk stratification. It provides evidence that clinically grounded features and validation design remain consequential even when newer architectures are available. [Al-Zaiti et al., ref. 11](https://doi.org/10.1038/s41591-023-02396-3).

For a proposed comparative study, regularized linear models, RBF SVMs and tree ensembles should remain meaningful controls. A complex representation that performs well with a simple classifier may offer a different accuracy–cost tradeoff from a larger neural network.

### 3.2 Deep learning

Deep networks learn representations jointly with a task objective. One-dimensional convolutional networks can learn local waveform patterns; temporal architectures and attention mechanisms can combine information across longer intervals. Their flexibility increases the need to distinguish patient-specific patterns from transferable clinical information.

Hannun et al. (2019) trained an arrhythmia network using 91,232 single-lead recordings from 53,549 patients across 12 rhythm classes. They reported average AUROC 0.97 and F1 0.837, compared with average cardiologist F1 0.780 on their adjudicated test task. These are rhythm-classification results, not OMI performance, and F1 0.837 does not imply a 16.3% misclassification rate. [Hannun et al., ref. 12](https://doi.org/10.1038/s41591-018-0268-3).

Gustafsson et al. (2026) used 540,372 ECGs from 225,824 patients for OMI detection and localization, with patient-separated random and temporal tests. Section 5 distinguishes internal from external performance. [Gustafsson et al., ref. 15](https://doi.org/10.1038/s41467-026-73023-1).

The methodological change is therefore broader than replacing a classifier: it includes learning directly from signals, using clinically meaningful reference standards and testing transportability across settings.

### 3.3 Recent foundation models and supervision strategies

Khattak et al. (2026) pretrained the CAPE contrastive ECG model on 5,203,269 recordings from four cohorts across three continents. Mixing cohorts improved performance in familiar distributions but could impair transfer through cohort-specific artifacts. Their within-cohort batching strategy addressed this problem. This September 2026 study makes the generalization gap concrete: increasing dataset size and diversity does not automatically eliminate shortcuts. [Khattak et al., ref. 16](https://doi.org/10.1371/journal.pdig.0001623).

Terminology requires care:

- **WST** means wavelet scattering transform, a signal representation.
- **WSL** can mean weakly supervised learning: learning from incomplete, coarse or imperfect labels.
- **Self-supervised learning** builds training objectives from the data themselves, such as contrasting related signal views, before adapting the representation to a clinical target.

Weak supervision is relevant when a recording has a diagnosis but individual beats or intervals do not. Assigning that diagnosis to every short window introduces a modelling assumption; a multiple-instance approach could instead aggregate window evidence at the recording level. This is a proposed direction, not evidence that every interval contains disease-specific information. [Weak-supervision framework, ref. 17](https://doi.org/10.1093/nsr/nwx106).

## 4. Signal-processing and hybrid modelling approaches

These approaches operate at different stages. VMD decomposes a waveform; WST and LFCC construct representations; Swin is an attention architecture; a variational quantum classifier is a trainable prediction component. They can be compared or combined, but their names do not imply that they solve the same problem.

### 4.1 Variational mode decomposition

VMD estimates a prescribed number of modes and their centre frequencies through a joint variational optimization. It seeks components with compact spectral bandwidths whose sum reconstructs the input. The method was introduced by Dragomiretskiy and Zosso in 2014. [VMD original paper, ref. 18](https://doi.org/10.1109/TSP.2013.2288675).

Its attraction for ECG research is an adaptive frequency decomposition. Candidate features include mode energies, envelope statistics and temporal descriptors. However, the number of modes, bandwidth penalty, initialization and stopping tolerance affect the resulting representation. A mode should not automatically be assigned a physiological interpretation, and decomposition alone does not guarantee denoising.

A defensible study would test whether VMD preserves diagnostically useful morphology and improves downstream classification relative to matched alternatives. Reconstruction error, convergence and classification error answer different questions. Failed or incomplete decompositions must contribute to the evaluation through a declared handling rule; silently excluding difficult signals can bias results.

### 4.2 Wavelet scattering transform

WST cascades wavelet filtering, a modulus nonlinearity and averaging. Its fixed filters provide multiscale representations with mathematical stability properties under specified signal deformations. The averaging scale controls the amount of local translation invariance. These properties do not imply invariance to every artifact or clinical robustness in every dataset. [Mallat, ref. 19](https://arxiv.org/abs/1101.2286).

Nahak et al. (2023) combined WST with an SVM for 17-class classification of 1,000 ten-second ECG fragments from 45 subjects. This supports WST as an established ECG representation rather than an entirely new proposal. The accessible abstract does not establish a patient-disjoint evaluation protocol; its accuracy is therefore reported with that limitation in Section 5. [Nahak et al., ref. 20](https://doi.org/10.1016/j.eswa.2023.120019).

An important proposed comparison is whether averaging that helps rhythm classification removes temporal detail needed for a different endpoint. WST averaging scales and VMD settings should consequently be selected within training data for each clinical task. WST is also distinct from a conventional wavelet scalogram used as an image input.

### 4.3 LFCC and Swin Transformers

Linear frequency cepstral coefficients summarize the log energy in linearly spaced spectral filters, usually followed by a discrete cosine transform. A compact formulation is:

$$
E_m(t)=\sum_f H_m(f)\left|\operatorname{STFT}(x)(t,f)\right|^2,
\qquad
c_k(t)=\operatorname{DCT}_k\!\left(\log(E_m(t)+\epsilon)\right).
$$

Here, $H_m$ is the m-th linear-frequency filter and $\epsilon>0$ stabilizes the logarithm. LFCC uses a linear frequency scale, whereas MFCC uses a mel scale. Normalization, retained coefficients and frame duration are part of the representation and must be specified. [LFCC definition and parameters, ref. 21](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.transforms.LFCC.html).

Swin Transformers use hierarchical representations and shifted local attention windows. The original architecture addressed computer vision; an ECG adaptation requires a justified mapping from signals to tokens. [Liu et al., ref. 22](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html).

Chen et al. (2024) applied Swin to wavelet time–frequency maps for arrhythmia detection. Their study is evidence for **wavelet–Swin**, not for the exact LFCC–Swin combination and not for OMI detection. [Chen et al., ref. 14](https://doi.org/10.3389/fcvm.2024.1401143).

**Proposed LFCC–Swin hypothesis:** local attention over cepstral sequences may learn useful temporal combinations while reducing input size. However, cepstral coefficient index is not the same as physical frequency or an image coordinate. A temporal attention model with coefficients and leads as channels should be compared with any two-dimensional tokenization.

Several information losses deserve explicit ablation. Magnitude spectra discard phase; retaining only part of the cepstrum loses spectral detail; per-record normalization may remove absolute voltage information. Whether these changes damage OMI discrimination is an empirical question. A waveform-preserving branch is a candidate control.

LFCC also has precedent in ECG biometric identification: Rabhi and Lachiri (2014) used cepstral information for person recognition. That task is distinct from disease detection. Its relevance here is a cautionary inference: a useful identity representation could facilitate leakage if one person's windows occur in both training and testing. [Rabhi and Lachiri, ref. 23](https://thesai.org/Downloads/Volume5No4/Paper_31-Human_Recognition_System_using_Cepstral_Information.pdf).

No primary clinical study establishing the exact LFCC–Swin combination for OMI was verified in this targeted search. It should be presented as a candidate approach; absence from this search does not establish that it is the first such method.

### 4.4 Quantum and hybrid quantum–classical classification

A variational quantum classifier (VQC) encodes a feature vector into a quantum state, applies trainable gates, measures observables and converts those measurements into predictions. The feature extractor can remain classical. Thus, VMD–VQC and WST–VQC are proposed hybrid pipelines whose representation and classifier contributions must be evaluated separately. [Circuit-centric classifiers, ref. 24](https://doi.org/10.1103/PhysRevA.101.032308).

A recent ECG example is Al-Shamasneh et al. (2026), who studied a quantum-enhanced convolutional approach and quantum generative augmentation on MIT-BIH and PTB-XL. Their methods describe simulated noisy execution using PennyLane's mixed-state simulator. This provides an experimental quantum-model example, but not evidence of clinical OMI performance or a hardware speed advantage. [Al-Shamasneh et al., ref. 25](https://doi.org/10.1186/s13677-026-00918-1).

The scientific challenge is to identify an advantage attributable to the quantum component. Appropriate controls include the same compressed features with logistic regression, SVMs and small neural networks; comparisons before and after compression; and circuit ablations. Resource reporting should include qubits, encoding, depth, measurements, optimization, preprocessing and actual execution hardware. Simulating a circuit on a classical computer does not demonstrate quantum acceleration.

Trainability is also architecture-dependent. McClean et al. established vanishing-gradient limitations for classes of randomly initialized quantum circuits; this is not a claim that every small VQC is untrainable. [McClean et al., ref. 26](https://doi.org/10.1038/s41467-018-07090-4).

Bowles et al.'s 2024 benchmarking preprint found that classical models often performed strongly against the tested quantum models. It supports demanding classical controls, rather than a universal conclusion about all quantum learning. The source is a preprint and should be identified as such. [Bowles et al., ref. 31](https://arxiv.org/abs/2403.07059).

## 5. How much error do existing methods have?

### 5.1 Define the error before comparing it

For binary classification, with true positives $TP$, true negatives $TN$, false positives $FP$ and false negatives $FN$:

$$
\begin{aligned}
\text{Overall error} &= \frac{FP+FN}{TP+TN+FP+FN}=1-\text{accuracy},\\
\text{False-negative rate (FNR)} &= \frac{FN}{TP+FN}=1-\text{sensitivity},\\
\text{False-positive rate (FPR)} &= \frac{FP}{TN+FP}=1-\text{specificity},\\
\text{False-discovery rate (FDR)} &= \frac{FP}{TP+FP}=1-\text{PPV}.
\end{aligned}
$$

These denominators differ. FNR measures the proportion of positive cases missed; FPR measures the proportion of negative cases incorrectly flagged; FDR measures the proportion of positive predictions that are wrong. None equals a mortality rate.

AUROC measures ranking across thresholds. Neither $1-\mathrm{AUROC}$ nor $1-F1$ is an overall misclassification rate. For multiclass and multilabel tasks, the averaging rule and definition of accuracy must be supplied before interpreting a complement as an error.

### 5.2 Published errors, with their evaluation context

Values labelled “derived” are arithmetic complements of the papers' rounded metrics, not new evaluations.

| Study and method | Target and evaluation context | Published metric | Error interpretation |
|---|---|---|---|
| de Chazal et al., 2004: morphology/interval features and discriminants [10] | MIT-BIH heartbeat classes; separate development and assessment recordings | SVEB sensitivity 75.9%, FPR 4.7%; VEB sensitivity 77.7%, FPR 1.2% | Derived class-specific FNR: **24.1% SVEB**, **22.3% VEB** |
| Nahak et al., 2023: WST–SVM [20] | Seventeen-class fragment classification; patient separation not established from the accessible abstract | Accuracy 98.90% | Derived reported overall error **1.10%**; independent-patient error remains unverified |
| Chen et al., 2024: wavelet–Swin [14] | MIT-BIH arrhythmia classification; intra-patient and grouped inter-patient analyses | Accuracy 99.34% and 98.37%, respectively | Derived errors **0.66%** and **1.63%**; inter-patient accuracy CI 97.75–98.99% corresponds to error CI **1.01–2.25%** |
| Du et al., 2026: median-waveform CNN [8] | ACS release's OMI test task | Accuracy 86.1%; sensitivity 69.7%; specificity 87.3%; PPV 27.7% | Derived overall error **13.9%**, FNR **30.3%**, FPR **12.7%**, FDR **72.3%** |
| Gustafsson et al., 2026: deep ECG model [15] | Acute TIMI-0 OMI versus all other classes in two internal SwED tests | C-statistic ≥0.95; sensitivity 87% at FPR 5% | Derived **FNR 13%** at that operating point; AUROC does not provide overall error |

Sources: [de Chazal](https://doi.org/10.1109/TBME.2004.827359), [Nahak](https://doi.org/10.1016/j.eswa.2023.120019), [Chen](https://doi.org/10.3389/fcvm.2024.1401143), [Du](https://doi.org/10.1038/s41597-026-07278-0), and [Gustafsson](https://doi.org/10.1038/s41467-026-73023-1). SVEB and VEB mean supraventricular and ventricular ectopic beats, respectively.

These rows are an evidence map, not a ranking. They differ in labels, populations, numbers of classes, evaluation units, input representations and thresholds. In particular, low arrhythmia error cannot be transferred to OMI, and a performance difference across separate papers cannot isolate a representation's effect.

External InCor C-statistics were 0.83 for OMI versus controls, 0.96 for OMI-STEMI and 0.82 for OMI-NSTEMI. These comparisons differ from the internal one-versus-all task. Endpoint composition and population changes therefore prevent attributing the difference solely to domain shift. [Gustafsson external validation, ref. 15](https://doi.org/10.1038/s41467-026-73023-1).

### 5.3 A direct comparison of older criteria and an OMI model

Herman et al.'s Table 4 compares methods within one cohort using maximum pre-angiography evidence per patient contact and validation-selected thresholds. Error complements below are derived. [Herman et al., ref. 13](https://doi.org/10.1093/ehjdh/ztad074).

| Method | Accuracy | Sensitivity | Specificity | Derived overall error | Derived FNR | Derived FPR |
|---|---:|---:|---:|---:|---:|---:|
| STEMI criteria | 83.6% | 32.5% | 97.7% | 16.4% | 67.5% | 2.3% |
| Expert ECG interpretation | 90.8% | 73.0% | 95.7% | 9.2% | 27.0% | 4.3% |
| OMI AI, optimal threshold | 90.9% | 80.6% | 93.7% | 9.1% | 19.4% | 6.3% |
| OMI AI, approximately STEMI-matched specificity | 90.6% | 65.8% | 97.5% | 9.4% | 34.2% | 2.5% |

At the optimal threshold, FNR was 48.1 percentage points lower than with STEMI criteria, while FPR was 4.0 points higher. AI sensitivity had a reported 95% CI of 76.8–84.0%. The matched-specificity row makes the threshold tradeoff clearer. This observational result does not establish reduced mortality.

### 5.4 Why accuracy alone can mislead

For prevalence $\pi$, sensitivity $Se$ and specificity $Sp$:

$$
\text{Overall error}=\pi(1-Se)+(1-\pi)(1-Sp),
\qquad
\mathrm{PPV}=\frac{\pi Se}{\pi Se+(1-\pi)(1-Sp)}.
$$

As an **illustrative calculation**, not a dataset result, assume 5% prevalence, 80% sensitivity and 95% specificity among 1,000 people. Expected counts are 40 true positives, 10 false negatives, 47.5 false positives and 902.5 true negatives. Accuracy is 94.25%, but PPV is only approximately 45.7%. A model that always predicts negative would achieve 95% accuracy while missing every positive case.

A comparative paper should consequently report confusion counts, sensitivity, specificity, PPV, NPV, precision–recall performance and calibration alongside accuracy. For rare OMI, sensitivity at a prespecified false-positive burden is a more informative primary comparison than whichever threshold maximizes accuracy.

## 6. Research gaps between earlier methods and the proposed approach

The following synthesis identifies questions for a new study. It does not imply that every earlier paper has every limitation, or that a new architecture automatically resolves them.

| Research gap | Limitation in some earlier or current designs | Candidate response | Evidence required |
|---|---|---|---|
| Representation choice | A fixed feature set may miss relevant temporal or spectral information | Compare morphology, WST, VMD and LFCC under identical conditions | Paired predictions on the same patients, with matched classifier controls |
| Patient dependence | Beats, windows or leads from one person can cross partitions | Group all observations by patient before any learned preprocessing | Auditable patient separation; uncertainty calculated at patient level |
| Source–diagnosis confounding | A disease class may identify its acquisition cohort | Test transfer to independently collected cohorts containing comparable classes | Performance within and across sources; sensitivity to device and cohort shifts |
| Diagnostic mismatch | Rhythm, STEMI, acute TIMI 0, TIMI 0/1 and broader adjudicated OMI are different targets | Define the reference standard before model development | Label-adjudication details, timing and sensitivity analyses under alternative definitions |
| Information loss | Compression, pooling and normalization may discard useful signal structure | Add waveform controls and vary each processing stage separately | Morphology-preservation checks and downstream error changes |
| Limited supervision | A record-level label does not locate abnormalities within a record | Compare recording-level pooling with weakly supervised aggregation | Performance on held-out patients; interval-level validation where available |
| Rare-positive errors | High accuracy can coexist with clinically important misses or many false alerts | Optimize and evaluate prespecified sensitivity–specificity tradeoffs | FNR, FPR, PPV, PR curves, calibration and workload implications |
| Cross-site transfer | Internal performance may depend on device, population or workflow | External and temporal evaluation; controlled pretraining comparisons | Site-specific estimates and uncertainty, with endpoint differences documented |
| Quantum attribution | A hybrid system can improve because of classical features or tuning | Compare the same representation with quantum and classical heads | Repeated-seed paired comparisons, encoding/compression ablations and resource costs |
| Clinical benefit | Better retrospective classification does not establish better care | Prospectively evaluate how predictions affect decisions | Time to treatment, unnecessary procedures, safety and patient outcomes |

The source-confounding concern follows directly from ECGData's construction. Recent foundation-model and external OMI studies show why it remains relevant even at much larger scale. [Dataset sources, refs. 3–7](https://github.com/mathworks/physionet_ECG_data); [Khattak et al., ref. 16](https://doi.org/10.1371/journal.pdig.0001623); [Gustafsson et al., ref. 15](https://doi.org/10.1038/s41467-026-73023-1).

The unresolved methodological question is not whether newer components can be assembled. It is whether their contribution survives rigorous controls and independent evaluation. VMD, scattering and Swin are established methods; an application combining them must specify what new hypothesis, capability or evidence it contributes.

## 7. A defensible research framework

### 7.1 Research question

**Do adaptive and multiscale ECG representations improve patient-independent classification and clinically relevant OMI sensitivity at a fixed false-positive burden, and does any improvement remain after controlling for classifier capacity, information compression, tuning and computational cost?**

ARR/CHF/NSR and OMI should be treated as separate tasks with separate models and conclusions. ECGData can support exploratory representation comparisons, while ACS/OMI addresses a different clinical endpoint. Performance across the two tasks should not be combined into a single “ECG accuracy.”

### 7.2 Proposed comparisons

| Purpose | Candidate comparisons |
|---|---|
| Establish a conventional baseline | Morphology/interval features with regularized linear models, SVMs and tree ensembles |
| Isolate representation effects | WST, VMD descriptors and LFCC summaries with the same classical classifiers |
| Isolate attention effects | LFCC with a simple temporal model versus LFCC–Swin; wavelet–Swin as a separate representation |
| Preserve waveform information | Raw-signal CNN/ResNet; LFCC–Swin with and without a waveform branch |
| Isolate quantum effects | Identical compressed WST/VMD features with VQC, logistic regression, SVM and a small neural network |
| Study pretraining | From-scratch and pretrained encoders under the same downstream split, with pretraining overlap checked |

This is a set of controlled alternatives, not a requirement to concatenate every transform into one system. Separate representation and classifier comparisons make a negative result interpretable: a useful transform may not require a complex classifier, and a good classifier may not benefit from a proposed transform.

### 7.3 Evaluation requirements

1. **Define the endpoint and unit.** Specify beat, window, recording or patient prediction. For OMI, state the angiographic definition, acute-versus-chronic distinction and interval between ECG and the reference assessment.
2. **Separate patients before fitting.** Keep repeated ECGs, leads and all derived windows together. Fit normalization, feature selection, dimensionality reduction, augmentation, tuning and thresholds within the development partitions.
3. **Freeze the final assessment.** Use nested patient-grouped validation where needed. Keep final test patients outside architecture selection, and identify any external pretraining overlap.
4. **Use matched comparisons.** Evaluate candidate systems on the same eligible patients with comparable tuning opportunities. Give classical controls both the compressed quantum input and, separately, their best justified uncompressed representation.
5. **Report variability.** Include every planned initialization seed, confidence intervals and paired differences. Use patient-cluster resampling where observations repeat; a large number of windows must not imply a large independent sample.
6. **Account for failures.** Report signal exclusions, transform non-convergence, missing leads, abstentions and coverage. A fallback decision rule should be declared before final testing.
7. **Evaluate useful operating points.** Report class-specific errors and, for OMI, sensitivity at prespecified specificity, PPV, PR-AUC, calibration and subgroup performance. Threshold selection must precede final test evaluation.
8. **Measure total cost.** Include signal processing, feature selection, training and inference; for quantum models, include encoding, shots and execution backend. Distinguish offline preprocessing from clinical latency.
9. **Test transfer and clinical relevance.** Evaluate temporal and external cohorts under compatible definitions. Prospective clinical assessment remains a separate stage from retrospective model comparison.

The resulting contribution could be evidence that one representation is more robust, that a simpler method is sufficient, or that an apparently promising component adds no measurable benefit. None of those outcomes should be excluded by selecting only favourable seeds or datasets.

## 8. Synthesis and limitations

The reviewed evidence supports continued investigation of learned ECG representations and more clinically targeted OMI detection. It also supports retaining conventional machine learning as a serious comparator. Reported error depends on the clinical label, unit of analysis, cohort and operating threshold; a single accuracy cannot summarize that evidence.

LFCC–Swin and VMD/WST–VQC provide testable research directions. The exact combinations require controlled validation before claims of improved OMI detection, quantum advantage or clinical benefit. The strongest proposed contribution is a reproducible evaluation that distinguishes representation effects from classifier effects while examining patient independence, information preservation and external transfer.

This narrative review is not exhaustive. Some studies were accessible only through abstracts, and no pooled effect estimate or formal risk-of-bias score is claimed. Published results are not independent replications. Global mortality figures motivate the clinical problem but cannot be used to calculate deaths prevented by the reviewed classifiers.

## References

Web sources were accessed on 11 October 2026. Dataset versions and mortality observation years are stated separately from publication dates.

1. **World Health Organization.** *The top 10 causes of death*. Updated 2 October 2026; estimates for 2023. [WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death).
2. **World Health Organization.** *Cardiovascular diseases (CVDs)*. Updated 31 July 2025; headline mortality estimate for 2022. [WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29).
3. **MathWorks.** *physionet_ECG_data*. Processed ECG distribution and preprocessing attribution. [Official distribution](https://github.com/mathworks/physionet_ECG_data).
4. **MathWorks.** *Signal Classification Using Wavelet-Based Features and Support Vector Machines*. [Official dataset example](https://www.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html).
5. **Moody G, Mark R.** *MIT-BIH Arrhythmia Database*, version 1.0.0. PhysioNet; 2005. [Dataset DOI: 10.13026/C2F305](https://doi.org/10.13026/C2F305).
6. **PhysioNet.** *BIDMC Congestive Heart Failure Database*, version 1.0.0. 2000. [Dataset DOI: 10.13026/C29G60](https://doi.org/10.13026/C29G60).
7. **Moody G.** *MIT-BIH Normal Sinus Rhythm Database*, version 1.0.0. PhysioNet; 1999. [Dataset DOI: 10.13026/C2NK5R](https://doi.org/10.13026/C2NK5R).
8. **Du X, Liu Y, Wang L, et al.** A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs. *Scientific Data*. 2026;13:1009. [DOI: 10.1038/s41597-026-07278-0](https://doi.org/10.1038/s41597-026-07278-0).
9. **Du X, et al.** *A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs*. Figshare; 2026. Version 1. [Data DOI: 10.6084/m9.figshare.29925314.v1](https://doi.org/10.6084/m9.figshare.29925314.v1).
10. **de Chazal P, O'Dwyer M, Reilly RB.** Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Transactions on Biomedical Engineering*. 2004;51(7):1196–1206. [DOI: 10.1109/TBME.2004.827359](https://doi.org/10.1109/TBME.2004.827359).
11. **Al-Zaiti SS, et al.** Machine learning for ECG diagnosis and risk stratification of occlusion myocardial infarction. *Nature Medicine*. 2023;29:1804–1813. [DOI: 10.1038/s41591-023-02396-3](https://doi.org/10.1038/s41591-023-02396-3).
12. **Hannun AY, Rajpurkar P, Haghpanahi M, et al.** Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*. 2019;25:65–69. [DOI: 10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3).
13. **Herman R, Meyers HP, Smith SW, et al.** International evaluation of an artificial intelligence–powered electrocardiogram model detecting acute coronary occlusion myocardial infarction. *European Heart Journal – Digital Health*. 2024;5(2):123–133. Published online 28 November 2023. [DOI: 10.1093/ehjdh/ztad074](https://doi.org/10.1093/ehjdh/ztad074).
14. **Chen S, Wang H, Zhang H, Peng C, Li Y.** A novel method of swin transformer with time-frequency characteristics for ECG-based arrhythmia detection. *Frontiers in Cardiovascular Medicine*. 2024;11:1401143. [DOI: 10.3389/fcvm.2024.1401143](https://doi.org/10.3389/fcvm.2024.1401143).
15. **Gustafsson S, Ribeiro AH, Gedon D, et al.** A deep learning ECG model for identification and localization of occlusion myocardial infarction. *Nature Communications*. 2026;17:4336. [DOI: 10.1038/s41467-026-73023-1](https://doi.org/10.1038/s41467-026-73023-1).
16. **Khattak GR, Patlatzoglou K, Barker J, et al.** Data distribution impacts the performance and generalisability of contrastive learning-based foundation models of electrocardiograms. *PLOS Digital Health*. 2026;5(9):e0001623. [DOI: 10.1371/journal.pdig.0001623](https://doi.org/10.1371/journal.pdig.0001623).
17. **Zhou Z-H.** A brief introduction to weakly supervised learning. *National Science Review*. 2018;5(1):44–53. Published online 2017. [DOI: 10.1093/nsr/nwx106](https://doi.org/10.1093/nsr/nwx106).
18. **Dragomiretskiy K, Zosso D.** Variational mode decomposition. *IEEE Transactions on Signal Processing*. 2014;62(3):531–544. [DOI: 10.1109/TSP.2013.2288675](https://doi.org/10.1109/TSP.2013.2288675).
19. **Mallat S.** Group invariant scattering. *Communications on Pure and Applied Mathematics*. 2012;65(10):1331–1398. [Author manuscript: arXiv:1101.2286](https://arxiv.org/abs/1101.2286).
20. **Nahak S, Pathak A, Saha G.** Fragment-level classification of ECG arrhythmia using wavelet scattering transform. *Expert Systems with Applications*. 2023;224:120019. [DOI: 10.1016/j.eswa.2023.120019](https://doi.org/10.1016/j.eswa.2023.120019).
21. **PyTorch contributors.** *torchaudio.transforms.LFCC*. TorchAudio 2.7 documentation. Technical definition, not clinical validation. [LFCC reference](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.transforms.LFCC.html).
22. **Liu Z, Lin Y, Cao Y, et al.** Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. *Proceedings of ICCV*. 2021:10012–10022. [Original conference paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html).
23. **Rabhi E, Lachiri Z.** Human recognition system using cepstral information. *International Journal of Advanced Computer Science and Applications*. 2014;5(4):220–223. [Original paper](https://thesai.org/Downloads/Volume5No4/Paper_31-Human_Recognition_System_using_Cepstral_Information.pdf).
24. **Schuld M, Bocharov A, Svore KM, Wiebe N.** Circuit-centric quantum classifiers. *Physical Review A*. 2020;101:032308. [DOI: 10.1103/PhysRevA.101.032308](https://doi.org/10.1103/PhysRevA.101.032308).
25. **Al-Shamasneh AR, Karim FK, Wang Y.** Quantum-enhanced ECG analysis for reliable and secure arrhythmia detection in medical IoT systems. *Journal of Cloud Computing*. 2026;15:90. [DOI: 10.1186/s13677-026-00918-1](https://doi.org/10.1186/s13677-026-00918-1).
26. **McClean JR, Boixo S, Smelyanskiy VN, Babbush R, Neven H.** Barren plateaus in quantum neural network training landscapes. *Nature Communications*. 2018;9:4812. [DOI: 10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4).
27. **Moody GB, Mark RG.** The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50. [DOI: 10.1109/51.932724](https://doi.org/10.1109/51.932724).
28. **Baim DS, Colucci WS, Monrad ES, et al.** Survival of patients with severe congestive heart failure treated with oral milrinone. *Journal of the American College of Cardiology*. 1986;7(3):661–670. [DOI: 10.1016/S0735-1097(86)80478-8](https://doi.org/10.1016/S0735-1097%2886%2980478-8).
29. **Goldberger AL, Amaral LAN, Glass L, et al.** PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215–e220. [DOI: 10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).
30. **Pollard T, Moody BE, Lehman L, et al.** PhysioNet as a global platform for biomedical research. *Nature Health*. 2026;1:792–795. [DOI: 10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).
31. **Bowles J, Ahmed S, Schuld M.** Better than classical? The subtle art of benchmarking quantum machine learning models. 2024. **Preprint**. [arXiv:2403.07059](https://arxiv.org/abs/2403.07059).
