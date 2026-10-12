# ECGData.mat and ACS/OMI: data structure, research context, citations and global mortality

Research date: **11 October 2026 (America/Toronto)**. Local verification occurred on 12 October UTC, which was still 11 October in Toronto. Mortality estimates below refer to their stated observation years, not to deaths counted during 2026.

## 1. Scope and principal findings

`ECGData.mat` supports classification of arrhythmia (ARR), congestive heart failure (CHF) and normal sinus rhythm (NSR). The ACS dataset supports several clinical labels; this project's selected task is **occlusion myocardial infarction (OMI) versus non-OMI**. These are different prediction problems. Neither local dataset supplies the survival outcomes and population denominator needed to estimate worldwide mortality.

The report combines direct inspection of the local files, the project's existing source and patient-identity audits, original dataset documentation and World Health Organization (WHO) sources. A new [verification record](ecgdata_acs_omi_data_review_2026-10-11_verification.json) records local file hashes, dimensions, class counts and patient separation. No model was trained or evaluated for this review.

| Property | ECGData.mat | ACS dataset, Figshare version 1 |
|---|---|---|
| Local prediction task | ARR / CHF / NSR | OMI / non-OMI |
| Full source size | 162 single-lead rows from 81 recordings | 19,955 twelve-lead ECG recordings |
| Independent patient identities | 80 | 18,909 |
| Signal length | 65,536 samples per row | Normally 5,000 samples per lead; two source records have 3,500 |
| Sampling frequency | 128 Hz after source preprocessing | Native 500 Hz |
| Duration | 512 seconds per row | Normally 10 seconds; two records are 7 seconds |
| Storage | MATLAB structure with `Data` and `Labels` | CSV metadata and WFDB-style `.hea` / `.dat` pairs |
| Patient identifiers | Recovered and verified in a separate project mapping | Supplied as `Patient_id` |
| Public mortality follow-up in the inspected files | Absent | Absent |

Counts and exceptions in this table are supported by the new local verification and the existing [ECGData identity report](../../../architects/patient_vmd_wst_vqc.md), [ACS download audit](acs_download_verification_2026-09-23.md) and [ACS loader audit](acs_loader_2026-09-24.md).

## 2. ECGData.mat

### 2.1 Origin and research population

The MathWorks distribution repackages signals from three PhysioNet databases. Its archive attribution describes physical scaling, resampling to 128 Hz, separation of the two source channels into individual rows, and truncation to 65,536 samples. The resulting data are a processed derivative, not the complete original databases. See the [MathWorks repository](https://github.com/mathworks/physionet_ECG_data), its `ECGData.zip` attribution file, and the [MathWorks dataset example](https://www.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html).

| Label | Original source | Rows in MAT file | Source recordings | Patients | Original acquisition context |
|---|---|---:|---:|---:|---|
| ARR | MIT-BIH Arrhythmia Database | 96 | 48 | 47 | Approximately half-hour, two-channel ambulatory recordings; originally 360 Hz |
| CHF | BIDMC Congestive Heart Failure Database | 30 | 15 | 15 | Approximately 20-hour, two-channel recordings from severe CHF patients; originally 250 Hz |
| NSR | MIT-BIH Normal Sinus Rhythm Database | 36 | 18 | 18 | Long-term recordings from subjects without significant arrhythmias |
| **Total** | Three separate source cohorts | **162** | **81** | **80** | All derivative rows have the same 128 Hz rate and 512-second duration |

Source descriptions: [MIT-BIH ARR v1.0.0](https://physionet.org/content/mitdb/1.0.0/), [BIDMC CHF v1.0.0](https://physionet.org/content/chfdb/1.0.0/) and [MIT-BIH NSR v1.0.0](https://physionet.org/content/nsrdb/1.0.0/). Row, recording and patient counts were also recalculated from the verified local mapping.

The CHF source contains 11 men and 4 women with NYHA class III–IV disease. The NSR source contains 5 men and 13 women referred to an arrhythmia laboratory. These are selected clinical cohorts; NSR should not be interpreted as a comprehensive certification of good health. [CHF documentation](https://physionet.org/content/chfdb/1.0.0/), [NSR documentation](https://physionet.org/content/nsrdb/1.0.0/).

### 2.2 Exact local structure

Direct inspection confirmed the following representation:

```text
ECGData.mat
└── ECGData                     1 × 1 MATLAB structure
    ├── Data                    162 × 65,536 double/float64 values
    └── Labels                  162 × 1 cell array of class strings

MATLAB row ranges:
    1–96       ARR
    97–126     CHF
    127–162    NSR

Project identity mapping, stored separately:
    ecgvmd/ecgdata_subjects.csv
    row → source database → source recording → channel → patient_id
```

Each matrix row is one lead signal. A column is a sample in time. The row duration is `65,536 / 128 = 512 seconds`, or 8 minutes 32 seconds. All inspected waveform values were finite. The MAT structure itself contains no patient ID, age, sex, death indicator or survival time; the project recovers source/patient identities separately. The loader's 128 Hz setting is supported by the distribution documentation, rather than a sampling-frequency field in the MAT structure.

The local [subject mapping](../../../ecgvmd/ecgdata_subjects.csv) and [loader](../../../ecgvmd/data.py) establish row identity using waveform hashes. This review successfully rechecked all 162 row hashes. The earlier source-matching work compared separate signal excerpts against PhysioNet, as documented in the [patient verification report](../../../architects/patient_vmd_wst_vqc.md).

**The 162 rows are not 162 independent patients.** In particular, MIT-BIH records 201 and 202 belong to one person. Both channels, both recordings where applicable, and every derived window must remain together during splitting. [MIT-BIH patient documentation](https://physionet.org/content/mitdb/1.0.0/).

### 2.3 What research this dataset supports

The corrected VMD/WST baseline uses 10 sampled, nonoverlapping 500-sample windows per row: **1,620 windows**, each **3.90625 seconds**, still drawn from only **80 patients**. Window sampling and five patient folds use seed 0. These are experiment-specific windows; they are not the native shape of the dataset. Other project studies use different windows and must retain their own protocols. [Corrected baseline protocol and report](../../../architects/patient_vmd_wst_vqc.md).

This dataset is useful for controlled signal-representation experiments, but its labels also identify which source database supplied the signal. A classifier can therefore exploit acquisition or cohort differences alongside disease-related morphology. Patient separation addresses patient leakage; it does not remove this source/diagnosis confounding. ARR is a broad source-cohort label, not an atrial-fibrillation label or a beat-by-beat diagnosis. The MAT file also does not retain the original databases' full annotation resources.

Accordingly, results should describe ARR/CHF/NSR classification on this selected cohort. They cannot establish OMI detection, global disease prevalence, survival prediction or mortality reduction. Historical row-grouped project scores should not be reported as patient-independent results. [Project source and evaluation audit](../../../architects/scattering_audit_2026-09-21.md).

## 3. ACS/OMI dataset

### 3.1 Origin and target definition

Du and colleagues assembled pre-angiography ECGs at the First Affiliated Hospital of Chongqing Medical University, China, from December 2015 to May 2024. The paper describes adults aged 19–99 and an ECG inclusion window of seven days before angiography. This is a selected hospital population, not a population survey. [Dataset paper](https://doi.org/10.1038/s41597-026-07278-0).

The paper defines OMI using **acute culprit-vessel occlusion with TIMI flow grade 0 or 1 before PCI**. It acknowledges that this does not include a broader adjudicated definition covering some spontaneously reperfused cases. OMI and STEMI are therefore different labels. [Definition and limitations](https://doi.org/10.1038/s41597-026-07278-0).

This project uses the supplied binary `OMI` field. `OMI=0` means non-OMI under that annotation rule, not a healthy heart. No new diagnosis is inferred from other metadata. [Project OMI protocol](../protocols/acs_omi_protocol_v1.md).

### 3.2 Files and waveform structure

The pinned data citation is [Figshare version 1, DOI 10.6084/m9.figshare.29925314.v1](https://doi.org/10.6084/m9.figshare.29925314.v1). Local archive identities and file layout are documented in the [download verification report](acs_download_verification_2026-09-23.md) and its [saved official Figshare metadata](acs_download_verification_2026-09-23_source.json).

```text
experiments/acs_omi_vmd_wst_vqc/data/
├── CSV.zip
│   └── CSV/
│       ├── train.csv            17,960 records; 28 columns
│       └── test.csv              1,995 records; 6 columns; no disease labels
└── ECG_row_data.zip
    └── row_data/
        ├── <record_id>.hea      Sampling, channels, gain/baseline metadata
        └── <record_id>.dat      Multiplexed signed 16-bit waveform samples

Optional source archive, not used in this project:
    ECG_median_data.zip          Median-waveform .med records

Eligible decoded ECG:
    shape = (5,000 time samples, 12 leads)
    sampling = 500 Hz
    duration = 10 seconds
    units = mV
```

The project returns leads in this order: **I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6**. It converts digital samples using `(digital value − baseline) / gain`. One recording becomes one observation; its twelve leads are combined as feature columns. The twelve leads are not twelve independent patients. [Validated loader](../loader.py), [loader audit](acs_loader_2026-09-24.md).

### 3.3 Metadata and labels

Direct inspection of `CSV.zip` confirmed these fields:

| Field group | Column names | Role in research |
|---|---|---|
| Patient and files | `Patient_id`, `ecg_row_record`, `ecg_med_record` | Group patients and join records to waveform files |
| Demographics and timing | `gender`, `age`, `Time_Interval` | Metadata; `Time_Interval` is ECG-to-angiography time in minutes, not survival follow-up |
| Diagnosis labels, training only | `AMI`, `OMI`, `CTO`, `NSTEMI`, `STEMI`, `UA` | Separate binary targets |
| Treatment/history flags, training only | `PCI`, `VF_VT`, `Paced`, `Prior_PCI` | Additional annotations; excluded from this project's waveform predictors |
| Vessel annotations, training only | `LM`, `PLAD`, `MLAD`, `DLAD`, `DB`, `PLCX`, `MLCX`, `DLCX`, `OM`, `PRCA`, `MRCA`, `DRCA` | Coronary-segment annotations; some values are category 2, so these are not all binary |

Timing meaning: [dataset documentation](https://doi.org/10.1038/s41597-026-07278-0). Schema and permitted label values: [local verification](ecgdata_acs_omi_data_review_2026-10-11_verification.json), [loader audit](acs_loader_2026-09-24.md).

AMI = acute myocardial infarction; CTO = chronic total occlusion; NSTEMI/STEMI = non-ST-elevation/ST-elevation myocardial infarction; UA = unstable angina; PCI = percutaneous coronary intervention. The diagnosis fields overlap and must not be added as if they were mutually exclusive classes.

| Official training label, before project exclusions | Positive ECGs | Negative ECGs |
|---|---:|---:|
| AMI | 2,679 | 15,281 |
| OMI | 1,151 | 16,809 |
| CTO | 957 | 17,003 |
| NSTEMI | 1,235 | 16,725 |
| STEMI | 1,442 | 16,518 |
| UA | 6,213 | 11,747 |

These counts were recalculated from the local training CSV. **OMI prevalence here is 1,151 / 17,960 = 6.41% of training ECGs**, not a patient-level prevalence, population incidence or mortality rate. Disease labels are absent from the public test CSV, so full-release positive counts cannot be reconstructed from these files.

### 3.4 Source cohort, exclusions and project split

| Partition | ECGs | Unique patients | OMI-positive ECGs | OMI-negative ECGs |
|---|---:|---:|---:|---:|
| Official training, before exclusions | 17,960 | 17,018 | 1,151 | 16,809 |
| Official test, before exclusions | 1,995 | 1,891 | Withheld | Withheld |
| **Full release** | **19,955** | **18,909** | Not publicly available for all records | Not publicly available for all records |
| Eligible development cohort | 17,905 | 16,967 | 1,146 | 16,759 |
| Project fit subset | 14,324 | 13,576 | 917 | 13,407 |
| Project internal validation | 3,581 | 3,391 | 229 | 3,352 |
| Eligible reserved test subset | 1,993 | 1,889 | Withheld | Withheld |

The rows describe nested populations and must not be summed. Official counts and development split counts were rechecked locally; eligibility counts come from the completed [full loader audit](acs_loader_2026-09-24.md). That audit found **55 training exclusions**: 50 records with a flat lead, three with a missing-sample marker and two short records. Two test records have flat leads. These are exclusions under the original **all-twelve-lead policy**, not a universal exclusion rule for every later study.

The official release also stores channel extrema in WFDB header fields normally used for the initial value and checksum. The project's loader explicitly checks this source convention. The original files remain unchanged. Both local ZIP SHA-256 hashes still match the earlier verified source identities; waveform quality was not rescanned for this review. [Source-format audit](acs_download_verification_2026-09-23.md).

The saved development split uses seed **20260924** and groups by patient. It is one fit/validation split produced from a five-fold grouping procedure, not five completed validation runs. The new check found zero patient overlap between fit, validation and official test. **114 development patients have different OMI labels across their ECGs**; their record-level labels are retained and all their ECGs stay in one partition. [Preparation report](acs_omi_preparation_2026-09-24.md), [new verification](ecgdata_acs_omi_data_review_2026-10-11_verification.json).

An always-negative prediction reaches `3,352 / 3,581 = 93.605%` internal-validation accuracy with **zero OMI sensitivity**. This explains the protocol's emphasis on average precision, sensitivity, specificity and patient-cluster uncertainty alongside accuracy. All preprocessing and model selection belong inside development partitions; the official test remains reserved. [OMI protocol](../protocols/acs_omi_protocol_v1.md).

## 4. Worldwide mortality and its relationship to these datasets

### 4.1 Verified global estimates

| Mortality measure | Observation year | Worldwide estimate | Source publication/update | Interpretation |
|---|---:|---:|---|---|
| **All causes of death** | **2023** | **About 61 million deaths** | WHO, 2 October 2026 | Total deaths across causes; not a cardiovascular-only total. [WHO causes of death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death) |
| **All cardiovascular diseases (CVDs)** | **2022** | **About 19.8 million deaths; approximately 32% of global deaths that year** | WHO, 31 July 2025 | Broad cardiovascular burden, not deaths attributed solely to ARR, CHF or OMI. [WHO CVD fact sheet](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29) |
| **Ischaemic heart disease (IHD)** | **2023** | **About 9.5 million deaths; approximately 16% of global deaths** | WHO, 2 October 2026 | Relevant context for coronary disease research; not an ACS-only or OMI-only count. [WHO causes of death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death) |
| Stroke | 2023 | About 6.8 million deaths | WHO, 2 October 2026 | Relevant to the broader CVD burden; neither dataset has a stroke classification target. [WHO causes of death](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death) |

The WHO CVD fact sheet also attributes **85% of CVD deaths to heart attack and stroke combined**. This combined percentage must not be described as the percentage due to heart attack alone, ACS or OMI. [WHO CVD fact sheet](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29).

These are rounded annual estimates. They are not a live 2026 death count. The cited fact sheets do not provide uncertainty intervals alongside these headline values. The CVD estimate and the other rows refer to different years and releases: do not divide the 2022 CVD count by the 2023 all-cause count, or add overlapping cause categories.

**Source freshness:** the live WHO causes-of-death page retrieved for this report is dated 2 October 2026 and reports 2023 estimates. Search previews still displayed an older 2021 summary. The opened page, rather than that stale preview, supplies the figures above. Its publication year must not be confused with its mortality observation year.

### 4.2 What can be said about ARR, CHF, ACS and OMI mortality

| Dataset label or condition | Appropriate mortality interpretation |
|---|---|
| ARR in ECGData | This broad label is not a single globally tabulated cause of death. A number for atrial fibrillation, ventricular arrhythmia or sudden cardiac death cannot automatically stand for the whole ARR cohort. |
| CHF in ECGData | Deaths among people with heart failure, deaths involving heart failure and deaths assigned to an underlying disease are different measures. A cohort survival percentage is not an annual global death count. |
| NSR in ECGData | NSR is a rhythm category, not a cause of death or a zero-risk survival label. |
| ACS | Coronary mortality provides context, but the global IHD total includes more than acute coronary presentations. It cannot be relabelled as an ACS death total. |
| OMI | No directly applicable worldwide OMI-specific annual death total was identified in the sources reviewed. The dataset's angiographic definition cannot be substituted for the broad IHD category. |

These distinctions follow from the inspected dataset labels and the difference between clinical conditions and mortality coding. WHO tabulates an underlying cause selected from the chain of conditions leading to death; counting every associated condition as an independent death would double-count people. [WHO cause-of-death framework](https://www.who.int/standards/classifications/classification-of-diseases/cause-of-death). WHO also distinguishes the broader IHD condition from acute ACS presentations. [WHO heart-attack explanation](https://www.who.int/news-room/fact-sheets/detail/heart-attack).

The original Baim et al. CHF publication concerns a treatment cohort and survival outcomes. It provides historical context for the BIDMC source, but those outcomes are not supplied as per-patient survival labels in `ECGData.mat` and do not estimate worldwide CHF mortality. [Original CHF publication](https://pubmed.ncbi.nlm.nih.gov/3950244/).

### 4.3 Research implications

Global cardiovascular mortality motivates research into ECG-based classification. A classification study measures a different outcome from death prevention. Demonstrating mortality prediction would require defined follow-up, vital status, time-to-event and censoring information; demonstrating that a model reduces deaths would additionally require evaluation of its use in clinical care.

For the current project, report patient-separated classification performance, every initialization seed, classical controls, exclusions and uncertainty. Do not convert accuracy, sensitivity, OMI prevalence or dataset sample counts into deaths prevented. Cross-dataset accuracy comparisons also cannot isolate the value of a transform when labels, populations, leads, durations and sampling rates differ.

## 5. How to cite the data correctly

For **ECGData.mat**, acknowledge the MathWorks derivative and its preprocessing, cite all three original PhysioNet datasets, and cite their associated publications. Cite the project's identity audit separately when reporting the recovered 80-patient structure. `ECGData.mat` is a filename, not a complete bibliographic citation.

For **ACS/OMI**, cite both the **versioned Figshare data DOI** and the **Scientific Data descriptor**. Describe OMI as the selected label in the ACS release; do not imply that this project collected a separate OMI dataset. Report project exclusions and patient partitions as project decisions, with links to the corresponding audits.

### Dataset and methodological references

1. **MathWorks.** *physionet_ECG_data* [processed ECG data distribution], including `ECGData.zip` and its `Modified_physionet_data.txt` attribution. [Official repository](https://github.com/mathworks/physionet_ECG_data). See also *Signal Classification Using Wavelet-Based Features and Support Vector Machines*: [official example](https://www.mathworks.com/help/wavelet/ug/ecg-classification-using-wavelet-features.html).

2. **Moody G, Mark R.** *MIT-BIH Arrhythmia Database*, version 1.0.0. PhysioNet; 2005. [Dataset DOI: 10.13026/C2F305](https://doi.org/10.13026/C2F305).

3. **PhysioNet.** *BIDMC Congestive Heart Failure Database*, version 1.0.0. Published 14 October 2000. [Dataset DOI: 10.13026/C29G60](https://doi.org/10.13026/C29G60). The source page does not name a dataset author; this entry uses the repository as the corporate source.

4. **Moody G.** *MIT-BIH Normal Sinus Rhythm Database*, version 1.0.0. PhysioNet; 1999. [Dataset DOI: 10.13026/C2NK5R](https://doi.org/10.13026/C2NK5R).

5. **Moody GB, Mark RG.** The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*. 2001;20(3):45–50. [DOI: 10.1109/51.932724](https://doi.org/10.1109/51.932724).

6. **Baim DS, Colucci WS, Monrad ES, et al.** Survival of patients with severe congestive heart failure treated with oral milrinone. *Journal of the American College of Cardiology*. 1986;7(3):661–670. [DOI: 10.1016/S0735-1097(86)80478-8](https://doi.org/10.1016/S0735-1097%2886%2980478-8).

7. **Goldberger AL, Amaral LAN, Glass L, et al.** PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215–e220. [DOI: 10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215). This is the historical PhysioNet citation included with the MathWorks derivative.

8. **Pollard T, Moody BE, Lehman L, et al.** PhysioNet as a global platform for biomedical research. *Nature Health*. 2026;1:792–795. [DOI: 10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z). The current PhysioNet source pages request this platform citation in addition to dataset-specific attribution.

9. **Du X, et al.** *A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs*. Figshare; 2026. Version 1. [Data DOI: 10.6084/m9.figshare.29925314.v1](https://doi.org/10.6084/m9.figshare.29925314.v1). The preserved official metadata records publication on 9 July 2026.

10. **Du X, Liu Y, Wang L, et al.** A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs. *Scientific Data*. 2026;13:1009. [Paper DOI: 10.1038/s41597-026-07278-0](https://doi.org/10.1038/s41597-026-07278-0).

### Mortality references

11. **World Health Organization.** *The top 10 causes of death*. Updated 2 October 2026; mortality estimates for 2023. [WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death).

12. **World Health Organization.** *Cardiovascular diseases (CVDs)*. Updated 31 July 2025; headline CVD mortality estimate for 2022. [WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29).

13. **World Health Organization.** *Cause of death*. [Mortality coding and underlying-cause framework](https://www.who.int/standards/classifications/classification-of-diseases/cause-of-death).

14. **World Health Organization.** *Heart attack*. Updated 10 September 2026. [Clinical terminology and context](https://www.who.int/news-room/fact-sheets/detail/heart-attack). Its older IHD mortality figure is not the source of the 2023 estimate in this report.

## 6. Verification record and remaining limits

The [machine-readable verification](ecgdata_acs_omi_data_review_2026-10-11_verification.json) records the local MAT and ZIP SHA-256 hashes, the patient-map hash, the saved ACS split hash, schemas, class counts and grouping checks. It rechecked ECGData waveform identity and ACS metadata/split arithmetic. Existing full-waveform quality audits were reused after verifying ACS archive identity; completed feature extraction and model fitting were not repeated.

External sources were consulted on the research date above. Live Figshare page/API access failed through the research browser, so the exact version, archive identities and dataset citation were checked against the preserved official API response from the September download audit. The publisher paper was accessible. This access limitation does not alter the local data counts, which were independently recalculated.

The report is a dataset and literature review. It creates no new model score, global mortality estimate, survival analysis or claim of quantum advantage. Further OMI-specific global mortality work would require a compatible epidemiological definition and an appropriate population source, neither of which can be recovered from these ECG classification files.
