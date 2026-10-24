# Experiment catalog

[Open the clickable experiment index](FILES.html).

As of **23 October 2026**, there are **9 completed main classification studies**
(6 ECGData and 3 ACS/OMI), plus **2 completed feasibility assessments that failed
their gates before classifier training**: **11 recent studies in total**.

There are also **25 historical numbered log entries (E1–E25)** and **5 older
unnumbered baseline/audit groups**. Those include setup checks, failed probes and
row-grouped comparisons, so adding them to the recent studies would not give a
meaningful count of completed, independent classification experiments. E18b is
part of E18. Two implementation/protocol-only proposals are listed separately.

One study includes its model arms, seeds, folds and retries. The matched KNN
diagnostic counts as a separate completed control study; its reused VQCs are not
new fits. The sensitivity study's saved-score and calibration parts count once.
The 5 unnumbered groups organize the existing workflows, not every notebook cell.

## Completed classification studies

| ID | Experiment | Status |
|---|---|---|
| ECG-01 | [Patient-grouped VMD versus WST with VQC](ecgdata/01_patient_vmd_wst_vqc/README.md) | complete |
| ECG-02 | [Matched KNN control and VQC training-fit diagnostic](ecgdata/02_matched_knn_control/README.md) | complete |
| ECG-03 | [Nested VQC architecture and training improvement](ecgdata/03_vqc_improvement/README.md) | complete |
| ECG-04 | [LFCC and temporal Swin baseline](ecgdata/04_lfcc_swin_v1/README.md) | complete |
| ECG-05 | [Thirty-second EMD/Hilbert morphology classifiers](ecgdata/05_emd_hilbert_30s/README.md) | complete |
| ECG-06 | [LFCC/Swin epoch selection and dropout](ecgdata/06_lfcc_swin_regularization_v2/README.md) | complete |
| ACS-01 | [ACS/OMI VMD versus WST with VQC and classical controls](acs_omi_vmd_wst_vqc/studies/01_vmd_wst_vqc/README.md) | complete |
| ACS-02 | [Balanced OMI training and LFCC/Swin comparison](acs_omi_vmd_wst_vqc/studies/02_balanced_training_and_swin/README.md) | complete |
| ACS-03 | [OMI sensitivity thresholds and calibration](acs_omi_vmd_wst_vqc/studies/03_sensitivity_calibration/README.md) | complete |

## Feasibility studies

Both assessments finished, but neither produced a trained classifier.

| ID | Experiment | Status |
|---|---|---|
| ACS-P01 | [All-lead EMD/Hilbert feasibility pilot](acs_omi_vmd_wst_vqc/studies/04_emd_hilbert_all_lead_pilot/README.md) | gate failed no classifier |
| ACS-P02 | [Lead-I EMD/Hilbert feasibility pilot](acs_omi_vmd_wst_vqc/studies/05_emd_hilbert_lead_i_pilot/README.md) | gate failed no classifier |

## Older work and proposals

- [Historical ECGData experiments and diagnostics](historical_ecgdata/README.md): E1–E25 and the 5 unnumbered groups.
- [ECGData learning-curve proposal](ecgdata/drafts/learning_curve/README.md): not run.
- [Original ACS attention implementation](acs_omi_vmd_wst_vqc/studies/drafts/original_attention_implementation/README.md): synthetic checks only under its initial protocol; subsequent balanced training is ACS-02.
- The larger-Swin/representation follow-up has not started and is not counted.

## Folder layout

```text
experiments/
  ecgdata/                         6 completed study folders; drafts separately
  acs_omi_vmd_wst_vqc/studies/      3 completed classifier studies and 2 pilot folders
  historical_ecgdata/              one folder per E-number or older workflow group
  catalog.json                    machine-readable inventory and canonical paths
```

Each study folder contains its own README and a clickable FILES.html page linking
to its report, protocol, code and saved artifacts where available. Original files stay at their
existing paths: several runners derive data/output paths from their source
locations, and frozen manifests record source hashes and artifact paths. This
organization adds a common browsing layout without invalidating those records.
Shared code and inputs appear in more than one folder without being duplicated.

Use the original report's commands from the repository root. Check completion
markers before running anything. No experiment was retrained for this catalog.
The [organization verification](organization_2026-10-23.json) records the checks
performed for this change; it is not a new scientific validity audit.

Large models, predictions, feature caches and raw data remain gitignored. The
indexes do not bundle them into Git. Fresh clones need the original artifacts;
the ordinary relative links work without Git symlink support. Symbolic-link
creation was unavailable on the Windows-backed filesystem, including an elevated
attempt, so no symbolic links were created.

Literature reviews, download checks, software tests and reproduction notebooks
are supporting records, not additional classifier experiments. Historical
row-grouped scores must not be described as patient-independent results.
