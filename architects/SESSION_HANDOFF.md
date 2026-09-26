# Session handoff — 2026-09-25

## New attention method — implementation only

Latest: the user approved **ECGData preparation, without training**, while keeping
ACS intact. The separate [ECGData architecture record](ecgdata_attention_preparation.md)
documents exact baseline window/fold reuse, the completed CPU LFCC cache, compact
three-class Swin and prepared training/evaluation runner. Original ACS settings
and data remain unchanged. No GPU work or classifier fitting was started.

The consolidated [architecture record](attention_method_architecture.md) documents
both implemented components, exact dimensions/settings, data safeguards, completed
verification, environment setup and pending work. Added at the user's request;
this documentation update did not start extraction or training.

The user requested a separate `attention method` folder for cepstral features and
a Swin Transformer, explicitly **without training**. See its
[handoff](../attention%20method/SESSION_HANDOFF.md) and
[implementation report](../attention%20method/reports/implementation.md).
LFCC/MFCC extraction, a temporal Swin adaptation, read-only ACS integration and
fit-only normalization are implemented. All 11 CPU checks passed; synthetic
forward inference produced finite logits without changing weights. The existing
ACS patient split hash/counts were checked read-only. No real-data extraction,
training or performance evaluation has run for this new method. Original VMD/WST
code/results remain unchanged. Temporary CPU PyTorch lives in
`/tmp/ecg-attention-deps`; the original environment was not modified.

## Current state

The completed VMD/WST experiments are saved and intact. After the user reported an accidental PC shutdown, all **930 files** listed in `results/vqc_improvement/artifact_inventory.csv` were checked against their saved SHA-256 hashes and sizes: **zero missing or changed files**. This check was repeated while creating this handoff. It describes that point in time; recheck if later changes or another interruption make integrity uncertain.

The improvement experiment finished at **2026-09-22 04:19:52 UTC** (00:19:52 EDT). All **90 inner fits and 30 outer fits** completed, with **8,280 epoch records**, **360 inner checkpoints**, and no failed training trials. All 13 regression tests passed before launch. Post-run validation checked patient separation, saved models/checkpoints, selection scores, preprocessing, and prediction coverage.

There is no unfinished training job from that experiment to resume. Process/session IDs from earlier chats are not durable checkpoints.

The active ACS / OMI study has its own workspace at `experiments/acs_omi_vmd_wst_vqc/`.
Read its [session handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) and
[workspace guide](../experiments/acs_omi_vmd_wst_vqc/README.md) before ACS work. It contains
its own code, tests, data, results, protocols, reports, and experiment log.
The original ECGData experiments described below remain in their existing locations.

Latest ACS update: GPU extraction **stopped at 2026-09-26 01:29:28 UTC** with
1,795/17,905 ECGs saved. Record 01985/V5 exhausted the frozen 32,000-iteration
limit. All 1,795 checkpoints passed integrity checks; a separate CPU reference
attempt reproduced the capped result. No production settings changed and the
run has not resumed. Read the [stop report](../experiments/acs_omi_vmd_wst_vqc/reports/gpu_extraction_stop_2026-09-25.md)
before proceeding. Investigate convergence, then document any amended retry
policy in a separate run that reuses verified checkpoints. No ACS model training
has started; earlier running-status/ETA reports are historical.

## Read these reports first

1. [VQC improvement results](vqc_improvement_results.md): final results, all candidate choices, uncertainty, learning curves, saved artifacts, and reproduction commands.
2. [Frozen improvement protocol](vqc_improvement_protocol.md): candidate family, optimizer, nested validation, seeds, and selection rules.
3. [Corrected VMD/WST comparison](patient_vmd_wst_vqc.md): verified patient mapping and original matched VQC baseline.
4. [KNN comparison and training-fit diagnostic](vqc_vs_spar_knn.md): classical controls on the same inputs and limits of the published-paper comparison.
5. [WST numerical audit](scattering_audit_2026-09-21.md): reference checks, source matching, and earlier corrections.
6. [Experiment log](../EXPERIMENT_LOG.md): research chronology. Older entries include superseded results; respect their corrections.

## Results to carry forward

Window metrics below are mean scores across three initialization seeds for each VQC. KNN is deterministic. These are patient-held-out results on the existing ECGData task.

| Front end | Classifier | Window accuracy | Window macro-F1 | Patient-vote accuracy |
|---|---|---:|---:|---:|
| VMD | Original VQC | 67.98% | 0.6489 | 80.00% |
| WST | Original VQC | 64.79% | 0.6192 | 78.33% |
| VMD | Nested compact/reupload VQC | 67.94% | 0.6552 | 77.50% |
| WST | Nested compact/reupload VQC | 63.05% | 0.6011 | 75.83% |
| VMD | Weighted KNN, k=10, inverse distance | 74.38% | 0.7008 | 78.75% |
| WST | Weighted KNN, k=10, inverse distance | 70.86% | 0.6494 | 82.50% |

**The attempted VQC improvement did not establish a better model.** All paired patient-bootstrap intervals for changes from the original VQC include zero. Keep the originals as reference models and retain the unsuccessful experiments. Do not promote the best single seed as the result.

The KNN rows use exactly the same selected 12 angle-scaled features as the VQC. Other KNN variants, including inverse-square weighting and full feature sets, are preserved in the diagnostic report. KNN does not dominate every per-class or patient metric.

The paired intervals for VMD versus WST also include zero. Neither the original fixed-classifier comparison nor the new tuned-pipeline comparison establishes transform superiority. There is no demonstrated quantum advantage.

## Dataset and correctness findings

- ECGData contains 162 lead rows from **81 recordings and 80 patients**, not 162 independent patients. MIT-BIH records 201 and 202 belong to the same patient.
- The source mapping in `ecgvmd/ecgdata_subjects.csv` / `.json` was verified against source ECG excerpts; the loader checks waveform hashes.
- The old row-grouped evaluation leaked patients. Older README/QUANTUM_STAGE numbers are historical exploratory results, not the current reference.
- The corrected experiments use 1,620 windows, five patient folds, 224 VMD descriptors versus 882 standard log-WST features, and training-only mRMR-12 plus angle scaling.
- Standard WST coefficients matched Kymatio exactly on the current 1,620-window configuration. Other reference/edge checks had errors around floating-point precision. A minor time-bin metadata calculation was corrected; the current T=64 coefficients were unaffected. This is numerical evidence, not proof of absence of all bugs.
- VMD convergence retries left zero iteration-capped windows in the corrected feature archive.
- Source snippets were previously downloaded to verify patient identities. They were not a replacement training dataset. The improvement run reused the existing feature archive.
- Class and source-database identity remain confounded in the original ARR/CHF/NSR cohort. The cohort has informed repeated research decisions; it is not untouched external validation.

The paper discussed with the user is **3-D Attractor Reconstruction for Enhanced ECG Classification of Arrhythmia and Congestive Heart Failure**, DOI **10.1109/JSEN.2025.3572080**. Its available abstract reports 94% validation and 93.2% test accuracy using weighted KNN with attractor-derived features. Its full patient split and feature protocol were not verified. Do not claim a replication, directly equate its score with ours, or accuse its authors of leakage.

## What was tested in the improvement run

The original baseline used a 12-qubit, two-layer VQC, 40 epochs, batch size 32, learning rate 0.05, balanced loss, and seeds 0/1/2.

The follow-up encoded all 12 features on six qubits with RY/RZ angles. It tested two blocks with one encoding, two blocks with repeated encoding, and three blocks with repeated encoding. The readout had 24 quantum observables and a linear three-class head, without a raw-feature bypass. Training used Adam with cosine learning-rate decay from 0.02 to 0.002 over 80 epochs.

For each front end and outer fold, three inner patient folds selected architecture and epoch from 20/40/60/80 using pooled inner macro-F1. Selected settings were refitted with three seeds. Both front ends had the same search budget. Outer fits never used test labels for tuning.

Six selections chose one encoding; four chose repeated encoding. Five chose 20 epochs, three 40, one 60, and one 80. More layers or longer training did not consistently help.

## Repository cleanup — 2026-09-22

The root `scratch/` folder was archived to `legacy/scratch/`, preserving both files
byte for byte. The script is historical interactive work; its window cache remains
local and gitignored. Removed 33 rebuildable Python bytecode files and the stale
`ecgvmd_bundle.zip` (34 files, 621,947 bytes total). Rebuild the ZIP with
`make_colab_bundle.py` when needed.

Dataset files, features, saved results, source-verification excerpts, reference
material, and distinct notebook backups were retained. The active package/scripts
and frozen experiment inputs were unchanged. All 930 saved improvement artifacts
were checked after cleanup. Details and per-file hashes are in the
[cleanup record](repository_cleanup_2026-09-22.md) and
[cleanup manifest](repository_cleanup_2026-09-22.json).

## Files and environment

| Location | Purpose |
|---|---|
| `results/patient_vmd_wst_vqc/` | Corrected baseline features, 30 original VQC fits, predictions, and summaries |
| `results/patient_knn_diagnostic/` | Matched KNN predictions and original VQC training-fit diagnostics |
| `results/vqc_improvement/` | All 120 new fits, checkpoints, histories, predictions, environment/source snapshots, and checksums |
| `architects/vqc_improvement_tables/` | Version-controlled result tables, all epoch histories, split assignments, predictions, and artifact inventory |
| `architects/vqc_improvement_figures/` | Comparison and learning-curve PNG/SVG files |
| `ecgvmd/reupload.py` | New experimental compact/reupload classifier |
| `scripts/improve_vqc.py` | Resumable, content-hashed nested experiment runner |
| `scripts/report_vqc_improvement.py` | Saved-model validation, metrics, and paired patient-bootstrap reporting |
| `scripts/archive_vqc_improvement.py` | Histories, predictions, figures, and artifact inventory export |
| `tests/test_reupload_vqc.py` | Circuit, derivative, checkpoint, optimizer, and patient-split tests |

Use the existing interpreter if available:

```bash
/home/jaydenlee/venvs/test-ecg-training/bin/python --version
```

Training used `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and ten worker processes. Versions and code hashes are in `results/vqc_improvement/environment.json` and `manifest.json`.

Large results/features are gitignored. A fresh clone may lack them even when the reports are present. Report missing artifacts honestly; do not silently regenerate a different experiment. Completed fits resume only under the same content-hashed manifest. Use a new output directory for changed code or settings. Documentation/reporting scripts can write files and should not be confused with a read-only integrity check.

## Read-only restart check

From the repository root, this uses only the Python standard library. It verifies completion and every file in the saved artifact inventory without retraining or modifying results:

```bash
python3 -B - <<'PY'
from pathlib import Path
import csv
import hashlib
import json

run = Path("results/vqc_improvement")
status = json.loads((run / "completed.json").read_text())
assert status["status"] == "complete", status
with (run / "artifact_inventory.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

problems = []
for row in rows:
    path = run / row["path"]
    if not path.is_file():
        problems.append((str(path), "missing"))
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
        problems.append((str(path), "changed"))

print(status)
print({"files_checked": len(rows), "problems": problems})
raise SystemExit(bool(problems))
PY
```

If files differ, inspect why before overwriting anything: an intentional later edit and corruption are different explanations. A missing completion marker requires inspection of trial sidecars and logs; the existence of some checkpoints alone does not mean a run completed.

## Current ACS work

Continue in [experiments/acs](../experiments/acs_omi_vmd_wst_vqc/README.md). The
[ACS handoff](../experiments/acs_omi_vmd_wst_vqc/SESSION_HANDOFF.md) records completed patient
splits, the two-record feature check, and the completed 16-record parallel pilot.
Full eight-worker extraction started on 2026-09-25; check the ACS status files
before resuming. The pilot suggests about three days for feature extraction
alone; ACS VQC training time remains unmeasured and no ACS model is fitted.
No completed experiment needs retraining because its files were reorganized.
The relocation preserved saved ACS manifests and all result bytes; the new runner
verifies the recorded import/path migration before resuming the original run.

## Earlier PTB-XL proposal — superseded by ACS preparation

The recommendation was **PTB-XL v1.0.3**, with 21,799 ten-second ECGs from 18,869 patients, using its official patient-separated folds: 1–8 training, 9 validation, 10 final test.

A proposed first task is **atrial fibrillation (AFIB) versus sinus rhythm (SR)**, initially lead II at 100 Hz. Label exclusions, handling of conflicting rhythm statements, and the exact protocol still need to be established. Sinus rhythm is a rhythm label, not a guarantee of overall cardiac health. This task changes the original ARR/CHF/NSR problem; scores would not be directly comparable. Retraining the method on it would test a new task, not externally validate the old three-class fitted model.

Both VMD and WST should receive the same records, leads, durations, patient splits, selection/scaling rules, and declared tuning budget. Update sampling-rate and duration assumptions explicitly; do not apply the old 128 Hz configuration or ECGData-specific source map blindly. Include matched classical controls and multiple VQC initialization seeds. Use training/validation data to establish the protocol and keep the test set untouched until final evaluation.

Chapman–Shaoxing was suggested as an alternative rhythm dataset; the original release covers 10,646 patients. If retaining CHF as a target is essential, reconsider dataset suitability rather than inventing an equivalent CHF label.

Sources checked on 2026-09-22:

- [PTB-XL v1.0.3 and recommended splits](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL label definitions](https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv)
- [Original Chapman–Shaoxing paper](https://www.nature.com/articles/s41597-020-0386-x)

These were earlier options. Continue from the verified ACS preparation described
above and the user's current request, without automatically starting training.
