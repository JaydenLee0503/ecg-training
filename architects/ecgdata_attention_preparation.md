# ECGData cepstral + Swin preparation — 2026-09-25

The user approved preparing the ECGData attention experiment while keeping the
original ACS data intact and leaving training unstarted. Preparation is complete.

| Component | Built or verified |
|---|---|
| Input adapter | Existing verified ECGData loader; 128 Hz, single lead per window, ARR/CHF/NSR |
| Matched comparison | Exact original 1,620 standardized windows and five patient folds; hashes and identities checked |
| Cepstral representation | 128-sample Hann / hop 32 / FFT 256; 32 linear bands, 0–64 Hz; c0–c15 |
| Prepared feature shape | 1,620 × 13 × 1 × 16; 162 saved row checkpoints |
| Compact temporal Swin | Widths 16/32, depths 2/2, heads 2/4, window 4/shift 2; **33,607 parameters** |
| Data safeguards | Outer-training-only normalization; patient separation; output writes restricted to new attention directories |
| Training implementation | Explicit command/flag only; 40 fixed epochs, seeds 0/1/2; epoch checkpoints, histories and saved predictions |
| Classical control | Training-only scaling and logistic regression over mean/std cepstra; implemented, not fitted |
| Reporting implementation | Per-seed window/patient-vote metrics; paired patient bootstrap; original VMD/WST predictions reused |
| Current execution status | CPU cache and forward-only checks complete; **no training or GPU work** |

The protocol uses the fixed architecture and epoch budget, with no outer-label
selection. Future hyperparameter tuning needs a separate inner-patient protocol.
This dataset contains 80 patients and has informed previous research decisions;
the experiment is exploratory, not independent external validation.

## Files

- [ECGData guide](../attention%20method/ECGDATA.md): commands, architecture and execution boundaries.
- [Frozen protocol](../attention%20method/ecgdata_protocol.json): exact settings and reference hashes.
- [CLI](../attention%20method/ecgdata.py): separate prepare, verify, dry-run, train and report actions.
- [Adapter/cache](../attention%20method/attention_ecg/ecgdata.py): exact window/fold reuse and resumable CPU extraction.
- [Training runner](../attention%20method/attention_ecg/ecgdata_training.py): future fitting, preprocessing and checkpoints.
- [Report runner](../attention%20method/attention_ecg/ecgdata_report.py): saved-prediction validation and uncertainty.
- [Tests](../attention%20method/tests/test_ecgdata.py): safeguards and numerical/report checks without model fitting.
- [Preparation report](../attention%20method/reports/ecgdata_preparation.md): completion evidence and limitations.

Generated cache: `attention method/features/ecgdata_lfcc_v1/` (gitignored).
Future training output: `attention method/results/ecgdata_lfcc_swin_v1/`
(not created during preparation).

## Evidence and pending work

Extraction completed in **5.981 seconds**. All **21 tests passed**, and a CPU
forward-only check on real fit-window features returned finite `[2,3]` logits
without changing weights. Completed reuse, partial-cache recovery, and corruption
rejection were exercised. No optimizer step or classifier fitting occurred.

Checksums confirmed **134 protected files unchanged**, including the original ACS
archives/settings/source, ECGData MAT, old comparison artifacts and original
attention ACS configuration and modules. See the
[preservation inventory](../attention%20method/reports/ecgdata_preservation.json)
and [validation record](../attention%20method/reports/ecgdata_validation.json).

The existing GPU workload was left untouched. CUDA compatibility/performance,
actual fitting, real training-checkpoint recovery and trained reporting remain
unverified. Do not start training under the current instruction. When training is
later requested, verify the intended device/environment and preserve all seeds,
negative results, exclusions, predictions and uncertainty. No model quality or
superiority over other feature/classifier methods has been established.
