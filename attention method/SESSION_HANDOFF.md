# Attention method handoff — 2026-09-25

**User explicitly said not to start training. No training has run.**

## Latest: ECGData preparation

The user approved adapting the attention pipeline to `ECGData.mat`, building its
CPU feature cache and preparing the training runner while leaving ACS intact.
See [ECGDATA.md](ECGDATA.md), [ecgdata_protocol.json](ecgdata_protocol.json) and
[preparation report](reports/ecgdata_preparation.md). The cache contains the exact
1,620 corrected baseline windows and patient folds, represented as 13 × 1 × 16
LFCC values each. The new compact model uses widths 16/32 and three outputs.

The explicit-only ECGData runner now implements future 15 Swin and five logistic
fits, checkpoints, saved predictions and patient-cluster reporting. Its optimizer
path has not run. Preparation and verification are separate commands and do not
start fitting. Keep the current GPU workload undisturbed. The original ACS
protocol and adapter are unchanged. The paragraphs below describe earlier ACS
implementation status; they do not negate the new ECGData runner.

## Earlier ACS implementation

Feature extraction and the temporal Swin model are implemented and pass 11 CPU
checks. Start with [README.md](README.md), [protocol.json](protocol.json), and the
[implementation report](reports/implementation.md). Defaults are LFCC sequences
for 12-lead, 500 Hz, 10-second ACS ECGs, with binary OMI output. All model weights
remain random; there is no trained checkpoint or predictive performance.

The existing split was verified by hash and patient boundaries, not regenerated.
Actual ACS signals have not been extracted for this method. Original VMD/WST
experiments and their frozen artifacts remain unchanged. This separate folder
name/location follows the user's explicit request.

Temporary CPU PyTorch dependencies are at `/tmp/ecg-attention-deps`; the existing
project interpreter supplies NumPy/SciPy. Commands in README run synthetic inference
and tests only. No executable training entry point or scheduled job exists.

Pending for a later training request: real-data engineering verification, resumable
extraction with provenance, persistent normalization packaging, declared training
budget, training runner and classical controls, full prediction/uncertainty reporting.
Keep official test patients reserved and retain all seeds 0/1/2 and failures.
