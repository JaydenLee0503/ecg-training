# Cepstral extraction and temporal Swin — architecture record

Update: the separate [ECGData preparation record](ecgdata_attention_preparation.md)
now documents a completed CPU feature cache and smaller three-class model with
an explicit-only training runner. No fitting has run. The ACS configuration
described below remains unchanged; its earlier pending-work statements are
specific to that implementation.

Recorded 2026-09-25. **Both components are implemented; no model training has
started.** Code lives in the user-requested `attention method/` folder. The default
task is ACS OMI versus non-OMI. Existing VMD/WST implementations and frozen
experiments remain unchanged.

## Built components

| Component | Implementation | Purpose |
|---|---|---|
| Cepstral extractor | [cepstral.py](../attention%20method/attention_ecg/cepstral.py) | Framewise LFCC, with optional MFCC |
| Temporal Swin classifier | [model.py](../attention%20method/attention_ecg/model.py) | Hierarchical shifted-window attention over cepstral frames |
| Data and normalization | [data.py](../attention%20method/attention_ecg/data.py) | Read-only ACS adapter, pinned patient split, fit-only statistics |
| Engineering check | [check.py](../attention%20method/check.py) | Synthetic forward inference without weight updates |
| Regression checks | [test_attention.py](../attention%20method/tests/test_attention.py) | Numerical reference, attention boundaries and data guards |
| Declared settings | [protocol.json](../attention%20method/protocol.json) | Input, features, model, split identity and future evaluation plan |

## Signal-processing path

The native input is a 10-second, 500 Hz, 12-lead ECG in mV, with the lead order
I, II, III, aVR, aVL, aVF, V1–V6. No additional filtering, resampling,
pre-emphasis, detrending or per-record amplitude normalization is introduced.

1. Frame each lead with 500 samples and a 125-sample hop: 37 frames per ECG.
2. Apply a periodic Hann window and a 512-point real FFT.
3. Calculate one-sided power spectral density, normalized by sampling rate and
   window energy, doubling non-DC/non-Nyquist bins.
4. Apply 40 unit-sum triangular bands over 0–100 Hz.
5. Take the natural logarithm with a fixed power floor of `1e-12`.
6. Apply orthonormal DCT-II and retain coefficients c0 through c19.

The feature tensor is **[37 time frames, 12 leads, 20 coefficients]**. Linear
frequency spacing (LFCC) is the declared default; mel spacing (MFCC) is an
implemented alternative requiring its own experiment settings. General input
lengths use right reflection padding to cover the final sample; the native ACS
length needs none. Frame positions and padding are returned as metadata.

Normalization accumulates a mean and population standard deviation for each
lead/coefficient over fit frames only. Standard deviations below `1e-8` become
1. Validation records cannot fit these statistics. This retains temporal order
instead of pooling the cepstra before the model.

## Classifier path

This is a **1-D adaptation of Swin**, not a pretrained image model or replication
of a particular ECG publication. One time frame is one patch, with its 240
lead/coefficient values projected to 64 channels.

| Stage | Tokens for default input | Channels | Blocks | Attention heads |
|---|---:|---:|---:|---:|
| Frame embedding and stage 1 | 37 | 64 | 2 | 2 |
| Pairwise merging and stage 2 | 19 | 128 | 2 | 4 |
| Pairwise merging and stage 3 | 10 | 256 | 2 | 8 |

Each stage alternates ordinary windows of 8 tokens with a shift of 4 tokens.
Blocks include learned relative position bias, LayerNorm, residual connections
and a GELU MLP. Explicit padding and key masks prevent artificial circular
connections between record endpoints. Odd-length pairwise merging pads one
token. Final normalization, temporal mean pooling and a linear head produce two
class logits. The model has **2,175,718 parameters**.

Hierarchy expands the temporal receptive field; it does not apply dense global
attention at every layer. Calls may have different sequence lengths, but all
records within a batch must have the same real frame count. Model weights are
random, and its outputs are not diagnostic predictions.

## Data boundaries and provenance

The adapter reuses the existing verified ACS loader and the frozen OMI split
read-only: **14,324 fit and 3,581 validation ECGs**, with no patient overlap.
The split CSV SHA-256 is
`d036f1eff7c350638e1e7e738d1ded6d017b4c7b62380a8debf89e89a3c5faba`.

Checks reject official test entries, duplicate records, patient crossings,
waveform/label identity changes, incompatible sampling/lead layouts and excluded
records. Raw data and old experiment outputs are not copied or overwritten.

## Verification actually completed

The recorded implementation run passed **11 tests in 0.392 seconds**, with no
failures or skips:

- Cepstral values compared against SciPy periodograms and an independently
  constructed cosine matrix; gain response, zeros, padding and mel spacing checked.
- Shifted attention connects neighboring windows without wrapping endpoints;
  padded keys are excluded.
- Finite outputs checked for frame counts 1, 7, 8, 9, 37 and 40; batch independence,
  model state save/load round-trip and unchanged weights verified.
- Patient, partition, identity and normalization guards tested with fixtures.
- Existing split checksum/counts checked read-only; code defaults match protocol.
- Synthetic end-to-end inference produced finite `[1, 2]` logits from
  `[37, 12, 20]` features, without changing weights.

These are engineering checks, not predictive validation or proof of absence of
bugs. Evidence and code hashes are in
[validation.json](../attention%20method/reports/validation.json), with details in
the [implementation report](../attention%20method/reports/implementation.md).
This architecture-record update did not rerun computation.

The existing interpreter supplies Python 3.12.3, NumPy 2.5.2 and SciPy 1.18.1.
CPU PyTorch 2.6.0+cpu was installed separately in `/tmp/ecg-attention-deps` after
the initial sandbox network attempt failed. The original environment was not
modified. Temporary dependencies may disappear after restart; reproduction
commands are in the [experiment README](../attention%20method/README.md).

## Pending work and interpretation

No real ECG feature extraction, optimizer, training loop, gradient update, trained
checkpoint, official-test prediction or model-quality metric has been produced.
GPU execution is unverified. Real-data engineering checks, resumable extraction,
persistent preprocessing/model packaging, classical controls, training and
statistical reporting remain pending. **Do not start training under the current
user instruction.**

Before a future run, declare optimizer/epoch/tuning budgets, retain seeds 0/1/2,
and save all splits, exclusions, source/code identities, settings, runtimes,
histories, predictions and failures. The plan uses OMI average precision as the
primary metric and paired patient-cluster uncertainty, with matched classical
controls and reserved official test patients.

Cepstral truncation loses spectral detail; magnitude processing loses phase and
polarity. Low-order cepstra describe spectral shape and are not explicit RR/HRV
measurements. Neither superiority of cepstra over conventional descriptors nor
Swin over CNNs/RNNs has been demonstrated here.

Architectural references used during implementation:
[original Swin paper](https://arxiv.org/abs/2103.14030),
[official Swin implementation](https://github.com/microsoft/Swin-Transformer),
and [SciPy DCT documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html).
