# Implementation record — 2026-09-25

User requested cepstral features and a Swin Transformer in a separate folder
named `attention method`, explicitly without starting training. This exact root
folder is used as requested; the older ACS-specific location rule does not move
this new experiment into the VMD/WST workspace. Existing scientific code, saved
models, protocols and data were not changed.

Implemented: framewise LFCC with optional MFCC; one-sided PSD, triangular bands,
natural log, orthonormal DCT-II, c0 retention and explicit tail metadata; temporal
Swin with relative bias, alternating shifted/unshifted attention, padding masks,
hierarchical pairwise merging and a binary head; read-only ACS adapter and pinned
patient-split validation; fit-only streaming normalization; synthetic CLI/tests.

Defaults: native 500 Hz, 10 seconds, 12 leads; 500-sample frames, hop 125, FFT 512,
40 bands over 0–100 Hz, 20 coefficients; 37 × 12 × 20 per ECG. Swin dimensions
64/128/256, depths 2/2/2, heads 2/4/8, window 8, shift 4, 2,175,718 parameters.
These settings are declared engineering choices, not optimized findings.

Validation completed:

- 11 tests passed in 0.392 seconds, zero failures or skips, CPU PyTorch 2.6.0+cpu.
- Cepstra compared with SciPy periodograms plus an independently constructed
  cosine transform; amplitude scaling, zeros, mel option and tail padding checked.
- Shifted attention connects neighboring ordinary windows without circular edge
  wraparound; padded keys excluded. Finite outputs checked at lengths 1/7/8/9/37/40.
- Batch independence, state-dictionary round-trip, and unchanged weights checked.
- Patient overlap, duplicate records, official test access, waveform mismatch,
  split checksum mismatch and validation-fitted normalization rejected in fixtures.
- Existing split hash verified read-only: 14,324 fit / 3,581 validation records,
  no patient overlap. Model and feature defaults match `protocol.json`.
- Synthetic full pipeline returned finite [1,2] logits from [37,12,20] features.
  Raw logits were not presented as diagnostic probabilities or performance.

Environment: existing Python 3.12.3 / NumPy 2.5.2 / SciPy 1.18.1. Torch was absent.
The first isolated pip attempt failed due to sandbox DNS restrictions. The
escalated retry installed CPU Torch and dependencies under `/tmp/ecg-attention-deps`.
The original virtual environment was unchanged. This temporary dependency folder
may disappear after restart. No GPU test or real ECG feature extraction was run.
Machine-readable checks, versions and code hashes are in `validation.json`.

No optimizer, gradient update, training loop, trained model, cohort feature cache,
test prediction or model-quality metric was produced. Real-data feature validation,
resumable extraction, persistent preprocessing/model packaging, classical controls,
training and statistical evaluation remain future work. A future execution must
declare optimizer/epochs/tuning budgets and retain all seeds and failures. No
superiority over traditional descriptors, CNNs or RNNs has been demonstrated.
