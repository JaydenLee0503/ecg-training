# Cepstral features + temporal Swin Transformer

**ECGData preparation added:** the separate [ECGData guide](ECGDATA.md) describes
the 128 Hz, single-lead, three-class protocol, completed CPU LFCC cache, smaller
Swin and explicit-only training/report runner. No model training has started.
The ACS settings below remain intact; their pending runner descriptions refer
to ACS, not the new ECGData implementation.

Separate experiment requested by the user. **No training has started.** This
directory contains the implemented feature extractor, randomly initialized model,
ACS adapter, normalization, synthetic checks, and a declared representation.
There is no optimizer, training loop, trained checkpoint, or diagnostic score.
The default task is the current ACS OMI versus non-OMI task. Generic feature/model
APIs can support other tasks only under a new declared input/split protocol.

## Pipeline

```text
10 s ECG, 500 Hz, 12 leads in mV
  -> 1 s periodic-Hann frames, 0.25 s hop (37 frames)
  -> 512-point FFT -> one-sided power spectral density
  -> 40 unit-sum triangular bands from 0 to 100 Hz
  -> natural log with floor 1e-12 -> orthonormal DCT-II
  -> retain c0 through c19: [time=37, leads=12, cepstra=20]
  -> normalize each lead/coefficient using FIT records only
  -> frame embedding: 240 channels -> 64
  -> regular/shifted temporal attention, window 8 / shift 4
  -> three stages: dimensions 64/128/256, two blocks each
  -> pairwise temporal merging: 37 -> 19 -> 10 tokens
  -> layer normalization -> mean pooling -> two class logits
```

The default is **linear-frequency cepstral coefficients (LFCC)**. Optional
`CepstralConfig(spacing='mel')` implements mel-frequency cepstral coefficients
(MFCC). Mel spacing is a speech-inspired alternative, not established here as
optimal for ECG. Changing spacing is a separate experiment, not a way to select
the best result using final-test patients. Defaults are fixed engineering choices,
not tuned or scientifically validated settings.

No speech pre-emphasis, detrending, resampling, or signal z-scoring is added.
All 12 lead channels and c0 are retained. Filter-bank rows are normalized to unit
sum; power uses Hann-energy and sampling-rate normalization, with doubled
non-DC/non-Nyquist bins. For general input lengths, the last frame is right
reflection-padded to cover the end; the native ACS length needs no padding.
Band endpoints are triangular zeros, so retaining c0 does not directly retain DC.

## What the attention model means

This is a **1-D temporal adaptation of Swin**, not a pretrained image Swin or a
reproduction of an ECG paper. One cepstral frame is one patch; lead/coefficient
coordinates are channels, not image pixels. Each stage alternates ordinary and
shifted windows and includes learned relative position bias, LayerNorm, residual
connections, and a GELU MLP. Pairwise merging expands temporal receptive fields;
the final pooling summarizes the whole record. It does not implement dense global
attention at every layer. Explicit padding/masking prevents circular attention
between the beginning and end of the ECG. Variable frame counts are supported
across calls; a batch must have a single true frame count without batch padding.

The source architecture uses shifted windows and hierarchy to connect local
regions efficiently: [Swin paper](https://arxiv.org/abs/2103.14030) and
[official implementation](https://github.com/microsoft/Swin-Transformer).
The cepstral transform uses [SciPy's orthonormal DCT-II](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html).

## Run checks, without training

Run from the repository root; quote the directory because its name contains a space.

```bash
/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/check.py' --features-only
```

The existing environment did not contain PyTorch. CPU PyTorch 2.6.0 was installed
in `/tmp/ecg-attention-deps` solely for development checks; it is temporary and
does not modify the original environment. While it exists:

```bash
PYTHONPATH=/tmp/ecg-attention-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/check.py'
PYTHONPATH=/tmp/ecg-attention-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s 'attention method/tests' -v
```

For a durable setup, create a separate virtual environment and install the
declared `requirements.txt`. A CPU-only install can use the official PyTorch CPU
index for `torch==2.6.0`; a future GPU environment needs its own tested installation
and recorded manifest. GPU execution has not been verified for this model.
Without PyTorch, feature/data tests run and model tests explicitly skip.

`check.py` uses synthetic signals, a fixed initialization seed, `eval()` and
`inference_mode()`. It checks finite output and unchanged parameters, and prints
shapes/settings. It neither reads patient signals nor saves predictions or weights.

## Connecting the existing ACS data

`attention_ecg/data.py` provides:

- `load_splits(path, expected_sha256=...)`: verify the saved split, reject official
  test entries, duplicate records, and patient overlap; never silently re-split.
- `extract_acs_record(record, splits, config)`: consume an `ACSRecord` from the
  existing verified loader, check lead order, sampling rate, eligibility, identity,
  label and waveform checksum, then return coefficients and provenance.
- `FitStandardizer.fit((record_id, features) iterator, splits)`: stream unique fit
  records only, accumulating one mean/std per lead and coefficient.
- `FitStandardizer.transform(features)`: apply frozen statistics to another ECG.

Add this directory and the repository root to Python's import path for API use.
The existing loader is imported read-only; raw archives and original experiment
outputs are not copied or changed. The pinned split in `protocol.json` contains
14,324 fit and 3,581 validation ECGs. The official test cohort stays reserved.
Missing local data/splits are an explicit error; they are gitignored in the parent
study. Real-data extraction and fitting have not been executed for this method.

## Scientific limits and future evaluation

Cepstral coefficients compactly summarize log spectral shape. Truncation and
filter-bank averaging discard information; magnitude power also loses waveform
polarity and phase. Low-order coefficients emphasize the spectral envelope and
are not explicit RR/HRV measurements. A one-second frame cannot resolve long
rhythm patterns by itself. The temporal model may learn patterns across frames,
but whether this preserves the morphology useful for OMI is untested.

Neither cepstral features outperforming conventional features nor Swin outperforming
CNNs/RNNs is an established result in this repo. A future experiment must compare
the same patients and inputs against pooled-cepstral logistic regression, matched
CNN/RNN models, and the existing VMD/WST pipelines once available. Save all seeds
0/1/2, patient assignments, exclusions, feature/scaler states, source/code hashes,
hyperparameters, runtimes, histories, predictions and failed trials. Use average
precision as primary for OMI, full threshold metrics, and paired patient-cluster
bootstrap intervals. Do not infer clinical diagnosis quality from synthetic checks.

Before any future training, implement a separately documented resumable extraction
and training/evaluation runner, freeze optimizer/epoch/tuning budgets for all arms,
and record that execution in a new results directory. This implementation does not
start or schedule that work. See `protocol.json` and `reports/implementation.md`.
