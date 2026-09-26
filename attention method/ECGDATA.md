# ECGData LFCC + small temporal Swin

This is a separate protocol from the original ACS setup. **Preparation is
complete; training has not started.** The original ACS raw archives, protocols,
adapter and model defaults are intact. No GPU was used for preparation.

## Frozen scientific settings

Reuse the corrected `results/patient_vmd_wst_vqc/` inputs: 1,620 windows, 162 lead
rows, 81 recordings and 80 patients, at 128 Hz. Each window is 500 samples, with
10 windows sampled per lead row using seed 0. Each window is z-scored exactly as
in the original comparison. Raw `.mat` waveform hashes are checked by the
existing loader; regenerated windows must match the saved baseline SHA-256.
Labels, lead-row identities, source identities and patient identities must align
with the baseline archive and fold CSV. The five outer patient folds are reused
byte-for-byte, not generated again.

Each single-lead window becomes **13 × 1 × 16** LFCC values:

| Setting | Value |
|---|---|
| Frame/hop | 128 / 32 samples: 1 s / 0.25 s |
| Hann/FFT | Periodic Hann / 256 points |
| Triangular bands | 32, linear spacing, 0–64 Hz |
| Cepstra | c0 through c15, orthonormal DCT-II |
| Log floor | Natural log, fixed power floor `1e-12` |
| Last frame | 12 reflected samples; starts and padding recorded |
| Normalization | Per coefficient, fitted on outer-training windows only |
| Model | Single-lead, 3 outputs: ARR, CHF, NSR |
| Swin stages | Widths 16/32, depths 2/2, heads 2/4 |
| Attention | Window 4, alternating shift 0/2; 13 -> 7 tokens |

The compact model is deliberately smaller than the ACS default. It has not been
selected using any new prediction score. Input representation and classifier
budgets differ from the VQC, so this is a complete-pipeline comparison.

The future fit protocol is fixed at 40 epochs, batch 32, AdamW (learning rate
0.001, weight decay 0.01), cosine decay to 0.0001, gradient norm clipping at 1,
and training-label-derived balanced cross entropy. Seeds are 0/1/2. There is no
architecture search, epoch selection or early stopping in v1, so no inner
validation is needed for selection. Future tuning would need inner patient folds
and a new protocol; outer predictions must not select settings.

Control: concatenate temporal means and population standard deviations of the
16 cepstra (32 features), then training-only StandardScaler and balanced logistic
regression (`C=1`, `lbfgs`, max 2,000 iterations). It has five deterministic fits;
Swin has 15 fits. **Neither control nor Swin has been fitted.**

## Prepared artifacts and commands

Run from the repository root. Preparation only needs the existing interpreter:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/ecgdata.py' prepare
/home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/ecgdata.py' verify
```

The completed cache is `attention method/features/ecgdata_lfcc_v1/`:

- `manifest.json`: protocol, source/data/reference hashes and numerical versions.
- `shards/`: 162 per-lead-row checkpoints, each with 10 feature windows, checksum,
  input hash, frame metadata and extraction runtime.
- `features.npz`: features and aligned labels, patients, recordings, lead rows,
  outer folds, original sample offsets and window IDs.
- `completed.json`: completion state and checksums for every saved artifact.

Completed runs verify/reuse their cache. Interrupted preparation reuses verified
shards. Changed inputs/code/settings require a new cache directory. Corruption
raises an error instead of silently recomputing a different experiment. The cache
is gitignored; the checked-in report records its identity but is not a backup.

For CPU forward inference only, while temporary PyTorch dependencies exist:

```bash
PYTHONPATH=/tmp/ecg-attention-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B 'attention method/ecgdata.py' dry-run
```

This reads two outer-training windows, uses fit-only normalization and random
weights, runs inference on CPU, and verifies that weights did not change. It
prints shapes, not diagnostic predictions. No optimizer or gradient update runs.

## Training runner, prepared for later use

The `train` subcommand is implemented but **must not be run yet under the current
instruction**. It requires both an explicit `--execute-training` flag and a
`--device cpu` or `--device cuda` choice; there is no default training action or
automatic GPU fallback. CPU-only PyTorch at `/tmp/ecg-attention-deps` cannot train
on CUDA. A later GPU run needs a separately verified CUDA-enabled environment
compatible with the RTX 5070, plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`. No new GPU
installation or workload was started.

Default future outputs go only to
`attention method/results/ecgdata_lfcc_swin_v1/`. Output guards reject original
data/experiment directories. Training holds an exclusive writer lock and pins
the cache, protocol, execution code, software versions and device. It saves
per-epoch model/optimizer/scheduler/RNG checkpoints, loss histories, per-fold
preprocessing, training/test indices, class weights, probabilities, predictions,
fit runtimes, warnings and failures. Completed fits are verified and reused.
Interrupted Swin fitting resumes at the last committed epoch, with deterministic
per-epoch shuffling. A checkpoint/marker mismatch stops for inspection; it is not
claimed to automatically recover a partially written checkpoint. Logistic fits
restart if interrupted before completion.

After a later successful training run, the separate `report` subcommand verifies
all 20 fits and exactly-once out-of-fold coverage, then calculates per-seed window
and patient-majority-vote metrics. Ties use ARR/CHF/NSR class order. It includes
paired comparisons to the saved original VMD/WST VQC predictions, without refitting
them. Primary metric is window macro-F1. Paired patient-cluster intervals use
2,000 draws (seed 20260925), averaging seeds inside each draw. They condition on
saved predictions and exclude retraining uncertainty. Single-class draws are
counted and skipped. This repeatedly studied cohort is not external validation.

Full optimizer execution, GPU execution, interruption during real fitting and
end-to-end trained reporting remain unverified because no training was allowed.
Unit tests cover partitioning, fit-only normalization, deterministic shuffle,
checkpoint serialization/corruption, write guards and metric/bootstrap logic
without fitting a classifier. See the
[preparation report](reports/ecgdata_preparation.md) and
[protocol](ecgdata_protocol.json).
