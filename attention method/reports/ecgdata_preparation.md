# ECGData attention preparation — 2026-09-25

**Completed: adapter, frozen protocol, CPU feature cache, compact Swin configuration,
and explicit-only training/report runners. No training started. ACS is intact.**

The user authorized these preparations while the GPU was occupied and requested
that the original ACS data remain intact. All new implementation and outputs are
under `attention method/`; architecture documentation is under `architects/`.

The adapter verified the original ECGData waveform/patient map and regenerated
the exact 1,620 standardized, 500-sample windows used in the corrected VMD/WST
comparison. Their SHA-256 is
`3eaffccfb4f7390ea677937bd233285d1a918f0810da527b3634566b952f9af9`.
The saved five-fold patient assignment hash is
`1d88466d8e2050de0f05fdf3f51b05bdbdd791da114f5e477ffeb472bcc650f5`.
Labels, patients, source recordings, lead rows and fold CSV entries agree with
the original feature archive. No new exclusions or resampling were introduced.

CPU feature extraction completed in **5.981 seconds**, producing
**[1,620 windows, 13 frames, 1 lead, 16 LFCC coefficients]**. The 162 per-row
checkpoints retain original window positions, per-row runtimes, frame starts,
right padding and checksums. The cache is
`attention method/features/ecgdata_lfcc_v1/`. Its local completion marker and
manifest are referenced by hash in [validation evidence](ecgdata_validation.json).

Settings are in [ecgdata_protocol.json](../ecgdata_protocol.json): 128 Hz input;
128-sample Hann frames, hop 32, FFT 256, 32 linear triangular bands to 64 Hz,
16 cepstra including c0. The final frame uses 12 reflected samples. The model
has widths 16/32, two blocks per stage, heads 2/4, window 4/shift 2, three output
classes and **33,607 parameters**. It remains randomly initialized.

Verification completed:

- All **21 tests passed in 0.318 seconds**, with no failures or skips. This includes
  the 11 original attention checks and 10 new ECGData checks, without fitting a
  classifier or executing an optimizer step.
- CPU forward-only inference on two real outer-training feature windows produced
  finite `[2,3]` logits and left weights unchanged. No diagnostic scores saved.
- A repeated `prepare` reused the completed cache. A separate instrumented check
  proved completed reuse did not call feature extraction.
- A temporary partial cache containing one completed row resumed by reusing that
  row and extracting the remaining 1,610 windows; it then rejected an intentionally
  corrupted shard. The temporary test cache was removed. Details are in
  [cache checks](ecgdata_cache_checks.json).
- Training without its explicit flag is rejected before accessing the cache or
  model. Unit tests cover patient separation, training-only normalization,
  deterministic shuffling, checkpoint serialization and corruption checks,
  exclusive writers, protected output paths, metrics and paired bootstrap logic.
- SHA-256/size checks before and after preparation found **zero changes in 134
  protected files**: both raw ACS ZIPs, its scientific/execution protocols and
  active Python source, the original ECGData MAT, the original comparison files,
  and the original attention ACS protocol/extractor/model/adapter. Evidence is in
  [preservation inventory](ecgdata_preservation.json). Active ACS generated results
  were not frozen or modified by this work.
- CLI help and `git diff --check` passed.

The future runner is fixed-budget: 15 Swin fits (five outer folds × seeds 0/1/2),
40 epochs each, and five deterministic pooled-cepstral logistic fits. No model or
epoch selection occurs on outer patients. It saves normalization, model and
optimizer state, shuffle/RNG state, histories, class weights, split identities,
probabilities, predictions, timings, warnings and failures. Reporting verifies
coverage and compares against both logistic and saved original VQC predictions,
with per-seed metrics and paired patient-cluster intervals.

**Not executed:** any classifier fit, optimizer/gradient update, CUDA operation,
GPU package installation, full trained reporting or model-performance evaluation.
The original CPU-only temporary Torch installation was reused for tests/inference.
Training interruption/recovery has serialization unit coverage but no real-fit
recovery demonstration. No predictive or runtime superiority is claimed; full
training time remains unmeasured. Leave the user's existing GPU job undisturbed.

The [ECGData guide](../ECGDATA.md) contains preparation, verification and dry-run
commands, and explains the later training/report entry points. The new cache is
gitignored; documentation alone does not preserve its generated files.
