# ACS GPU VMD benchmark — 2026-09-25

The user requested a GPU speed test after the CPU pilot suggested roughly three
days for full feature extraction. This benchmark is separate from OMI model
evaluation and leaves the interrupted production run at its 115 checkpoints.
The workspace remains `experiments/acs/`; the earlier rename did not complete.

## Method and pre-run checks

The [declared benchmark protocol](../protocols/gpu_vmd_benchmark_v1.md) specifies
the same 16 fit ECGs as the CPU pilot, double precision, frozen VMD settings and
retries, numerical acceptance limits, a fresh eight-worker CPU baseline and
three GPU batched repetitions. CPU FFT and descriptors are retained; CUDA
accelerates the iterative VMD updates. No source/test-label or model selection
changes are involved. Timings exclude WST, source loading and output-file writes.

All four opt-in GPU numerical tests passed in 1.802 seconds. They cover DC and
multitone inputs, a nonzero dual update, zero/constant signals, odd-length
trimming, initialization and cap boundaries. All 31 existing ACS tests also
passed in 4.872 seconds. These tests do not prove absence of all software bugs.

The host exposes an RTX 5070 with 12,227 MiB VRAM, driver 610.88, and an existing
CUDA 12.8 installation. The first CuPy install attempt failed because the
sandbox could not resolve the package server. The approved retry installed
CuPy 14.2.0, cuda-pathfinder 1.8.2 and NumPy 2.5.2 under `/tmp/acs-gpu-deps-v1`.
The original Python environment and CPU-run source files were not modified.
GPU tests and the benchmark use the required device/filesystem permissions.

## Initial two-record comparison

The two original engineering ECGs (17045 and 11678, all 24 leads) passed the
real-data check. Their stored float32 feature values, names, final iteration
counts and retry limits matched the CPU exactly. Maximum waveform-mode error
was 6.0004e-15 mV; maximum centre-frequency error was 1.1013e-13 Hz, within the
predeclared tolerances. The single-process CPU took 54.412 seconds, versus
16.814 seconds for the GPU using the same four-lead batch size. This initial
comparison is not the eight-worker/batched-GPU throughput result.

## Completed 16-record throughput comparison

The benchmark completed at **22:12:38 UTC (18:12:38 EDT)**. All three GPU
trials passed every declared numerical check.

| Execution | Records / leads | Measured wall time |
|---|---:|---:|
| CPU, eight processes, four leads per batch | 16 / 192 | 222.416 s |
| GPU, all 192 leads batched, repetition 0 | 16 / 192 | 11.998 s |
| GPU, all 192 leads batched, repetition 1 | 16 / 192 | 11.670 s |
| GPU, all 192 leads batched, repetition 2 | 16 / 192 | 11.656 s |
| GPU median | 16 / 192 | **11.670 s** |

The median GPU speedup over the fresh eight-worker CPU baseline is **19.06×**.
This corresponds to about **4,936 ECGs/hour**, or **3.63 hours** extrapolated to
17,905 ECGs for **VMD plus its CPU descriptors**. It excludes WST, source
loading, checkpoint serialization and all classifier training. The production
pipeline has not been timed on GPU, so this is not a promised complete-runtime
estimate. The sample contains only 16 fit ECGs; CPU timing was measured once,
and GPU timing three times in one process. Larger or harder cases may differ.

All **48 GPU-to-saved-CPU record comparisons** (16 records × 3 repetitions)
have exactly equal stored float32 feature values: maximum absolute error 0.
Names, final iteration counts and retry limits also match exactly. An independent
post-run check additionally matched every attempt by record, lead, limit,
iteration count and capped status: **288 attempts, including 96 capped
intermediate attempts, in each GPU trial**, with no final capped leads. Maximum
final iteration count is 7,583. The first two records' float64 modes have the
small differences reported above; feature equality does not imply bit-identical
intermediate calculations or prove software has no bugs.

GPU context initialization took 0.387 seconds and synthetic compilation/warmup
0.248 seconds in this process; source verification/loading took 38.386 seconds.
These are separately recorded, outside the transform timing table. CuPy reports
CUDA build/linked runtime 12.9, local toolkit/NVRTC 12.8, driver API 13.3 and
device compute capability 12.0. The exact configuration is saved with the run.

## Artifacts, integrity and next step

The [machine-readable report](gpu_vmd_benchmark_2026-09-25.json) includes the
manifest and all declared settings, hardware/environment, each record's patient
and label, input hash, CPU duration, convergence counts/limits, every GPU feature
comparison, all timing repetitions and the 48-artifact checksum inventory.
VMD is deterministically initialized; GPU warmup uses seed 20260925, and the
random device tests use seeds 23 and 19. There are no model predictions or model
initialization results in this benchmark.

Local artifacts are in `results/gpu_vmd_benchmark_v1/`: `events.jsonl` saves the
stage messages, `status.json` confirms completion, and `manifest.json` pins the
setup. The NPZ/JSON pairs retain per-variant output vectors, reference modes,
iteration/attempt diagnostics and durations. Hardware, CuPy configuration,
dependency installation and source input hashes are also saved.

A repeated benchmark command verified all **48 saved artifacts** and returned
without recomputation. A separate check verified all **115 production CPU
checkpoints**, the frozen production manifest/source settings and every GPU
trial's attempt diagnostics, with no problems. The original NumPy environment,
shared numerical code, frozen protocol and saved CPU outputs remain unchanged.

The benchmark supports implementing a separate resumable GPU extraction runner
with a new execution manifest, logged batch/checkpoint behavior and the same
scientific settings. That production integration is **pending**. The old CPU
run remains interrupted at 115 records; no full GPU extraction or ACS model
training has started. No result here establishes clinical model performance,
transform superiority or quantum advantage.
