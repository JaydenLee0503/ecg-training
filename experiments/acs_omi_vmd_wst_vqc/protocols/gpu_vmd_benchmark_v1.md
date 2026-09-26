# GPU VMD benchmark v1 — 2026-09-25

This is an engineering test requested by the user, separate from OMI model
evaluation. It uses the same 16 fit ECGs selected without labels for the CPU
pilot. The scientific protocol, patient split, all 12 native 500 Hz leads,
float64/complex128 arithmetic, VMD settings and convergence retry policy stay
fixed. The [JSON specification](gpu_vmd_benchmark_v1.json) was written before
real-ECG timings. Production extraction stays stopped during benchmarking.

The experimental implementation is [gpu_vmd.py](../gpu_vmd.py). CPU NumPy FFT
and reconstruction are retained. A CUDA kernel performs the iterative updates,
with one 256-thread block per lead and at most 64 iterations per launch. CUDA
FMA is disabled and fast math is not enabled. Each lead retains its own stopping
condition. At a retry, capped leads restart from the same initialization as the
CPU pipeline. The existing CPU descriptor code still computes all 28 features
per mode in groups of four leads. This is a hybrid CPU/GPU implementation.

Before timing, four device tests compare against the CPU solver on multitone/DC
signals, nonzero dual updates, zeros/constants, odd-length inputs, initialization
and iteration-cap boundaries. Real-data checks then compare both saved VMD
features and the individual modes/centre frequencies of the two original
engineering records. All 16 ECGs are checked against their saved CPU features.

Predeclared tolerances are feature absolute 1e-6 / relative 1e-5, waveform-mode
absolute 1e-8 / relative 1e-7, and centre-frequency absolute 1e-6 Hz / relative
1e-7. Final iteration counts and retry limits must match exactly. Names, record
order and lead order must also match. A mismatch is retained and prevents
recommending replacement of production features, even if a speedup is observed.

The timing comparisons are:

1. Single-process CPU versus four-lead GPU batches on the first two pilot ECGs.
2. A fresh eight-process CPU baseline on all 16 ECGs, using the original
   four-lead batching, measured once.
3. All 192 leads batched on the GPU, measured three times after synthetic
   compilation/warmup, with all times and their median reported.

Wall-clock GPU timings synchronize execution and include host/device transfers,
retries, reconstruction and CPU descriptors. CPU pool startup/shutdown is
included in its full-baseline wall time. Source loading/identity checks and
artifact serialization are excluded from both transform timings; GPU context
initialization and warmup are recorded separately. WST is not timed here.
An extrapolated cohort duration is a small-sample VMD estimate, not a promised
end-to-end duration or VQC-training estimate.

CuPy 14.2.0, cuda-pathfinder 1.8.2 and a matching NumPy 2.5.2 wheel were installed
under `/tmp/acs-gpu-deps-v1`, using the existing CUDA 12.8 installation. The
original Python environment was not modified. The runner adds that isolated
directory to its import path and records package versions, hardware, CuPy
configuration and the package-install report. Temporary dependencies may need
recreation after cleanup/restart.

```bash
/home/jaydenlee/venvs/test-ecg-training/bin/python -B -m pip install --target /tmp/acs-gpu-deps-v1 --report /tmp/acs-gpu-install-v1.json cupy-cuda12x==14.2.0 numpy==2.5.2
PYTHONPATH=/tmp/acs-gpu-deps-v1 ACS_GPU_TESTS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUPY_CACHE_DIR=/tmp/acs-cupy-test-cache /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest experiments.acs.tests.test_gpu_vmd -v
/home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/benchmark_gpu.py
```

Default outputs go to `results/gpu_vmd_benchmark_v1/`. An unchanged completed
run verifies its saved artifact hashes and returns without recomputing. An
incomplete or changed run requires a new `--out` directory. The CPU production
run is locked through a read-only handle during timing, without rewriting its
manifest, status or saved vectors. A completed benchmark does not launch full
GPU extraction, restart the CPU run or train classifiers.

The timing approach follows [CuPy's performance guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html);
isolated wheel installation follows its [installation documentation](https://docs.cupy.dev/en/stable/install.html).
