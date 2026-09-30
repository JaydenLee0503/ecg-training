## Resume after sleep or restart

The latest run is `results/omi_v1_gpu_retry256k/`, paused at the user's request on September 29 EDT with **12,916/17,905 ECGs saved (72.1%)** and 4,989 pending. The active batch finished before extraction stopped; the downstream workflow also exited. No ACS model has been trained. Resume the same extraction tomorrow, then restart the workflow after extraction reports `running`. See the [handoff](SESSION_HANDOFF.md) and [resume guide](reports/resume_acs_2026-09-28.md). The official test remains reserved.

# ACS / OMI experiment

This folder contains the new ACS study, separate from the original ECGData
ARR/CHF/NSR experiments. Start with [SESSION_HANDOFF.md](SESSION_HANDOFF.md).
The plain-language experiment plan is in [PLAN.md](PLAN.md), and all log
locations are explained in [LOGS.md](LOGS.md).

Historical first GPU stop: extraction stopped at 1,795/17,905 ECGs because record
01985/V5 exhausted the 32,000-iteration convergence limit. All saved checkpoints
passed verification, and a CPU reference attempt reproduced the capped result.
See the [stop report](reports/gpu_extraction_stop_2026-09-25.md) and handoff before
resuming that historical run. Separate validated retry protocols subsequently
resolved this stop and the later 04124/V5 stop; see the latest status above.

The separate [GPU benchmark](reports/gpu_vmd_benchmark_2026-09-25.md) is complete:
16 ECGs took 222.416 seconds on eight CPU workers versus 11.670 seconds median
on the RTX 5070, with exactly matching stored VMD features and convergence
counts. This 19.06× result covers VMD/descriptors only. Its cohort extrapolation
is 3.63 hours before WST, source/file overhead and model training. The separate resumable
GPU runner passed integrated validation but later stopped at the convergence
safeguard described above; the old CPU run is preserved.

```text
experiments/acs_omi_vmd_wst_vqc/
  loader.py          Verified ACS archive reader
  pipeline.py        Matched inputs, splits, features and classifier interfaces
  scripts/           Experiment runner, loader audit and download verifier
  tests/             ACS-specific tests
  protocols/         Declared OMI experiment settings
  reports/           Audits, preparation results and machine-readable summaries
  data/              Original CSV.zip and ECG_row_data.zip (gitignored)
  results/           Loader audits and saved OMI run (gitignored)
  provenance/        Move inventory, frozen source snapshots and hash mapping
  EXPERIMENT_LOG.md  ACS chronology, including failures and exclusions
```

The original ECGData notebooks, scripts, protocols and saved models remain in
their existing locations. Numerical VMD/WST, feature descriptors, selection and
classifiers remain shared in the repository's `ecgvmd/` package; ACS imports them
without maintaining a second copy.

## Current status

The [OMI protocol](protocols/acs_omi_protocol_v1.md) and patient split are saved:
14,324 fit ECGs and 3,581 validation ECGs, with no patient overlap. A two-ECG
engineering check passed for both front ends. A separate eight-worker,
16-record extraction pilot passed in 234.593 seconds, including exact serial
and Kymatio comparisons. Full extraction started on 2026-09-25; ACS model
training has not started. See the
[parallel extraction report](reports/parallel_extraction_2026-09-25.md) for
measurements and limitations, and the saved extraction status for live progress.

## Commands

Run these from the **repository root**, using the existing environment. The
default dataset path is this workspace's `data/`, independently of the shell's
working directory.

Verify the completed engineering check and reuse its checkpoints:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs_omi_vmd_wst_vqc/scripts/experiment.py smoke --out experiments/acs_omi_vmd_wst_vqc/results/omi_v1
```

Run ACS tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/acs-matplotlib /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/acs_omi_vmd_wst_vqc/tests -v
```

Follow [SESSION_HANDOFF.md](SESSION_HANDOFF.md) for monitoring and resume commands.
The active GPU run is `results/omi_v1_gpu_retry256k/`. Earlier CPU/GPU runs remain
preserved. The separate `finish_development256k` workflow handles verification,
bundling, fitting and reporting after extraction succeeds; inspect its status
before starting another workflow. The extraction command alone does not train models.

Only a **new** experiment directory needs preparation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs_omi_vmd_wst_vqc/scripts/experiment.py prepare --out experiments/acs_omi_vmd_wst_vqc/results/NEW_RUN
```

Import the loader or shared ACS interfaces from the repository root:

```python
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset
from experiments.acs_omi_vmd_wst_vqc.pipeline import fit_preprocessor, make_classifiers

with ACSDataset(target="OMI") as dataset:
    record = next(dataset.iter_records("train"))
```

## Saved provenance

The saved scientific protocol, patient assignments, raw data and all completed
results were moved without changing their bytes. Historical JSON reports still
contain the original paths and source hashes; these describe the runs as executed.
`provenance/before_reorganization/` preserves those five original ACS source files
as text snapshots. They are evidence, not alternate active implementations.

The runner accepts the existing manifest only through the exact recorded
[layout migration](provenance/layout_2026-09-24.json), checking both the frozen
original sources and relocated code. Later code changes still require a separate
run. [The move inventory](provenance/relocation.json) maps all old locations to
new ones. Do not rewrite old manifests to disguise a changed experiment.

Data and results are local and gitignored. Committing this workspace saves its
code and documentation, not a backup of the dataset or generated artifacts.

The later descriptive rename is recorded in
[descriptive_rename_v1.json](provenance/descriptive_rename_v1.json), with original
sources in `provenance/before_descriptive_rename/`. Old historical paths are
mapped explicitly; no compatibility symlink or mutable result manifest is used.
# Current continuation — 2026-09-27 EDT

The separate `results/omi_v1_gpu_retry128k/` run resumes the original extraction
with validated bounded extra retries. Original checkpoints remain unchanged.
Read [SESSION_HANDOFF.md](SESSION_HANDOFF.md) and the
[retry protocol](protocols/gpu_extraction_retry128k_v1.md) before starting a writer.

Live extraction status: `results/omi_v1_gpu_retry128k/extraction_status.json`.
Downstream workflow: `results/omi_development_workflow_v1/status.json` and stage logs.
The workflow waits for complete extraction, then verifies/bundles features, trains
all declared models and reports internal-validation metrics. It stops on failures.
Model output: `results/omi_models_v1_retry128k/`. Official test data remain reserved.
