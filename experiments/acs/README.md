# ACS / OMI experiment

This folder contains the new ACS study, separate from the original ECGData
ARR/CHF/NSR experiments. Start with [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

```text
experiments/acs/
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
engineering check passed for both front ends. Full feature extraction and model
training have not started. See the
[preparation report](reports/acs_omi_preparation_2026-09-24.md) for actual timings,
convergence, tests and limitations.

## Commands

Run these from the **repository root**, using the existing environment. The
default dataset path is this workspace's `data/`, independently of the shell's
working directory.

Verify the completed engineering check and reuse its checkpoints:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py smoke --out experiments/acs/results/omi_v1
```

Run ACS tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/acs-matplotlib /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/acs/tests -v
```

The next full extraction, when deliberately started, is resumable:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py extract --out experiments/acs/results/omi_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py bundle --out experiments/acs/results/omi_v1
```

Serial VMD extraction is expensive; the two-record timing extrapolated to roughly
5.5 days for the full cohort. This is a rough estimate, not a representative
benchmark. No long run was started during the folder reorganization.

Only a **new** experiment directory needs preparation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py prepare --out experiments/acs/results/NEW_RUN
```

Import the loader or shared ACS interfaces from the repository root:

```python
from experiments.acs.loader import ACSDataset
from experiments.acs.pipeline import fit_preprocessor, make_classifiers

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
