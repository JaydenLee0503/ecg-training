# ACS loader implementation and validation — 2026-09-24

Paths and commands updated for the ACS workspace reorganization; scientific
results and machine-readable historical records are unchanged. See
[the workspace guide](../README.md).

## Outcome

Implemented a reader for the verified ACS Figshare v1 archives in
[`experiments/acs/loader.py`](../loader.py), with an audit command in
[`experiments/acs/scripts/prepare.py`](../scripts/prepare.py). All **19,955 ECGs** were
checked using the default policy: request all 12 leads, require 5,000 samples,
and exclude records with a flat requested lead or a missing sample in a requested
lead. **17,905 training ECGs and 1,993 test ECGs are eligible; 57 are excluded.**
Every decision is recorded. The original ZIPs were neither extracted nor modified.

This completes the loader, not an ACS model experiment. No VMD/WST features,
train/validation folds, fitted preprocessing, classical models, or VQCs were
created. OMI was used to check label loading; the scientific target, lead inputs,
and model protocol remain to be selected. The previous ECGData experiments and
their frozen inputs are unchanged.

## Input and output contract

- Reads `experiments/acs/data/CSV.zip` and `experiments/acs/data/ECG_row_data.zip` directly. Both complete SHA-256
  hashes must match the verified v1 source manifest on opening. See the
  [download audit](acs_download_verification_2026-09-23.md) for source provenance.
- Validates metadata schemas, unique recording IDs, matching raw/median IDs,
  one waveform/header pair per record, and disjoint official train/test patients.
- Decodes multiplexed little-endian signed format 16. Uses each lead's explicit
  baseline and gain to return physical samples as `(digital - baseline) / gain`.
- Eligible signals are read-only `float64` arrays in **mV**, shape
  **(5000, number of requested leads)**, at the native **500 Hz**. Leads are
  selected by name and returned in the requested order.
- Each record exposes `record_id`, `patient_id`, official `split`, and an optional
  binary `label`. Supported targets are `AMI`, `OMI`, `CTO`, `NSTEMI`, `STEMI`,
  and `UA`. They are separate binary annotations, not mutually exclusive classes.
  Without an explicit target, labels are `None`. Test labels are always `None`.
- Demographic, vessel, and treatment annotations are not returned as predictors.
  Vessel annotations can contain category 2; only the six diagnosis targets are
  required to be binary.

The loader checks the known source convention in which header initial-value and
checksum fields contain channel maxima and minima. It accepts this convention
only when those values match the actual signal, and records the convention for
each ECG. Correct standard first-value/checksum fields are also supported and
tested. An unrecognized convention, incomplete sample frame, unsupported format,
or source identity mismatch raises an error. Headers are not rewritten.

Quality policy `acs-v1-strict-selected-leads-1` excludes a non-10-second record,
any flat requested lead, or any format-16 missing marker (`-32768`) in a requested
lead. Flatness is assessed on valid samples. Flags for all source leads are
retained, including unrequested leads. There is no filtering, padding,
imputation, normalization, or resampling. This policy catches the declared
conditions; it does not certify clinical signal quality or absence of all bugs.

## Usage

Run from the repository root with the existing Python environment. No new
dependency installation is needed.

```python
from experiments.acs.loader import ACSDataset

with ACSDataset(target="OMI") as ds:
    record = next(ds.iter_records("train"))
    signal = record.signal_mV       # (5000, 12), mV
    lead_ii = record.lead("II")     # (5000,), mV
    patient = record.info.patient_id
    label = record.info.label
    fs = record.fs                 # 500.0 Hz
    decisions = dict(ds.decisions) # inspected records only
```

`iter_records()` defaults to the training partition and yields only eligible
records. Exclusions remain in `ds.decisions`; iteration must be exhausted to
audit a whole partition. `inspect(record_id)` returns a decision and sets
`signal_mV=None` for an excluded record. `load_record(record_id)` instead raises
`ACSExcluded`, whose `decision` records the reason. `ds.summary()` distinguishes
metadata totals from records inspected so far. Use a context manager to close
the archive; create a separate reader per worker process.

For a small batch that can feed later feature extraction:

```python
from itertools import islice
import numpy as np
from experiments.acs.loader import ACSDataset

with ACSDataset(target="OMI") as ds:
    batch = list(islice(ds.iter_records("train"), 32))
    W = np.stack([r.lead("II") for r in batch])
    y = np.array([r.info.label for r in batch], dtype=int)
    groups = np.array([r.info.patient_id for r in batch])
    fs = batch[0].fs
```

This illustrates the API; CSV-order sampling is not a study/split protocol. The
example still applies the default all-12-lead eligibility policy. Passing
`leads=("II",)` explicitly changes eligibility to that lead and must be recorded
as a separate input policy. Further splits must group by patient, with feature
selection, scaling, and tuning confined to their training/validation partitions.
Pass **500 Hz** explicitly to later signal processing; the old 128 Hz settings
are not applicable automatically.

The existing `run_pipeline.py` and `scripts/matched_vqc.py` remain ECGData-specific.
This reader is an input interface for a new ACS runner, not a switch that makes
those original experiment commands run ACS. Lead fusion, windowing, preprocessing,
matched feature budgets, and classifier settings still require a declared protocol.

## Full-dataset validation

The successful audit completed on **2026-09-24 at 22:24:28 EDT**
(`2026-09-25T02:24:28.414068+00:00`), in **43.703 seconds**. Every eligible output
passed shape and finiteness checks. All 19,955 records matched the source extrema
header convention. No patient crossed the official partition boundary.

| Partition | Source ECGs | Source patients | Eligible ECGs | Eligible patients | Excluded ECGs |
|---|---:|---:|---:|---:|---:|
| Training | 17,960 | 17,018 | 17,905 | 16,967 | 55 |
| Test | 1,995 | 1,891 | 1,993 | 1,889 | 2 |

Training exclusions: 50 flat-lead records, 3 missing-marker records, and 2 short
records (`03228`, `14262`). Test exclusions: 2 flat-lead records. These sets do
not overlap under the default policy. Their union exactly matches the independent
prior download audit's quality flags. Eligible training OMI labels are **1,146
positive and 16,759 negative**; these are record counts, not independent patient
counts. All test label cells in the ledger are blank.

A future official test evaluation must explicitly handle the two excluded test
records through its declared coverage/abstention or fallback policy and any
official submission requirements. A result on 1,993 eligible records cannot be
silently described as a result on the full 1,995-record test partition. No test
disease labels were obtained or used, and no predictive performance was measured.

All **11 focused unit tests passed** (0.067 seconds). Synthetic fixtures cover
physical scaling, signed multiplexed samples, lead order, standard and source
headers, patient overlap, hidden labels, malformed/duplicate metadata, archive
hash mismatch, short signals, missing markers, flat leads, invalid API arguments,
and nonbinary vessel annotations. An independent post-run ledger check verified
all 19,955 unique IDs, train/test separation, label availability, eligible record
and patient counts, all 57 exclusions, output checksums, and source-code hashes.
These tests support the implemented contract; they do not establish bug-free
software or clinical validity.

## Failures and corrections

1. The first audit launch failed to create `experiments/acs/results/loader/` because the shell
   saw a read-only workspace mount. It stopped before the data audit. The required
   write was rerun with approved escalation.
2. The first actual data run stopped on record `18543`: the initial implementation
   incorrectly required every clinical annotation to be binary. Inspection showed
   that vessel annotations legitimately contain category 2. Validation was narrowed
   to the six binary diagnosis targets, and a regression test was added. The failed
   run's status is retained at
   `experiments/acs/results/loader/2026-09-24_all_leads_omi/status.json`.
3. The corrected run used a new directory,
   `experiments/acs/results/loader/2026-09-24_all_leads_omi_v2/`, and finished successfully.
   No failed output or source dataset was overwritten.

## Saved artifacts and reproduction

- [Version-controlled audit summary](acs_loader_2026-09-24.json): exact copy of
  the completed run summary, including policy, source and code hashes, environment,
  counts, runtime, and every excluded record with reasons.
- `experiments/acs/results/loader/2026-09-24_all_leads_omi_v2/summary.json`: original summary.
- `experiments/acs/results/loader/2026-09-24_all_leads_omi_v2/records.csv`: all 19,955 decisions,
  including per-waveform hashes and source flags, not just the exclusions.
- `experiments/acs/results/loader/2026-09-24_all_leads_omi_v2/status.json`: completion marker.

The ledger's SHA-256 is
`2b9763ca7cee01418385fd29d979e890e911238572d6f38244edcb5eae78e908`.
Data and results directories are gitignored; a fresh clone may not contain the
archives or full ledger. The JSON report contains the exact validated code hashes.

```bash
/home/jaydenlee/venvs/test-ecg-training/bin/python -B -m unittest discover -s experiments/acs/tests -p test_loader.py -v
/home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/prepare.py --target OMI --output experiments/acs/results/loader/NEW_RUN
```

The audit command refuses an existing output directory. Select a new directory
for a justified rerun or changed lead policy; do not rerun simply because a new
chat started. Omitting `--target` produces a label-free audit. No random seed,
model predictions, accuracy, or confidence interval applies to this deterministic
loader check. Those must be saved in the next, separately documented experiment.
