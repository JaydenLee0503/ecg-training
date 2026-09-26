# Where this experiment records its work

All paths below are relative to this experiment workspace. Start with
[PLAN.md](PLAN.md) for the research question and steps, and
[SESSION_HANDOFF.md](SESSION_HANDOFF.md) for the latest documented state.

## Human-readable records saved in Git

| Location | What it records |
|---|---|
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Dated decisions, work completed, failures, exclusions, validation and pending work |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | Current state, restart/resume instructions and next work for a new chat |
| [protocols/](protocols/) | Frozen scientific settings, split/initialization/bootstrap seeds, metrics and execution rules |
| [reports/](reports/) | Detailed data audits, preparation, pilot timings, numerical comparisons and machine-readable summaries |
| [provenance/](provenance/) | Source snapshots, layout changes and artifact inventories/checksums |

The [parallel extraction report](reports/parallel_extraction_2026-09-25.md)
and its [JSON summary](reports/parallel_extraction_2026-09-25.json) describe the
completed pilot. Counts captured in a report are dated snapshots; they do not
update automatically as the full extraction proceeds.

## Detailed local computation records

The CPU feature run is `results/omi_v1_parallel/`. It is currently interrupted
for folder maintenance, with 115 completed ECG checkpoints; consult the latest
status rather than assuming it is running.

| File or pattern inside that run | Contents |
|---|---|
| `extraction_status.json` | Latest full-run state, completed/resumed/new record counts, workers, timestamps and elapsed time; overwritten as progress changes |
| `pilot_status.json` | Completed 16-record pilot, reference comparisons, convergence and measured throughput |
| `manifest.json` and `manifest.sha256` | Frozen code, environment, protocol, parent-run identity and split/source hashes |
| `protocol.json` and `execution_protocol.json` | Exact scientific and process-execution settings for this run |
| `splits.csv` | Every development ECG's patient, fit/validation assignment, label and waveform hash |
| `source_records.csv` | Source audit ledger, including quality decisions and exclusions |
| `features/<record_id>_attempts.jsonl` | Append-only VMD attempt events: timestamp, worker PID, lead, iteration limit/count and whether the attempt reached its cap |
| `features/<record_id>.json` | Completed ECG identity, input/feature hashes, convergence diagnostics, VMD/WST durations, feature counts and any reference comparison |
| `features/<record_id>.npz` | Actual VMD/WST vectors and feature names; completion JSON and checksum are required before reuse |
| `failure_<timestamp>.json` / `failures_<timestamp>.json` | Retained exception traces when a run fails or is interrupted; these files exist only when such an event occurs |

Older evidence is retained separately: `results/omi_v1/smoke/` contains the
original two-record serial check, and `results/loader/` contains the initial
failed audit plus its completed `_v2` replacement. Do not merge their status
files or mistake a pilot for completed full extraction.

To inspect current progress from the repository root:

```bash
cat experiments/acs_omi_vmd_wst_vqc/results/omi_v1_parallel/extraction_status.json
```

The terminal's printed progress is not currently saved as a complete console
transcript. The durable record is the structured status, per-ECG metadata,
attempt events, failure traces and dated reports listed above. An individual
attempt log can include attempts from an interrupted run; the completion
metadata identifies the successful result. A saved `running` status alone
does not prove that a process survived a PC restart.

## GPU engineering benchmark records

The separate GPU speed test writes to `results/gpu_vmd_benchmark_v1/`.
`events.jsonl` saves its stage/timing messages, `status.json` its current state,
`manifest.json` its frozen setup, and `hardware.json` / `cupy_config.txt` /
`dependency_install.json` its environment. Per-variant NPZ/JSON files preserve
outputs and VMD attempt/iteration diagnostics. On completion, `summary.json`
contains all timings, comparisons and artifact checksums; exceptions are saved
in `failure.json`. See the [GPU report](reports/gpu_vmd_benchmark_2026-09-25.md).
These are speed/correctness records, not trained-model performance results.

## GPU production extraction records (stopped at convergence safeguard)

The run is `results/omi_v1_gpu/`; preparation, pilot and recovery checks passed.
It subsequently stopped with 1,795 completed ECGs at record 01985/V5's convergence
limit. `failure_1790386168619869105.json` preserves the failure traceback and batch.
The separate `results/vmd_cap_01985_cpu_diagnostic_v1/` directory retains the CPU
reference reproduction, declared diagnostic settings and source/result hashes.
See the [stop report](reports/gpu_extraction_stop_2026-09-25.md).
Its `manifest.json` embeds the scientific/execution settings, code/environment/
hardware hashes, split/source identities and all imported CPU checkpoint hashes.
`preparation.json` records import completion; `pilot_status.json` and `pilot/`
record the integrated CPU/GPU/reference comparisons; `extraction_status.json`
records full-run progress. `pilot_batches.jsonl`, `extract_batches.jsonl` and
`sessions.jsonl` record batch timings and completed/interrupted sessions.

`features/<record_id>.npz` and `.json` store features and completion metadata.
Each metadata file states its CPU-import or GPU-computation origin. Imported
attempt logs use `<record_id>_source_attempts.jsonl`; new GPU attempt logs use
`<record_id>_attempts.jsonl`. Their hashes are checked on resume. Failures retain
`failure_<timestamp>.json`, active IDs and traceback. `recovery_verification.json` records the verified 131-record stop/resume check.
The [launch report](reports/gpu_extraction_launch_2026-09-25.md) links the completed
validation evidence. Monitor `extraction_status.json` for current full-run counts.

## Model-training records

**No ACS model-training histories, fitted-model checkpoints, validation
predictions or performance tables exist yet.** These must be implemented and
saved by the fitting/reporting stage, including every seed, epoch history,
settings, elapsed times, predictions, metrics and uncertainty. Do not report
planned files as completed results.

Raw archives in `data/` and generated artifacts in `results/` are gitignored.
Reports and source code in Git do not back up the dataset or feature archives;
a fresh clone needs those local artifacts separately.
