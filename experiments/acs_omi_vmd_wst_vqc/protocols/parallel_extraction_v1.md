# Parallel ACS extraction protocol — 2026-09-25

This is an execution change to the frozen [OMI v1 scientific protocol](acs_omi_protocol_v1.md).
The numerical code, patient split, source identity, exclusions, leads, units,
sampling rate, VMD convergence policy and WST settings are unchanged. The old
serial runner, manifest and completed outputs remain intact. No classifier
training, tuning or official-test processing is included.

The [execution specification](parallel_extraction_v1.json) pins the parent
manifest and scientific protocol hashes. Prepare a separate `omi_v1_parallel`
directory by copying the verified protocol, split and source ledger without
rerandomizing. Record new code hashes, environment, worker count and pilot IDs.
Changes to those settings require a separate directory.

Use eight spawned worker processes, each restricted to one numerical-library
thread. The coordinator verifies and opens the original ZIPs once, checks every
ECG against its frozen split row, and sends its decoded signal to a worker.
At most one record per worker is in flight. Each worker appends its own VMD
attempt log; the coordinator writes the completed feature archive atomically,
then its completion metadata and checksum. An OS file lock prevents two
coordinators from writing to the same run. The lock is released on process exit.

Before full extraction, use the first 16 **fit** records in the original
label-independent hash order. This includes the two previous engineering
records, 11678 and 17045. Compare their parallel features, feature names,
physical input hashes, VMD attempts, iterations and limits exactly against the
saved serial outputs. Also repeat their Kymatio check. Other pilot records
provide additional timing/convergence observations without classifier scoring.
Do not change settings based on validation predictions; none are computed.

A failure stops submission of new records. Already running records finish and
their successful checkpoints are retained. Preserve all failure traces and
attempt logs. Do not silently exclude a nonconverging ECG from either arm or
label a partial run complete. Interrupted records without completion metadata
are recomputed on restart; verified completed records are reused.

The full stage requires a complete pilot. It extracts all 17,905 eligible
official-training ECGs and reuses pilot checkpoints under the same parallel
manifest. The 1,995 official test records remain reserved. Worker timings and
wall-clock throughput are engineering observations, not model accuracy or proof
of quantum advantage. Record how many pilot samples informed any runtime
extrapolation, including launch and source-verification overhead.

Commands from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/extract_parallel.py prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/extract_parallel.py pilot
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/extract_parallel.py extract
```

`prepare` refuses an existing directory. The other two commands resume it.
Defaults are eight workers and `experiments/acs/results/omi_v1_parallel`.
Status/progress files are `pilot_status.json` and `extraction_status.json`;
each complete feature record is stored under `features/`. These are separate
from the old serial runner's statuses. Full-feature bundling and model fitting
must verify this new manifest when implemented; do not point the old bundle
command at the parallel directory.
