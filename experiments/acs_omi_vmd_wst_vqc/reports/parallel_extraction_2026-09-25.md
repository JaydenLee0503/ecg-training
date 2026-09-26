# Parallel extraction — 2026-09-25

## Scope and implementation

Added a separate process-based runner under the
[parallel execution protocol](../protocols/parallel_extraction_v1.md). The
[scientific OMI protocol](../protocols/acs_omi_protocol_v1.md), source archives,
patient split, loader, transform implementation and serial runner are unchanged
from commit `89e324f`. A read-only restart check verified the two original serial
checkpoints before new computation.

The new run is `experiments/acs/results/omi_v1_parallel/`. It copies the existing
protocol, patient assignments and source ledger without creating new folds.
All 17,905 eligible official-training ECGs remain in scope: 14,324 fit and 3,581
internal-validation ECGs. Official test ECGs remain reserved. No model has been
trained, and no accuracy or quantum-advantage result is claimed.

The host exposes 16 logical CPUs and about 21 GB of available RAM. This execution
uses eight spawned processes and one numerical-library thread per worker. The
coordinator validates the source ZIPs once, decodes one record per available
worker, and checks its identity against the frozen split before submission.
Workers log per-lead attempts; the coordinator atomically saves feature vectors
and their completion metadata. An OS file lock excludes simultaneous writers.
Failures stop new submissions while retaining completed in-flight records.

## Verification before the pilot

All **31 ACS tests passed** in **5.332 seconds**. Eight new tests exercise actual
spawned processes, bounded work submission, stop-on-failure behavior, completion
of already-running records, source identity boundaries, label-independent pilot
selection, exact serial reference comparison, checkpoint corruption, and writer
lock release. The existing 23 loader/pipeline/migration tests remain passing.

The first preparation attempt failed at directory creation because the sandbox
exposed the result location as read-only. The approved rerun completed and saved
the separate manifest. The pilot also uses the required file-write permission.
Original experiment files were not changed to bypass their provenance checks.

## Pilot and full-run status

The 16 pilot IDs, selected from fit patients without labels, are:
`17045, 11678, 19330, 15014, 14525, 08564, 06496, 07202, 00407, 18957, 19703,
16283, 05554, 17903, 07316, 13784`.

**Pilot completed successfully** at 20:18:27 UTC (16:18:27 EDT). Its measured
elapsed time was **234.593 seconds**, or **245.53 ECGs/hour**, including source
verification and process startup. All 16 records produced both feature vectors;
all 192 VMD leads converged, with 96 capped intermediate attempts and no final
capped leads. The maximum iteration count was 7,583. Both reference records
matched the serial arrays, feature names, physical inputs and convergence
diagnostics exactly, and both repeated Kymatio comparisons had zero maximum
absolute error. This is numerical evidence, not proof of absence of all bugs.
The pilot included one OMI-positive record; selection did not use labels.

A repeated pilot command verified all 16 saved records without recomputing.
An independent read-only check at 21:06:42 UTC verified the parallel and parent
manifests, frozen code/settings, patient separation, split/ledger identities and
all 26 then-completed feature checkpoints, with no problems. Shared numerical
code, loader, pipeline, serial runner and scientific protocol remain unchanged
from `89e324f`.

**Full extraction started** at 21:03:57 UTC (17:03:57 EDT), with eight workers,
reusing the 16 pilot checkpoints. The archived status snapshot at 21:06:57 UTC
contains 28 completed records of 17,905. This is a progress snapshot, not a
completion claim. Official test records remain reserved; no ACS classifiers
have been trained or scored. The extraction command ends after features; it
does not automatically start training.

Live status is in `results/omi_v1_parallel/extraction_status.json` relative to
the ACS workspace; completed pilot status remains in `pilot_status.json`.
Individual feature checkpoints and attempt logs are in `features/`. See the
execution protocol for resume commands; do not treat a partial run as complete.
After an interruption, inspect the saved status and resume the same extraction
command. Completed records are verified and reused; incomplete records restart.
Do not run a second coordinator while one is active.

## Runtime estimate and remaining work

At the pilot's measured throughput, 17,905 ECGs extrapolate to **72.92 hours
(about three days) of feature extraction**, predominantly VMD computation.
This estimate comes from only 16 records; convergence, CPU contention and
interruptions can change actual duration. It is not a training-time estimate
or a controlled speedup comparison with the earlier two-record serial check.

The scientific protocol subsequently calls for six VQC fits: VMD and WST each
with initialization seeds 0/1/2, 40 epochs per fit, plus classical controls.
ACS VQC training time has **not been benchmarked**, so there is no measured
end-to-end completion estimate. Full-feature bundling for the parallel manifest,
model fitting, saved predictions/histories and statistical reporting still need
implementation. Keep feature selection/scaling inside the fit partition and
retain the existing split and scientific settings.

The [machine-readable report](parallel_extraction_2026-09-25.json) records exact
run/code hashes, environment, patient/class counts, all pilot identities and
per-record runtimes/checksums, reference errors, convergence summaries, tests,
the full-run progress snapshot and the unknown training time. Detailed attempt
logs and feature archives remain local and gitignored. No model performance,
uncertainty interval or quantum-advantage claim is available yet.

## Hardware follow-up

After the user questioned the three-day estimate, read-only hardware inspection
identified an AMD Ryzen 7 8700F (8 physical / 16 logical cores) and an NVIDIA
GeForce RTX 5070 with 12,227 MiB reported VRAM, driver 610.88. The first
`nvidia-smi` call was blocked by the sandbox; the approved read-only query
outside it succeeded. GPU availability does not mean the current extractor
uses it: `ecgvmd/vmd.py` uses NumPy CPU arrays and float64/complex128 arithmetic.

Summing the 16 pilot workers' recorded transform durations gives 1,496.997
seconds for VMD and 3.466 seconds for WST. Thus VMD accounts for 99.77% of their
combined transform time. These are summed worker durations, not elapsed run
time, and include effects of concurrent CPU work. VMD is the priority for an
acceleration benchmark; no GPU speedup has been measured or promised.

Recommended next investigation: benchmark an accelerated VMD implementation
on the same saved engineering inputs, validate numerical agreement and
convergence against the CPU reference at the declared precision/settings, and
record transfer/startup costs separately from repeated execution. A changed
backend needs its own documented execution protocol and output directory.
The full CPU extraction remains active; this hardware inspection did not
switch its backend or start a GPU experiment.

[CuPy's benchmarking guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html)
requires accounting for asynchronous execution and initialization/compilation
overhead. For the later six-qubit VQC, a GPU speedup should also be measured:
[PennyLane's simulator benchmarks](https://pennylane.ai/blog/2024/03/hpc-4-u-and-me/)
show workload-dependent CPU/GPU crossover, not universal GPU superiority.
These external examples are not ACS performance measurements.
