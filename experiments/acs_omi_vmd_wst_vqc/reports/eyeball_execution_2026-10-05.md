# Rotational morphology execution — 2026-10-05 EDT

**Continuation:** the 16k run subsequently failed after nine completed leads.
The completed 64k diagnostic and resumed assessment are recorded in the
[October 8 continuation](eyeball_continuation_2026-10-08.md). The running-state
paragraph below is a historical snapshot, not an instruction to restart it.

The user authorized the proposed experiment and then explicitly authorized the
EMD package download. Work continues from the research review; this is a
10-second independent adaptation, not a verified paper reproduction.

## Setup and preservation

Installed EMD-signal / PyEMD 1.6.4 with `--no-deps` into
`results/eyeball_dependencies_v1/`. Existing NumPy 2.5.2, SciPy 1.18.1 and
Python 3.12.3 remain in use. The first sandbox download failed DNS. The first
network escalation was interrupted before installation; a later user-authorized
retry succeeded. Initial import produced a Matplotlib cache-location warning;
subsequent commands use `MPLCONFIGDIR=/tmp/eyeball-mpl`.
The [setup record](eyeball_setup_2026-10-05.json) contains package file hashes.

The [preflight check](eyeball_preflight_2026-10-05.json) verified 1,542 artifact
entries from the completed original ACS, balanced OMI and ECGData Swin runs,
with zero missing or changed files. It records 1,424 unique protected artifacts
and original source files for a later preservation check. No completed model
or feature archive was retrained or rewritten.

## Implemented numerical choices

- Four IMFs from the pinned PyEMD solver, kept separately from the residual.
  Instrumentation records sifting counts and cap events without changing solver
  arithmetic. Native 500 Hz physical mV input; no waveform normalization,
  resampling or additional filter.
- Native-length FFT Hilbert transform; discard 0.25 seconds at each endpoint.
  Signed instantaneous frequency uses unwrapped phase and a centered gradient.
  Low-envelope masks and minimum phase-support requirements are explicit in
  the engineering protocol. No inherited clipping to nonnegative frequency.
- Four mean frequencies, four mean envelopes, energy-weighted global frequency,
  root-sum-square global envelope, and two geometric coordinates.
- Geometry uses the sum of the four analytic signals and the exact arc-length
  centroid of its open piecewise-linear path. This is a declared continuous
  interpolation convention, not an assertion that the paper used this exact
  implementation. Degenerate paths fail explicitly.
- Negative instantaneous frequency fractions, invalid-phase support, original
  waveform identities, every attempt, per-lead runtime, reconstruction error
  and process high-water memory are saved.

All eight focused numerical/provenance tests passed after the final pre-pilot
edit, in 1.145 seconds. They cover known analytic signals and energy summaries,
nonuniform trajectory sampling, rejection cases, EMD instrumentation parity,
exact repeat extraction, patient selection independent of outcomes, and changed
artifact refusal. [Test log](eyeball_tests_2026-10-05.log).
These checks do not prove absence of bugs or clinical utility.

## Initial pilot failure, retained

`results/eyeball_omi_pilot_v1/` froze the initial
[1,000-iteration specification](../protocols/eyeball_omi_engineering_v1.json).
The 32 fit patients and one ECG each were selected by the declared identifier
hashes without using labels. At 2026-10-06T02:32:27Z, the pilot stopped on the
first selected record, **17086 / P18802 / Lead I**. The third component reached
the cap after 999 actual sifts; the library's loop checks the limit before its
next sift. No baseline lead completed and no model was fitted.

The saved diagnostics record candidate sift counts 60, 506, 999 (capped), 651.
The library continued decomposition after the cap; our wrapper rejected the
entire output rather than silently accepting it. Failure JSON, attempt ledger,
source snapshots and [execution log](eyeball_pilot_2026-10-05.log) remain saved.

## Bounded cap diagnosis and separate retry

`results/eyeball_cap_17086_v1/` declared trial limits 4,000 and 16,000 on that
same fit ECG, stopping once resolved. The 4,000 bound resolved it with sift counts
**60, 506, 1,715, 53**. A separate invocation of the unmodified pinned package
returned exactly equal IMFs and residual. Reconstruction maximum error was zero.
Extraction took 1.7066 seconds; extraction plus reference comparison took 2.9503
seconds. The 16,000 diagnostic trial was not needed and was not run. There were
no captured warnings. See the [diagnostic log](eyeball_cap_17086_2026-10-05.log).

A new [retry specification](../protocols/eyeball_omi_engineering_retry16k_v1.json)
uses a prospective 16,000 ceiling with all convergence criteria unchanged.
Its separate output is `results/eyeball_omi_pilot_retry16k_v1/`. It reuses the
exact original selected patients and ECGs; it does not replace the difficult
case. At 02:36:21 UTC it had completed the first nine Lead-I ECGs. Read its
status and completion marker for later progress; this paragraph is a snapshot.

The planned remainder is all 384 baseline leads, exact repeat checks, 224
Lead-I perturbations and a fixed-sample geometry gallery. Classifier fitting
remains conditional on the engineering outcome. There are no new OMI performance
estimates at this snapshot. Official test patients remain reserved.
