# ACS OMI protocol v1 — declared 2026-09-24

Paths and commands updated for the ACS workspace reorganization; scientific
results and machine-readable historical records are unchanged. See
[the workspace guide](../README.md).

## Research question and scope

Compare VMD descriptors + VQC against standard WST features + the same VQC for
the supplied binary **OMI versus non-OMI** label. The user selected OMI. A negative
OMI label does not mean a healthy patient. The dataset paper defines its OMI
annotation from angiography and notes limits to that definition; this experiment
uses the supplied labels without relabeling or adding diagnoses.
[Source: dataset paper](https://www.nature.com/articles/s41597-026-07278-0).

This is a first fixed-budget, internal-validation experiment on a new task. It
does not externally validate the old ARR/CHF/NSR classifier. It is not a replication
of the dataset authors' median-waveform CNN, and there is no novelty or quantum
advantage claim. Prior unsuccessful VQC experiments remain part of the record.

The [machine-readable protocol](acs_omi_protocol_v1.json) defines the implemented
settings. It is saved with each run along with source/code hashes and package
versions. Changing those settings or implementation requires a separate run.

## Cohort, input and split

Use verified Figshare v1 `CSV.zip` and `ECG_row_data.zip`, with the existing
[loader policy and exclusions](../reports/acs_loader_2026-09-24.md). Retain all 12 leads in
named order and the full 10 seconds, at native 500 Hz in mV. No extra filters,
resampling, per-record amplitude normalization, beat selection, or window
augmentation is introduced. This retains physical amplitudes but does not remove
baseline drift or other artifacts beyond the loader's declared exclusions.

There are 17,905 eligible official-training ECGs from 16,967 patients. One ECG
becomes one feature row; leads are concatenated as columns, never treated as
independent observations. Use every eligible record, without class resampling.

Sort records by ID; apply `StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=20260924)` grouped by patient. Fold 0 is the internal validation
partition; folds 1–4 form the fit partition. This gives an approximately 80/20
split, not a five-fold performance estimate. Save every assignment before feature
extraction. No patient may cross partitions and each partition must contain both
labels. [Splitter documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Keep the official 1,995 test ECGs reserved. This runner rejects their signals
and produces neither their features nor predictions. Test labels are withheld.
The 2 test quality exclusions remain visible in the saved source ledger; a later
official evaluation needs its own declared coverage/submission handling. No
online submission is part of this protocol implementation.

## Matched representations

| Setting | VMD | WST |
|---|---|---|
| ECGs, labels, leads, duration, units, splits | Identical | Identical |
| Transform | 8 modes per lead, alpha=2000, tau=0, DC mode, uniform initialization, tolerance 1e-7 | J=8, Q=(8,1), T=256, orders 0/1/2 |
| Per-lead reduction | Existing 28 descriptors per mode; no rhythm/global extras | Log orders 1/2 with epsilon 1e-12, then mean over time bins; signed linear order 0 |
| Lead fusion | Concatenate in loader lead order | Same |
| Downstream budget | Training-fitted mRMR to 12 features | Same |
| Angle scaling | Training-fitted StandardScaler, then tanh times pi/2 | Same |

VMD yields 2,688 candidate features per ECG. The WST dimension is recorded from
the implementation during validation. T=256 at 500 Hz is a 0.512-second averaging
scale before pooling. Pooling is the **mean of logged coefficients**, not the log
of their mean. It sacrifices temporal detail and does not make WST a lossless
representation. The choice keeps candidate dimensions manageable at 12 leads.
The comparison concerns these complete feature pipelines; their candidate
representations differ even with the shared final feature budget.

VMD parameters are declared for this 500 Hz experiment, not claimed to reproduce
the physical frequency behavior of the old 128 Hz experiment. Retry only capped
leads at iteration limits 2,000 / 4,000 / 8,000 / 16,000 / 32,000, from the same
deterministic initialization. Log every attempt. If a lead remains capped, stop
and retain the failure; do not discard its record from either arm or accept it
silently. Nonfinite features likewise stop the run, without zero replacement.

## Classifiers and evaluation plan

The first VQC is the existing compact implementation: 6 qubits encoding all 12
features using RY/RZ, 2 trainable blocks, encoding once (`reupload=False`), X/Y/Z
and neighboring ZZ readout, and a binary linear softmax head. It has 86 parameters
(36 circuit and 50 head parameters). This reduces simulator state size relative
to the old 12-qubit model; prior results did not establish superior accuracy.
Both arms use identical settings and initialization seeds **0, 1, 2**, with all
seed results retained.

Use 40 epochs, batch 32, Adam with cosine learning rate 0.02 to 0.002 over those
40 epochs, balanced class weights from fit labels, and analytic `default.qubit`
with backpropagation. The existing implementation uses shuffle seed `seed+10000`
and weighted cross-entropy normalized within each batch. There is no early
stopping, validation-driven epoch choice, or hyperparameter search in v1.

Controls use the exact same 12 selected and angle-scaled features: distance-weighted
Euclidean KNN with k=10 and logistic regression with C=1, balanced class weights,
lbfgs, and 2,000 maximum iterations. Also report an always-negative classifier
and a constant fit-prevalence ranking score. These are matched compressed-input
controls; they do not establish performance against all classical methods.

Primary metric: ECG-level **average precision**, explicitly the non-interpolated
precision–recall summary. Also report ROC AUC, balanced accuracy, sensitivity,
specificity, precision, F1, accuracy, and confusion matrices at fixed score
threshold 0.5. Class-weighted scores are not assumed clinically calibrated.
Do not optimize the threshold on these validation scores and call them final-test
performance. Record predictions, scores, selected columns, fitted preprocessing,
all model seeds/weights, histories, runtimes, warnings and failed trials.

Report 95% paired patient-cluster bootstrap intervals using 2,000 resamples and
seed 20260925. Resample patients and retain their ECGs together, applying the same
multiplicities to both arms and every seed. Average per-seed metric differences;
do not count three seeds as independent patient samples. Discard and count any
single-class bootstrap draw for ranking metrics. These intervals condition on the
saved predictions and do not include retraining uncertainty. Patient diagnosis
voting is not defined: repeated ECGs can have different supplied labels.

## Implementation and execution stages

`experiments/acs/pipeline.py` provides shared split, feature, train-only preprocessing,
and unfitted-classifier interfaces. `experiments/acs/scripts/experiment.py` implements split
preparation, a deterministic engineering check, resumable full feature extraction,
and assembly of aligned matrices. Model fitting and statistical reporting are
the next stage; this command does not train models.

The engineering check uses the two fit ECGs with the smallest SHA-256 of
`protocol_id:record_id`, without looking at labels. It checks all 12 leads against
Kymatio at the actual ACS WST settings, with atol=1e-12 and rtol=1e-10. Its small
timings are engineering observations, not accuracy results or reliable full-run
benchmarks. Its artifacts are separate from the full feature archive.

From the repository root, using a new run directory for `prepare`:

The existing `omi_v1` run is already prepared. Skip `prepare` when checking or
resuming it. For a fresh reproduction, replace the output directory in every
command below with a new name.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py prepare --out experiments/acs/results/omi_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py smoke --out experiments/acs/results/omi_v1
```

For the subsequent full extraction, after reviewing the engineering outcome:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py extract --out experiments/acs/results/omi_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B experiments/acs/scripts/experiment.py bundle --out experiments/acs/results/omi_v1
```

Use a single writer per run directory. Resuming verifies manifest, environment,
split, ledger, code, and every completed record's feature checksum. Complete
record checkpoints are reused; a record interrupted before its completion marker
is recomputed with its previous attempt log retained. No full-run bundle is
produced from the engineering subset. Original ECGData commands are unchanged.

See the separate preparation report for actual completion status, counts, tests,
failures and timing. Protocol declarations alone are not completed results.
