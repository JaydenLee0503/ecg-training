This experiment tests improvements to the VQC on the existing ECGData dataset.
The candidate set and selection procedure below are declared before scoring any
candidate on ECG validation data. Every trial and result will be retained.

**Question.** Can repeated encoding and a richer quantum measurement readout improve
the 12-feature VQC's patient-held-out window macro-F1? Accuracy and patient voting
are secondary outcomes. No target of 94% is imposed.

**Fixed inputs.** Reuse the exact converged VMD descriptors, standard WST features,
1,620 windows, 80 patient IDs, and five outer folds from
`results/patient_vmd_wst_vqc`. Fit mRMR-12 and the existing angle scaler on each
training partition. No data downloads or changes to transform settings.

**Candidate family.** Six qubits encode all 12 selected features, with one RY and
one RZ data angle per wire. Each trainable block uses Rot gates and a ring of CNOTs.
Measure X, Y, and Z on each wire plus the six neighboring ZZ products (24 outputs),
then use a linear three-class softmax head. These extra observables measure more
properties of the state; ZZ measurements are not by themselves proof of entanglement.
No classical hidden layer or raw-feature bypass is added.

| Candidate | Trainable blocks | Data encoding | Quantum + head parameters |
|---|---:|---|---:|
| compact_once | 2 | Before the first block only | 36 + 75 = 111 |
| reupload_2 | 2 | Before every block | 36 + 75 = 111 |
| reupload_3 | 3 | Before every block | 54 + 75 = 129 |

The original 12-qubit classifier has 111 parameters. The compact two-block models
retain that count while changing how information enters and leaves the circuit.
Fewer qubits also reduce simulator state size, making nested tuning practical.
This is a comparison of complete classifier configurations; only compact_once versus
reupload_2 isolates repeated encoding within the new family.

**Optimization.** Adam, batch 32, initial learning rate 0.02, cosine decay toward
0.002 over 80 epochs, balanced class weights derived only from training labels.
The per-batch weighted cross-entropy convention matches the original classifier.
Initialization seed 0 for inner trials. Separate shuffle RNG with seed + 10000
makes minibatch order identical across candidate architectures. Evaluate epochs
20, 40, 60, and 80. Outer retraining at a selected epoch retains the same 80-epoch
learning-rate schedule prefix. Record every epoch's loss and gradient norm.

Use analytic `default.qubit` with autograd backpropagation for the six-qubit models.
Before ECG scoring, a synthetic 96-row/six-Adam-step comparison reproduced the
`lightning.qubit` adjoint fit to 1e-9 while reducing total fit time from 2.21 s to
0.29 s (including initialization; this is a small timing check, not a hardware
benchmark). The first exploratory synthetic timings with adjoint were 0.22–0.33 s
per step. This backend choice changes execution, not the declared circuit.

**Nested selection.** For each outer fold and front end, split only its training
patients into three inner folds (`StratifiedGroupKFold`, seed 100 + outer fold).
Refit selection and scaling independently within every inner training fold. Train
all three candidates through 80 epochs, saving validation predictions at all four
checkpoints. Choose the candidate/epoch with highest pooled inner out-of-fold
macro-F1. Exact ties favor fewer blocks, fewer epochs, then declaration order.
Neither outer labels nor outer predictions enter this choice.

Refit the selected configuration on all outer-training patients with seeds 0, 1,
and 2; evaluate the outer test patients once after each fit. Candidate grids and
budgets are identical for VMD and WST, though each may select different settings.
There are 90 inner fits and 30 outer fits. Synthetic checks and timing measurements
may precede the run; they will not be used to choose ECG classifier settings.

**Controls and reporting.** Compare the resulting out-of-fold predictions against
the saved original VQC and fixed weighted-KNN controls. Report per-seed accuracy,
balanced accuracy, macro-F1, per-class metrics, confusion matrices, patient voting,
and paired patient-bootstrap intervals for changes. Repeated seeds are not treated
as independent patients. Preserve failed/non-improving trials, learning histories,
selected columns, split IDs, fitted pipelines, weights, hashes, software versions,
timestamps, and runtimes. Record engineering failures separately from scientific
results. Publish a narrative report and a machine-readable trial ledger.

**Interpretation limits.** The old five-fold results have already informed this
research direction. Nested selection prevents new within-run tuning on the outer
folds, but this remains follow-up research on the same 80 patients, not an untouched
external validation. Class/source-database confounding remains. Simulator runtime
and accuracy do not establish quantum advantage or hardware performance.

Repeated encoding is motivated by [Pérez-Salinas et al.](https://arxiv.org/abs/1907.02085)
and the [PennyLane implementation](https://pennylane.ai/demos/tutorial_data_reuploading_classifier).
The nesting principle follows [scikit-learn's model-selection guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
