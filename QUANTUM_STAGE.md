# The quantum stage — what was built, what was measured, what it means

Companion to [README.md](README.md#the-quantum-stage--the-original-design), which
describes the *plan*. This
file records what happened when the plan met the data. Written 2026-08-29,
**concluded 2026-08-30**.

Both paths are now finished: the quantum kernel (path A, with and without entanglement)
and the variational classifier (path B). Every number below is leak-free — feature
selection, angle scaling *and* kernel bandwidth are all fitted inside the training fold —
and reproduces from the commands in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

**Several numbers in earlier versions of this file have been superseded**, including the
headline. The full audit trail is in
[EXPERIMENT_LOG.md § Superseded numbers](EXPERIMENT_LOG.md#superseded-numbers--the-audit-trail).

---

## The one-line result

**Every quantum model tried here reaches parity with a comparable classical one and
beats none of them. The feature map that scores best is provably classical, and adding
entanglement makes it worse, not better.**

Three independent routes reached that conclusion: a product-state kernel, an entangled
IQP kernel, and a trainable variational circuit. All measured differences against
classical rivals are inside the noise floor, except the variational classifier's, which
is outside it in the *wrong* direction.

That is a real result, not a failure. It is also what the literature predicts — see
[Prior art](#prior-art).

### Every leak-free number, one table

Full 1620 windows, 162 records, record-wise 5-fold, k=12 selected in-fold.

| model | macro-F1 | quantum? |
|---|---:|---|
| **product-cosine kernel** (angle map, nested bandwidth) | **0.7286** | no — provably a product state |
| MLP, 32 hidden | 0.7148 | no |
| RandomForest 400 | 0.7130 | no |
| **IQP kernel** (entangled, nested bandwidth) | **0.7127** | yes |
| **VQCClassifier** (12 qubits, depth 2, 3 seeds) | **0.6428** ± 0.0166 | yes |
| *(reference: RF on all 236 features)* | *0.7861* | no |

The noise floor on this dataset is **0.0271** (E12), measured by reseeding one model.
Everything in the top four is inside it of everything else. The VQC is not.

### Every metric, not just macro-F1

Added 2026-09-06 (E25). Macro-F1 ranks models; it does not say what one *does*, and on a
59/19/22 prior the thing you most need to know is which class a model quietly abandoned.
Full panel, segment level, same folds:

| model | acc | bal-acc | macro-F1 | macro-sens | macro-spec | CHF sens / prec |
|---|---:|---:|---:|---:|---:|---:|
| angle kernel, nested bw | 0.7747 | 0.7193 | **0.7286** | 0.7193 | 0.8603 | 0.583 / 0.697 |
| MLP, 32 hidden | 0.7568 | 0.7105 | 0.7148 | 0.7105 | 0.8522 | 0.627 / 0.667 |
| RandomForest 400 | 0.7586 | 0.6988 | 0.7130 | 0.6988 | 0.8455 | 0.573 / 0.708 |
| IQP kernel, nested bw | 0.7673 | 0.6962 | 0.7127 | 0.6962 | 0.8496 | **0.513** / 0.703 |
| VQC, 3 seeds | 0.657–0.688 | 0.658–0.691 | 0.627–0.660 | 0.658–0.691 | 0.821–0.835 | 0.630–0.723 / **0.443–0.513** |

Record level and every confusion matrix: `results/all_metrics.md`, regenerated in seconds
from saved predictions by `scripts/metrics_table.py`.

**A paired bootstrap over the 162 records** (5000 resamples, same records drawn for every
model, so the interval is on the difference) is a sharper instrument than the noise floor
and agrees with it:

| vs RandomForest | Δ macro-F1 | 95% CI | P(beats RF) |
|---|---:|---|---:|
| angle kernel | +0.0156 | [−0.0158, +0.0463] | 0.835 |
| **IQP kernel** | **−0.0003** | **[−0.0333, +0.0302]** | **0.509** |
| VQC (worst → best seed) | −0.0860 → −0.0528 | all three entirely below 0 | ≤ 0.010 |

**The entangled kernel is a coin flip against a random forest.** All three VQC intervals
lie wholly below zero — the first interval-based confirmation that its deficit is real
rather than seed noise.

Two things the panel shows that macro-F1 could not:

**IQP's entire deficit is CHF.** It beats RF on raw accuracy (0.7673 vs 0.7586) while
losing on balanced accuracy and macro-F1, because it buys that accuracy by leaning on the
59% majority: CHF sensitivity 0.513 against RF's 0.573, with 126 of 300 CHF windows going
to ARR. Its CHF precision is fine — when it says CHF it is right, it just says it too
rarely. Going from the product map to the entangled one costs 0.583 → 0.513 in CHF
sensitivity. That is what "entanglement does not help here" looks like concretely.

**The VQC is not uniformly worse — it is differently calibrated.** It has the *highest*
CHF sensitivity of any model tried (0.630–0.723 vs RF's 0.573) and the *lowest* CHF
precision (0.443–0.513 vs 0.708). Its macro-*sensitivity* nearly matches RF's while its
macro-F1 sits 0.05–0.09 below: the gap is precision, not recall. It over-calls the
minority class. A reader with only the macro-F1 would conclude the VQC finds less; it
finds more, and is wrong more often when it does.

Neither revises a conclusion. Parity for the kernels, a real deficit for the VQC, no
quantum model ahead of a classical one on any aggregate metric.

---

## What was built

`ecgvmd/quantum.py`, as sklearn estimators so the project's existing honesty machinery
(`StratifiedGroupKFold` on record id, `MRMRSelector` fitting in-fold) applies unchanged:

| object | what it is |
|---|---|
| `TanhAngleScaler` | standardise → `tanh` → `× π/2`. Bounded rotation angles. |
| `angle_kernel_qnode` | RY angle embedding fidelity, on `lightning.qubit`, analytic |
| `iqp_kernel_qnode` | second-order Pauli-Z (ZZ / IQP) map — the entangled one |
| `product_angle_kernel` | closed form of the angle kernel (see below) |
| `gram_matrix` | Gram, symmetric case exploits the triangle and the unit diagonal |
| `QuantumKernelSVC` | `SVC(kernel="precomputed")` on any of the three maps |
| `iqp_state_qnode` | the IQP map returning `qml.state()` rather than a pair overlap |
| `state_gram` | the whole Gram as one matmul — 5.5 s at n=1620 against ~1.5 h pairwise |
| `vqc_qnode` | angle embed → `StronglyEntanglingLayers` → one `<Z>` per qubit |
| `VQCClassifier` | the variational classifier, with a linear read-out head |

Verified: Gram matches brute force to 1e-15, symmetric, unit diagonal, PSD; closed form
matches the statevector simulator to 2.8e-16 with identical predictions; the scaler
stays bounded under 1000× out-of-distribution input.

Depends on `pennylane==0.45.1` and `pennylane-lightning==0.45.0`. Both resolve cleanly
against the existing pins — no numpy/scipy/sklearn downgrade.

---

## Measured results

### Path A — the kernels, nested (the reportable protocol)

Bandwidth is chosen by an inner record-wise CV on training records only; the test fold
chooses nothing. `scripts/kernel_nested_bw.py`.

| map | nested (leak-free) | transductive best | optimism from the leak |
|---|---:|---:|---:|
| **angle / product-cosine** | **0.7286** | 0.7356 | +0.0070 |
| **IQP (entangled)** | **0.7127** | 0.7355 | **+0.0227** |

**Entanglement provides no measurable benefit, and the transductive protocol was hiding
that.** Scored the old way the two maps are indistinguishable — 0.7356 against 0.7355, a
gap of 0.0001. Scored honestly the entangled map lands **0.0159 below** the classically
tractable one. That is the opposite sign to the n=324 probe's +0.0028, and at 0.59x the
noise floor the correct statement is not "angle wins" but "entanglement buys nothing
detectable here".

**The entangled map is 3.2x more sensitive to the bandwidth leak** (+0.0227 against
+0.0070): its inner folds settled on bw=0.35 while the transductive sweep preferred 0.50.
The richer map has more capacity to exploit a hyperparameter tuned on the test folds.
**Any feature-map comparison that tunes transductively will systematically flatter the
more expressive map.** That is probably the most transferable result in this project.

### Path B — the variational classifier

40 epochs = 1640 Adam steps, batch 32, lr 0.05, depth 2, three seeds. The budget comes
from a single-fold training curve (E19), not from running the five-fold at several budgets
and keeping the best.

| model | macro-F1 |
|---|---:|
| RF, mRMR-12 in-fold | 0.7130 |
| **VQCClassifier, 3 seeds** | **0.6428**  sd 0.0166, range [0.6271, 0.6602] |

**The VQC does not reach parity, and this gap is not noise** — 0.0702 is 2.6x the noise
floor and 4.2x the model's own across-seed spread. Every seed lost to every classical
rival.

Two distinct claims, both measured, because they are easy to conflate:

* *the circuit is subtractive relative to its own head* — **false**. On an identical fold
  the VQC scores 0.6733 against 0.6343 for the same linear head with the circuit removed:
  **+0.039**. An earlier claim to the contrary is retracted.
* *the VQC beats classical baselines on these features* — **false**, by 2.6 noise floors.

A circuit that helps the head bolted to it, inside a model that loses to a random forest.

**Why it loses — and it is not what you would guess.** `VQCClassifier` measures `<Z_i>` on
each of 12 qubits, but the state has 4096 amplitudes and single-qubit marginals are exactly
the observables blind to inter-qubit correlation. The ansatz spends 24 CNOTs building
correlations and reads out through a channel that cannot see them. Reading the *same
trained circuit* three ways, one fold (E24):

| representation | RF macro-F1 |
|---|---:|
| raw 12 features — the ceiling | 0.7358 |
| the 12 `<Z_i>` the VQC actually uses | 0.6892 |
| the 66 discarded `<Z_i Z_j>` correlations | **0.7229** |
| all 78 observables | **0.7320** |

The discarded observables score *better* than the kept ones, and a full readout recovers to
within 0.0038 of the raw features. **The bottleneck is the readout, not the circuit** — the
information survives the encoding and dies at the measurement. The recoverable gain, +0.043,
clears the noise floor.

This does not change the conclusion: 0.7320 still only reaches the raw features' 0.7358, so
a fully-read-out VQC would land at parity, not advantage — and `<Z_i Z_j>` expansion is
functionally a nonlinear feature expansion, which classical models do more cheaply. What it
changes is the **attribution**. The deficit is a design choice inherited from the reference
paper, not a property of quantum models, and **the 0.6428 above understates this
architecture family by roughly 0.05**. Numbers from E24 are single-fold and must not be
placed beside the five-fold table.

### The noise floor, which is the point

Running the **same** product-cosine model across 8 different subsample/CV seeds:

| | mean | sd | range |
|---|---:|---:|---:|
| product-cosine | 0.6860 | 0.0271 | 0.6298 – 0.7308 |
| SVC-rbf | 0.6940 | 0.0320 | 0.6468 – 0.7659 |

**0.0271 is the number every margin above has to clear.** Only the VQC's deficit does.

## Five findings worth carrying forward

### 1. Plain angle encoding is not quantum

`AngleEmbedding` with RY and no entangling gates prepares a **product state**,
`⊗ᵢ (cos(xᵢ/2)|0⟩ + sin(xᵢ/2)|1⟩)`. The overlap therefore factorises across qubits and
the fidelity kernel has an exact closed form:

```
|⟨φ(x)|φ(y)⟩|²  =  Πᵢ cos²((xᵢ − yᵢ)/2)
```

Verified against the simulator to **1.7e-16**, and **1634× faster** as five numpy
operations. This is exact, not an approximation.

The consequence: a quantum kernel built on plain angle encoding is classically tractable
*by construction*. It is a shift-invariant product-cosine kernel, a close cousin of the
RBF. No amount of tuning will make it exhibit quantum structure, and reporting it as a
quantum result would be wrong. It remains a perfectly good classical kernel — **0.7286
nested**, in about a second, is still the best 12-feature number this project has — it
is simply not evidence about quantum machine learning.

**Entanglement in the feature map is the only thing that makes a quantum kernel
quantum.** `IQPEmbedding`'s `ZZ(φ(xᵢ,xⱼ))` phases are what break the factorisation;
confirmed by a 0.287 maximum deviation from the product form. `QuantumKernelSVC` uses
the closed form for `embedding="angle"` and the simulator for `embedding="iqp"`.

### 2. Bandwidth and scaler are coupled, and the obvious default is the worst setting

Two failure modes bracket the usable range. Off-diagonal Gram entries collapsing toward
0 → the matrix approaches the identity and the SVM memorises. Collapsing toward 1 → no
discrimination survives. Measured on 324 windows, 12 in-fold selected features:

| scaling | mean off-diag | median | frac > 0.01 | |
|---|---:|---:|---:|---|
| MinMax `[0, π]` | 0.185 | 0.108 | 0.842 | |
| `tanh`, scale=1.00 | 0.042 | 0.001 | 0.319 | **identity regime — unusable** |
| `tanh`, scale=0.75 | 0.107 | 0.034 | 0.681 | |
| `tanh`, scale=0.50 | 0.295 | 0.244 | 0.998 | usable |
| `tanh`, scale=0.35 | 0.518 | 0.509 | 1.000 | usable |
| `tanh`, scale=0.25 | 0.703 | 0.711 | 1.000 | |
| `tanh`, scale=0.15 | 0.878 | 0.885 | 1.000 | **all-ones regime — unusable** |

`TanhAngleScaler(scale=1.0)` — the natural-looking default — sits in the dead zone.
Standardising to unit variance spreads pairwise angles far wider than min/max scaling
does on these skewed features, so the scaler that fixes the wrap-around problem
(below) simultaneously pushes the kernel toward the identity. **The two choices cannot
be made independently.** The IQP map has its own band, around 0.4–0.6.

The score curve follows the geometry: 0.6460 at bw=1.0, rising to a peak near 0.25–0.35,
then collapsing to 0.4694 at 0.10 and 0.2481 at 0.05. It does not degrade gracefully.

**Superseded in part.** Those figures come from an n=324 probe. At the full 1620 the peak
sits at **bw=0.50**, not 0.35 — the shipped default was carried over from the smaller
sample and never re-checked, which cost the published headline about 0.011. Bandwidth is
now chosen per fold (findings 4 and 5); do not hard-code it again.

### 3. Min/max angle scaling has a silent wrap-around bug

With `MinMaxScaler` fitted inside the training fold, **0.31% of test-fold values land
outside `[0, π]`, worst case 1.40π**. Since `⟨Z⟩ = cos(θ)`, a feature 40% past the
training maximum rotates *past* the south pole and reads back as mid-range
(`cos(1.4π) ≈ −0.31`) instead of extreme (`cos(π) = −1`). Monotonicity inverts precisely
on the outlier samples — the ones most likely to be diagnostically interesting.

`TanhAngleScaler` borrows the bounding trick from the dressed circuit of Mari et al.
(2020): standardise, `tanh`, scale by `π/2`. Bounded for *any* input, so there is no
train/test range coupling at all. This is the one component of the reference paper's
method 2 that transferred directly.

### 4. Entanglement does not help, and the leaky protocol was concealing it

The whole point of the IQP map is that `ZZ(phi(x_i,x_j))` phases break the product-state
factorisation — it is the only genuinely quantum kernel here. Measured leak-free at the
full 1620 it scores **0.7127**, against **0.7286** for the classically-tractable angle
map. Entanglement costs 0.0159 rather than buying anything.

The reason this went unnoticed is worth stating separately, because it generalises: under
the *transductive* protocol the two maps look identical (0.7356 vs 0.7355). The entangled
map's advantage was entirely an artefact of tuning its bandwidth on the test folds, which
benefits it 3.2x more than it benefits the simpler map. **A more expressive model gains
more from a leaked hyperparameter, so any leaky comparison is biased toward
expressiveness — precisely the axis such comparisons are usually trying to measure.**

### 5. The pairwise Gram is the wrong algorithm on a simulator

`iqp_kernel_qnode` answers "what is `<phi(x)|phi(y)>`?" one pair at a time — 1,311,390
circuit evaluations at n=1620, about 1.5 h per fold and ~43 h for a nested run. But on a
simulator the state is available directly: prepare each `|phi(x)>` once and the entire
Gram is a single matrix product.

```
PSI = stack([state(x) for x in X])      # (n, 2**n_qubits) complex
G   = |conj(PSI) @ PSI.T|**2            # the whole Gram, one matmul
```

Measured at n=1620, 12 qubits: **4.67 s** for the state preparations plus **0.81 s** for
the product — **5.48 s** against ~1.5 h. The full nested IQP run (150 kernel fits) took
18 minutes instead of ~43 hours. Verified against the pairwise path to **3.1e-17** with
identical predictions, unit diagonal, symmetry and PSD preserved. `state_gram`.

**Two caveats, and the first one matters.** This is *not* a tractability result. Unlike
`product_angle_kernel`, which proved the angle map prepares a product state and is
therefore classical by construction, this is a simulator implementation detail: the IQP
map remains genuinely entangled, and on hardware you would still pay per-pair overlap
estimation. Do not report it as evidence that IQP is classically simulable.

Second, it dies on memory before it dies on time. The state matrix is
`n x 2**n_qubits` complex128: 1.7 GB at 16 qubits, 27 GB at 20, **434 GB at 24**. Past
~18 qubits the pairwise path — slow but O(1) in memory — is the only option.

---

## Prior art

Olvera, Montiel Ross & Rubio, *EEG-based motor imagery classification with quantum
algorithms*, Expert Systems With Applications 247 (2024) 123354,
[doi:10.1016/j.eswa.2024.123354](https://doi.org/10.1016/j.eswa.2024.123354).
The PDF in this repo is `Quantum fetre_Seletn.pdf`.

Two methods. **Method 1** is an adaptive quantum genetic algorithm used as a *wrapper
feature selector* — 120 features as a 120-bit chromosome across 10 registers of 12
qubits, depth-3 circuits, fitness = classifier F1. It is the direct analogue of this
project's `MRMRSelector` and remains unbuilt here. If it is built, its fitness call must
live inside the training fold or it reintroduces exactly the selection leak this project
is built to avoid.

**Method 2** is a dressed variational classifier: CNN feature extractor → `Linear(n→n_q)`
→ `tanh` → `× π/2` → VQC (4 qubits, depth 6) → `Linear(n_q→K)`. Adopted here: the
`tanh`/`× π/2` bounding, and the linear postprocessing head. Rejected: the learned
`Linear(236→n_q)` front end, which would put trainable dense layers on both sides of the
circuit and make the quantum layer's contribution unattributable; 4 qubits, when 12 costs
nothing on a simulator and the ablation prices the narrowing at 0.724 → 0.661; and depth
6, which was empirical for their problem.

Note the paper labels VQC1's embedding "amplitude" while describing Hadamards followed
by `Ry(x)` on 4 qubits for 4 inputs — that is angle encoding. Only VQC2 (16 inputs,
4 qubits) is genuinely amplitude-encoded.

**Their headline finding is parity.** Hybrid EEGNet 83.82% against classical baselines
at 83–88%; hybrid ATCNet 85.56% against classical ATCNet 85.85%. Their conclusion is
that "the quantum neural network behaves similarly to the fully connected softmax
layer." Every hybrid was statistically indistinguishable from its classical twin. The
kernel result recorded above reaches the same conclusion by a different route.

Their subject-dependent (85.56%) versus subject-independent (73.73%) gap also
independently corroborates this project's record-wise splitting rule — same ~12-point
magnitude as the +0.12 macro-F1 illusion documented in the README.

---

## Status and next steps

**Concluded.** Everything this file was opened to answer.

| question | answer |
|---|---|
| Does the angle map separate the classes? | Yes, about as well as an RBF — 0.7286 nested. |
| Is it quantum? | **No.** It prepares a product state and has an exact closed form (finding 1). |
| Does entanglement help? | **No.** IQP nested scores 0.7127, *below* the angle map (finding 4). |
| Does a trainable circuit help? | **No.** 0.6428, losing to a random forest by 2.6 noise floors. |
| Does the circuit help the head it is attached to? | **Yes**, +0.039 — but the whole model still loses. |
| *Why* does it lose? | Mostly a lossy readout, not the circuit: 12 marginals out of a 4096-amplitude state, with the entanglement-sensitive observables never measured (E24). Fixing it is worth ~+0.05 — to parity, not advantage. |

**The project's question is answered three independent ways, and they agree.** No further
quantum experiment on these features is likely to change it.

**Open, in priority order.**

1. **If stage 4 is ever reopened, fix the readout first.** Measure `<Z_i Z_j>` alongside
   `<Z_i>` and widen the head to match — one circuit, more observables, cheap on a
   simulator, and worth about +0.05 on E24's evidence. It would move the VQC to roughly
   classical parity. Do this before anything else in stage 4; every other tuning knob is
   smaller than the readout.
2. **The CNN falsification test** (method 2's front end) — see below. Tests whether
   *learned* features beat the 236 hand-crafted ones; it is a question about the classical
   half and would not change any conclusion above.
3. **The quantum genetic feature selector** (method 1) is untouched. If it is built, its
   fitness call must live inside the training fold, or it reintroduces exactly the
   selection leak this project is built to avoid.
4. **Reproduce or retire the RF baseline of 0.7243.** It does not reproduce at n=1620
   (0.7130 today, identical folds) and no artefact of the original run survives. Do not
   cite it again until someone traces it.
5. **Register width.** The mRMR ablation says k=24 scores 0.740 against 0.724 at k=12 —
   but that gain is 0.016, *inside* the 0.0271 noise floor, and 24 qubits is out of reach
   for training on this hardware (E23: 20 qubits is 399x the cost of 12, and `state_gram`
   needs 434 GB at 24). Before anyone buys a GPU for this: rerun the mRMR-k sweep with
   more seeds and find out whether k=24's advantage is real. That costs minutes.

**Not open.** Bandwidth nesting (done, both maps), the full-scale IQP run (done, 18 min),
and stage 4 (done). See
[EXPERIMENT_LOG.md § Superseded numbers](EXPERIMENT_LOG.md#superseded-numbers--the-audit-trail)
for what these replaced.

### The CNN falsification test

The reference paper's method 2 learns its features with a CNN instead of selecting them.
Before adopting that here, run the cheap version that can kill it in an afternoon.

**The test.** A small 1D CNN over the stored IMF tensor `__imfs`, shape
`(1620, 8, 500)` — eight IMF channels, 500 samples. VMD replaces the paper's wavelet
preprocessing; the CNN replaces the 28 hand-crafted descriptors. Same
`StratifiedGroupKFold` on `groups`, same macro-F1.

**The question.** Does it beat **0.7861** — the RF on all 236 hand-crafted features? If
learned features cannot beat engineered ones on this data, method 2's front end is dead
here and the cost of finding out was one day.

**Why it is likely to fail, and why that is worth knowing anyway.** The effective sample
size is **162 records, not 1620 windows** — windows from one recording are
near-duplicates, which is why `groups` exists at all. A CNN with even a few thousand
parameters learning from 162 independent units will memorise recordings. The reference
paper avoids this by training one model *per subject* on ~720 trials each; even so its
cross-subject accuracy falls from 85.6% to 73.7%. This project is permanently in that
harder cross-record setting.

The `u8` confound compounds it. That mode scores 0.743 alone and is plausibly recording
hardware rather than physiology. Hand-crafted features make the confound *measurable* —
drop `u8`, observe the 0.080 cost. A CNN on the raw modes will find the same fingerprint
faster and expose no equivalent handle. Losing that visibility is a real cost, not a
stylistic preference.

**Priority: below the full-scale IQP run and the stage-4 variational classifier, above
the quantum genetic selector.** If it does beat 0.7861, the dressed VQC head from
method 2 becomes worth building on top of it, and the ordering changes.

**Caveat inherited from the classical stage.** Three of the twelve selected features come
from `u8`, the highest-frequency mode, which the ablation suggests may encode recording
hardware rather than physiology. A quarter of the register may be reading which database
a record came from. Every number here inherits that, and should be read as an upper
bound.
