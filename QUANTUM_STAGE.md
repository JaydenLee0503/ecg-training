# The quantum stage — what was built, what was measured, what it means

Companion to [README.md](README.md#the-quantum-stage), which describes the *plan*. This
file records what happened when the plan met the data. Written 2026-08-29.

Everything here is "path A" — the quantum kernel. No variational circuit has been built
yet. Every number came out of `scripts/quantum_kernel_probe.py` or the module it
imports, and reproduces.

---

## The one-line result

**A quantum kernel on these features reaches parity with a classical one, and the
feature map that scores best is provably classical.** The measured differences between
quantum and classical models are roughly a tenth of the noise floor.

That is a real result, not a failure. It is also what the literature predicts — see
[Prior art](#prior-art).

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
| `QuantumKernelSVC` | `SVC(kernel="precomputed")` on either map |

Verified: Gram matches brute force to 1e-15, symmetric, unit diagonal, PSD; closed form
matches the statevector simulator to 2.8e-16 with identical predictions; the scaler
stays bounded under 1000× out-of-distribution input.

Depends on `pennylane==0.45.1` and `pennylane-lightning==0.45.0`. Both resolve cleanly
against the existing pins — no numpy/scipy/sklearn downgrade.

---

## Measured results

### Full 1620 windows, record-wise 5-fold CV, mRMR-12 selected in-fold

| model | macro-F1 | wall clock |
|---|---:|---:|
| **product-cosine kernel** (angle map, bw=0.35) | **0.7246** | 1 s |
| RF, mRMR-12 in-fold — the project's stated baseline | 0.7243 | 3 s |
| product-cosine kernel, bw=0.25 | 0.7129 | 1 s |
| product-cosine kernel, bw=0.15 | 0.7023 | 1 s |
| *(reference: RF on all 236 features)* | *0.7861* | |

The 12-feature quantum-derived kernel lands exactly on the 12-feature random forest.
Both are ~0.06 below the all-features ceiling, which is the documented cost of reducing
to a qubit-sized register.

### 324 windows (2/record, all 162 records), identical rows and folds

| model | macro-F1 | wall clock |
|---|---:|---:|
| IQP (entangled) kernel, bw=0.50 | 0.7032 | 906 s |
| product-cosine kernel, bw=0.35 | 0.7004 | 0 s |
| SVC-rbf, standardised, untuned | 0.6971 | 0 s |
| IQP (entangled) kernel, bw=0.40 | 0.6911 | 918 s |
| SVC-rbf, `gamma`/`C` tuned in-fold | 0.6784 | 2 s |

### The noise floor, which is the point

Running the **same** product-cosine model across 8 different subsample/CV seeds:

| | mean | sd | range |
|---|---:|---:|---:|
| product-cosine | 0.6860 | 0.0271 | 0.6298 – 0.7308 |
| SVC-rbf | 0.6940 | 0.0320 | 0.6468 – 0.7659 |

**The entangled kernel's margin over the classical one is +0.0028.** The same model
reseeded moves by up to 0.101. The margin is one tenth of a standard deviation and one
thirty-sixth of the observed range. At this sample size the models are indistinguishable,
and any ranking among them is noise.

Do not report a winner at n=324. A definitive IQP comparison needs the full 1620, which
is ~7.5 hours (1.3M pairs × 4.11 ms × 5 folds).

---

## Three findings worth carrying forward

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
quantum result would be wrong. It remains a perfectly good classical kernel — 0.7246 in
one second is the best 12-feature number this project has — it is simply not evidence
about quantum machine learning.

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

**Answered.** Path A. The angle map separates the classes about as well as an RBF; it is
classically tractable; the entangled map gains nothing detectable at n=324.

**Open.**

1. **IQP at the full 1620** (~7.5 h). The n=324 comparison is suggestive, not conclusive.
   Worth one overnight run before any claim about entanglement is published.
2. **The variational classifier (stage 4).** Still worth building — a trainable model is
   a different class from a fixed kernel, and it is the project's stated goal — but
   calibrate the expectation to ~0.70–0.72. `AngleEmbedding` → `StronglyEntanglingLayers`
   *is* entangled, so the stage-4 circuit does not inherit finding 1.
3. **Bandwidth is currently selected transductively.** It was chosen from Gram statistics
   over the whole subsample. Acceptable for a probe; for a published number it must be
   nested inside the fold or fixed a priori and declared as such.
4. **The CNN falsification test** (method 2's front end) — see below.
5. **The quantum genetic feature selector** (method 1) is untouched.

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
