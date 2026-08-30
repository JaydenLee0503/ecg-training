# Experiment log — the quantum stage

A running lab notebook. Newest entries at the bottom. Every row is something that was
actually run; failures and bugs are recorded alongside results, because a number that
came from a broken configuration is the most dangerous kind.

Conclusions and tables live in [QUANTUM_STAGE.md](QUANTUM_STAGE.md); this file is the
chronology.

**On timestamps.** E1-E15 were run without capturing the wall clock, so only their
**order and measured durations** are recorded - both of which are exact. Clock times are
deliberately absent rather than reconstructed; an earlier draft of this file estimated
them from durations and produced times that were provably wrong (some in the future).
The session ran between the commits `c52944c` (2026-08-29 20:20) and the present. From
**E16 onward every entry carries a real timestamp read from the system clock** at the
moment of recording.

**Standing definitions.** Unless a row says otherwise: features are
`features/fixed_K8_a2000_L500_it500_eeb2053b.npz`, block `VMD modes + rhythm` (236
features), reduced by `MRMRSelector(k=12)` **fitted inside each training fold**, scored
as macro-F1 under `StratifiedGroupKFold(5)` on the record id. "n=324" means 2 windows
per record, "n=486" means 3, "n=1620" means 10 — always all 162 records.

**Reference points.** RF on all 236 features = 0.7861. RF on mRMR-12 in-fold = 0.7243.
MLP on mRMR-12 in-fold = 0.7061.

---

## 2026-08-29

### E1 — Readiness check · 6 s

MLP on the existing handoff file and on in-fold selected features, to confirm the
classical→quantum interface works before building anything.

| | macro-F1 |
|---|---:|
| MLP on `quantum_*_q12.npz` `X` (transductively leaky) | 0.7368 |
| MLP, mRMR-12 selected in-fold, angle-scaled | 0.7061 |

**Verdict.** Interface sound. The leaky number sits just above the 0.7243 RF target and
the honest one just below, exactly as expected. No bug in encoding, scaling, or split.

### E2 — Angle wrap-around diagnostic · 3 s

Question: does in-fold `MinMaxScaler` to `[0, pi]` ever produce out-of-range test angles?

**Result.** 60 / 19440 values (0.31%) fall outside `[0, pi]`; worst overshoot 1.40*pi.

**Verdict.** Real bug, small incidence. Since `<Z> = cos(t)`, an overshooting feature
reads back as mid-range instead of extreme — monotonicity inverts on exactly the outlier
samples. Motivated `TanhAngleScaler`. Recorded in QUANTUM_STAGE.md finding 3.

### E3 — Environment · ~2 min

`pennylane==0.45.1` + `pennylane-lightning==0.45.0` dry-run then installed. Clean resolve
against the existing pins; no numpy/scipy/sklearn downgrade. `qiskit`/
`qiskit-machine-learning` also checked and would resolve cleanly — not installed.

### E4 — Kernel correctness and throughput · 1 min

Gram vs brute force: max error 1.11e-15. Symmetric, unit diagonal, PSD (min eigenvalue
0.485). Parameter broadcasting works. Throughput flat at **~0.47 ms/pair** from batch 64
to 16384 (memory-bound, not call-bound).

### E5 — First stage-2 probe · 240 s — **INVALID, wrong bandwidth**

`scripts/quantum_kernel_probe.py`, n=486, `TanhAngleScaler(scale=1.0)`.

| model | macro-F1 |
|---|---:|
| RF, all 236 features | 0.7490 |
| RF, mRMR-12 | 0.6883 |
| SVC-rbf, standardised | 0.7048 |
| SVC-rbf, tanh-scaled (matched control) | 0.6794 |
| QuantumKernelSVC, bandwidth=1.0 | 0.6460 |

Gram off-diagonal: mean 0.0419, median 0.0013, frac>0.01 = 0.325.

**Verdict. Do not cite the 0.6460.** The Gram was in the identity regime, so the score
reflects kernel concentration, not class separability. An earlier concentration check
had been run on MinMax-scaled data and was assumed to carry over to the tanh scaler; it
does not. Lesson: re-measure Gram geometry whenever the scaler changes.

### E6 — Gram geometry sweep · 90 s

n=324, 12 in-fold features. Off-diagonal mean / median / frac>0.01:

| scaling | mean | median | frac>0.01 | |
|---|---:|---:|---:|---|
| MinMax `[0, pi]` | 0.185 | 0.108 | 0.842 | |
| tanh, scale=1.00 | 0.042 | 0.001 | 0.319 | identity regime |
| tanh, scale=0.75 | 0.107 | 0.034 | 0.681 | |
| tanh, scale=0.50 | 0.295 | 0.244 | 0.998 | usable |
| tanh, scale=0.35 | 0.518 | 0.509 | 1.000 | usable |
| tanh, scale=0.25 | 0.703 | 0.711 | 1.000 | |
| tanh, scale=0.15 | 0.878 | 0.885 | 1.000 | all-ones regime |

**Verdict.** Usable band ~0.35–0.50. Both failure modes are live. QUANTUM_STAGE.md
finding 2.

### E7 — Bandwidth CV sweep · 730 s

n=486, same controls as E5.

| bandwidth | macro-F1 |
|---:|---:|
| 0.50 | 0.6887 |
| 0.35 | 0.6968 |
| 0.25 | 0.7039 |

**Verdict.** Monotone increase as bandwidth falls. Best (0.7039) beats the matched
control (0.6794) and draws level with the untuned standardised RBF (0.7048).
Hypothesised at the time that the kernel was degenerating toward a classical RBF —
partly right, see E9.

### E8 — Product-state factorisation · 1 s — **key finding**

Tested whether the RY angle kernel has a closed form.

```
|<phi(x)|phi(y)>|^2  ==  prod_i cos^2((x_i - y_i)/2)
```

**Result.** Max absolute difference vs the statevector simulator **1.665e-16**; closed
form **1634x faster**.

**Verdict.** `AngleEmbedding` with no entangling gates prepares a product state, so the
kernel factorises and is classically tractable *by construction*. Every number in E5 and
E7 is a classical product-cosine kernel. Motivated `product_angle_kernel` and the
`embedding="iqp"` path. QUANTUM_STAGE.md finding 1.

### E9 — Deeper bandwidths + tuned rivals · 720 s

| model | macro-F1 |
|---|---:|
| QuantumKernelSVC bw=0.15 | 0.7029 |
| QuantumKernelSVC bw=0.10 | 0.4694 |
| QuantumKernelSVC bw=0.05 | 0.2481 |

**Verdict.** The E7 degeneration hypothesis was wrong in its specifics — the score does
not plateau at the RBF value, it **peaks near 0.2 and then collapses** as the Gram
saturates to all-ones. There is a genuine operating point.

The tuned-RBF half of this run **crashed**: `GridSearchCV` nested inside a `Pipeline`
does not receive `groups` (`ValueError: Pipeline.fit does not accept the groups
parameter`). Had it silently succeeded with a non-grouped inner CV it would have leaked
records across inner folds. Rerun in E10 with a manual outer loop.

### E10 — IQP geometry, tuned rivals, full-scale product kernel · 400 s

IQP Gram geometry (n=162): mean 0.036 @ bw=1.0, **0.237 @ bw=0.5**, 0.654 @ 0.25,
0.912 @ 0.12 → usable band ~0.4–0.6, so IQP is viable at 12 qubits.

Tuned classical rivals, `gamma`/`C` selected in-fold, n=486: standardised **0.7110**,
tanh(0.25) **0.7051**.

**Product-cosine kernel on the full n=1620** (closed form, so now affordable):

| bandwidth | macro-F1 | wall clock |
|---:|---:|---:|
| **0.35** | **0.7246** | 1 s |
| 0.25 | 0.7129 | 1 s |
| 0.15 | 0.7023 | 1 s |

**Verdict.** 0.7246 lands on the RF mRMR-12 baseline of 0.7243. Best 12-feature number
in the project, and it takes one second. Also: once the RBF is tuned properly it edges
out the angle kernel at n=486 (0.7110 vs 0.7039), so the earlier "+0.024 over control"
was hyperparameter budget, not kernel structure.

### E11 — Entangled IQP vs classical rivals · 1830 s

n=324, identical rows and folds.

| model | macro-F1 | wall clock |
|---|---:|---:|
| IQP (entangled), bw=0.50 | 0.7032 | 906 s |
| product-cosine, bw=0.35 | 0.7004 | 0 s |
| SVC-rbf, standardised, untuned | 0.6971 | 0 s |
| IQP (entangled), bw=0.40 | 0.6911 | 918 s |
| SVC-rbf, `gamma`/`C` tuned in-fold | 0.6784 | 2 s |

**Verdict.** Entanglement bought **+0.0028 for 906 seconds**. Note also that in-fold
tuning *hurt* the RBF at this sample size (0.6784 vs 0.6971) — the inner grid search
overfits at n=324.

### E12 — Noise floor · 60 s — **the number that interprets E11**

The same product-cosine model across 8 subsample/CV seeds:

| model | mean | sd | range |
|---|---:|---:|---:|
| product-cosine | 0.6860 | 0.0271 | 0.6298 – 0.7308 |
| SVC-rbf | 0.6940 | 0.0320 | 0.6468 – 0.7659 |

**Verdict.** A single model reseeded moves by up to **0.101**. E11's quantum margin of
+0.0028 is one tenth of a standard deviation. At n=324 these models are
indistinguishable and no ranking among them is meaningful. **Path A concluded: parity.**

### E13 — VQC throughput · 3 min

`AngleEmbedding` → `StronglyEntanglingLayers` → 12 `<Z>`, `diff_method="adjoint"`.
Gradient wall-clock per step:

| depth | batch 32 | batch 128 | batch 512 |
|---:|---:|---:|---:|
| 2 | 0.414 s | **0.370 s** | 14.6 s |
| 4 | 1.99 s | 5.68 s | 21.8 s |

**Verdict.** Depth 2 at batch 128 is the operating point; depth 4 is ~15x worse and
batch 512 thrashes memory. Implies a full n=1620 five-fold run costs ~19 min, so stage 4
can run at full scale without subsampling.

### E14 — VQC first smoke test · 148 s — **FAILED, model collapsed**

n=324, one fold, 30 epochs, batch 128, lr 0.01, no class weighting.

```
loss 1.1083 -> 0.8683      train macro-F1 0.2451      test macro-F1 0.2516
```

**Verdict.** The model predicts **ARR on every sample**. "Always predict ARR" scores
exactly 0.2481 on this class distribution. Two causes:

1. **Step budget.** 259 training windows at batch 128 = 3 steps/epoch; 30 epochs = **90
   Adam steps** for 111 parameters. The batch size had been chosen from E13's throughput
   benchmark without checking how many steps it left at this sample size.
2. **Class imbalance.** 59% ARR / 19% CHF / 22% NSR. Loss 0.868 is *below* the prior
   entropy 0.9566, so the circuit was genuinely learning — but argmax could not overcome
   the prior.

Fixes: `class_weight="balanced"` is now the default, and `n_steps_` is computed and
reported at fit time so the budget is visible rather than implicit.

**This is the entry to remember.** Run at full scale without the smoke test, this would
have produced ~0.245 across five folds and looked exactly like the "VQC underperforms,
parity confirmed" result that had been forecast all session. A prediction of parity makes
a bad number *easier* to accept, not harder.

### E15 — VQC learning-rate and step-budget sweep · 737 s then killed

n=486, one fold (train 387 / test 99), `class_weight="balanced"`. Timed out at 3000 s
after the first of three configurations; the other two never ran.

| config | steps | loss | train F1 | test F1 | gap |
|---|---:|---:|---:|---:|---:|
| lr 0.05, 120 ep, batch 32 | 1560 | 1.081 → 0.439 | 0.8196 | 0.5812 | +0.2384 |

**Verdict.** The E14 collapse is fixed — the model learns (train 0.82) and predicts all
three classes. But it now **overfits hard**: a 0.24 train/test gap, with 111 parameters
against 387 training windows drawn from roughly 130 records. Test 0.5812 is far short of
the ~0.70 parity threshold.

Diagnosis is sample size, not hyperparameters: at full n=1620 a training fold is 1296
windows, 3.3x more data against the same parameter count. Going to full scale rather
than tuning further at n=486.

### E16 — VQC training curve at full scale · started 23:02:07, killed at 3000 s timeout (epoch 30 of 100)

n=1620, one fold (train 1296 / test 324), lr 0.05, batch 128, 100 epochs, evaluated
every 5 epochs. Added `eval_set` / `eval_every` to `VQCClassifier` for this — it records
train F1, val F1 and the gap into `history_` during training, so the overfitting curve is
visible instead of inferred from a single endpoint.

Looking for: where val F1 peaks, and whether it then decays (overfitting) or plateaus
(capacity-limited). That determines the epoch budget for the five-fold run.

Curve as far as it ran, recorded 2026-08-29 23:17:02 while still running:

| epoch | steps | loss | train F1 | val F1 | gap |
|---:|---:|---:|---:|---:|---:|
| 5 | 55 | 0.7940 | 0.6288 | 0.5498 | +0.0790 |
| 10 | 110 | 0.7305 | 0.6813 | 0.5503 | +0.1311 |
| 15 | 165 | 0.6901 | 0.6856 | 0.5523 | +0.1333 |
| 20 | 220 | 0.6718 | 0.7062 | 0.5532 | +0.1530 |
| 25 | 275 | 0.6579 | 0.6990 | 0.5577 | +0.1412 |
| 30 | 330 | 0.6432 | 0.6937 | 0.5126 | +0.1811 |

Killed by the 3000 s timeout at epoch 30; final state recorded 2026-08-29 23:48:20.

**Verdict.** Val is **flat at ~0.55** across all 30 epochs and never trends upward.
Train stalls too, peaking at 0.7062 and then drifting *down* while the loss keeps
falling — the loss is being reduced on the weighted objective without translating into
better decisions. Not the
predicted overfitting-after-a-peak shape, and **not sample size** — n=486 gave test
0.5812 (E15), n=1620 gives 0.5532. Train reaches only 0.706, so the model underfits even
the training set. That points at capacity or optimisation, not generalisation, and
prompted E17.

### E17 — The linear-head control · recorded 2026-08-29 23:17:02 · 20 s — **the control that should have come first**

The question E16 could not answer: is the *circuit* contributing anything? Hold the head
fixed and remove the circuit. Same 12 in-fold features, same `TanhAngleScaler`, same
folds, n=1620.

| model | macro-F1 |
|---|---:|
| MLP, 32 hidden (nonlinear head, no circuit) | 0.7212 |
| **LogisticRegression (linear head, no circuit)** | **0.6771** |
| VQC = circuit + linear head (E16, epoch 20) | 0.5532 |

**Verdict.** The circuit is a **net negative**. A plain linear model on the same twelve
inputs scores 0.6771; inserting the variational circuit before that same linear head
drops it to 0.5532. The circuit is discarding class-relevant information rather than
adding representational power.

This control isolates the circuit's contribution by holding the head fixed — exactly the
argument used to reject the reference paper's dressed front end (a trainable layer on
both sides makes the quantum contribution unattributable). The argument was applied to
their architecture and not to ours. It should have been the first stage-4 experiment,
before `VQCClassifier` was written.

Still open at the time of writing: whether 0.5532 is an expressivity ceiling or an
optimisation failure. **Answered by E18 — it is not expressivity.**

### E18 — Capacity probe (overfit 96 samples) · recorded 2026-08-30 00:10:39

Can the circuit memorise a tiny set? 96 training windows, 32 per class, depth 2, 200
epochs, batch 32, lr 0.05. Generalisation is irrelevant here — the only question is
whether the model *can* fit.

| model | TRAIN F1 on 96 samples |
|---|---:|
| linear head alone, no circuit | 0.8636 |
| **depth 2, 600 steps** (loss 1.1112 → 0.1695) | **0.9792** |
| **depth 4, 600 steps** (loss 1.0999 → 0.0426) | **1.0000** |

**Verdict — this reverses the E16/E17 reading.** The depth-2 circuit memorises 96
samples almost perfectly (0.9792) and *beats* the linear head on the same data (0.8636);
depth 4 fits them exactly (1.0000). **There is no expressivity ceiling at either
depth.** 72 parameters are ample.

So E16's train F1 of 0.7062 was **undertrained, not capacity-limited**: it ran only 330
Adam steps at batch 128 before the timeout, while this probe used 600 steps at batch 32
on a far smaller problem. The correct reading of stage 4 so far is that the VQC is a
**high-variance model that fits but does not generalise** on 1290 windows drawn from 162
records — not that it cannot represent the task.

**Consequences for what has been claimed.**

* The E16 conclusion "capacity or optimisation, not generalisation" was **wrong in the
  direction it pointed**. Capacity is fine.
* E17's finding stands as measured (circuit 0.5532 vs no circuit 0.6771 at 330 steps),
  but must **not** be reported as "the circuit is inherently subtractive". It is
  subtractive *at that training budget*.
* **Nothing about stage 4 is concluded.** The full-scale run needs a step budget of the
  right order — hundreds of epochs, not 30 — before any VQC number is reportable.

### E18b — Depth 4 · **DID RUN** — logged as NOT RUN in error, corrected 2026-08-30

The entry here read "NOT RUN — killed by the shutdown before the depth-4 arm started".
That was wrong. `results/e18_capacity_probe.txt` has the line, written at 00:10:40:

```
depth 4,  600 steps: loss 1.0999 -> 0.0426 | TRAIN F1 1.0000  (520s)
```

The E18 entry above is stamped 00:10:39 — the log was written one second before the
depth-4 line landed, and never re-read. Depth 4 reaches **train F1 1.0000** in 520 s,
1.7x depth 2's 307 s, not the ~15x that E13 predicted. It is folded into the table above.

**Process note.** Reading a run's output while it is still writing, and not re-reading it
after the process exits, is how this happened. Both E16 and E18 were logged from partial
output. Log from the finished file, not from the terminal mid-run.

**Two runs of E18 were lost to operator error before this one produced anything**: the
first (300 epochs, 2400 s timeout) was sized below the ~36 min the workload needed, and
its output was piped through `grep`, which block-buffers — when SIGTERM killed the
pipeline `grep` died without flushing, so the already-computed depth-2 result was
destroyed. **Do not pipe a long run through `grep`; write to a file with `python -u`.**

---

### E19 — The step budget, done properly · 2026-08-30 12:26–12:51 · 1475 s — **reverses E17**

n=1620, one fold (train 1290 / test 330), depth 2, batch 32, lr 0.05, 100 epochs =
**4100 Adam steps** — 12.4x E16's 330. `OMP_NUM_THREADS=4`, `python -u` to a file.

| epoch | steps | loss | train F1 | val F1 | gap |
|---:|---:|---:|---:|---:|---:|
| 10 | 410 | 0.6559 | 0.7150 | 0.6691 | +0.0459 |
| 20 | 820 | 0.6154 | 0.7173 | 0.6476 | +0.0697 |
| 30 | 1230 | 0.5836 | 0.7417 | 0.6500 | +0.0917 |
| 40 | 1640 | 0.5614 | 0.7157 | 0.6491 | +0.0666 |
| 50 | 2050 | 0.5558 | 0.7455 | 0.6606 | +0.0849 |
| 60 | 2460 | 0.5518 | 0.7444 | 0.6334 | +0.1110 |
| 70 | 2870 | 0.5582 | 0.7583 | 0.6535 | +0.1048 |
| **80** | **3280** | 0.5435 | 0.7687 | **0.6733** | +0.0954 |
| 90 | 3690 | 0.5528 | 0.7622 | 0.6663 | +0.0959 |
| 100 | 4100 | 0.5557 | 0.7156 | 0.6099 | +0.1058 |

**Val F1 is 0.6691 by step 410 and 0.6733 at its best.** E16 measured 0.55 flat and
concluded the circuit was broken. It was measuring its own step budget.

### Matched controls, identical fold, identical rows and scaler

E17 compared a *single-fold* VQC against rivals; on this fold `LogisticRegression`
scores 0.6343, not the 0.6771 E17 reports, so **E17's rivals were not computed on the
VQC's fold**. Recomputed here on the same 1290/330 split:

| model | val F1 (same fold) |
|---|---:|
| LogisticRegression — the VQC's own head, circuit removed | 0.6343 |
| **VQC depth 2, 410 steps** | **0.6691** |
| **VQC depth 2, 3280 steps** | **0.6733** |
| MLP 32 hidden, no circuit | 0.7008 |
| RandomForest 400 | 0.7198 |

**Verdict — E17 is retracted on two independent grounds.** Its VQC number came from a
run at 8% of the needed step budget, *and* its baselines came from different folds.
Holding both fixed, **the circuit is additive, not subtractive: +0.039 over the same
linear head without it.** The claim "the circuit is discarding class-relevant
information" was an artefact of both errors and must not be repeated.

**What stands.** The VQC still loses to the MLP (0.7008) and the RF (0.7198) on the same
twelve features. The honest reading is the one path A reached by another route: the
quantum model attains parity with a comparable classical model and does not beat a good
one.

**The curve is flat after step 410.** Best-at-3280 beats step-410 by +0.0042 — one sixth
of E12's 0.0271 noise floor — for 8x the compute, and epoch 100 falls to 0.6099. The
argmax at epoch 80 is noise, not a peak. **Do not spend 3280 steps on the five-fold.**
Epoch 30–40 (1230–1640 steps) sits in the flat region with the smaller train/val gap and
is the defensible budget.

### E20 — Stage 4, the five-fold · 2026-08-30 16:15-16:52 · 2218 s — **concludes path B**

The budget came from E19 (40 epochs = 1640 steps, inside the flat region, *not* E19's
argmax at 80 — see E19 on why that argmax is noise). Three seeds, because E19's single
fold wandered 0.6099-0.6733. n=1620, depth 2, batch 32, lr 0.05, `n_jobs=5` /
`OMP_NUM_THREADS=1`. Rivals share the rows and folds exactly.

| model | macro-F1 |
|---|---:|
| product-cosine kernel, bw=0.35 | 0.7246 |
| *(E21, nested bandwidth — the defensible kernel number)* | *0.7286* |
| MLP, mRMR-12 in-fold | 0.7148 |
| RF, mRMR-12 in-fold | 0.7130 |
| **VQCClassifier, 3 seeds** | **0.6428**  sd 0.0166, range [0.6271, 0.6602] |

**Verdict — the VQC does not reach parity, and this one is not noise.** The gap to the RF
is **0.0702**: 2.6x E12's 0.0271 noise floor and 4.2x the VQC's own across-seed sd of
0.0166. Every seed lands below every classical rival. Path B is concluded.

**This does not reinstate E17.** Two separate claims, both now measured:

* *The circuit is subtractive relative to its own head* — **false** (E19: +0.039 over the
  same linear head, identical fold). E17 stays retracted.
* *The VQC as an architecture beats classical baselines on these features* — **false**
  (this run). It loses to a random forest by 2.6 noise floors.

A circuit that helps the head bolted to it, inside a model that loses to a random forest,
is exactly the shape of the result the reference paper reports and the shape path A
reached independently. **Both paths now agree: quantum reaches parity with a comparable
classical model and does not beat a good one.**

Cost note: 735 s per seed against E16's 2773 s for a *tenth* the steps — the
`OMP_NUM_THREADS` fix (see `ecgvmd/quantum.py`) and `--n-jobs 5` together, worth ~12x.

### E21 — Bandwidth nested in-fold · 2026-08-30 16:33 · 26 s — **closes open item 3**

Open item 3 in this file and in QUANTUM_STAGE.md: every kernel number was scored with a
bandwidth chosen from Gram statistics over the *whole* sample, then cross-validated. Same
class of error as fitting `MRMRSelector` outside the fold. `scripts/kernel_nested_bw.py`
picks the bandwidth by an inner record-wise CV on the training records only, refits on the
full training fold, and predicts the held-out fold. The test fold chooses nothing.

Full 1620, k=12, outer 5-fold / inner 4-fold, both `StratifiedGroupKFold` on record id.

| bandwidth | transductive macro-F1 (the published protocol) |
|---:|---:|
| 0.15 | 0.7023 |
| 0.25 | 0.7129 |
| **0.35** | **0.7246**  <- the published headline |
| **0.50** | **0.7356**  <- actually the best |
| 0.75 | 0.7321 |
| 1.00 | 0.7233 |

| protocol | macro-F1 |
|---|---:|
| **nested, leak-free — the reportable number** | **0.7286** |
| transductive best (bw=0.50) | 0.7356 |
| transductive at the published bw=0.35 | 0.7246 |

Per-fold bandwidths chosen: `[0.75, 0.50, 0.75, 0.50, 0.25]`.

**Three findings.**

**1. The leak was real but small.** Choosing bandwidth on the test folds is worth
**+0.0070**, about a quarter of E12's 0.0271 noise floor. It never explained the result.

**2. The honest number is *higher* than the published one, not lower — 0.7286 against
0.7246.** Because bw=0.35 was never the best setting. It came from E6's Gram geometry at
n=324, and was carried to n=1620 without re-checking: at full scale bw=0.50 scores 0.7356
and 0.75 scores 0.7321. The published headline was scored at a suboptimal bandwidth, which
happens to have cancelled most of the transductive optimism. Two errors pointing opposite
ways is not a defence of either.

**3. Bandwidth is genuinely fold-dependent** — three distinct values across five folds —
which is why nesting costs so little. A single global bandwidth is the wrong model of the
hyperparameter.

**Consequences.** `QUANTUM_STAGE.md`'s headline becomes **0.7286, nested**, and the
comparison is against RF 0.7130 (see E20) rather than the unreproducible 0.7243: a margin
of **+0.0156**, still inside the noise floor, so **the parity conclusion is unchanged and
is now defensible**. Open item 3 is closed for the angle kernel. It remains open for IQP,
which has its own band around 0.4-0.6 and has never been run nested — do that as part of
the full-scale IQP run, not after it.

---

## Open items

1. **IQP at the full n=1620** — **no longer ~7.5 h; measured at 5.5 s.** `iqp_kernel_qnode`
   uses the adjoint trick, one circuit *per pair* (1,311,390 of them). On a simulator the
   statevectors can be taken directly and the whole Gram is one matmul: prepare each
   `|phi(x)>` once (1620 circuits, 4.67 s, 106 MB), then `G = |conj(PSI) @ PSI.T|**2`
   (0.81 s). Verified against the pairwise path at n=60 to **7.1e-15**, unit diagonal,
   symmetric, PSD. Exact, like `product_angle_kernel` — but this one applies to the
   *entangled* map, so it costs nothing scientifically.
   **Consequence:** nested-bandwidth IQP at full scale is ~10 min rather than ~43 h, so
   IQP can be held to E21's standard instead of the "fixed a priori" fallback.
   To do: move `gram_matrix` onto the statevector path, keeping the pairwise version as
   the correctness oracle, then run IQP nested. This is the last open question that could
   still change a conclusion.
2. ~~**Stage 4 full five-fold.**~~ **Closed by E20** — VQC 0.6428 vs RF 0.7130.
3. ~~**Bandwidth is selected transductively.**~~ **Closed for the angle kernel by E21**
   (nested = 0.7286). Still open for IQP — nest it *inside* the full-scale run, not after.
4. **CNN falsification test** — see QUANTUM_STAGE.md.
5. **Quantum genetic feature selector** (paper method 1) — untouched.
