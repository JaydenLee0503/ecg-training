# Wavelet scattering vs VMD as the front end to the quantum classifier

**Historical protocol:** the results below grouped separated ECG leads by row,
not by verified patient. Do not use them as patient-independent performance estimates
or as a definitive method ranking. See [the corrected comparison](patient_vmd_wst_vqc.md)
for the verified source mapping and matched patient-grouped rerun.

*Run 2026-09-21. Code: `ecgvmd/scatter.py`, `scripts/wst_vs_vmd_vqc.py`. Artefacts:
`results/wst_vs_vmd_vqc.{csv,log}`, `results/wst_vs_vmd_vqc_preds.npz`,
`architects/scatter_check.csv`, `architects/morlet_bank_J6_Q8.csv`,
`architects/classical_reference.csv`.*

---

## 1. What was built

`ecgvmd/scatter.py` had one working backend and three TODOs. All three are now
implemented, and the hand-rolled cascade is the default.

| function | what it is |
| --- | --- |
| `_morlet_bank` | Analytic Morlet filters in Fourier, plus the `(xi, sigma, j)` schedule that decides how many there are |
| `_gauss_lowpass` | The averaging filter, at an arbitrary invariance scale `T` |
| `scatter_batch` | The cascade by hand: pad, FFT, three orders of convolve-modulus-subsample, unpad |
| `scatter_features` | `ScatterResult` -> `(X, names)`, the pair the rest of the pipeline eats |
| `scatter_check` | Asserts `scatter_batch` == `scatter_kymatio` over six configurations |
| `morlet_bank_table` | The bank as a table, for reading the support constraint in numbers |

### It agrees with the reference exactly

The point of writing the cascade by hand is only worth anything if it is provably the
same transform. `scatter_check` runs both backends over six configurations and asserts
agreement below `1e-12`:

| J | Q | T | max_order | paths | bins | max abs diff |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | (8, 1) | 64 | 2 | 126 | 7 | **0.0** |
| 6 | (8, 1) | 64 | 1 | 39 | 7 | **0.0** |
| 5 | (8, 1) | 32 | 2 | 84 | 15 | **0.0** |
| 6 | 8 | 64 | 2 | 126 | 7 | **0.0** |
| 4 | (4, 2) | 16 | 2 | 56 | 31 | 1.4e-16 |
| 7 | (12, 1) | 128 | 2 | 251 | 3 | 8.3e-17 |

Bit-identical at the project's own settings and at three others; float64 round-off from
FFT operation order at the remaining two. Path order, `xi` and `sigma` match as well, so
the path tables are interchangeable. `scatter_batch` is also **1.4x faster** — 0.169s vs
0.233s for 200 windows, median of 5, filter-bank cache warm on both. That is not the
reason to prefer it, but it removes the last argument for keeping kymatio in the hot
path; it stays in the repo as the thing `scatter_check` compares against.

The hand-rolled version is now the one to read when a question comes up about the
transform, because the three things that actually shape the output are visible in it:

- **The L1-in-time normalisation.** Under L2, wide low-frequency filters would answer a
  given amplitude more strongly than narrow ones and every order-1 coefficient would
  carry its filter's scale as well as the signal's content.
- **The `j2 > j1` pruning.** Only ever scatter a fast modulation onto a slower one. This
  one condition is what keeps order 2 at 87 paths instead of 38 x 38 = 1444.
- **The subsampling schedule.** `k1 + k1_J` and `k1 + k2 + k2_T` both sum to `log2_T`,
  which is why the output is a dense `(B, P, n_bins)` array rather than a ragged list.

### The support constraint, in numbers

`morlet_bank_table()` (full table in `architects/morlet_bank_J6_Q8.csv`) shows why a
nominal `J*Q = 48` bank comes out as 38 filters at J=6 / Q=8:

```
 n     xi_hz  sigma_hz  j  support             region
 0 55.728751  2.898014  0       24         constant-Q
...
30  4.142064  0.215396  3      294         constant-Q
31  3.624306  0.200000  3      317 constant-bandwidth
...
37  0.517758  0.200000  5      317 constant-bandwidth
```

Support grows with every constant-Q step, and by n=30 it is 294 — a full support of 588,
already past the 500-sample window and living entirely on the padding (the cascade pads
to 1024, not 512). Then it stops: the constant-bandwidth region pins sigma at
`sigma_min`, so the last seven filters share one envelope and only their carrier moves.
The flat tail is the constraint being respected, not hit.

### Two things that were wrong and are now fixed

1. **`scatter_kymatio` violated its own contract at `max_order=1`.** kymatio sizes its
   meta by `max_order`, so it returned a `(P, 1)` `xi` where `ScatterResult` documents
   `(P, 2)`, and `ScatterResult.meta()` raised `IndexError` on `xi[:, 1]`. Found by
   `scatter_check`, which compares metas. Fixed with `_two_cols`, which NaN-pads.

2. **The module docstring's order-0 measurement did not reproduce.** It claimed S0 ranges
   `-0.20 to +0.14`, 57% negative. Measured: `-1.12 to +1.13`, 50.6% negative on the
   1620-window set, and both backends agree to the digit. The range scales with T
   (`-1.86/+1.65` at T=32, `-0.04/+0.05` at T=500) and the sign split is 0.50 +/- 0.03
   throughout — no J/T produces the old numbers. The *conclusion* that paragraph drew is
   untouched and was always the point: order 0 is signed, so it cannot go through a log.
   The docstring now carries the corrected numbers and a note that the old ones were
   wrong.

### The one real design decision in `scatter_features`

Orders 1 and 2 span **5.3 decades** on this data (1.19e-06 to 2.47e-01) and are strongly
right-skewed within each path. The default takes the log of them, which is standard
(Andén & Mallat 2014) and matters more here than in a tree pipeline because
`TanhAngleScaler` standardises and then saturates: on a raw skewed column almost every
value lands in one flat tail of the tanh and the qubit stops encoding anything. Measured
median absolute per-column skew: **1.26 linear -> 0.47 log**.

Order 0 is left **linear, not dropped**. `log(S0)` is NaN half the time, and dropping the
row throws away the window's local mean trajectory — on a z-scored window, baseline
wander, which is a real discriminant between these classes. The scaler standardises each
column separately anyway, so one linear column among 125 log columns costs nothing.

---

## 2. The protocol

The question is about the **front end**, so everything after it is held identical, not
merely similar:

- the same 1620 windows in the same row order;
- the same `StratifiedGroupKFold(5, shuffle=True, random_state=0)` splits — record-wise,
  which on this data is worth ~0.12 macro-F1 of illusion if you get it wrong;
- the same in-fold `MRMRSelector(k=12)` and `TanhAngleScaler`, both inside the pipeline
  so nothing is fitted on a test fold;
- the same 12-qubit depth-2 `StronglyEntanglingLayers` circuit, 111 parameters;
- the same budget — 40 epochs, lr 0.05, batch 32 — and the same three seeds.

**The epoch budget is VMD's, used unchanged.** 40 came from a training curve on VMD
features (e19). Re-tuning it for scattering and not for VMD would hand scattering an
advantage unrelated to the front end; tuning both against these five folds would select
on the test folds. So both get 40, and the curve is reported separately as a diagnostic.

**The scattering block was declared before the run.** `WST log` — every path, every time
bin, log on orders 1 and 2 — is the primary, because it is the standard form of the
transform and `scatter_features`' default. The bin-mean and linear variants are scored
with RF only, as context, and never with the VQC.

**Row alignment is asserted, not assumed.** The VMD block is read from the August feature
`.npz`; the scattering block is computed here from `segment()`. Two separate paths to the
same 1620 rows, so the script checks `y` and `groups` elementwise and dies if they
differ. This caught a real trap immediately: that `.npz` was built with
`n_per_record=10` (1620 windows) while `CFG` defaults to `0` (all 21222). Segmenting
under `CFG` produced 21222 rows and the script refused to continue. It now reads the
`Config` back out of the `.npz`, which is what `FeatureBundle.save` stores it for.

Confirmation that this worked: RF on the VMD block scores **0.7130** here, matching the
August e25 run to four decimals on the same folds.

---

## 3. Results

### 3a. Classical reference — complete

> **Superseded in part by §7.** The blocks compared here differ in the decomposition AND
> in the feature extraction; §7 separates them and reverses the conclusion drawn from this
> table. Read §7 before quoting any number in §3a as a front-end result.

RF, identical rows and folds, `mRMR-12` fitted inside each training fold.
Artefact: `architects/classical_reference.csv`.

| block | dim | RF, no selection | RF, mRMR-12 |
| --- | --- | --- | --- |
| VMD modes + rhythm | 236 | **0.7892** | **0.7130** |
| WST log | 882 | 0.7338 | 0.6483 |
| WST log, bin-mean | 126 | 0.7193 | 0.6543 |
| WST linear | 882 | 0.7303 | 0.6669 |

*(The 0.7892 is RF-400 with default settings, not the project's `rf()` helper, which adds
`class_weight="balanced_subsample"` and `min_samples_leaf=2` and is the source of the
0.7861 quoted in QUANTUM_STAGE.md. Same conclusion, different estimator — do not mix the
two numbers in one table.)*

**Scattering loses to VMD on this data, and it loses before selection is involved.**
−0.055 macro-F1 with all features, widening to −0.065 at k=12. The deficit is
concentrated in CHF, the minority class: CHF sensitivity is 0.573 under VMD and 0.433
under WST log. That is the failure mode `full_metrics` exists to catch — a model quietly
abandoning CHF while overall accuracy stays respectable (0.7099 vs 0.7586).

### 3b. The selection bottleneck is measurable and specific

`mRMR-12` on the 882-column block picks, per fold, only **7, 6, 6, 9, 6 distinct paths**
— a mean of **6.8 paths for 12 columns**. It is spending roughly five of the twelve
qubits on neighbouring time bins of paths it already holds. The `0.1` damping term in
`mrmr_select`'s `F(f) / (redundancy + 0.1)` cannot suppress this: adjacent bins of one
scattering path are correlated but not identically, so they keep scoring well.

The bin-mean block is the control, and it behaves exactly as that story predicts. It has
12/12 distinct paths by construction, and it is **worse without selection** (0.7193 vs
0.7338 — averaging away the time axis genuinely discards information) yet **better with
it** (0.6543 vs 0.6483). The crossover is the bottleneck, isolated.

This matters beyond scattering: any front end that emits a time × frequency grid will hit
the same thing, because `mrmr_select` has no notion that two columns are the same
descriptor at adjacent times.

### 3c. VMD + VQC — on disk from e25, reused

`results/e25_vqc_fivefold.csv`, run 2026-09-06 at the identical protocol: same 1620 rows,
same `StratifiedGroupKFold(5, shuffle=True, random_state=0)`, k=12, depth 2, 40 epochs,
lr 0.05, batch 32, seeds 0/1/2.

| seed | macro-F1 | accuracy | record acc | sens ARR | sens CHF | sens NSR |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6412 | 0.6660 | 0.7840 | 0.649 | 0.723 | 0.664 |
| 1 | 0.6271 | 0.6574 | 0.7222 | 0.654 | 0.630 | 0.689 |
| 2 | 0.6602 | 0.6883 | 0.7716 | 0.682 | 0.670 | 0.719 |
| **mean** | **0.6428** | 0.6706 | 0.7593 | 0.662 | 0.674 | 0.691 |
| sd | 0.0166 | | | | | |

Re-running this arm was ~36 minutes of compute for a number already measured, so it is
reused rather than recomputed. The licence for reusing it is that RF on the same VMD block
reproduces e25's **0.7130** to four decimals here, on the same folds — the rows and splits
are provably the same objects.

### 3d. WST + VQC — complete

`results/wst_vqc.csv`, run 2026-09-21 at the protocol of §2, seeds 0/1/2. The VMD arm of
this table is §3c, reused; the licence is that RF on the VMD block reproduced **0.7130**
to four decimals on these same folds in this same run.

| seed | macro-F1 | accuracy | record acc | sens ARR | sens CHF | sens NSR |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6136 | 0.6346 | 0.7284 | 0.597 | 0.643 | 0.728 |
| 1 | 0.6124 | 0.6364 | 0.7284 | 0.614 | 0.607 | 0.722 |
| 2 | 0.6446 | 0.6704 | 0.7469 | 0.656 | 0.630 | 0.742 |
| **mean** | **0.6236** | 0.6471 | 0.7346 | 0.622 | 0.627 | 0.731 |
| sd | 0.0183 | | | | | |

**Scattering − VMD = −0.0193 macro-F1** (pooled seed sd 0.0175, so 1.10 pooled sd; all
three seeds negative; paired t p=0.044, n=3, indicative only).

**This does not clear the noise floor.** The project's floor is 0.0271, measured across 8
subsample/CV seeds of the same product-cosine model (QUANTUM_STAGE.md). At 0.0193 the gap
is inside it. The direction is consistent across all three seeds, but by the standard this
project applies everywhere else, scattering and VMD are **not distinguishable under the
VQC**.

### The gap is 3.4x smaller under the VQC than under RF

This is the one thing in this document that is not just a smaller version of §3a.

| | VMD | WST log | gap |
| --- | --- | --- | --- |
| RF, mRMR-12 | 0.7130 | 0.6483 | **−0.0648** — clears the floor |
| VQC, 3 seeds | 0.6428 | 0.6236 | **−0.0193** — does not |

And the deficit changes location. Under RF it was minority-class abandonment: CHF
sensitivity 0.573 on VMD against 0.433 on scattering. Under the VQC that collapse mostly
does not occur.

| CHF sensitivity | VMD | WST log | gap |
| --- | --- | --- | --- |
| RF | 0.573 | 0.433 | −0.140 |
| VQC | 0.674 | 0.627 | **−0.048** |

The VQC lifts CHF sensitivity on scattering features from 0.433 to 0.627. This is very
likely `class_weight="balanced"` rather than anything quantum — it raises CHF on *both*
front ends — but it helps scattering roughly 3x more, and that is what closes the gap. The
per-class pattern flips with it: under the VQC scattering is *better* on NSR (+0.040) and
worse on ARR (−0.040) and CHF (−0.048), where under RF it lost on all three.

A second asymmetry, unexplained: the VQC's deficit against its own RF is −0.0702 on VMD
but only −0.0247 on scattering. Whatever the VQC fails to extract, it fails less on
scattering features.

An earlier attempt at this ran VMD first and was killed 7 minutes in, before any VQC
finished, yielding nothing. The script now runs the PRIMARY (scattering) arm first for
that reason — the arm worth protecting from an interruption is the one nobody has
measured yet. The 2026-09-21 rerun was itself killed at the VQC section boundary with only
the RF table written (`results/wst_vqc.killed-run.log`); the per-seed CSV flush is what
made the restart cost one seed rather than the whole run.

---

## 4. The two diagnostics, both now run

Both were flagged in §3d as able to invalidate the reading. Both were run 2026-09-21, each
alone on the box. Neither leaves the budget question open; one substantially changes the
conclusion.

### 4a. The budget — not an artefact

`--curve --epochs 60 --eval-every 5`, fold 0, both front ends
(`results/wst_vs_vmd_curve_*.csv`). The worry was that 40 epochs came from a VMD training
curve, so a scattering arm still climbing at 40 would be measuring the budget.

| front end | best val-F1 | at epoch | at 40 | at 60 |
| --- | --- | --- | --- | --- |
| VMD | 0.6718 | 15 | 0.6491 | 0.6334 |
| WST log | 0.6837 | 25 | 0.6535 | 0.6342 |

**Neither arm is still improving at 40** — both peak well before it and decline after. The
budget is generous rather than tight, and the §3d gap is not a budget artefact.

Two observations in passing, neither promoted to a score (single fold, evaluated on its
own test half):

* **Scattering overfits harder.** Train-F1 runs 0.75-0.77 against VMD's 0.72-0.74, with a
  train/val gap about 1.5x VMD's. Same 12 qubits, same 111 parameters — scattering's
  selected columns are simply easier to memorise, which is what §3b would predict of a
  block containing near-duplicate features.
* **Scattering's peak is the higher one** (0.6837 at epoch 25 vs 0.6718 at epoch 15),
  inverting the five-fold ordering.

Both arms peak before 40, so the shipped budget mildly understates both VQC numbers. It
does not bias the comparison — both pay it equally.

### 4b. The selection — this is where the gap lives

`--vqc-blocks binmean --seeds 3` (`results/wst_vqc_binmean.csv`). The bin-mean block is
126 features, one per scattering path, so mRMR gets **12/12 distinct paths by
construction** and cannot spend qubits on adjacent time bins.

| arm | features | macro-F1 | sd | vs VMD |
| --- | --- | --- | --- | --- |
| WST log | 882 | 0.6236 | 0.0183 | -0.0193 |
| **WST log, bin-mean** | **126** | **0.6415** | **0.0091** | **-0.0013** |
| VMD modes + rhythm | 236 | 0.6428 | 0.0166 | — |

**Bin-mean recovers +0.0180 of the -0.0193 gap — about 93% — and lands at VMD parity**
(-0.0013, 0.10 pooled sd), with the tightest seed spread of the three arms.

This is the answer §3b was pointing at. The scattering deficit under the VQC is a
**selection artefact, not a transform deficit**. Given 12 genuinely distinct descriptors,
scattering matches VMD.

**The effect is specific to the quantum arm.** Under RF the identical control barely moves:

| | WST log | bin-mean | delta | VMD |
| --- | --- | --- | --- | --- |
| RF, mRMR-12 | 0.6483 | 0.6543 | +0.0060 | 0.7130 |
| VQC, 3 seeds | 0.6236 | 0.6415 | **+0.0180** | 0.6428 |

The mechanism is architectural: the VQC's 12 selected features *are* its 12 qubits, a hard
budget with no redundancy to spare, while RF can still exploit correlated columns across
400 trees. A near-duplicate column costs the VQC a qubit and costs RF almost nothing.

Per-class, bin-mean has the best NSR sensitivity of any arm tested (0.784, against 0.731
for WST log and 0.691 for VMD); it remains below VMD on ARR and CHF.

### 4c. What is and is not a result

Every pairwise difference among the three VQC arms is inside the project's 0.0271 noise
floor, the +0.0180 recovery included. Individually, none of them is a result, and this
document should not be read as establishing an ordering among the three under the VQC.

What carries weight is coherence rather than any single margin: bin-mean lands exactly
where the §3b selection measurement predicted, its variance drops by half, and the RF/VQC
contrast has a mechanical explanation that was stated before the run rather than after it.

## 5. Recommendation

> **Withdrawn in part by §7.** The bullet below asserting VMD as the better front end was
> written before the descriptor confound was controlled. §7c restates what survives.

**Revised from the pre-diagnostic draft.** The earlier reading — that scattering is simply
the weaker front end — is an RF-era result and does not survive the change of classifier.

* **Under RF, VMD is better and the margin is real**: -0.0648 at k=12, clearing the noise
  floor. Nothing here disturbs that.
* **Under the VQC, the three arms are not distinguishable**, and the apparent -0.0193
  scattering deficit is attributable to `mrmr_select` rather than to the transform: remove
  the time-bin redundancy and it goes away (§4b).

So "scattering is the worse front end" should not be quoted as a general claim. It is true
of RF at k=12 and it is not established for the VQC at all.

Keep `scatter.py` regardless: 1.4x faster than kymatio, provably the same transform, and a
second front end sharing every downstream component — which is what made this comparison
an afternoon rather than a rewrite.

**The actionable finding is still §3b, now with a measured price tag.** `mrmr_select`
wastes about 40% of a 12-qubit budget on near-duplicate columns whenever the feature matrix
has a repeated-descriptor structure, and §4b prices that waste at **+0.0180 macro-F1** on
the scattering block. VMD's 236 features are 8 modes x 28 descriptors and have exactly the
same structure, so **the VMD block is probably paying a similar tax and nobody has
checked**. A mode-mean control on VMD is the obvious next run, and on §4b's evidence it is
worth more than any further front-end comparison.

## 6. Reproduce

```bash
# the cross-check that licenses the hand-rolled backend (seconds)
python -c "import ecgvmd as E; E.scatter_check()"

# the classical reference table
OMP_NUM_THREADS=1 python scripts/wst_vs_vmd_vqc.py --no-vqc --n-jobs 5

# the scattering arm of the VQC comparison (~36 min, 3 seeds)
OMP_NUM_THREADS=1 python scripts/wst_vs_vmd_vqc.py --vqc-blocks wst --n-jobs 5 \
    --seeds 3 --epochs 40 --lr 0.05 --batch-size 32 --out results/wst_vqc.csv

# the budget diagnostic, S4a - run it ALONE, it oversubscribes otherwise
OMP_NUM_THREADS=4 python scripts/wst_vs_vmd_vqc.py --curve --epochs 60 --eval-every 5

# the descriptor confound control, S7 (~4 min) - THE one that reverses S3a
OMP_NUM_THREADS=4 python scripts/wst_descriptor_probe.py --seeds 5

# the selection control, S4b (~36 min, 3 seeds) - 126 features, 12/12 distinct paths
OMP_NUM_THREADS=1 python scripts/wst_vs_vmd_vqc.py --vqc-blocks binmean --n-jobs 5 \
    --seeds 3 --epochs 40 --lr 0.05 --batch-size 32 --out results/wst_vqc_binmean.csv
```

Run the three VQC jobs one at a time. Five folds at `n_jobs=5` already saturates a 16-core
box; two such jobs, or a `--curve` alongside one, thrash and distort both. The 2026-09-21
session lost a run to exactly that.

`requirements.txt` now pins `kymatio==0.3.0`. It is needed only by `scatter_kymatio` and
`scatter_check`; `scatter_batch` is pure numpy/scipy.

## 7. The confound that invalidates §3a's conclusion

Added 2026-09-21, after §3-§5 were written. It reverses their headline.

### 7a. The comparison varied two things at once

`RF | VMD modes + rhythm` against `RF | WST log` differs in the decomposition **and** in
the feature extraction, and §3a attributed the whole gap to the first.

* VMD contributes 8 adaptive modes **and 28 descriptors per mode** — entropies, TKEO,
  waveform length, Hjorth parameters, permutation entropy.
* The scattering block contributes 126 paths and **no descriptors at all**. A scattering
  coefficient `|x * psi1| * phi` *is already a summary*: a time-averaged modulus, i.e. an
  energy.

That this is the live axis is measured, not assumed. mRMR selecting 12 from the VMD block
picks (60 picks over 5 folds):

| picks | descriptor | family |
| --- | --- | --- |
| 18 | `spec_entropy` | complexity |
| 12 | `tkeo` | complexity |
| 9 | `wave_len` | complexity |
| 6 | `shannon` | complexity |
| 4 | `spec_bandwidth` | spectral spread |
| **2** | **`rel_energy`** | **energy** |

**26 of 60 picks are entropy-family; 2 are energy.** Scattering can express the family
that is almost never chosen, and cannot express the one that is.

Note also that `centre_hz` and `peak_hz` are picked **0** times, so the adaptive-centre
argument for VMD — plausible a priori, since a fixed bank's centres are constants carrying
zero variance — is **not** what is happening. It was checked and it is not the mechanism.

### 7b. The control: same descriptors, different decomposition

`scripts/wst_descriptor_probe.py`. The order-1 envelopes `|x * psi1|` exist inside the
cascade at full time resolution; `scatter_batch` subsamples them and convolves with `phi`,
and *that averaging* is what destroys the within-band structure entropy and TKEO read.
Recomputed before the averaging and fed to `mode_features` — the same 28 descriptors — the
decomposition is the only thing left varying (38 fixed Morlet bands vs 8 adaptive modes).

Paired across 5 CV seeds, since the single-split margin sits inside the noise floor:

| block | dim | k=all | k=12 |
| --- | --- | --- | --- |
| VMD modes | 227 | 0.7630 +/- 0.0136 | 0.7122 +/- 0.0137 |
| WST raw paths | 882 | 0.7219 +/- 0.0137 | 0.6650 +/- 0.0179 |
| **WST env + descriptors** | 1064 | **0.7563 +/- 0.0085** | **0.7383 +/- 0.0091** |

| paired difference | k=all | k=12 |
| --- | --- | --- |
| WST env+desc − VMD | -0.0067 +/- 0.0094, p=0.19, 1/5 | **+0.0260 +/- 0.0107, p=0.006, 5/5** |
| WST raw − VMD | -0.0411 +/- 0.0084, p=0.0004, 0/5 | -0.0472 +/- 0.0100, p=0.0005, 0/5 |
| WST env+desc − WST raw | +0.0344, p=0.0002, 5/5 | +0.0732, p=0.0001, 5/5 |

**At full dimensionality the gap vanishes (parity, p=0.19). At k=12 the sign reverses and
scattering wins by +0.0260 on every one of 5 seeds (p=0.006).**

And the CHF deficit — the clinically expensive failure this project flagged as the reason
to prefer VMD — is entirely an artefact of the raw-path form. CHF sensitivity at k=12,
mean over 5 seeds:

| block | CHF sens |
| --- | --- |
| VMD modes | 0.572 +/- 0.038 |
| **WST env + descriptors** | **0.576 +/- 0.034** |
| WST raw paths | 0.427 +/- 0.021 |

### 7c. What §3a actually established

Restated precisely:

* **True, and unchanged:** scattering *in its standard form* — raw time-averaged
  coefficients — is a worse front end than VMD-with-descriptors, by 0.041-0.047 macro-F1,
  p<0.001, every seed.
* **False:** that this is a property of the scattering transform. Give both decompositions
  the same descriptor bank and they are equivalent at full dimensionality and scattering is
  **better** under a 12-feature budget.

The §3a/§5 claim "nothing here argues for replacing VMD with scattering" is withdrawn. On
the k=12 evidence the argument now runs the other way.

### 7d. Why scattering wins specifically at k=12

It has 38 bands to VMD's 8 modes — nearly 5x the distinct objects to draw 12 descriptors
from. At full dimensionality VMD's adaptivity is competitive; under a tight selection
budget scattering's finer frequency resolution is what pays. That the advantage appears
exactly at k=12 matters here beyond the classical table: **k=12 is the qubit count.**

### 7e. What has not changed

The §3b selection tax survives intact, merely relocated. mRMR picks only **5.8 distinct
bands per 12** from the envelope block (5, 5, 5, 7, 7 per fold) — the same ~40% waste, now
spent on several descriptors of one band rather than adjacent time bins of one path. The
+0.0260 above is therefore achieved *while still paying it*.

**Caveat on dimensional matching.** The envelope block is 1064 columns against VMD's 227,
so the k=all row is not dimensionally matched and should be read as parity rather than as a
measurement. The k=12 row *is* matched — both capped at 12 — and that is where the result
lives. 76 of the 1064 columns are zero-variance (`peak_hz`, `zcr`, constants for a fixed
bank) and are inert.

### 7f. The run this now implies

The VQC operates at exactly k=12, where `WST env + descriptors` is the strongest front end
measured (0.7383 vs VMD's 0.7122). Every quantum number in §3 was produced on a front end
that is **not** the best available at its own budget. The obvious next run is the VQC on
this block, against the 0.6428 VMD arm, at the protocol of §2. Not yet run.
