# ECG arrhythmia classification with Variational Mode Decomposition

Three exploratory notebooks merged into one pipeline. The classical signal-processing
half is finished and measured; the quantum deep-learning half plugs in at a documented
interface (`features/quantum_*.npz`).

**Task.** 162 ECG recordings (8.5 min each, 128 Hz) in three classes — **ARR**
(arrhythmia, 96 records), **CHF** (congestive heart failure, 30), **NSR** (normal sinus
rhythm, 36). Predict the class from the signal.

**Approach.** Cut each recording into short windows, decompose each window into `K`
intrinsic mode functions with VMD, reduce each IMF to 28 descriptors, select a
qubit-sized subset, and classify.

---

## Status — what is done, and what to run

**Finished and measured.** Loading, segmentation, the VMD solver and its convergence
bookkeeping, the 28-descriptor IMF feature extraction, record-wise cross-validation, the
VMD-vs-control ablation, the alpha sweep, and mRMR reduction to 12 features. Every number
in this README came out of that code and reproduces.

**In progress.** The quantum stage. `ecgvmd/quantum.py` holds the feature maps, the
kernel estimator and the variational classifier; `requirements.txt` pins PennyLane.
Results so far are in [QUANTUM_STAGE.md](QUANTUM_STAGE.md), the run-by-run chronology in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Headline: the quantum kernel reaches parity with
classical, and plain angle encoding turns out to be classically tractable by
construction. The classical→quantum handoff is unchanged:
`ecgvmd/select.py::quantum_ready` scales features into rotation angles and notebook 3
writes `features/quantum_<signature>_q12.npz`, though the estimators in
`ecgvmd/quantum.py` read the *full* feature file and select in-fold instead, which is the
honest path. See [The quantum stage](#the-quantum-stage) for the original plan.

**Current artefacts** (both on disk, both gitignored):

| file | contents |
|---|---|
| `features/fixed_K8_a2000_L500_it500_eeb2053b.npz` | 1620 windows x 282 features + raw IMF tensor |
| `features/quantum_fixed_K8_a2000_L500_it500_eeb2053b_q12.npz` | the 12 selected features, angle-scaled |

### Commands, with measured wall-clock

`V=~/venvs/test-ecg-training/bin/python`, single core, from the project root.

| what | command | time |
|---|---|---|
| wiring check, 3 windows/record | `$V run_pipeline.py --smoke` | 35 s |
| **reproduce the tables below** | `$V run_pipeline.py --n-per-record 10` | 1 min 17 s |
| every window (21222 of them) | `$V run_pipeline.py` | 8 min 47 s |
| the convergence experiment | `$V scripts/alpha_sweep.py` | ~40 min |

Two things about that table are easy to get wrong.

**`run_pipeline.py` with no arguments does not reproduce the results below.** Bare, it
segments *every* window — 21222 of them — and scores slightly differently (VMD+rhythm
0.777 rather than 0.786, and the everything-block rises to 0.794). The published tables
are the `--n-per-record 10` run, 1620 windows, which is also what `FAST = True` means in
notebook 2. `FAST = True` is not a smoke setting; it is the shipped configuration.

**Only notebook 3 writes `quantum_*.npz`.** `run_pipeline.py` stops at the full feature
file. If you need the quantum handoff rebuilt, run
`03_baseline_and_quantum_prep.ipynb` — there is no CLI path to it yet.

Read [Three things that will bite you](#three-things-that-will-bite-you) before trusting
any number here.

---

## Layout

```
ecgvmd/                 the library — import this, don't copy-paste from notebooks
  config.py             every tunable, in one frozen dataclass
  data.py               ECGData.mat loading (Colab-aware)
  vmd.py                the two solvers + convergence bookkeeping
  segment.py            fixed-grid and beat-aligned windowing, R-peak detection
  features.py           IMF feature extraction (the main event)
  evaluate.py           record-wise cross-validation, leakage measurement
  select.py             mRMR reduction to a qubit-sized feature set
  quantum.py            feature maps, quantum kernel, variational classifier

01_vmd_core.ipynb                    what VMD is, does the solver work, did it converge
02_imf_features.ipynb                windows -> IMFs -> feature matrix   [the main one]
03_baseline_and_quantum_prep.ipynb   is VMD earning its cost, and the quantum handoff

run_pipeline.py         command-line extraction, for when you don't want a notebook
scripts/alpha_sweep.py  the convergence/alpha experiment
scripts/quantum_kernel_probe.py   the quantum-kernel gate, with Gram diagnostics

QUANTUM_STAGE.md        what the quantum stage measured, and what it means
EXPERIMENT_LOG.md       run-by-run chronology, failures included
make_colab_bundle.py    zips the package for upload to Colab
legacy/                 the three original notebooks, superseded, kept for provenance

ECGData.mat             the data (70 MB, gitignored - distribute out of band)
features/               generated .npz artefacts  (gitignored)
results/                generated .csv results    (gitignored)
```

The three original notebooks and their pre-patch `.bak` copies now live in `legacy/`,
kept for provenance. Nothing in the active pipeline imports from there — see
`legacy/README.md` for what each one contributed.

---

## Running it in VS Code (WSL Ubuntu)

The environment is already set up at `~/venvs/test-ecg-training`, and
`.vscode/settings.json` points VS Code at it.

1. **Open the folder in VS Code connected to WSL.** From the WSL terminal:
   ```bash
   cd /mnt/d/Projects/test-ecg-training
   code .
   ```
   The window title should say `[WSL: Ubuntu]`. If it doesn't, use
   *Ctrl+Shift+P → WSL: Reopen Folder in WSL*.

2. **Open `02_imf_features.ipynb`** and click **Select Kernel** (top right) →
   *Jupyter Kernel...* → **`test-ecg-training`**.

   If it isn't listed, pick *Python Environments...* and choose
   `/home/jaydenlee/venvs/test-ecg-training/bin/python`.

3. **Run All.** Notebooks are at the project root and `jupyter.notebookFileRoot` is set
   to the workspace folder, so `ECGData.mat` resolves without any path fiddling.

Start with `02_imf_features.ipynb` — it is the one that produces something. `FAST = True`
in its config cell is 10 windows per record (1620 total) and takes a couple of minutes;
it is the setting that produced every table in this README, not a smoke mode.
`FAST = False` is all 21222 windows — about 9 minutes for extraction and scoring, longer
in the notebook because `KEEP_IMFS = True` also writes the raw mode tensor.

**If the kernel is missing entirely**, rebuild it:
```bash
python3 -m venv ~/venvs/test-ecg-training
~/venvs/test-ecg-training/bin/pip install -r requirements.txt
~/venvs/test-ecg-training/bin/python -m ipykernel install --user \
    --name test-ecg-training --display-name test-ecg-training
```

### Without a notebook

```bash
cd /mnt/d/Projects/test-ecg-training
V=~/venvs/test-ecg-training/bin/python

$V run_pipeline.py --smoke                    # 35 s, checks the wiring
$V run_pipeline.py --n-per-record 10          # 1 min, reproduces the published tables
$V run_pipeline.py                            # every window (21222), 9 min
$V run_pipeline.py --seg-mode beat --n-per-record 25 --keep-imfs
$V scripts/alpha_sweep.py                     # the convergence experiment, ~40 min
```

`run_pipeline.py --help` lists every option. The output `.npz` is what notebook 3 and
the quantum stage read. It does **not** write the quantum handoff file — only notebook 3
does that.

---

## Running it in Google Colab

Colab has numpy/scipy/sklearn/pandas/matplotlib preinstalled, so there is nothing to
install. The only question is how the runtime gets the `ecgvmd` package and the 70 MB
`.mat` file. Two options.

### Option A — Google Drive (recommended)

Do this once and every future session is a two-click start.

1. Locally: `python make_colab_bundle.py` → produces `ecgvmd_bundle.zip`.
2. In Drive, create the folder `MyDrive/test-ecg-training/`.
3. Unzip the bundle into it, and upload `ECGData.mat` alongside. You should end up with
   `MyDrive/test-ecg-training/ecgvmd/`, `.../run_pipeline.py`, `.../ECGData.mat`.
4. Upload the notebooks too (or open them from Drive with *File → Open notebook → Drive*).
5. In Colab, add **one cell at the very top** and run it before anything else:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   The notebook's own bootstrap cell then finds `MyDrive/test-ecg-training` on its own —
   it is already in the search path.

6. **Runtime → Run all.**

### Option B — upload each session

1. Locally: `python make_colab_bundle.py`.
2. Open the notebook in Colab (*File → Upload notebook*).
3. Run the bootstrap cell. It will not find the package and will open an upload dialog —
   select `ecgvmd_bundle.zip`.
4. Upload `ECGData.mat` the same way when the data cell prompts (this is the slow part,
   70 MB every session, which is why Option A exists).

### Colab notes

- **The bootstrap cell is the same code in both environments.** It tries `import ecgvmd`,
  then walks up from the working directory, then checks the two Colab locations, then
  falls back to an upload prompt. Nothing needs editing.
- **A GPU does not help.** VMD is FFT-bound on the CPU; the batched solver already uses
  numpy's vectorised path. Pick a standard CPU runtime.
- **Colab disconnects at ~90 minutes idle.** `FAST = False` on the free tier is a gamble;
  either keep `FAST = True`, or write the `.npz` to Drive so a disconnect isn't fatal:
  ```python
  out = "/content/drive/MyDrive/test-ecg-training/features/run1.npz"
  ```
- **Save results back to Drive**, not to `/content` — the latter vanishes with the runtime.

---

## What the pipeline does

```
ECGData.mat  ──load_ecgdata──▶  162 x 65536 float64, labels, record_ids
             ──segment──────▶  (B, 500) z-scored windows + labels + GROUPS
             ──extract_features─▶  VMD each window into K IMFs, 28 descriptors each
             ──evaluate─────▶  record-wise cross-validated macro-F1
             ──MRMRSelector─▶  12 features, qubit-sized
             ──quantum_ready▶  scaled to [0, pi]  ──▶  the quantum stage (not built)
```

**Feature blocks produced** (K=8):

| block | dim | what it is |
|---|---|---|
| `VMD modes` | 227 | 28 descriptors x 8 modes + 3 global |
| `rhythm only` | 9 | rate and HRV from R-peaks on the QRS-band modes |
| `VMD modes + rhythm` | 236 | the main set |
| `control (no VMD)` | 46 | the same descriptors on the *undecomposed* window — the thing VMD has to beat |
| `VMD + rhythm + control` | 282 | everything |

---

## Three things that will bite you

**1. Split by record, never by segment.** Segments from one recording are
near-duplicates. A random split lets the model recognise the patient instead of the
pathology, and on this dataset that is worth about **+0.12 macro-F1 of pure illusion**.
`groups` (the record id) travels with the data everywhere and
`StratifiedGroupKFold` is the default. Notebook 3 measures the gap explicitly.

**2. Check the capped fraction.** VMD's ADMM solver stops on `tol` *or* on `max_iter`. If
it stopped on `max_iter`, those are not the VMD modes — they are wherever the iteration
happened to be. Every `VMDResult` carries `.capped_fraction` and warns above 5%.
Counter-intuitively, **lower alpha converges more slowly** — a weak bandwidth penalty
leaves the centre frequencies sliding around each other for far longer:

`scripts/alpha_sweep.py` runs the whole grid. Fixed-grid windows, K=8, 1620 windows,
all 162 records, record-wise CV:

| alpha | % capped @500 | macro-F1 @500 | % capped @2000 | macro-F1 @2000 | residual energy |
|---:|---:|---:|---:|---:|---:|
| 5 | 100.0 | **0.808** | 99.4 | **0.801** | 0.00% |
| 50 | 99.5 | 0.798 | 64.6 | 0.784 | 0.01% |
| 200 | 93.7 | 0.761 | 15.3 | 0.774 | 0.08% |
| 500 | 75.6 | 0.780 | 3.4 | 0.757 | 0.32% |
| 2000 | 34.2 | 0.786 | **0.6** | 0.776 | 1.86% |
| 8000 | 21.2 | 0.723 | **0.0** | 0.723 | 11.11% |

Three things follow.

**alpha=5 never converges, and it does not matter.** Quadrupling the budget leaves it still
99.4% capped and moves macro-F1 only from 0.808 to 0.801. The score does not depend on where
you truncate, so the earlier "alpha=5 is best" finding is real, not an artefact of stopping
early. But the modes it produces are defined by `max_iter`, so that number is part of the
method and has to be reported.

**Only the low end is stuck.** With `max_iter=2000`, alpha=200 falls to 15% capped, alpha=2000
to 0.6% and alpha=8000 to zero. A genuinely converged decomposition is available — it just
isn't at alpha=5.

**Converging harder does not help classification.** Among fully converged settings the best is
alpha=2000 at 0.776, still 0.025 below unconverged alpha=5. And letting alpha=2000 itself run to
convergence moves it 0.786 → 0.776, i.e. slightly *worse* (though that gap is within noise on
162 records). Convergence buys interpretability, not accuracy.

So: **alpha=5 for the best score, alpha=2000 with `max_iter=2000` for a decomposition you can
defend as solved.** The shipped default is alpha=2000 / `max_iter=500` — a middle position, and
the one to change first if you want either property outright.

**3. Residual norm ratio is not residual energy.** `recon_norm_ratio` is
‖x − Σuₖ‖ / ‖x‖. The unexplained *energy* fraction is its **square**. A norm ratio of
0.21 means 4.3% of the energy is missing, not 21%. Both are exposed on `VMDResult`
because conflating them overstates the error roughly five-fold.

**And one to stay suspicious of:** the three classes come from three different source
databases, and the ablation says the confound is real. The highest-frequency mode `u8`,
centred well above the physiological ECG band, scores **0.743 on its own** against 0.786
for all eight modes together; removing it costs 0.080 (0.707 without, 0.786 with). The
most likely explanation is that the model recognises recording hardware — different
amplifiers, different anti-aliasing filters — rather than cardiac pathology. Settling it
needs a held-out recording from a fourth database, which this dataset cannot provide.
**Report both numbers and read the headline as an upper bound.**

---

## Measured results

1620 windows from all 162 records, K=8, alpha=2000, five-fold record-wise CV.

| feature block | dim | segment macro-F1 | record accuracy |
|---|---:|---:|---:|
| rhythm only | 9 | 0.564 | 0.741 |
| control (no VMD) | 46 | 0.648 | 0.741 |
| VMD modes | 227 | 0.767 | 0.852 |
| **VMD modes + rhythm** | 236 | **0.786** | **0.864** |
| VMD + rhythm + control | 282 | 0.775 | 0.852 |

VMD earns its cost: 0.786 against 0.648 for the same statistics computed without it.

| model | segment macro-F1 | record accuracy |
|---|---:|---:|
| HistGradientBoosting | 0.794 | 0.846 |
| Logistic regression (scaled) | 0.792 | 0.889 |
| Random forest | 0.786 | 0.864 |
| Extra trees | 0.780 | 0.833 |

A four-point spread across four very different estimators, against a fourteen-point gap
between feature blocks. The features are the work; the classifier is nearly a detail.

Reducing to a qubit-sized set (mRMR, selected **inside** each training fold):

| features kept | 4 | 8 | 12 | 16 | 24 | 236 |
|---|---:|---:|---:|---:|---:|---:|
| segment macro-F1 | 0.661 | 0.705 | 0.724 | 0.722 | 0.740 | 0.786 |

**12 is the shipped operating point.** It beats 8 by 0.019 for four more qubits, which
cost nothing on a simulator. The dip at 16 is noise, not a ceiling — 24 is better again,
so if the register ever gets cheaper, keep widening it.

---

## The quantum stage

Not built yet. This section is the design: the full chain, then the corrections to the
step list it was drafted from.

### The chain, end to end

```
 ECGData.mat                162 records x 65536 samples, 128 Hz
     |
 [1] segment                (1620, 500)  z-scored windows  + y + groups
     |
 [2] VMD                    (1620, 8, 500)  eight IMFs per window
     |
 [3] descriptors            (1620, 236)  28 per mode + rhythm
     |
 [4] mRMR select            (1620, 12)   <- the qubit budget is decided HERE
     |
 [5] scale to angles        (1620, 12) in [0, pi]     MUST be fit on the train fold
     |
 ================= classical | quantum ==================
     |
 [6] feature map            ONE of: angle (RY) | amplitude | ZZ second-order
     |                      -> a 12-qubit state |phi(x)>
     |
     +---- path A: quantum kernel ----> K[i,j] = |<phi(xi)|phi(xj)>|^2 -> SVM -> class
     |                                  (no trainable quantum parameters at all)
     |
     +---- path B: variational -------> [7] ansatz: parameterised rotations
                                            + entangling layer, repeated d times
                                        [8] measure <Z> per qubit
                                        [9] classical head -> 3 logits -> softmax
                                        [10] cross-entropy -> parameter-shift -> Adam
     |
 [11] evaluate              StratifiedGroupKFold on `groups`, macro-F1
                            target to beat: 0.7243 (in-fold mRMR-12)
```

Steps 1–5 exist and run. Step 5 exists in two forms: `quantum_ready` (correct, refits
per call) and the `X` array baked into `quantum_*.npz` (fitted on all data — see the
caveat below). Steps 6–11 are **partly built** — see
[QUANTUM_STAGE.md](QUANTUM_STAGE.md) for what was measured and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for the chronology. Path A (the quantum kernel)
is concluded; path B (the variational circuit) is in progress.

### Critique of the drafted step list

The list this was built from read: *quantum state → VMD → encode into qubits → angle
encoding → amplitude encoding → ZZ feature map → variational quantum circuit
(parameterised gates, entanglement layers, trainable parameters) → measure expectation
values → classification.* The shape is right. Six things in it are wrong or missing, in
descending order of how much damage they do.

**1. The three encodings are alternatives, not consecutive steps.** This is the one that
matters. Angle encoding, amplitude encoding and the ZZ feature map all occupy slot [6];
you choose one, or you benchmark them against each other. Running them in series is not
a thing — the second would overwrite the state the first prepared. And ZZ is not a
sibling of angle encoding but a *superset* of it: the second-order Pauli-Z expansion of
Havlicek et al. is single-qubit rotations (angle encoding) followed by entangling
`ZZ(phi(x_i, x_j))` phases. Writing them as a sequence hides that.

For this project the choice is close to forced:

| map | qubits for 12 features | depth | verdict here |
|---|---|---|---|
| angle (RY) | 12 | 1 layer | **start here** — already implemented, trivially interpretable |
| ZZ, 2nd order | 12 | 1 + 66 two-qubit gates per rep | the interesting one; try after angle works |
| amplitude | 4 | O(2^n) state prep | **skip** — see below |

**Amplitude encoding is the wrong choice at this scale and should be dropped.** Its
selling point is exponential compression, 2^n features into n qubits. With 12 features it
saves eight qubits on a simulator where twelve qubits costs nothing, in exchange for a
state-preparation circuit far deeper than everything else combined. Worse, it is lossy in
a way that matters here: normalising each row to unit L2 norm discards the row's overall
magnitude, and several selected features (`u5_tkeo`, `u5_wave_len`, `u6_tkeo`,
`u6_wave_len`) are
*energy* measures whose absolute scale is the class signal. `quantum_ready(mode=
"amplitude")` will happily do it; the classifier will be worse and it will not be obvious
why. It earns its place only if the feature budget grows past ~64.

**2. Feature selection is missing from the list, and it is where the qubit count comes
from.** "VMD → encode into qubits" skips a step that is doing real work. VMD produces
236 features; nothing encodes 236 features onto near-term hardware. `MRMRSelector` cuts
that to 12, and that number *is* the qubit count under angle encoding. It also costs
something measurable — 0.786 macro-F1 at 236 features against 0.724 at 12 — so it belongs
in the diagram where the loss can be seen, not left implicit.

**3. "Quantum state" is not a first step.** It is the output of step [6], not an input to
anything. The chain starts at the signal.

**4. Expectation values → classification is underspecified, and as written it only does
two classes.** A single `<Z>` on one qubit is a scalar in [-1, 1]: one number, one
decision boundary, two classes. This problem has three (ARR / CHF / NSR). The options,
in order of preference:

* measure `<Z>` on all eight qubits, feed the 8-vector to a small classical linear head,
  softmax over 3 logits — one circuit per sample, trains stably, and it is the honest
  hybrid model;
* measure `<Z>` on three designated qubits and softmax those directly — fewer classical
  parameters, but the three outputs are correlated in an uncontrolled way;
* three one-vs-rest circuits — triples the cost, and the class imbalance (96/30/36
  records) makes each binary problem badly skewed.

Take the first. Also decide *analytic vs shot-based*: on a simulator, take exact
expectation values. Sampling noise at 1024 shots is roughly +/-0.03 on each `<Z>`, which
is the same order as the differences between the classifiers in the table above, and it
will make every result unreadable.

**5. The training loop is absent entirely.** "Trainable parameters" names the parameters
but not what trains them. Needed: cross-entropy loss, parameter-shift-rule gradients (2
circuit evaluations per parameter per step — budget for it), Adam at ~0.01, minibatches
of 32, and a fixed seed for the initial parameters. A `RealAmplitudes`-style ansatz on 8
qubits with linear entanglement and d=2 is ~24 parameters, which is small enough that
barren plateaus are not yet the problem; full (all-to-all) entanglement at d=6 is where
gradients start vanishing, so add depth only against measured validation gain.

**6. The evaluation step is missing, and it is the point of the exercise.** Nothing in the
list says how the model is scored. Two constraints are non-negotiable here:

* **split on `groups`.** Record-wise, `StratifiedGroupKFold`, exactly as the classical
  side does. A random split is worth ~+0.12 macro-F1 of pure illusion on this dataset.
* **compare against `classical_baseline_f1_k` = 0.7243**, not `classical_baseline_f1` =
  0.7861. The first is mRMR-12 selected in-fold — the same 12 features the circuit sees.
  The second is all 236 features and is not the quantum model's competition.

Also worth knowing before committing to path B: **the ZZ feature map has a well-known
failure mode**. As feature dimension grows, kernel values concentrate — off-diagonal
entries collapse toward zero, the Gram matrix approaches the identity, and the SVM
memorises the training set. At 12 features it is usually still fine, but plot the Gram
matrix before trusting the score.

  **Measured, and the standard advice needs a caveat.** Both ends fail. Shrinking the
  data does pull the Gram off the identity, but overshoot it and every entry collapses
  toward 1.0 instead, which is the *opposite* failure — no discrimination left. On these
  features the usable band is narrow and the natural default sits outside it; see
  [QUANTUM_STAGE.md](QUANTUM_STAGE.md) finding 2 for the sweep.

### Two things to do before writing any circuit

1. **A classical MLP on the same 12 features.** If a small `MLPClassifier` on
   `quantum_*.npz` does not land near 0.724, the problem is in the encoding or the split,
   not the quantum layer — and it is enormously easier to find now.
2. **Path A before path B.** A quantum kernel plus `SVC(kernel="precomputed")` has no
   trainable quantum parameters, no optimiser, and no barren plateaus. It answers "does
   this feature map separate the classes at all" in one afternoon. Only if it does is a
   variational circuit worth building.

### The handoff file

Notebook 3 writes `features/quantum_<signature>_q12.npz`. That file is the interface; the
quantum model reads it and never reaches back into VMD.

| array | shape | meaning |
|---|---|---|
| `X` | (n, q) | features scaled to [0, pi], ready for angle encoding |
| `X_raw` | (n, q) | the same features unscaled |
| `y` | (n,) | ARR / CHF / NSR |
| `groups` | (n,) | record id — **split on this** |
| `feature_names` | (q,) | which descriptors were chosen |
| `enc_lo`, `enc_hi` | (q,) | the min/max limits used for the angle scaling |
| `encoding` | scalar | `"angle"` — the limits above are specific to it |
| `n_qubits` | scalar | 12 |
| `classical_baseline_f1` | scalar | 0.7861 — all 236 features, honest CV |
| `classical_baseline_f1_k` | scalar | 0.7243 — mRMR-12 in-fold — **the fair comparison** |
| `full_feature_file` | str | path to the full feature `.npz` |

The twelve features currently selected are `u5_tkeo`, `u6_shannon`, `u8_spec_entropy`,
`u5_spec_entropy`, `u6_tkeo`, `u4_spec_entropy`, `u5_wave_len`, `u8_spec_bandwidth`,
`u6_wave_len`, `u6_spec_entropy`, `u8_shannon`, `recon_norm_ratio`. Two things to note.
**Three of the twelve come from `u8`** — the mode the ablation says may be recording
hardware rather than physiology — so a quarter of the register may be encoding which
database a recording came from. And `recon_norm_ratio` is not a physiological descriptor
at all: it measures how much signal energy VMD failed to reconstruct on that window. It
is a decomposition-quality number, and it is plausibly another device fingerprint. Watch
both when interpreting any result.

mRMR is greedy, so the first eight entries are exactly the previous `q8` set; the last
four are what widening the register bought.

**Two caveats that matter.**

*The file is transductively leaky.* Both the mRMR selection and the min/max scaling in it
are fitted on the whole dataset. That is accepted so the file is a fixed, inspectable
artefact — but cross-validating a model *on this file* therefore gives an optimistic
number. To stay honest, point the quantum stage at `full_feature_file` and wrap the model
as `make_pipeline(MRMRSelector(k=12), quantum_model)`; selection then happens inside each
training fold.

*The [0, pi] range is angle-specific.* `enc_lo`/`enc_hi` were chosen for RY rotations. The
ZZ feature map conventionally takes data on a different range and applies its own
`phi(x_i) = x_i`, `phi(x_i, x_j) = (pi - x_i)(pi - x_j)`; feeding it [0, pi] data without
thinking gives a degenerate second-order term wherever a feature sits near pi. Re-scale
from `X_raw` for any map other than angle.

---

## Requirements

`requirements.txt` pins the versions this was developed against. Nothing exotic: numpy,
scipy, scikit-learn, pandas, matplotlib, jupyter. No `vmdpy` — the solver is in
`ecgvmd/vmd.py` and was verified against it to 2e-16.
