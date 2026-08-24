# ECG arrhythmia classification with Variational Mode Decomposition

Three exploratory notebooks merged into one pipeline. The classical signal-processing
half is finished and measured; the quantum deep-learning half plugs in at a documented
interface (`features/quantum_*.npz`).

**Task.** 162 ECG recordings (8.5 min each, 128 Hz) in three classes — **ARR**
(arrhythmia, 96 records), **CHF** (congestive heart failure, 30), **NSR** (normal sinus
rhythm, 36). Predict the class from the signal.

**Approach.** Cut each recording into short windows, decompose each window into `K`
intrinsic mode functions with VMD, reduce each IMF to 28 descriptors, and classify.

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

01_vmd_core.ipynb                    what VMD is, does the solver work, did it converge
02_imf_features.ipynb                windows -> IMFs -> feature matrix   [the main one]
03_baseline_and_quantum_prep.ipynb   is VMD earning its cost, and the quantum handoff

run_pipeline.py         command-line extraction, for when you don't want a notebook
scripts/alpha_sweep.py  the convergence/alpha experiment
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
in its config cell takes about 3 minutes; `FAST = False` processes every window and takes
roughly 40 minutes on one core.

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

$V run_pipeline.py --smoke                    # ~1 min, checks the wiring
$V run_pipeline.py                            # every window, K=8, alpha=2000
$V run_pipeline.py --seg-mode beat --n-per-record 25 --keep-imfs
$V scripts/alpha_sweep.py                     # the convergence experiment
```

`run_pipeline.py --help` lists every option. The output `.npz` is what notebook 3 and
the quantum stage read.

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
             ──MRMRSelector─▶  8–16 features, qubit-sized
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

---

## The quantum stage

Notebook 3 ends by writing `features/quantum_<signature>_q8.npz`. That file is the
interface; the quantum model reads it and never reaches back into VMD.

| array | shape | meaning |
|---|---|---|
| `X` | (n, q) | features scaled to [0, π], ready for angle encoding |
| `X_raw` | (n, q) | the same features unscaled |
| `y` | (n,) | ARR / CHF / NSR |
| `groups` | (n,) | record id — **split on this** |
| `feature_names` | (q,) | which descriptors were chosen |
| `enc_lo`, `enc_hi` | (q,) | the min/max limits used for the angle scaling |
| `classical_baseline_f1` | scalar | all 236 features, honest CV |
| `classical_baseline_f1_k` | scalar | mRMR-8 selected **in-fold** — the fair comparison |
| `full_feature_file` | str | path to the full feature `.npz` |

**One caveat that matters.** The mRMR selection and the min/max scaling in that file are
fitted on the whole dataset. That is a small transductive leak, accepted so the file is a
fixed, inspectable artefact. Cross-validating a model *on this file* therefore gives an
optimistic number. To stay honest, point the quantum stage at `full_feature_file` and wrap
the model as `make_pipeline(MRMRSelector(k=8), quantum_model)` — selection then happens
inside each training fold, and `classical_baseline_f1_k` is the number to compare against.

Before adding a variational circuit, run a small classical MLP on the same 8 features and
confirm it lands near the mRMR-8 row in notebook 3. If it doesn't, the problem is in the
encoding, not the quantum layer — much easier to find now than later.

---

## Requirements

`requirements.txt` pins the versions this was developed against. Nothing exotic: numpy,
scipy, scikit-learn, pandas, matplotlib, jupyter. No `vmdpy` — the solver is in
`ecgvmd/vmd.py` and was verified against it to 2e-16.
