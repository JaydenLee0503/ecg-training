# Reproduce LFCC + Swin from scratch

Self-contained bundle for training the LFCC + temporal Swin (`ecgdata_lfcc_small_swin_v1`)
from the raw `ECGData.mat`: window extraction, LFCC features, 15 Swin fits (seeds 0/1/2 ×
five patient folds, 40 epochs) and results. Nothing else from the repository is needed.

## Colab

1. Zip this folder as `reproduce_swin.zip` (the zip contains the `reproduce_swin/` folder).
2. In Colab choose a GPU runtime, upload the zip to `/content`, and open `lfcc_swin_training.ipynb`.
3. Uncomment and run the first code cell (unzip + pinned numpy/scipy/scikit-learn/pandas),
   restart the session, then run all cells.

Outputs are written inside the bundle: `attention method/features/ecgdata_lfcc_v1/` and
`attention method/results/ecgdata_lfcc_swin_retrain/`. Fits checkpoint every epoch; re-running
the training cell resumes.

## Contents

| Path | Purpose |
|---|---|
| `lfcc_swin_training.ipynb` | The notebook |
| `ECGData.mat` | Raw ECGs (162 lead rows, 128 Hz) |
| `ecgvmd/` | ECGData loader, verified patient map, window segmentation |
| `results/patient_vmd_wst_vqc/` | Frozen windows and patient folds the extraction is checked against |
| `attention method/ecgdata_protocol.json` | Frozen LFCC, model and training settings |
| `attention method/attention_ecg/` | LFCC extractor, Swin model, training and metrics code |

Do not edit these files: the code checks their hashes. They are byte-identical copies of the
repository originals as of 2026-10-17. `ECGData.mat` and `results/` are gitignored, so a Git
checkout of this folder contains only the code; the local folder has everything.

A different GPU or PyTorch version gives slightly different weights, so expect results close to,
not bit-identical with, the original 83.35% window accuracy / 0.8006 macro-F1 / 94.17% patient vote.
A one-fit CPU test of this folder reproduced every original predicted class for that fit.
