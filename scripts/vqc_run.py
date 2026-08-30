#!/usr/bin/env python
"""Stage 4: the variational quantum classifier, cross-validated against matched rivals.

The circuit here is genuinely entangled — `StronglyEntanglingLayers` interleaves
parameterised rotations with a ring of CNOTs, so the state does not factorise and the
`product_angle_kernel` shortcut that made the *kernel* classical does not apply. Whatever
this measures, it measures about a quantum model.

Three things this script is careful about.

**The epoch budget is fixed before the run, not chosen from it.** `--epochs` should come
from a training curve on a single fold (`--curve`), not from running the five-fold at
several budgets and keeping the best — that selects on the test folds. The curve mode
exists so the budget has an honest provenance.

**Everything is selected in-fold.** `MRMRSelector` and `TanhAngleScaler` sit inside the
pipeline, so the 12 features and the scaling limits are fitted on training data only.

**The rivals see identical rows and folds.** A quantum number compared against a
baseline computed on different data is not a comparison.

    python scripts/vqc_run.py --curve                 # one fold, training curve
    python scripts/vqc_run.py --epochs 60             # the five-fold run
    python scripts/vqc_run.py --epochs 60 --seeds 3   # repeat, to see the spread
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecgvmd as E
from ecgvmd.quantum import QuantumKernelSVC, TanhAngleScaler, VQCClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(path, block):
    if path is None:
        c = [p for p in glob.glob("features/*.npz") if "quantum" not in p]
        if not c:
            sys.exit("no feature .npz found — run run_pipeline.py first")
        path = max(c, key=os.path.getmtime)
    f = np.load(path, allow_pickle=True)
    return path, f[f"X::{block}"], f["__y"], f["__groups"]


def subsample(y, g, n_per_record, seed):
    if not n_per_record:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate(
        [rng.choice(np.flatnonzero(g == r), min(n_per_record, (g == r).sum()),
                    replace=False) for r in np.unique(g)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--block", default="VMD modes + rhythm")
    ap.add_argument("--n-per-record", type=int, default=0, help="0 = all windows")
    ap.add_argument("--k", type=int, default=12, help="qubits = selected features")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=1, help="repeat the whole CV this often")
    ap.add_argument("--curve", action="store_true",
                    help="one fold with periodic evaluation, to choose --epochs")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--out", default="results/vqc_run.csv")
    args = ap.parse_args()

    path, X, y, g = load(args.features, args.block)
    idx = subsample(y, g, args.n_per_record, 0)
    X, y, g = X[idx], y[idx], g[idx]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {path}")
    print(f"block {args.block!r}: {X.shape}, {len(np.unique(g))} records, "
          f"classes {dict(zip(*np.unique(y, return_counts=True)))}")

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)

    # ---- curve mode: one fold, so --epochs has an honest provenance --------------
    if args.curve:
        tr, te = next(cv.split(X, y, g))
        sel = E.MRMRSelector(k=args.k).fit(X[tr], y[tr])
        sca = TanhAngleScaler().fit(sel.transform(X[tr]))
        Xtr = sca.transform(sel.transform(X[tr]))
        Xte = sca.transform(sel.transform(X[te]))
        print(f"curve mode: train {len(Xtr)} / test {len(Xte)}\n", flush=True)
        t = time.time()
        m = VQCClassifier(n_layers=args.layers, epochs=args.epochs, lr=args.lr,
                          batch_size=args.batch_size, seed=0, verbose=True,
                          eval_set=(Xte, y[te]), eval_every=args.eval_every).fit(Xtr, y[tr])
        h = pd.DataFrame(m.history_)
        os.makedirs("results", exist_ok=True)
        h.to_csv("results/vqc_curve.csv", index=False)
        best = h.loc[h["val_f1"].idxmax()]
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] done in {time.time()-t:.0f}s")
        print(f"best val_f1 {best.val_f1:.4f} at epoch {int(best.epoch)} "
              f"({int(best.steps)} steps), train {best.train_f1:.4f}, gap {best.gap:+.4f}")
        print("wrote results/vqc_curve.csv")
        print("\nUse that epoch as --epochs for the five-fold run. Do NOT run the "
              "five-fold at several budgets and keep the best — that selects on test.")
        return

    # ---- the five-fold run, with matched rivals ----------------------------------
    rows = []

    def score(name, est, note=""):
        t = time.time()
        p = cross_val_predict(est, X, y, cv=cv, groups=g)
        f1 = f1_score(y, p, average="macro")
        dt = time.time() - t
        print(f"  {name:<44s} {f1:.4f}  ({dt:5.0f}s) {note}", flush=True)
        rows.append({"model": name, "macro_f1": f1, "seconds": round(dt, 1),
                     "n_windows": len(X), "k": args.k,
                     "recorded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    print("\nclassical rivals (identical rows and folds):", flush=True)
    score("RF, mRMR-12 in-fold",
          make_pipeline(E.MRMRSelector(k=args.k),
                        RandomForestClassifier(400, random_state=0, n_jobs=-1)))
    score("MLP, mRMR-12 in-fold",
          make_pipeline(E.MRMRSelector(k=args.k), StandardScaler(),
                        MLPClassifier((32,), max_iter=800, random_state=0)))
    score("product-cosine kernel, bw=0.35",
          make_pipeline(E.MRMRSelector(k=args.k), TanhAngleScaler(scale=0.35),
                        QuantumKernelSVC()), "<- classical, despite the name")

    print(f"\nVQC ({args.k} qubits, depth {args.layers}, {args.epochs} epochs, "
          f"lr {args.lr}, batch {args.batch_size}):", flush=True)
    for s in range(args.seeds):
        score(f"VQCClassifier, seed={s}",
              make_pipeline(E.MRMRSelector(k=args.k), TanhAngleScaler(),
                            VQCClassifier(n_layers=args.layers, epochs=args.epochs,
                                          lr=args.lr, batch_size=args.batch_size, seed=s)))

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] wrote {args.out}")
    vq = df[df.model.str.startswith("VQC")].macro_f1
    if len(vq) > 1:
        print(f"VQC across {len(vq)} seeds: mean {vq.mean():.4f}  sd {vq.std():.4f}  "
              f"range [{vq.min():.4f}, {vq.max():.4f}]")


if __name__ == "__main__":
    main()
