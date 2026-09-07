#!/usr/bin/env python
"""Stage 2 of the quantum plan: does the angle feature map separate the classes at all?

This is "path A" - a quantum kernel handed to `SVC(kernel="precomputed")`. There are
no trainable quantum parameters, no optimiser, and no barren plateaus, so it answers
the only question worth asking before a variational circuit is worth building. If the
kernel cannot separate ARR/CHF/NSR here, a VQC on the same feature map will not either.

Three things this script is careful about, each of which is a way to get a wrong answer:

**Selection happens inside the fold.** The pipeline is
`MRMRSelector(k) -> TanhAngleScaler -> QuantumKernelSVC`, cross-validated with
`StratifiedGroupKFold` on the record id. Reading `features/quantum_*.npz` directly
instead would inherit that file's transductive leak.

**The classical control runs on the identical rows and folds.** A quantum score from a
486-window subsample compared against the 1620-window classical baseline (0.7061) is
not a comparison. The controls here see exactly the same data.

**Subsampling is per record, not random.** Random rows would drop whole records, and
the group split needs all 162 of them.

    python scripts/quantum_kernel_probe.py                 # 3 windows/record, 486 total
    python scripts/quantum_kernel_probe.py --n-per-record 10   # the full 1620, ~10 min/fold
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecgvmd as E
from ecgvmd.quantum import QuantumKernelSVC, TanhAngleScaler, gram_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def subsample_per_record(y, groups, n_per_record, seed=0):
    """Row indices keeping at most `n_per_record` windows from each record."""
    rng = np.random.default_rng(seed)
    keep = []
    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        take = min(n_per_record, len(idx))
        keep.append(rng.choice(idx, take, replace=False))
    return np.sort(np.concatenate(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None,
                    help="feature .npz (default: newest in features/)")
    ap.add_argument("--block", default="VMD modes + rhythm")
    ap.add_argument("--n-per-record", type=int, default=3)
    ap.add_argument("--k", type=int, default=12, help="qubits = selected features")
    ap.add_argument("--bandwidths", type=float, nargs="+", default=[1.0])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/quantum_kernel_probe.csv")
    ap.add_argument("--gram-png", default="results/quantum_gram.png")
    args = ap.parse_args()

    path = args.features
    if path is None:
        import glob
        cands = [p for p in glob.glob("features/*.npz") if "quantum" not in p]
        if not cands:
            sys.exit("no feature .npz found - run run_pipeline.py first")
        path = max(cands, key=os.path.getmtime)
    print(f"features: {path}")

    f = np.load(path, allow_pickle=True)
    X_all = f[f"X::{args.block}"]
    names = f[f"names::{args.block}"]
    y_all, g_all = f["__y"], f["__groups"]

    sub = subsample_per_record(y_all, g_all, args.n_per_record, args.seed)
    X, y, g = X_all[sub], y_all[sub], g_all[sub]
    print(f"block {args.block!r}: {X_all.shape} -> subsample {X.shape}, "
          f"{len(np.unique(g))} records, classes {dict(zip(*np.unique(y, return_counts=True)))}")

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=args.seed)
    rows = []

    def score(name, est, note=""):
        t = time.time()
        pred = cross_val_predict(est, X, y, cv=cv, groups=g)
        dt = time.time() - t
        row = E.metrics_row(name, y, pred, groups=g, seconds=round(dt, 1), k=args.k,
                            note=note)
        print(f"  {name:<44s} macro-F1 {row['macro_f1']:.4f}   ({dt:6.1f}s) {note}")
        print(f"    acc {row['accuracy']:.4f}  macro-sens {row['macro_sensitivity']:.4f}"
              f"  macro-spec {row['macro_specificity']:.4f}  "
              + "  ".join(f"{c} sens {row[f'sens_{c}']:.3f}" for c in E.CLASS_ORDER))
        rows.append(row)
        return row["macro_f1"]

    print("\nclassical controls (identical rows, identical folds):")
    score("RF (all features, no selection)",
          RandomForestClassifier(400, random_state=args.seed, n_jobs=-1))
    score(f"RF, mRMR-{args.k} in-fold",
          make_pipeline(E.MRMRSelector(k=args.k),
                        RandomForestClassifier(400, random_state=args.seed, n_jobs=-1)))
    score(f"SVC-rbf, mRMR-{args.k}, standardised",
          make_pipeline(E.MRMRSelector(k=args.k), StandardScaler(), SVC(C=args.C)))
    score(f"SVC-rbf, mRMR-{args.k}, tanh-angle scaled",
          make_pipeline(E.MRMRSelector(k=args.k), TanhAngleScaler(), SVC(C=args.C)),
          "<- the matched control")

    print(f"\nquantum kernel ({args.k} qubits, RY angle embedding, analytic):")
    for bw in args.bandwidths:
        score(f"QuantumKernelSVC, mRMR-{args.k}, bandwidth={bw:g}",
              make_pipeline(E.MRMRSelector(k=args.k), TanhAngleScaler(scale=bw),
                            QuantumKernelSVC(C=args.C)),
              f"bw={bw:g}")

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # ---- the Gram matrix, which must be looked at before any score is trusted -------
    sel = E.MRMRSelector(k=args.k).fit(X, y)
    print(f"\nselected: {', '.join(np.asarray(names)[sel.idx_])}")
    order = np.argsort(y, kind="stable")
    Xsel = sel.transform(X)[order]

    G = None
    for bw in args.bandwidths:
        Gb = gram_matrix(TanhAngleScaler(scale=bw).fit_transform(Xsel))
        off = Gb[np.triu_indices(len(Gb), 1)]
        flag = ""
        if off.mean() < 0.05:
            flag = "  <- concentrated toward 0, the SVM is memorising"
        elif off.mean() > 0.90:
            flag = "  <- concentrated toward 1, no discrimination left"
        print(f"Gram off-diagonal @ bandwidth={bw:<5g} mean {off.mean():.4f}  "
              f"median {np.median(off):.4f}  frac>0.01 {(off > 0.01).mean():.3f}{flag}")
        if G is None:
            G = Gb  # plot the first, which is the one the caller led with

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        im = axes[0].imshow(G, cmap="magma", vmin=0, vmax=1)
        axes[0].set_title(f"Gram matrix, {args.k} qubits (sorted by class)")
        bounds = np.cumsum([np.sum(y == c) for c in E.CLASS_ORDER])[:-1]
        for b in bounds:
            axes[0].axhline(b - .5, c="cyan", lw=.8)
            axes[0].axvline(b - .5, c="cyan", lw=.8)
        fig.colorbar(im, ax=axes[0], fraction=.046)
        axes[1].hist(off, bins=60, color="#2471a3")
        axes[1].set_title("off-diagonal distribution")
        axes[1].set_xlabel("kernel value")
        fig.tight_layout()
        fig.savefig(args.gram_png, dpi=130)
        print(f"wrote {args.gram_png}")
    except Exception as e:  # plotting is a convenience, not the result
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
