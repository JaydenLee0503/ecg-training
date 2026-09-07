#!/usr/bin/env python
"""Nested bandwidth selection for the quantum kernel — the leak-free version.

Every kernel number in QUANTUM_STAGE.md chose its bandwidth *transductively*: the
`TanhAngleScaler(scale=...)` was picked from Gram statistics computed over the whole
subsample, then scored with record-wise CV. That is the same class of error as fitting
`MRMRSelector` outside the fold — the thing this project is built to avoid — and it is
open item 3 in both QUANTUM_STAGE.md and EXPERIMENT_LOG.md.

This script fixes it the explicit way. For each outer fold: run an inner record-wise CV
*on the training records only*, pick the bandwidth that wins there, refit on the full
training fold, predict the held-out fold. The test fold never participates in choosing
anything. sklearn's GridSearchCV would need `groups` routed through the inner splitter,
which is fiddly enough to hide mistakes, so the loop is written out.

The comparison that matters is `nested` against `transductive`. If nested comes in
materially lower, the published 0.7246 was partly an artefact of choosing the
hyperparameter on the test folds, and the headline has to move.

The nested predictions are scored with the full panel — accuracy, per-class sensitivity
and specificity, F1 and the confusion matrix, at segment and record level — and saved
beside the CSV so any further metric can be recomputed without re-running the kernel.

    python scripts/kernel_nested_bw.py                    # the full 1620
    python scripts/kernel_nested_bw.py --n-per-record 2   # quick, n=324
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
from ecgvmd.quantum import QuantumKernelSVC, TanhAngleScaler

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

BANDWIDTHS = (0.15, 0.25, 0.35, 0.50, 0.75, 1.00)


def load(path, block):
    if path is None:
        c = [p for p in glob.glob("features/*.npz") if "quantum" not in p]
        if not c:
            sys.exit("no feature .npz found — run run_pipeline.py first")
        path = max(c, key=os.path.getmtime)
    f = np.load(path, allow_pickle=True)
    return path, f[f"X::{block}"], f["__y"], f["__groups"]


def subsample(y, g, n_per_record, seed=0):
    if not n_per_record:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate(
        [rng.choice(np.flatnonzero(g == r), min(n_per_record, (g == r).sum()),
                    replace=False) for r in np.unique(g)]))


def pipe(k, bw, embedding="angle"):
    return make_pipeline(E.MRMRSelector(k=k), TanhAngleScaler(scale=bw),
                         QuantumKernelSVC(embedding=embedding))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--block", default="VMD modes + rhythm")
    ap.add_argument("--n-per-record", type=int, default=0, help="0 = all windows")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedding", default="angle",
                    choices=["angle", "iqp", "iqp-state"],
                    help="'iqp-state' is the same map as 'iqp' via statevectors, ~1000x "
                         "faster; 'iqp' is the pairwise oracle and is hours at n=1620")
    ap.add_argument("--out", default="results/kernel_nested_bw.csv")
    args = ap.parse_args()

    path, X, y, g = load(args.features, args.block)
    idx = subsample(y, g, args.n_per_record, args.seed)
    X, y, g = X[idx], y[idx], g[idx]
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {path}")
    print(f"block {args.block!r}: {X.shape}, {len(np.unique(g))} records")
    print(f"embedding: {args.embedding}\nbandwidth grid: {list(BANDWIDTHS)}\n", flush=True)

    outer = StratifiedGroupKFold(5, shuffle=True, random_state=args.seed)
    t0 = time.time()

    # ---- nested: bandwidth chosen inside each training fold ----------------------
    pred = np.empty(len(y), dtype=object)
    picked, rows = [], []
    for fold, (tr, te) in enumerate(outer.split(X, y, g)):
        inner = StratifiedGroupKFold(args.inner_folds, shuffle=True,
                                     random_state=args.seed)
        scores = {}
        for bw in BANDWIDTHS:
            ip = np.empty(len(tr), dtype=object)
            for itr, ite in inner.split(X[tr], y[tr], g[tr]):
                m = pipe(args.k, bw, args.embedding).fit(X[tr][itr], y[tr][itr])
                ip[ite] = m.predict(X[tr][ite])
            scores[bw] = f1_score(y[tr], ip.astype(str), average="macro")
        best = max(scores, key=scores.get)
        picked.append(best)
        m = pipe(args.k, best, args.embedding).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        print(f"  fold {fold}: chose bw={best:.2f}  "
              + "  ".join(f"{b:.2f}={scores[b]:.4f}" for b in BANDWIDTHS), flush=True)
        rows.append({"fold": fold, "chosen_bw": best,
                     **{f"inner_f1_bw{b:.2f}": scores[b] for b in BANDWIDTHS}})
    nested_pred = pred.astype(str)
    nested = f1_score(y, nested_pred, average="macro")

    # ---- transductive: the published protocol, for the comparison ----------------
    trans, trans_pred = {}, {}
    for bw in BANDWIDTHS:
        p = np.empty(len(y), dtype=object)
        for tr, te in outer.split(X, y, g):
            p[te] = pipe(args.k, bw, args.embedding).fit(X[tr], y[tr]).predict(X[te])
        trans[bw] = f1_score(y, p.astype(str), average="macro")
        trans_pred[bw] = p.astype(str)
    best_trans = max(trans, key=trans.get)

    dt = time.time() - t0
    print(f"\n  transductive (bandwidth picked on the full sample — the published way):")
    for b in BANDWIDTHS:
        mark = "  <- best, this is what gets reported" if b == best_trans else ""
        print(f"    bw={b:.2f}  {trans[b]:.4f}{mark}")
    print(f"\n  NESTED  (bandwidth picked inside each training fold) : {nested:.4f}")
    print(f"  TRANSDUCTIVE best                                    : {trans[best_trans]:.4f}")
    print(f"  optimism from choosing bandwidth on the test folds    : "
          f"{trans[best_trans] - nested:+.4f}")
    print(f"  bandwidths chosen per fold: {picked}")
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] done in {dt:.0f}s")

    # ---- the full panel on the nested (leak-free) predictions --------------------
    name = f"{args.embedding} kernel, nested bw"
    print()
    print(E.metrics_report(y, nested_pred, g, title=name))

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df["nested_overall_f1"] = nested
    df["transductive_best_f1"] = trans[best_trans]
    df["transductive_best_bw"] = best_trans
    df["n_windows"] = len(X)
    df["recorded"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    metrics_path = args.out.replace(".csv", "_metrics.csv")
    pd.DataFrame([
        E.metrics_row(name, y, nested_pred, groups=g, protocol="nested",
                      embedding=args.embedding, k=args.k, seconds=round(dt, 1),
                      recorded=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        E.metrics_row(f"{args.embedding} kernel, transductive bw={best_trans:.2f}",
                      y, trans_pred[best_trans], groups=g, protocol="transductive",
                      embedding=args.embedding, k=args.k, seconds=round(dt, 1),
                      recorded=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]).to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path} — accuracy, sensitivity, specificity, F1, confusion")

    pred_path = args.out.replace(".csv", "_preds.npz")
    np.savez_compressed(pred_path, __y=np.asarray(y).astype(str),
                        __groups=np.asarray(g), nested=nested_pred,
                        **{f"transductive_bw{b:.2f}": trans_pred[b] for b in BANDWIDTHS})
    print(f"wrote {pred_path} — out-of-fold predictions")


if __name__ == "__main__":
    main()
