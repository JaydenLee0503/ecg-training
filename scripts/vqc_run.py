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

Every model is scored with the full panel — accuracy, per-class sensitivity and
specificity, F1 and the confusion matrix, at segment and record level — and its
out-of-fold predictions are saved next to the CSV. Reporting macro-F1 alone cost this
project a baseline it could not reproduce (EXPERIMENT_LOG.md, "still unresolved"); with
the predictions on disk any metric can be recomputed without paying 37 minutes again.

**The fitted models are kept too.** `cross_val_predict` throws every estimator away, so
until now a 37-minute run left no trained weights behind and every follow-up question
meant training again. Each fold's fitted pipeline goes to `<out>_models/` as joblib, and
the VQC's 111 parameters also as a plain .npz — see `VQCClassifier.save`.

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

import joblib
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
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
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="folds in parallel. Pair n_jobs=5 with OMP_NUM_THREADS=1: "
                         "the 64 KB statevector does not parallelise across gates, "
                         "so parallelise across folds instead. Do not oversubscribe.")
    ap.add_argument("--out", default="results/vqc_run.csv")
    ap.add_argument("--no-save-models", action="store_true",
                    help="skip writing the per-fold fitted pipelines and VQC weights")
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

    preds = {}
    splits = list(cv.split(X, y, g))
    fold_of = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(splits):
        fold_of[te] = k
    model_dir = args.out.replace(".csv", "_models")

    def fit_fold(est, tr, te):
        """One fold, fitted and predicted. Returns the model as well as the prediction —
        which is the whole reason this is a hand-written loop and not cross_val_predict."""
        m = clone(est).fit(X[tr], y[tr])
        return m, m.predict(X[te])

    def save_fold(name, m, k):
        slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(m, f"{model_dir}/{slug}_fold{k}.joblib", compress=3)
        last = m[-1] if hasattr(m, "steps") else m
        if isinstance(last, VQCClassifier):
            last.save(f"{model_dir}/{slug}_fold{k}_weights.npz")

    def score(name, est, note=""):
        t = time.time()
        fitted = Parallel(n_jobs=args.n_jobs)(
            delayed(fit_fold)(est, tr, te) for tr, te in splits)
        p = np.empty(len(y), dtype=object)
        for k, ((tr, te), (m, pf)) in enumerate(zip(splits, fitted)):
            p[te] = pf
            if not args.no_save_models:
                save_fold(name, m, k)
        p = p.astype(str)
        dt = time.time() - t
        preds[name] = p
        row = E.metrics_row(name, y, p, groups=g, seconds=round(dt, 1), k=args.k,
                            recorded=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"  {name:<44s} {row['macro_f1']:.4f}  ({dt:5.0f}s) {note}", flush=True)
        print(f"    acc {row['accuracy']:.4f}  bal-acc {row['balanced_accuracy']:.4f}  "
              f"macro-sens {row['macro_sensitivity']:.4f}  "
              f"macro-spec {row['macro_specificity']:.4f}  "
              f"record acc {row['record_accuracy']:.4f}", flush=True)
        print("    " + "  ".join(f"{c}: sens {row[f'sens_{c}']:.3f} / "
                                 f"spec {row[f'spec_{c}']:.3f}" for c in E.CLASS_ORDER),
              flush=True)
        rows.append(row)

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
    pred_path = args.out.replace(".csv", "_preds.npz")
    np.savez_compressed(pred_path, __y=np.asarray(y).astype(str),
                        __groups=np.asarray(g), __fold=fold_of, **preds)
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] wrote {args.out}")
    print(f"wrote {pred_path} — out-of-fold predictions, for recomputing any metric")
    if not args.no_save_models:
        print(f"wrote {model_dir}/ — per-fold fitted pipelines (.joblib) and, for the "
              f"VQC, its trained parameters (.npz)")

    for r in rows:
        print()
        print(E.metrics_report(y, preds[r["model"]], g, title=r["model"]))

    vq = df[df.model.str.startswith("VQC")].macro_f1
    if len(vq) > 1:
        print(f"\nVQC across {len(vq)} seeds: mean {vq.mean():.4f}  sd {vq.std():.4f}  "
              f"range [{vq.min():.4f}, {vq.max():.4f}]")
        for c in E.CLASS_ORDER:
            s = df[df.model.str.startswith("VQC")][f"sens_{c}"]
            print(f"  {c} sensitivity across seeds: mean {s.mean():.4f}  "
                  f"sd {s.std():.4f}")


if __name__ == "__main__":
    main()
