#!/usr/bin/env python
"""Wavelet scattering vs VMD as the front end to the same quantum classifier.

This is the historical exploratory runner, including asymmetric rhythm features and
the separate Morlet-envelope descriptor arm. Use scripts/matched_vqc.py for the
corrected two-arm, patient-grouped comparison with converged VMD.

The question is about the FRONT END, not about the classifier. So everything after the
front end is held fixed and identical: the same 1620 windows in the same order, the same
`StratifiedGroupKFold(5, random_state=0)` splits, the same in-fold `MRMRSelector(k=12)`,
the same `TanhAngleScaler`, the same 12-qubit depth-2 circuit, the same 40-epoch budget,
the same three seeds. Only the matrix they are handed differs.

**Row alignment is asserted, not assumed.** The VMD block is read from the feature .npz
that `run_pipeline.py` wrote in August; the scattering block is computed here from
`segment()`. Those are two separate paths to the same 1620 rows, and if they ever disagree
the comparison is meaningless, so the script checks `y` and `groups` elementwise and dies
if they differ. That is cheaper than re-running VMD (30 min) and strictly safer than
trusting that the config has not drifted.

**The epoch budget is VMD's, used unchanged.** 40 epochs came from a training curve on
VMD features (`results/vqc_curve.csv`, EXPERIMENT_LOG e19). Re-tuning it for scattering
and not for VMD would hand scattering an advantage that has nothing to do with the front
end; tuning both against these same five folds would select on the test folds. So both
get 40 and `--curve` reports, separately and as a diagnostic only, whether scattering was
still improving when the budget ran out. Do not promote a curve number to the table.

**The scattering block is declared before the run, not chosen after it.** `WST log` -
every path, every time bin, log on orders 1 and 2 - is the primary. It is the standard
form of the transform (Anden & Mallat 2014) and it is what `scatter_features` defaults
to. The bin-mean and linear variants are scored with RF only, as context for reading the
primary, and never with the VQC.

    OMP_NUM_THREADS=1 python scripts/wst_vs_vmd_vqc.py --n-jobs 5
    OMP_NUM_THREADS=4 python scripts/wst_vs_vmd_vqc.py --curve
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
from ecgvmd.quantum import TanhAngleScaler, VQCClassifier

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

#: The block the VQC is run on, fixed before the run. See the module docstring.
PRIMARY = "WST log"
VMD_BLOCK = "VMD modes + rhythm"
#: The S7 block: order-1 envelopes + VMD's 28 descriptors. Best measured at k=12.
WSTDESC = "WST env + descriptors"


def load_vmd(path=None, block=VMD_BLOCK):
    """The VMD block, its labels/groups, AND the Config it was built under.

    Reading the config back out of the .npz rather than assuming `CFG` is not
    defensiveness for its own sake: this artefact was written with `n_per_record=10`
    (1620 windows), while `CFG` defaults to 0, which means all 21222. Segmenting under
    the wrong one produces a different set of rows, and then nothing about the
    comparison holds. `FeatureBundle.save` stores the config for exactly this reason.
    """
    import json
    if path is None:
        c = [p for p in glob.glob("features/*.npz") if "quantum" not in p]
        if not c:
            sys.exit("no feature .npz found - run run_pipeline.py first")
        path = max(c, key=os.path.getmtime)
    f = np.load(path, allow_pickle=True)
    cfg = E.Config(**json.loads(str(f["__config"])))
    return path, f[f"X::{block}"], f["__y"], f["__groups"], cfg


def build_blocks(args):
    """Every design matrix in the comparison, on one provably shared set of rows."""
    vmd_path, Xv, y_v, g_v, cfg = load_vmd(args.features)
    print(f"VMD block {VMD_BLOCK!r}: {Xv.shape} from {vmd_path}")
    print(f"  its config: {cfg.summary()} | n_per_record={cfg.n_per_record}")

    ds = E.load_ecgdata(args.mat)
    W, y, g = E.segment(ds, cfg)
    print(f"segmented: {W.shape} from {len(np.unique(g))} patients")

    # The whole comparison rests on this.
    if W.shape[0] != len(y_v):
        sys.exit(f"row count differs: {W.shape[0]} segmented vs {len(y_v)} in the .npz, "
                 f"under the .npz's own config. Something other than segmentation "
                 f"changed - do not paper over this.")
    if not np.array_equal(np.asarray(y).astype(str), np.asarray(y_v).astype(str)):
        raise ValueError("Labels differ between segment() and the .npz; rows are not aligned")
    if not np.array_equal(np.asarray(g), np.asarray(g_v)):
        raise ValueError("Feature archive does not use verified patient groups. "
                         "Use scripts/matched_vqc.py for the corrected comparison.")
    print("row alignment: labels and groups match the .npz elementwise")

    t = time.time()
    res = E.scatter_batch(W, J=args.J, Q=(args.Q1, args.Q2), T=args.T,
                          max_order=args.max_order, fs=cfg.fs)
    print(f"\n{res!r}\n  {time.time() - t:.1f}s for {W.shape[0]} windows "
          f"({res.bin_seconds:.2f} s bin spacing, "
          f"{res.invariance_seconds:.2f} s averaging scale)")

    blocks = {VMD_BLOCK: Xv}
    blocks[PRIMARY] = E.scatter_features(res, log=True)[0]
    blocks["WST log, bin-mean"] = E.scatter_features(res, log=True, reduce="mean")[0]
    blocks["WST linear"] = E.scatter_features(res, log=False)[0]

    # The block architects/wst_vs_vmd_vqc.md S7 shows is the strongest at k=12 - the
    # order-1 envelopes carrying VMD's own 28 descriptors, rather than raw time-averaged
    # coefficients. S7f: every quantum number above was produced on a front end that is
    # not the best available at the VQC's own budget.
    from ecgvmd.features import mode_features
    env, xi_hz = E.order1_envelopes(W, J=args.J, Q=(args.Q1, args.Q2), T=args.T,
                                    fs=cfg.fs)
    blocks[WSTDESC] = mode_features(
        env, np.broadcast_to(xi_hz, (len(W), len(xi_hz))), cfg.fs, x=W, prefix="w")[0]
    _, names = E.scatter_features(res, log=True)
    return blocks, np.asarray(y), np.asarray(g), res, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None, help="the VMD feature .npz")
    ap.add_argument("--mat", default=None)
    ap.add_argument("-J", type=int, default=6)
    ap.add_argument("--Q1", type=int, default=8)
    ap.add_argument("--Q2", type=int, default=1)
    ap.add_argument("-T", type=int, default=64)
    ap.add_argument("--max-order", type=int, default=2)
    ap.add_argument("--k", type=int, default=12, help="qubits = selected features")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--curve", action="store_true",
                    help="diagnostic: one fold with periodic eval, both front ends")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-vqc", action="store_true", help="RF table only")
    ap.add_argument("--vqc-blocks", choices=["both", "wst", "vmd", "binmean", "wstdesc"],
                    default="both",
                    help="which arms get the VQC. 'wst' is for when the VMD arm is "
                         "already on disk from an earlier run at this exact protocol "
                         "(results/e25_vqc_fivefold.csv) and paying for it again buys "
                         "nothing. 'binmean' is the selection control: 126 features, "
                         "12/12 distinct paths by construction, so if it beats 'wst' "
                         "the residual gap is mrmr_select on a time x frequency grid "
                         "and not the transform.")
    ap.add_argument("--out", default="results/wst_vs_vmd_vqc.csv")
    args = ap.parse_args()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] wst_vs_vmd_vqc")
    blocks, y, g, res, names = build_blocks(args)
    print("\nblocks:")
    for k, X in blocks.items():
        print(f"  {k:<22s} {X.shape[1]:4d} features")
    print(f"  classes {dict(zip(*np.unique(y, return_counts=True)))}")

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    splits = list(cv.split(blocks[VMD_BLOCK], y, g))
    fold_of = np.empty(len(y), dtype=int)
    for i, (_, te) in enumerate(splits):
        fold_of[te] = i

    # ---- curve mode: diagnostic only, never promoted to the table ----------------
    if args.curve:
        tr, te = splits[0]
        for bname in (VMD_BLOCK, PRIMARY):
            X = blocks[bname]
            sel = E.MRMRSelector(k=args.k).fit(X[tr], y[tr])
            sca = TanhAngleScaler().fit(sel.transform(X[tr]))
            Xtr, Xte = (sca.transform(sel.transform(X[i])) for i in (tr, te))
            print(f"\n--- curve, {bname} ({X.shape[1]} -> {args.k}) ---", flush=True)
            m = VQCClassifier(n_layers=args.layers, epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch_size, seed=0, verbose=True,
                              eval_set=(Xte, y[te]),
                              eval_every=args.eval_every).fit(Xtr, y[tr])
            h = pd.DataFrame(m.history_)
            os.makedirs("results", exist_ok=True)
            slug = "".join(c if c.isalnum() else "_" for c in bname).strip("_")
            h.to_csv(f"results/wst_vs_vmd_curve_{slug}.csv", index=False)
            b = h.loc[h.val_f1.idxmax()]
            print(f"  best val_f1 {b.val_f1:.4f} at epoch {int(b.epoch)}, "
                  f"train {b.train_f1:.4f}, gap {b.gap:+.4f}; "
                  f"at epoch {args.epochs}: val {h.val_f1.iloc[-1]:.4f}")
        print("\nDIAGNOSTIC ONLY. One fold, and its test half was used to evaluate. "
              "Do not read these as scores.")
        return

    rows, preds = [], {}

    def fit_fold(est, tr, te, X):
        return clone(est).fit(X[tr], y[tr]).predict(X[te])

    def score(name, est, X, note=""):
        t = time.time()
        out = Parallel(n_jobs=args.n_jobs)(
            delayed(fit_fold)(est, tr, te, X) for tr, te in splits)
        p = np.empty(len(y), dtype=object)
        for (tr, te), pf in zip(splits, out):
            p[te] = pf
        p = p.astype(str)
        dt = time.time() - t
        preds[name] = p
        row = E.metrics_row(name, y, p, groups=g, seconds=round(dt, 1), k=args.k,
                            dim=X.shape[1],
                            recorded=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        rows.append(row)
        print(f"  {name:<44s} {row['macro_f1']:.4f}  ({dt:5.0f}s) {note}", flush=True)
        print(f"    acc {row['accuracy']:.4f}  bal-acc {row['balanced_accuracy']:.4f}  "
              f"macro-sens {row['macro_sensitivity']:.4f}  "
              f"macro-spec {row['macro_specificity']:.4f}  "
              f"record acc {row['record_accuracy']:.4f}", flush=True)
        print("    " + "  ".join(f"{c}: sens {row[f'sens_{c}']:.3f} / "
                                 f"spec {row[f'spec_{c}']:.3f}" for c in E.CLASS_ORDER),
              flush=True)
        # Flush to disk after every model. A VQC seed costs ~12 minutes; losing a
        # finished one because the run was interrupted later is pure waste.
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        np.savez_compressed(args.out.replace(".csv", "_preds.npz"),
                            __y=np.asarray(y).astype(str), __groups=np.asarray(g),
                            __fold=fold_of, **preds)

    print("\nclassical reference - RF on mRMR-12, identical folds:", flush=True)
    for bname, X in blocks.items():
        score(f"RF mRMR-12 | {bname}",
              make_pipeline(E.MRMRSelector(k=args.k),
                            RandomForestClassifier(400, random_state=0, n_jobs=-1)), X)

    # PRIMARY first, deliberately. The scattering arm is the new information; the VMD
    # arm reproduces a number already on disk. If the run is cut short, the half that
    # survives should be the half nobody has yet.
    vqc_blocks = {"both": (PRIMARY, VMD_BLOCK), "wst": (PRIMARY,),
                  "vmd": (VMD_BLOCK,),
                  "binmean": ("WST log, bin-mean",),
                  "wstdesc": (WSTDESC,)}[args.vqc_blocks]

    if not args.no_vqc:
        print(f"\nVQC ({args.k} qubits, depth {args.layers}, {args.epochs} epochs, "
              f"lr {args.lr}, batch {args.batch_size}) on "
              f"{', '.join(vqc_blocks)} - each seed is written to the CSV as it lands:",
              flush=True)
        for bname in vqc_blocks:
            for s in range(args.seeds):
                score(f"VQC seed={s} | {bname}",
                      make_pipeline(E.MRMRSelector(k=args.k), TanhAngleScaler(),
                                    VQCClassifier(n_layers=args.layers,
                                                  epochs=args.epochs, lr=args.lr,
                                                  batch_size=args.batch_size, seed=s)),
                      blocks[bname])

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pred_path = args.out.replace(".csv", "_preds.npz")
    np.savez_compressed(pred_path, __y=np.asarray(y).astype(str), __groups=np.asarray(g),
                        __fold=fold_of, **preds)
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] wrote {args.out} and {pred_path}")

    # ---- the comparison, stated as a difference of means over seeds --------------
    if not args.no_vqc:
        print("\n" + "=" * 78)
        print(f"VQC across {args.seeds} seeds, same folds, same circuit, same budget:")
        summary = {}
        for bname in vqc_blocks:
            v = df[df.model.str.endswith(bname) & df.model.str.startswith("VQC")]
            summary[bname] = v
            print(f"\n  {bname} ({blocks[bname].shape[1]} features -> {args.k} qubits)")
            print(f"    macro-F1  mean {v.macro_f1.mean():.4f}  sd {v.macro_f1.std():.4f}"
                  f"  range [{v.macro_f1.min():.4f}, {v.macro_f1.max():.4f}]")
            print(f"    record acc mean {v.record_accuracy.mean():.4f}")
            for c in E.CLASS_ORDER:
                print(f"    {c} sens mean {v[f'sens_{c}'].mean():.4f} "
                      f"sd {v[f'sens_{c}'].std():.4f}")
        if PRIMARY in summary and VMD_BLOCK in summary:
            a, b = summary[PRIMARY].macro_f1, summary[VMD_BLOCK].macro_f1
            d = a.mean() - b.mean()
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            print(f"\n  scattering - VMD = {d:+.4f} macro-F1  "
                  f"(pooled seed sd {pooled:.4f}, n={len(a)} seeds each)")
            print("  With three seeds per arm this is a direction, not a p-value. A "
                  "difference inside\n  one pooled sd is not a result.")
        else:
            missing = [b for b in (PRIMARY, VMD_BLOCK) if b not in summary]
            print(f"\n  {', '.join(missing)} not run here - compare by hand against the "
                  f"arm on disk,\n  and only if its protocol matches this one exactly.")
        print("=" * 78)

    for r in rows:
        print()
        print(E.metrics_report(y, preds[r["model"]], g, title=r["model"]))


if __name__ == "__main__":
    main()
