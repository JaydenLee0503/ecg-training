#!/usr/bin/env python
"""Step 2 of the merge plan: settle convergence before anything depends on it.

The open question inherited from the source notebooks: `Arrythmia_v2` reported that
alpha=5 classifies better than alpha=2000, but at alpha=5 essentially every segment
exhausts `max_iter=500` without meeting `tol`. So the "optimum" might be an artefact
of a truncated solve rather than a property of the decomposition.

This script re-runs the sweep at max_iter=500 AND max_iter=2000, for both segmentation
modes, and records:

  * the capped fraction and median iteration count (did it actually converge?)
  * the residual energy fraction (how much of the signal is left over?)
  * record-wise cross-validated macro-F1 (does it classify better?)

Results append to results/alpha_sweep.csv as they land, so a long run can be inspected
while it is still going.

    python scripts/alpha_sweep.py                    # full sweep
    python scripts/alpha_sweep.py --fast             # 4 records/class, quick check
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[5, 50, 200, 500, 2000, 8000])
    ap.add_argument("--max-iters", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--seg-modes", nargs="+", default=["fixed", "beat"])
    ap.add_argument("--n-per-record", type=int, default=10)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results/alpha_sweep.csv")
    args = ap.parse_args()

    if args.fast:
        args.alphas, args.max_iters, args.n_per_record = [5, 2000], [500, 2000], 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds = E.load_ecgdata()
    print(ds, flush=True)

    rows = []
    for seg_mode in args.seg_modes:
        base = E.CFG.replace(seg_mode=seg_mode, K=args.K,
                             n_per_record=args.n_per_record)
        W, y, g = E.segment(ds, base)
        print(f"\n=== {seg_mode}: {W.shape[0]} windows, "
              f"{len(np.unique(g))} records ===", flush=True)

        for max_iter in args.max_iters:
            for alpha in args.alphas:
                cfg = base.replace(alpha=alpha, max_iter=max_iter)
                t0 = time.time()
                fb = E.extract_features(W, y, g, cfg, with_control=False, verbose=False)
                X, _ = fb["VMD modes + rhythm"]
                sc = E.evaluate(E.rf(cfg), X, y, g, cfg)

                # residual on a subsample - cheap, and only needs the modes
                sub = W[::max(1, len(W) // 128)]
                res = E.vmd_batch(sub, alpha=alpha, tau=cfg.tau, K=cfg.K, dc=cfg.dc,
                                  tol=cfg.tol, max_iter=max_iter, fs=cfg.fs)

                row = {
                    "seg_mode": seg_mode, "alpha": alpha, "max_iter": max_iter,
                    "K": args.K, "n_windows": len(W),
                    "capped_frac": fb.capped_fraction,
                    "median_iters": float(np.median(fb.iters)),
                    "resid_energy_frac": float(res.residual_energy_fraction(sub).mean()),
                    "resid_norm_ratio": float(res.residual_norm_ratio(sub).mean()),
                    "segment_macroF1": sc["segment macro-F1"],
                    "record_acc": sc["record acc"],
                    "seconds": time.time() - t0,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(f"  alpha={alpha:<7g} max_iter={max_iter:<5d} "
                      f"capped={100 * row['capped_frac']:5.1f}%  "
                      f"med_it={row['median_iters']:6.0f}  "
                      f"resid_E={row['resid_energy_frac']:.4f}  "
                      f"F1={row['segment_macroF1']:.4f}  "
                      f"rec={row['record_acc']:.4f}  "
                      f"[{row['seconds']:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
