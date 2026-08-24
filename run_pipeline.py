#!/usr/bin/env python
"""End-to-end VMD -> IMF features, from the command line.

The notebooks are for looking at things; this is for producing the artefact. It does
the expensive part once and writes a `.npz` that the notebooks (and, later, the
quantum model) load in a second.

Examples
--------
    # quick check that everything is wired up (~1 min)
    python run_pipeline.py --smoke

    # the real thing: every window of every record, K=8 modes
    python run_pipeline.py --out features/fixed_K8_a2000.npz

    # beat-aligned windows, 25 per record, and keep the raw IMF tensor for a DL model
    python run_pipeline.py --seg-mode beat --n-per-record 25 --keep-imfs \
        --out features/beat_K8_a2000.npz

    # skip the classifier, just extract
    python run_pipeline.py --no-eval
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import ecgvmd as E


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--mat", default=None, help="path to ECGData.mat (auto-detected if omitted)")
    g.add_argument("--seg-mode", choices=["fixed", "beat"], default="fixed")
    g.add_argument("--seg-len", type=int, default=500)
    g.add_argument("--n-per-record", type=int, default=0,
                   help="windows sampled per record; 0 = all of them (fixed mode only)")
    g.add_argument("--seg-stride", type=int, default=1)

    g = p.add_argument_group("VMD")
    g.add_argument("-K", "--modes", type=int, default=8)
    g.add_argument("-a", "--alpha", type=float, default=2000.0)
    g.add_argument("--tau", type=float, default=0.0)
    g.add_argument("--no-dc", action="store_true", help="do not pin mode 0 at 0 Hz")
    g.add_argument("--tol", type=float, default=1e-7)
    g.add_argument("--max-iter", type=int, default=500)
    g.add_argument("--chunk", type=int, default=64)

    g = p.add_argument_group("output")
    g.add_argument("-o", "--out", default=None, help="output .npz (default: features/<signature>.npz)")
    g.add_argument("--keep-imfs", action="store_true",
                   help="also store the raw (B, K, N) mode tensor - large, but it is what a DL model eats")
    g.add_argument("--no-control", action="store_true", help="skip the no-VMD control block")
    g.add_argument("--no-eval", action="store_true", help="extract only, do not cross-validate")
    g.add_argument("--smoke", action="store_true", help="3 windows/record - wiring check only")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    cfg = E.Config(
        seg_mode=args.seg_mode, seg_len=args.seg_len,
        n_per_record=3 if args.smoke else args.n_per_record,
        seg_stride=args.seg_stride,
        K=args.modes, alpha=args.alpha, tau=args.tau, dc=not args.no_dc,
        tol=args.tol, max_iter=args.max_iter, chunk=args.chunk,
    )
    verbose = not args.quiet

    t0 = time.time()
    ds = E.load_ecgdata(args.mat)
    print(ds)
    print(cfg.summary())

    W, y, g = E.segment(ds, cfg)
    print(f"\nsegmented: {W.shape[0]} windows x {W.shape[1]} samples from "
          f"{len(np.unique(g))} records")
    print("  class balance:", dict(zip(*np.unique(y, return_counts=True))))

    fb = E.extract_features(W, y, g, cfg, with_control=not args.no_control,
                            keep_imfs=args.keep_imfs, verbose=verbose)
    print("\n" + fb.describe())

    out = args.out or os.path.join("features", cfg.signature() + ".npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fb.save(out)
    print(f"\nwrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    if not args.no_eval:
        import pandas as pd
        print("\ncross-validating (record-wise splits)...")
        rows = E.compare_blocks(fb, cfg)
        df = pd.DataFrame(rows).set_index("feature set")
        print(df.round(4).to_string())
        df.to_csv(out.replace(".npz", "_scores.csv"))

    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
