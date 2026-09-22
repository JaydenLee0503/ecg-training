#!/usr/bin/env python
"""Is scattering's deficit the TRANSFORM, or the statistics taken from it?

§3a of `architects/wst_vs_vmd_vqc.md` measured scattering losing to VMD by 0.055-0.065
macro-F1 and concluded it is the weaker front end. That comparison is not clean, and this
script is the control it was missing.

**The two blocks differ in two ways at once, not one.** VMD contributes 8 adaptive modes
AND 28 descriptors per mode - entropies, TKEO, waveform length, Hjorth parameters. The
scattering block contributes 126 paths and *no* descriptors: a scattering coefficient
`|x * psi1| * phi` is already a summary, a time-averaged modulus, i.e. an energy. So the
comparison confounds the decomposition with the feature extraction.

That this matters is not a guess. mRMR selecting 12 from the VMD block picks complexity
and entropy descriptors 26 times in 60 and `rel_energy` twice - it almost never wants the
one family scattering can express.

**The control.** The order-1 envelopes `|x * psi1|` exist inside the cascade at full time
resolution; `scatter_batch` then subsamples them and convolves with `phi`, which is what
destroys the within-band structure entropy and TKEO measure. Recompute them before that
averaging and feed them to `mode_features` - the *same* 28 descriptors VMD gets - and the
decomposition is the only thing left varying.

Scored across 5 CV seeds because the single-split margin is inside the project's noise
floor; the paired difference across seeds is the honest yardstick.

    OMP_NUM_THREADS=4 python scripts/wst_descriptor_probe.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecgvmd as E
from ecgvmd.features import mode_features
from ecgvmd import order1_envelopes

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--seeds", type=int, default=5, help="CV seeds, for the paired spread")
    ap.add_argument("--out", default="results/wst_desc_seeds.csv")
    args = ap.parse_args()

    path = args.features or max(
        [x for x in glob.glob("features/*.npz") if "quantum" not in x],
        key=os.path.getmtime)
    f = np.load(path, allow_pickle=True)
    cfg = E.Config(**json.loads(str(f["__config"])))
    y, g = f["__y"], f["__groups"]

    ds = E.load_ecgdata(None)
    W, ys, gs = E.segment(ds, cfg)
    assert np.array_equal(np.asarray(ys).astype(str), np.asarray(y).astype(str)), \
        "rows are not aligned with the .npz - the comparison would be meaningless"
    X = np.asarray(W, dtype=np.float64)

    env, xi_hz = order1_envelopes(X, fs=cfg.fs)
    print(f"order-1 envelopes {env.shape}, {xi_hz.max():.1f}-{xi_hz.min():.2f} Hz")
    desc, names = mode_features(env, np.broadcast_to(xi_hz, (len(X), len(xi_hz))),
                                cfg.fs, x=X, prefix="w")
    dead = desc.std(0) < 1e-12
    print(f"WST envelope descriptors: {desc.shape}, {dead.sum()} zero-variance "
          f"(peak_hz/zcr are constants for a FIXED filter bank - harmless)")

    res = E.scatter_batch(W, J=6, Q=(8, 1), T=64, max_order=2, fs=cfg.fs)
    blocks = {"VMD modes": f["X::VMD modes"],
              "WST raw paths": E.scatter_features(res, log=True)[0],
              "WST env + descriptors": desc}

    rows = []
    for seed in range(args.seeds):
        sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                  .split(blocks["VMD modes"], y, g))
        for name, Xb in blocks.items():
            for k in (None, 12):
                est = (RandomForestClassifier(400, random_state=0, n_jobs=-1) if k is None
                       else make_pipeline(E.MRMRSelector(k=12),
                                          RandomForestClassifier(400, random_state=0,
                                                                 n_jobs=-1)))
                pr = cross_val_predict(est, Xb, y, cv=sp, groups=g, n_jobs=1)
                r = E.metrics_row(name, y, pr, groups=g)
                rows.append(dict(seed=seed, block=name, dim=Xb.shape[1],
                                 k=("all" if k is None else "12"),
                                 macro_f1=r["macro_f1"], sens_CHF=r["sens_CHF"]))
        print(f"  cv seed {seed} done", flush=True)

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    np.savez_compressed("results/wst_env_descriptors.npz", X=desc,
                        names=np.asarray(names), y=np.asarray(y).astype(str),
                        groups=np.asarray(g))

    from scipy import stats
    for kk in ("all", "12"):
        piv = df[df.k == kk].pivot(index="seed", columns="block", values="macro_f1")
        print(f"\n--- k={kk}, paired across {args.seeds} CV seeds ---")
        for b in piv.columns:
            print(f"  {b:24s} mean {piv[b].mean():.4f}  sd {piv[b].std():.4f}")
        for a, b in [("WST env + descriptors", "VMD modes"),
                     ("WST raw paths", "VMD modes"),
                     ("WST env + descriptors", "WST raw paths")]:
            d, t = piv[a] - piv[b], stats.ttest_rel(piv[a], piv[b])
            print(f"  {a:24s} - {b:22s} {d.mean():+.4f} +/- {d.std():.4f}  "
                  f"t={t.statistic:+.2f} p={t.pvalue:.4f}  {int((d > 0).sum())}/{len(d)}")
    print(f"\nwrote {args.out} and results/wst_env_descriptors.npz")


if __name__ == "__main__":
    main()
