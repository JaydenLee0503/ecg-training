#!/usr/bin/env python
"""Consolidate every saved out-of-fold prediction into one metric table.

The quantum scripts each write their own `*_preds.npz`; this reads all of them and
produces the comparison that actually answers "is the quantum model better?" — the full
panel per model at both levels, plus a paired bootstrap against a chosen baseline.

Two reasons this is a script and not a notebook cell. It reads *predictions*, so it costs
seconds rather than the hours the models cost, and it can be re-run after any new run
lands. And the comparison it makes is paired: every bootstrap resample draws the same
records for every model, so the interval is on the *difference*, which is far tighter
than comparing two independent confidence intervals by eye.

Bootstrapping over records, not windows, is the only defensible choice here — windows
from one recording are near-duplicates, so resampling them would understate the spread
by roughly the same factor that a random train/test split inflates the score.

    python scripts/metrics_table.py
    python scripts/metrics_table.py --baseline "RF, mRMR-12 in-fold" --n-boot 5000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecgvmd as E

#: Array keys inside a prediction file that need a clearer label out of context.
RENAME = {"rf": "RF, mRMR-12 in-fold"}

#: `nested` means a different feature map in each kernel file, so name it by the file.
TAG_LABEL = {
    "kernel_nested_bw": "angle kernel, nested bw",
    "e25_iqp_nested": "IQP kernel, nested bw",
}


def collect(paths):
    """Every (model, predictions) pair across the prediction files, aligned on rows."""
    ref_y = ref_g = None
    models = {}
    for p in sorted(paths):
        f = np.load(p, allow_pickle=False)
        y, g = f["__y"], f["__groups"]
        if ref_y is None:
            ref_y, ref_g = y, g
        elif len(y) != len(ref_y) or not np.array_equal(y, ref_y) \
                or not np.array_equal(g, ref_g):
            print(f"  skipping {p}: different rows than the first file "
                  f"(n={len(y)} vs {len(ref_y)}) — cannot be compared row-wise")
            continue
        tag = os.path.basename(p).replace("_preds.npz", "")
        for k in f.files:
            if k.startswith("__") or k.startswith("transductive"):
                continue
            name = TAG_LABEL.get(tag, tag) if k == "nested" else RENAME.get(k, k)
            models[name] = f[k].astype(str)
    return ref_y, ref_g, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=None,
                    help="prediction .npz files (default: results/*_preds.npz)")
    ap.add_argument("--baseline", default="RF, mRMR-12 in-fold",
                    help="the model every other one is compared against")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/all_metrics")
    args = ap.parse_args()

    paths = args.preds or glob.glob("results/*_preds.npz")
    if not paths:
        sys.exit("no results/*_preds.npz found — run the quantum scripts first")
    y, g, models = collect(paths)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {len(models)} models over "
          f"{len(y)} windows / {len(np.unique(g))} records\n")

    rows = []
    for name, p in models.items():
        rows.append(E.metrics_row(name, y, p, groups=g))
    df = pd.DataFrame(rows)

    # ---- paired bootstrap over records ------------------------------------------
    base = args.baseline if args.baseline in models else None
    if base is None:
        print(f"baseline {args.baseline!r} not among the saved models; "
              f"skipping the paired comparison\n")
    else:
        recs = np.unique(g)
        rows_of = {r: np.flatnonzero(g == r) for r in recs}
        rng = np.random.default_rng(args.seed)
        picks = [np.concatenate([rows_of[r] for r in
                                 rng.choice(recs, len(recs), replace=True)])
                 for _ in range(args.n_boot)]

        def mf1(p, idx):
            return E.full_metrics(y[idx], p[idx])["macro_f1"]

        bl = models[base]
        stats = {}
        for name, p in models.items():
            if name == base:
                stats[name] = (0.0, 0.0, 0.0, 0.5)
                continue
            d = np.array([mf1(p, i) - mf1(bl, i) for i in picks])
            lo, hi = np.percentile(d, [2.5, 97.5])
            stats[name] = (d.mean(), lo, hi, (d > 0).mean())
        df["delta_macro_f1_vs_baseline"] = [stats[n][0] for n in df.model]
        df["delta_ci_lo"] = [stats[n][1] for n in df.model]
        df["delta_ci_hi"] = [stats[n][2] for n in df.model]
        df["p_beats_baseline"] = [stats[n][3] for n in df.model]
        df["baseline"] = base
        df["n_boot"] = args.n_boot

    df["recorded"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("macro_f1", ascending=False)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"{args.out}.csv", index=False)

    # ---- a markdown twin, so the numbers can go straight into the docs -----------
    panels = {n: E.full_metrics(y, p, g) for n, p in models.items()}

    def table(level, title):
        """`level` is "" for segment or "record" for the majority-vote panel."""
        cols = ["accuracy", "balanced_accuracy", "macro_f1", "macro_sensitivity",
                "macro_specificity"]
        out = [f"### {title}", "",
               "| model | acc | bal-acc | macro-F1 | macro-sens | macro-spec | "
               + " | ".join(f"{c} sens / spec" for c in E.CLASS_ORDER) + " |",
               "|---|" + "---:|" * (5 + len(E.CLASS_ORDER))]
        for name in df.model:
            m = panels[name][level] if level else panels[name]
            pc = m["per_class"]
            out.append("| " + name + " | "
                       + " | ".join(f"{m[c]:.4f}" for c in cols) + " | "
                       + " | ".join(f"{pc[c]['sensitivity']:.3f} / "
                                    f"{pc[c]['specificity']:.3f}"
                                    for c in E.CLASS_ORDER) + " |")
        return "\n".join(out)

    md = [f"# Metric panel — every model, every metric",
          "",
          f"Generated by `scripts/metrics_table.py` on "
          f"{datetime.now():%Y-%m-%d %H:%M}, from the out-of-fold predictions in "
          f"`results/*_preds.npz`. {len(y)} windows, {len(np.unique(g))} records, "
          f"record-wise 5-fold, everything fitted in-fold.",
          "",
          table("", "Segment level"),
          "",
          table("record", "Record level (majority vote)"),
          ""]
    if base is not None:
        md += [f"### Paired bootstrap vs `{base}` ({args.n_boot} resamples of the "
               f"{len(np.unique(g))} records)", "",
               "| model | Δ macro-F1 | 95% CI | P(beats baseline) |",
               "|---|---:|---|---:|"]
        for _, r in df.iterrows():
            if r.model == base:
                continue
            md.append(f"| {r.model} | {r.delta_macro_f1_vs_baseline:+.4f} | "
                      f"[{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}] | "
                      f"{r.p_beats_baseline:.3f} |")
        md.append("")
    md += ["### Confusion matrices (rows = true, order "
           + ", ".join(E.CLASS_ORDER) + ")", ""]
    for name, p in models.items():
        md += [f"**{name}** — segment, then record:", "", "```",
               E.metrics_report(y, p, g, title=name), "```", ""]
    with open(f"{args.out}.md", "w") as fh:
        fh.write("\n".join(md))

    print(df[["model", "accuracy", "macro_f1", "macro_sensitivity",
              "macro_specificity", "record_accuracy"]].to_string(index=False))
    print(f"\nwrote {args.out}.csv  — one flat row per model, every metric")
    print(f"wrote {args.out}.md   — the same as markdown tables, ready for the docs")


if __name__ == "__main__":
    main()
