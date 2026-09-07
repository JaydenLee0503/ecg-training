"""Evaluation protocol.

One rule dominates everything else here: **split by record, never by segment.**
Segments from the same recording are near-duplicates. A random split puts some of a
patient's segments in train and the rest in test, and the model recognises the patient
rather than the pathology. Measured on this dataset that is worth about +0.12 macro-F1
of pure illusion.

`StratifiedGroupKFold` on the record id is therefore the default and `evaluate` refuses
to run without groups unless you explicitly ask for the leaky variant.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CFG, CLASS_ORDER, Config

__all__ = ["group_cv", "naive_cv", "record_vote", "evaluate", "leakage_gap", "rf",
           "default_models", "compare_blocks", "report", "per_class_metrics",
           "full_metrics", "metrics_row", "metrics_report", "cm_string"]


def group_cv(cfg: Config | None = None):
    cfg = cfg or CFG
    return StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)


def naive_cv(cfg: Config | None = None):
    """The WRONG splitter. Provided only so the leakage gap can be measured."""
    cfg = cfg or CFG
    return StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)


def record_vote(pred, groups, truth):
    """Majority vote of segment predictions within each record.

    A clinician diagnoses a recording, not a 3.9-second window, so this is the metric
    that corresponds to the actual task.
    """
    pred, groups, truth = np.asarray(pred), np.asarray(groups), np.asarray(truth)
    recs = np.unique(groups)
    rp, ry = [], []
    for r in recs:
        m = groups == r
        v, c = np.unique(pred[m], return_counts=True)
        rp.append(v[c.argmax()])
        ry.append(truth[m][0])
    return np.array(ry), np.array(rp)


def rf(cfg: Config | None = None, n: int = 400):
    cfg = cfg or CFG
    return RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=cfg.seed,
                                  class_weight="balanced_subsample", min_samples_leaf=2)


def default_models(cfg: Config | None = None) -> dict:
    cfg = cfg or CFG
    return {
        "RandomForest": rf(cfg),
        "ExtraTrees": ExtraTreesClassifier(400, n_jobs=-1, random_state=cfg.seed,
                                           class_weight="balanced_subsample",
                                           min_samples_leaf=2),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=cfg.seed),
        "LogReg (scaled)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=cfg.seed)),
    }


def evaluate(model, X, y, groups, cfg: Config | None = None, cv=None,
             use_groups: bool = True, return_pred: bool = False,
             full: bool = False):
    """Cross-validated segment metrics plus the record-level majority vote.

    Set `use_groups=False` only to demonstrate leakage; the numbers it produces are
    not valid estimates of anything.

    `full=True` returns `full_metrics` instead of the four-number summary — accuracy,
    sensitivity, specificity, F1 and confusion matrices at both levels. The compact form
    stays the default because `compare_blocks` tabulates it one row per feature block.
    """
    cfg = cfg or CFG
    cv = cv if cv is not None else (group_cv(cfg) if use_groups else naive_cv(cfg))
    pred = cross_val_predict(model, X, y, groups=groups if use_groups else None,
                             cv=cv, n_jobs=1)
    if full:
        m = full_metrics(y, pred, groups)
        return (m, pred) if return_pred else m
    ry, rp = record_vote(pred, groups, y)
    scores = {
        "segment acc": accuracy_score(y, pred),
        "segment bal-acc": balanced_accuracy_score(y, pred),
        "segment macro-F1": f1_score(y, pred, average="macro"),
        "record acc": accuracy_score(ry, rp),
        "n": len(y),
    }
    return (scores, pred) if return_pred else scores


def leakage_gap(X, y, groups, cfg: Config | None = None, model=None):
    """Same features, same model, two splitters. The difference is the leakage."""
    cfg = cfg or CFG
    model = model or rf(cfg)
    leaky = evaluate(model, X, y, groups, cfg, use_groups=False)
    honest = evaluate(model, X, y, groups, cfg, use_groups=True)
    return {
        "random split (LEAKY)": leaky["segment macro-F1"],
        "record-wise split (honest)": honest["segment macro-F1"],
        "inflation": leaky["segment macro-F1"] - honest["segment macro-F1"],
    }


def compare_blocks(bundle, cfg: Config | None = None, model=None, blocks=None):
    """Run `evaluate` over every feature block in a FeatureBundle.

    Returns a list of dicts; wrap in `pd.DataFrame(...).set_index('feature set')`.
    """
    cfg = cfg or bundle.cfg
    rows = []
    for name in (blocks or bundle.keys()):
        X, _ = bundle[name]
        s = evaluate(model or rf(cfg), X, bundle.y, bundle.groups, cfg)
        rows.append({"feature set": name, "dim": X.shape[1], **s})
    return rows


# --------------------------------------------------------------------------------
# The full metric panel: accuracy, sensitivity, specificity, F1, confusion matrix
# --------------------------------------------------------------------------------
#
# Until now every quantum run in this project reported macro-F1 and nothing else, which
# is enough to rank models and not enough to say what a model does. A 59% ARR / 19% CHF
# / 22% NSR prior makes the omission expensive: the failure mode here is a model that
# quietly abandons CHF, and macro-F1 shows that only as a number that is lower than you
# hoped. Sensitivity and specificity per class name the class it abandoned.
#
# Both are one-vs-rest. For CHF, "negative" means ARR *or* NSR:
#
#     sensitivity  TP / (TP + FN)   how much of this class the model finds
#     specificity  TN / (TN + FP)   how much of everything else it keeps out
#
# The pair has to be read together. "Always predict ARR" scores specificity 1.000 on CHF
# and NSR while finding none of either, so a high specificity alone is not evidence of
# anything on a skewed prior.


def _ovr_counts(y, pred, labels):
    """One-vs-rest TP/FP/FN/TN per class, from the multiclass confusion matrix."""
    C = confusion_matrix(y, pred, labels=labels)
    tp = np.diag(C).astype(float)
    fn = C.sum(1) - tp
    fp = C.sum(0) - tp
    tn = C.sum() - tp - fn - fp
    return C, tp, fp, fn, tn


def per_class_metrics(y, pred, labels=None):
    """Per-class sensitivity, specificity, precision, F1 and support.

    Returns `{class: {metric: value}}` in `CLASS_ORDER`, which both prints readably and
    flattens into a CSV row. A class with no support gets `nan` sensitivity rather than
    a silent zero — absent is not the same as missed.
    """
    labels = list(labels if labels is not None else CLASS_ORDER)
    C, tp, fp, fn, tn = _ovr_counts(y, pred, labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        f1 = np.where(np.nan_to_num(prec) + np.nan_to_num(sens) > 0,
                      2 * prec * sens / (prec + sens), 0.0)
    return {c: {"sensitivity": float(sens[i]), "specificity": float(spec[i]),
                "precision": float(prec[i]), "f1": float(f1[i]),
                "support": int(tp[i] + fn[i]),
                "tp": int(tp[i]), "fp": int(fp[i]),
                "fn": int(fn[i]), "tn": int(tn[i])}
            for i, c in enumerate(labels)}


def full_metrics(y, pred, groups=None, labels=None):
    """Every metric the project reports, at segment and (with `groups`) record level.

    `balanced_accuracy` and `macro_sensitivity` are the same quantity — macro-averaged
    recall — computed two ways and kept both because the two names appear in different
    halves of the literature. If they ever disagree, something is wrong upstream.

    Pass the out-of-fold predictions from `cross_val_predict`, not in-sample ones.
    """
    labels = list(labels if labels is not None else CLASS_ORDER)
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    pc = per_class_metrics(y, pred, labels)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=labels),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=labels),
        "macro_sensitivity": float(np.nanmean([v["sensitivity"] for v in pc.values()])),
        "macro_specificity": float(np.nanmean([v["specificity"] for v in pc.values()])),
        "n": int(len(y)),
        "labels": labels,
        "per_class": pc,
        "confusion": confusion_matrix(y, pred, labels=labels),
    }
    if groups is not None:
        ry, rp = record_vote(pred, groups, y)
        rpc = per_class_metrics(ry, rp, labels)
        out["record"] = {
            "accuracy": accuracy_score(ry, rp),
            "balanced_accuracy": balanced_accuracy_score(ry, rp),
            "macro_f1": f1_score(ry, rp, average="macro", labels=labels),
            "weighted_f1": f1_score(ry, rp, average="weighted", labels=labels),
            "macro_sensitivity": float(np.nanmean([v["sensitivity"] for v in rpc.values()])),
            "macro_specificity": float(np.nanmean([v["specificity"] for v in rpc.values()])),
            "n": int(len(ry)),
            "per_class": rpc,
            "confusion": confusion_matrix(ry, rp, labels=labels),
        }
    return out


def cm_string(C) -> str:
    """Confusion matrix as one CSV-safe field: rows `;`-separated, cells `,`-separated."""
    return ";".join(",".join(str(int(v)) for v in row) for row in np.asarray(C))


def metrics_row(name, y, pred, groups=None, labels=None, **extra):
    """`full_metrics` flattened into one row, for appending to a results CSV.

    Keeps the confusion matrices as `cm` / `record_cm` strings (see `cm_string`) so a
    saved row reconstructs every count without a second artefact. Scripts that used to
    write `{"model": name, "macro_f1": f1}` can write this instead; `macro_f1` keeps its
    name and meaning, so old columns still line up.
    """
    labels = list(labels if labels is not None else CLASS_ORDER)
    m = full_metrics(y, pred, groups, labels)
    row = {"model": name}
    row.update({k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                  "weighted_f1", "macro_sensitivity",
                                  "macro_specificity")})
    for c in labels:
        for k in ("sensitivity", "specificity", "precision", "f1"):
            row[f"{k[:4]}_{c}"] = m["per_class"][c][k]
        row[f"support_{c}"] = m["per_class"][c]["support"]
    row["cm"] = cm_string(m["confusion"])
    if "record" in m:
        r = m["record"]
        row.update({f"record_{k}": r[k] for k in ("accuracy", "balanced_accuracy",
                                                  "macro_f1", "macro_sensitivity",
                                                  "macro_specificity")})
        row["record_cm"] = cm_string(r["confusion"])
    row["n_windows"] = m["n"]
    row.update(extra)
    return row


def _panel(m, labels) -> list[str]:
    """The per-class table plus the confusion matrix, for one level of aggregation."""
    w = max(len(c) for c in labels) + 1
    out = [f"  {'':<{w}}  {'sens':>6} {'spec':>6} {'prec':>6} {'F1':>6} {'n':>6}"]
    for c in labels:
        v = m["per_class"][c]
        out.append(f"  {c:<{w}}  {v['sensitivity']:6.3f} {v['specificity']:6.3f} "
                   f"{v['precision']:6.3f} {v['f1']:6.3f} {v['support']:6d}")
    out.append(f"  {'macro':<{w}}  {m['macro_sensitivity']:6.3f} "
               f"{m['macro_specificity']:6.3f} {'':>6} {m['macro_f1']:6.3f} "
               f"{m['n']:6d}")
    out.append(f"  accuracy {m['accuracy']:.4f}   balanced accuracy "
               f"{m['balanced_accuracy']:.4f}   macro-F1 {m['macro_f1']:.4f}")
    out.append("  confusion (rows = true, cols = predicted, order "
               + ", ".join(labels) + "):")
    C = np.asarray(m["confusion"])
    for c, row in zip(labels, C):
        out.append(f"    {c:<4} " + " ".join(f"{int(v):6d}" for v in row))
    return out


def metrics_report(y, pred, groups=None, title: str = "", labels=None) -> str:
    """The printable panel: accuracy, sensitivity, specificity, F1, confusion matrix.

    Segment level always; record level too when `groups` is given, which is the metric
    that corresponds to the clinical task — a recording is diagnosed, not a 3.9 s window.
    """
    labels = list(labels if labels is not None else CLASS_ORDER)
    m = full_metrics(y, pred, groups, labels)
    out = [f"=== {title} ===" if title else "", "segment level:"]
    out += _panel(m, labels)
    if "record" in m:
        out += ["", f"record level (majority vote, {m['record']['n']} records):"]
        out += _panel(m["record"], labels)
    return "\n".join(x for x in out if x != "")


def report(y, pred, groups, title: str = ""):
    """sklearn's classification report at both levels, then the full metric panel.

    Kept as the notebooks call it; `metrics_report` is the same panel without the
    sklearn text, and `full_metrics` is the same numbers as a dict.
    """
    ry, rp = record_vote(pred, groups, y)
    out = [f"=== {title} ===" if title else "",
           "segment level:", classification_report(y, pred, digits=3),
           "record level (majority vote):", classification_report(ry, rp, digits=3),
           metrics_report(y, pred, groups,
                          title="sensitivity / specificity / confusion")]
    return "\n".join(x for x in out if x)
