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
           "default_models", "compare_blocks"]


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
             use_groups: bool = True, return_pred: bool = False):
    """Cross-validated segment metrics plus the record-level majority vote.

    Set `use_groups=False` only to demonstrate leakage; the numbers it produces are
    not valid estimates of anything.
    """
    cfg = cfg or CFG
    cv = cv if cv is not None else (group_cv(cfg) if use_groups else naive_cv(cfg))
    pred = cross_val_predict(model, X, y, groups=groups if use_groups else None,
                             cv=cv, n_jobs=1)
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


def report(y, pred, groups, title: str = ""):
    """Printed classification report at both segment and record level."""
    ry, rp = record_vote(pred, groups, y)
    out = [f"=== {title} ===" if title else "",
           "segment level:", classification_report(y, pred, digits=3),
           "record level (majority vote):", classification_report(ry, rp, digits=3),
           "record confusion matrix (rows=true, order " + ",".join(CLASS_ORDER) + "):",
           str(confusion_matrix(ry, rp, labels=CLASS_ORDER))]
    return "\n".join(x for x in out if x)
