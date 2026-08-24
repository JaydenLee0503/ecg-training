"""Feature selection - the bridge to the quantum stage.

A variational quantum classifier encodes one feature per qubit (angle encoding) or
log2(d) qubits (amplitude encoding). Either way the practical budget is single-digit
to low-double-digit features, not the ~250 that `extract_features` produces. This
module reduces the design matrix to a qubit-sized subset *honestly*: the ranking is
computed inside the training fold only, so the selection itself cannot leak.

Two selectors:

* `rank_anova`  - univariate F, fast and interpretable.
* `mrmr_select` - maximum relevance, minimum redundancy: greedily add the feature with
                  the best relevance-minus-mean-correlation score. Matters here because
                  the 28 descriptors of one mode are heavily correlated with each other,
                  and a pure top-F list will happily pick eight views of the same thing.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

from .config import EPS

__all__ = ["rank_anova", "mrmr_select", "MRMRSelector", "quantum_ready"]


def rank_anova(X, y):
    """Feature indices ordered by descending univariate ANOVA F. Non-finite F -> last."""
    X = np.asarray(X, dtype=float)
    var_ok = X.std(0) > 1e-12
    F = np.full(X.shape[1], -np.inf)
    if var_ok.any():
        f, _ = f_classif(X[:, var_ok], y)
        F[var_ok] = np.nan_to_num(f, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return np.argsort(-F), F


def mrmr_select(X, y, k: int = 8):
    """Greedy mRMR. Returns the chosen column indices, in the order they were chosen.

    Score of a candidate f given the already-chosen set S:
        F(f) / (mean_{s in S} |corr(f, s)| + eps)
    Relevance is the ANOVA F; redundancy is mean absolute Pearson correlation.
    """
    X = np.asarray(X, dtype=float)
    order, F = rank_anova(X, y)
    k = min(k, X.shape[1])

    Xs = (X - X.mean(0)) / (X.std(0) + EPS)
    chosen = [int(order[0])]
    pool = [int(i) for i in order[1:] if np.isfinite(F[i])]

    while len(chosen) < k and pool:
        C = np.abs(Xs[:, pool].T @ Xs[:, chosen] / len(Xs))     # (n_pool, n_chosen)
        red = C.mean(1)
        score = F[pool] / (red + 0.1)                            # 0.1 damps the divide
        chosen.append(pool.pop(int(np.argmax(score))))
    return np.array(chosen, dtype=int)


class MRMRSelector(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapper, so selection happens *inside* the CV fold.

    Putting `mrmr_select` outside cross-validation would rank features using the test
    fold's labels. Wrapped like this, `make_pipeline(MRMRSelector(8), clf)` is honest.
    """

    def __init__(self, k: int = 8, method: str = "mrmr"):
        self.k = k
        self.method = method

    def fit(self, X, y):
        if self.method == "mrmr":
            self.idx_ = mrmr_select(X, y, self.k)
        else:
            self.idx_ = rank_anova(X, y)[0][:self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"f{i}" for i in self.idx_])
        return np.asarray(input_features)[self.idx_]


def quantum_ready(X, idx=None, mode: str = "angle"):
    """Scale a (already-selected) design matrix into the range a circuit expects.

    `angle`     -> [0, pi], the usual RY rotation-angle range.
    `amplitude` -> unit L2 norm per row, for amplitude encoding.

    Returns `(Xq, params)`; feed `params` to the same function at inference time so the
    test set is scaled with the training set's limits rather than its own.
    """
    Z = np.asarray(X, dtype=float)
    if idx is not None:
        Z = Z[:, idx]
    if mode == "angle":
        lo, hi = Z.min(0), Z.max(0)
        Xq = np.pi * (Z - lo) / (hi - lo + EPS)
        return Xq, {"mode": mode, "lo": lo, "hi": hi}
    if mode == "amplitude":
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / (n + EPS), {"mode": mode}
    raise ValueError(f"unknown encoding mode {mode!r}")
