#!/usr/bin/env python
"""Is the VQC's deficit the circuit, or the way it is read out?

`VQCClassifier` measures `<Z_i>` on each of 12 qubits and hands the 12 numbers to a
linear head. But the state has 2**12 = 4096 amplitudes, and single-qubit marginals are
exactly the observables that **cannot see inter-qubit correlation** — entanglement lives
in `<Z_i Z_j>` and above. So the ansatz spends 24 CNOTs building correlations and then
reads out through a channel blind to them.

This script tests that. It trains the VQC once, then reads the *same trained circuit*
three ways — marginals only, correlations only, both — and asks what a strong classical
model can extract from each. If the correlations carry signal the marginals do not, the
bottleneck is the readout, not the circuit, and the stage-4 number is attributable to a
design choice rather than to anything about quantum models.

One fold, one seed: the comparisons here are within-fold only and are NOT comparable to
the five-fold numbers in E20. The fold used runs high (RF on raw features scores 0.7358
here against 0.7130 across five folds).

    python scripts/readout_probe.py          # ~10 min, most of it training the VQC
"""
from __future__ import annotations

import glob
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecgvmd as E
from ecgvmd.quantum import TanhAngleScaler, VQCClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_QUBITS = 12


def main():
    import pennylane as qml

    c = [p for p in glob.glob("features/*.npz") if "quantum" not in p]
    if not c:
        sys.exit("no feature .npz found — run run_pipeline.py first")
    f = np.load(max(c, key=os.path.getmtime), allow_pickle=True)
    X, y, g = f["X::VMD modes + rhythm"], f["__y"], f["__groups"]

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    tr, te = next(cv.split(X, y, g))
    sel = E.MRMRSelector(k=N_QUBITS).fit(X[tr], y[tr])
    sca = TanhAngleScaler().fit(sel.transform(X[tr]))
    Xtr = sca.transform(sel.transform(X[tr]))
    Xte = sca.transform(sel.transform(X[te]))
    print(f"fold: train {len(Xtr)} / test {len(Xte)}\n", flush=True)

    print("training the VQC (40 epochs) to get its learned weights...", flush=True)
    t = time.time()
    m = VQCClassifier(n_layers=2, epochs=40, batch_size=32, lr=0.05, seed=0).fit(Xtr, y[tr])
    print(f"  done in {time.time()-t:.0f}s   VQC itself: "
          f"{f1_score(y[te], m.predict(Xte), average='macro'):.4f}\n", flush=True)

    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    pairs = list(itertools.combinations(range(N_QUBITS), 2))

    @qml.qnode(dev)
    def obs(x, w):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
        return ([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
                + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs])

    def feats(A):
        out = np.array([np.asarray(obs(x, m.w_)) for x in A])
        return out[:, :N_QUBITS], out[:, N_QUBITS:]

    Ztr, ZZtr = feats(Xtr)
    Zte, ZZte = feats(Xte)
    print(f"readout shapes: <Z> {Ztr.shape}   <ZZ> {ZZtr.shape}\n")

    def score(name, A, B, est):
        est.fit(A, y[tr])
        print(f"  {name:<52s} {f1_score(y[te], est.predict(B), average='macro'):.4f}")

    rf = lambda: RandomForestClassifier(400, random_state=0, n_jobs=-1)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    print("what a strong classical model can extract from each representation:")
    score("RF on the raw 12 angle-scaled features", Xtr, Xte, rf())
    score("RF on the circuit's 12 <Z> outputs", Ztr, Zte, rf())
    score("RF on the 66 <ZiZj> correlations", ZZtr, ZZte, rf())
    score("RF on <Z> + <ZiZj>  (78 obs)",
          np.hstack([Ztr, ZZtr]), np.hstack([Zte, ZZte]), rf())
    print("\nlinear models, to isolate the head:")
    score("LogReg on raw 12", Xtr, Xte, lr())
    score("LogReg on <Z>  (= the VQC's own architecture)", Ztr, Zte, lr())
    score("LogReg on <Z> + <ZiZj>",
          np.hstack([Ztr, ZZtr]), np.hstack([Zte, ZZte]), lr())


if __name__ == "__main__":
    main()
