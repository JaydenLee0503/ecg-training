"""The quantum stage - feature maps, kernels, and the estimators that wrap them.

Everything here is an sklearn estimator. That is deliberate and it is the whole point
of the module: the honesty machinery this project depends on - `StratifiedGroupKFold`
on the record id, `MRMRSelector` ranking features inside the training fold only - is
sklearn's pipeline protocol. Write the quantum model as an estimator and all of it
applies unchanged:

    make_pipeline(MRMRSelector(k=12), TanhAngleScaler(), QuantumKernelSVC())

Write it as a loop in a notebook that reads `features/quantum_*.npz` directly and you
inherit that file's transductive leak (its mRMR ranking and its min/max limits were
fitted on all 1620 windows), and every number you produce is quietly optimistic.

Two measured facts shaped the design.

**Angle scaling must be bounded, not min/max.** With `MinMaxScaler` fitted in-fold,
0.31% of test-fold values land outside [0, pi], the worst at 1.40*pi. Since <Z> = cos(t),
a feature 40% past the training maximum rotates *past* the south pole and reads back as
mid-range (cos(1.4pi) = -0.31) rather than extreme (cos(pi) = -1). Monotonicity breaks
exactly on the outliers. `TanhAngleScaler` borrows the bounding trick from the dressed
circuit of Mari et al. (2020) - standardise, tanh, scale by pi/2 - which saturates
instead of wrapping.

**Bandwidth is not optional, and it interacts with the scaler.** The known failure mode
of quantum kernels is that off-diagonal Gram entries collapse toward zero as dimension
grows, the matrix approaches the identity, and the SVM memorises. There is a second,
less-discussed failure at the other end: entries collapse toward 1.0 and the kernel
stops discriminating at all. Both are live here. Measured on 324 windows of 12 in-fold
selected features (mean / median off-diagonal, fraction above 0.01):

    MinMax [0, pi]        0.185  0.108   0.842
    tanh, scale=1.00      0.042  0.001   0.319    <- identity regime, unusable
    tanh, scale=0.75      0.107  0.034   0.681
    tanh, scale=0.50      0.295  0.244   0.998    <- usable
    tanh, scale=0.35      0.518  0.509   1.000    <- usable
    tanh, scale=0.25      0.703  0.711   1.000
    tanh, scale=0.15      0.878  0.885   1.000    <- all-ones regime, unusable

Note that `TanhAngleScaler(scale=1.0)` - the default, and the natural-looking choice -
sits in the dead zone. Standardising to unit variance spreads pairwise angles much
wider than min/max scaling does on these skewed features, so the scaler that fixes the
wrap-around problem simultaneously pushes the kernel toward the identity. The two
choices are coupled and neither can be made in isolation.

`scripts/quantum_kernel_probe.py` prints these statistics and plots `gram_`. Look at
them before trusting any score: a concentrated kernel yields a bad number for reasons
that have nothing to do with whether the classes are separable.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

__all__ = ["TanhAngleScaler", "angle_kernel_qnode", "iqp_kernel_qnode",
           "product_angle_kernel", "gram_matrix", "QuantumKernelSVC",
           "VQCClassifier"]

#: Pairs pushed through one broadcast call. Measured flat at ~0.47 ms/pair from 64 to
#: 16384 (memory-bound, not call-bound), so this is chosen for footprint, not speed.
CHUNK = 4096


class TanhAngleScaler(BaseEstimator, TransformerMixin):
    """Scale features into bounded rotation angles: standardise -> tanh -> * (pi/2).

    Output is in (-pi/2, pi/2) by construction, for *any* input, including test values
    far outside the range seen during `fit`. That is the property `MinMaxScaler` lacks
    and the reason this exists - see the module docstring for the measured failure.

    `scale` sets the bandwidth of the feature map, and for kernel use the default of
    1.0 is *wrong* - it puts the Gram matrix in the identity regime (see the module
    docstring for the measured sweep). The usable band is roughly 0.35-0.50. The
    default stays 1.0 because it is the natural range for a variational circuit, where
    the concern is expressivity rather than pairwise fidelity; kernels should pass an
    explicit value and check the Gram statistics.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        return np.tanh(Z) * (np.pi / 2) * self.scale

    def get_feature_names_out(self, input_features=None):
        return self.scaler_.get_feature_names_out(input_features)


def angle_kernel_qnode(n_qubits: int):
    """A QNode returning |<phi(x1)|phi(x2)>|^2 for RY angle embedding.

    Built as U(x1) followed by U-dagger(x2); the fidelity is then the probability of
    measuring the all-zeros state. Analytic (`shots=None`) on purpose: sampling noise
    at 1024 shots is about +/-0.03 per entry, the same order as the differences this
    project is trying to measure.
    """
    import pennylane as qml

    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))

    return circuit


def iqp_kernel_qnode(n_qubits: int, n_repeats: int = 1):
    """A QNode for the second-order Pauli-Z (ZZ / IQP) feature map of Havlicek et al.

    Unlike `angle_kernel_qnode`, this one is *not* a product state: the entangling
    `ZZ(phi(x_i, x_j))` phases correlate the qubits, the overlap does not factorise,
    and there is no closed form. This is the map worth simulating - see
    `product_angle_kernel` for why the angle map is not.

    ZZ conventionally takes data on a different range than the RY rotations do. Feeding
    it [0, pi] data gives a degenerate second-order term wherever a feature sits near
    pi, so re-scale from raw features rather than reusing angle-encoding limits.
    """
    import pennylane as qml

    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.IQPEmbedding(x1, wires=range(n_qubits), n_repeats=n_repeats)
        qml.adjoint(qml.IQPEmbedding)(x2, wires=range(n_qubits), n_repeats=n_repeats)
        return qml.probs(wires=range(n_qubits))

    return circuit


def product_angle_kernel(A, B):
    """Closed form of the RY angle-embedding fidelity kernel: prod_i cos^2((a_i-b_i)/2).

    `AngleEmbedding` with no entangling gates prepares a product state, so the overlap
    factorises across qubits and the "quantum" kernel collapses to this one-line
    numpy expression. Verified against the statevector simulator to 1.7e-16, and about
    1600x faster.

    The consequence is worth stating plainly: **a quantum kernel built on plain angle
    encoding is classically tractable by construction.** It is a shift-invariant
    product-cosine kernel, a close cousin of the RBF, and no amount of tuning will make
    it exhibit quantum structure. Reporting it as a quantum result would be wrong. It
    remains a perfectly good classical kernel and a fair baseline - it is simply not
    evidence about quantum machine learning.

    Entanglement in the feature map is what breaks the factorisation; see
    `iqp_kernel_qnode`.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    return np.prod(np.cos((A - B) / 2.0) ** 2, axis=1)


def _fidelities(qnode, A, B, chunk: int = CHUNK):
    """Evaluate the kernel on paired rows of A and B, batched through broadcasting."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    out = np.empty(len(A), dtype=float)
    for s in range(0, len(A), chunk):
        e = min(s + chunk, len(A))
        probs = np.asarray(qnode(A[s:e], B[s:e]))
        # A single-row chunk comes back un-batched as (2**n,) rather than (1, 2**n).
        out[s:e] = probs[:, 0] if probs.ndim == 2 else probs[0]
    return out


def gram_matrix(X, Y=None, qnode=None, chunk: int = CHUNK, kernel_fn=None):
    """Kernel matrix between X and Y (or X with itself, exploiting symmetry).

    `kernel_fn` takes paired row blocks and returns their fidelities - pass
    `product_angle_kernel` for the closed form. Otherwise `qnode` is simulated,
    defaulting to the RY angle map.

    The symmetric case evaluates only the upper triangle and fills the diagonal with
    exact 1.0 rather than computing it - `<phi(x)|phi(x)> = 1` identically, and paying
    n circuit evaluations to rediscover that is waste.
    """
    X = np.asarray(X, dtype=float)
    if kernel_fn is None:
        qnode = qnode or angle_kernel_qnode(X.shape[1])
        kernel_fn = lambda a, b: _fidelities(qnode, a, b, chunk)

    if Y is None:
        n = len(X)
        iu = np.triu_indices(n, k=1)
        vals = kernel_fn(X[iu[0]], X[iu[1]])
        G = np.eye(n)
        G[iu] = vals
        G[(iu[1], iu[0])] = vals
        return G

    Y = np.asarray(Y, dtype=float)
    ii, jj = np.meshgrid(np.arange(len(X)), np.arange(len(Y)), indexing="ij")
    vals = kernel_fn(X[ii.ravel()], Y[jj.ravel()])
    return vals.reshape(len(X), len(Y))


def iqp_state_qnode(n_qubits: int, n_repeats: int = 1):
    """The IQP feature map, returning the prepared state rather than a pair overlap.

    `iqp_kernel_qnode` answers "what is <phi(x)|phi(y)>?" one pair at a time, which costs
    n(n-1)/2 circuit evaluations - 1,311,390 of them at n=1620, roughly 1.5 h per fold.
    On a simulator the state itself is available, so preparing each |phi(x)> *once* and
    contracting turns the whole Gram into a single matrix product. See `state_gram`.
    """
    import pennylane as qml

    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(x):
        qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=n_repeats)
        return qml.state()

    return circuit


def state_gram(A, B=None, state_qnode=None, n_repeats: int = 1):
    """Fidelity Gram via explicit statevectors: |<phi(a)|phi(b)>|^2 as one matmul.

    Exact, not an approximation - it is the same quantity `gram_matrix` computes
    pairwise, reassociated. Verified against the pairwise path to 7.1e-15 at n=60, with
    unit diagonal, symmetry and PSD preserved. Measured at n=1620, 12 qubits: 4.67 s for
    the state preparations plus 0.81 s for the product, against ~1.5 h pairwise.

    The catch is memory, and it binds sooner than you would guess: the state matrix is
    n x 2**n_qubits complex128, so 1620 windows costs 1.7 GB at 16 qubits, 27 GB at 20
    and 434 GB at 24. Past ~18 qubits this trick stops being available and the pairwise
    path - slow but O(1) in memory - is the only option.

    Unlike `product_angle_kernel` this is *not* evidence the map is classically
    tractable. It is a simulator implementation detail: on hardware you would still pay
    per-pair overlap estimation, and the IQP map remains entangled either way.
    """
    A = np.asarray(A, dtype=float)
    qn = state_qnode or iqp_state_qnode(A.shape[1], n_repeats)
    PA = np.stack([np.asarray(qn(x)) for x in A])
    PB = PA if B is None else np.stack([np.asarray(qn(x))
                                        for x in np.asarray(B, dtype=float)])
    G = np.abs(PA.conj() @ PB.T) ** 2
    if B is None:
        np.fill_diagonal(G, 1.0)          # exact by construction; kill rounding drift
    return G


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """SVC on a quantum kernel. No trainable quantum parameters at all.

    This is "path A": it answers *does this feature map separate the classes* without
    an optimiser, without parameter-shift gradients, and without any risk of barren
    plateaus. Only if the answer is yes is a variational circuit worth building.

    After `fit`, `gram_` holds the training Gram matrix. Plot it. A matrix that looks
    like the identity means the kernel has concentrated and the SVM is memorising the
    training set; the score will look fine and mean nothing.
    """

    def __init__(self, C: float = 1.0, embedding: str = "angle", n_repeats: int = 1,
                 chunk: int = CHUNK, class_weight=None, exact_angle: bool = True):
        self.C = C
        self.embedding = embedding
        self.n_repeats = n_repeats
        self.chunk = chunk
        self.class_weight = class_weight
        self.exact_angle = exact_angle

    def _setup(self, n_features):
        """Choose between the closed form and a simulated circuit.

        `angle` has an exact classical closed form (`product_angle_kernel`) because its
        feature map prepares a product state, so it is computed directly unless
        `exact_angle=False` forces the simulator for cross-checking. `iqp` entangles
        and has no such shortcut.
        """
        self.gram_fn_ = None
        if self.embedding == "angle":
            self.qnode_ = None if self.exact_angle else angle_kernel_qnode(n_features)
            self.kernel_fn_ = product_angle_kernel if self.exact_angle else None
        elif self.embedding == "iqp":
            self.qnode_ = iqp_kernel_qnode(n_features, self.n_repeats)
            self.kernel_fn_ = None
        elif self.embedding == "iqp-state":
            # same map as "iqp", same numbers, ~1000x faster - see `state_gram`
            self.qnode_ = iqp_state_qnode(n_features, self.n_repeats)
            self.kernel_fn_ = None
            self.gram_fn_ = lambda A, B=None: state_gram(A, B, state_qnode=self.qnode_)
        else:
            raise ValueError(f"unknown embedding {self.embedding!r}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.X_train_ = X
        self.n_qubits_ = X.shape[1]
        self._setup(self.n_qubits_)
        self.gram_ = (self.gram_fn_(X) if self.gram_fn_ is not None else
                      gram_matrix(X, qnode=self.qnode_, chunk=self.chunk,
                                  kernel_fn=self.kernel_fn_))
        self.svc_ = SVC(kernel="precomputed", C=self.C,
                        class_weight=self.class_weight).fit(self.gram_, y)
        self.classes_ = self.svc_.classes_
        return self

    def _test_kernel(self, X):
        check_is_fitted(self, "svc_")
        X = np.asarray(X, dtype=float)
        if self.gram_fn_ is not None:
            return self.gram_fn_(X, self.X_train_)
        return gram_matrix(X, self.X_train_, qnode=self.qnode_, chunk=self.chunk,
                           kernel_fn=self.kernel_fn_)

    def predict(self, X):
        return self.svc_.predict(self._test_kernel(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_kernel(X))


# --------------------------------------------------------------------------------
# Path B: the variational classifier
# --------------------------------------------------------------------------------

def vqc_qnode(n_qubits: int, n_layers: int):
    """Angle embedding, entangling ansatz, one <Z> per qubit.

    Unlike the angle *kernel*, this circuit is genuinely entangled:
    `StronglyEntanglingLayers` interleaves parameterised rotations with a ring of CNOTs,
    so the state does not factorise and `product_angle_kernel`'s shortcut does not apply
    here. The embedding alone was never quantum; the ansatz is.

    `diff_method="adjoint"` computes all gradients in roughly one backward pass instead
    of the parameter-shift rule's two circuit evaluations per parameter per step. On
    hardware you would pay parameter-shift; on a simulator paying it is waste.
    """
    import pennylane as qml

    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(x, w):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Variational quantum classifier with a linear read-out head.

        angle embed -> StronglyEntanglingLayers(depth) -> <Z> per qubit
                    -> Linear(n_qubits -> n_classes) -> softmax

    Measuring `<Z>` on *all* qubits and passing the vector to a small classical head is
    the honest multi-class arrangement. A single `<Z>` is a scalar in [-1, 1]: one
    decision boundary, two classes. This problem has three (ARR / CHF / NSR).

    The head is deliberately a single linear layer with no hidden units. Anything richer
    - and in particular a trainable layer *before* the circuit, as in the dressed circuit
    of Mari et al. (2020) - lets the classical parameters absorb the work and makes the
    quantum layer's contribution unattributable. Here the circuit sees the selected
    features directly, so it is the only thing that can explain a difference.

    Depth 2 on 12 qubits is 72 quantum parameters plus 39 classical: small enough that
    barren plateaus are not yet the problem. Buy depth only against measured validation
    gain.

    **Set `OMP_NUM_THREADS=4`.** The statevector is 2**12 * 16 B = 64 KB and fits in L2,
    so there is not enough work per gate to feed many threads. Leaving the variable unset
    lets OpenMP take all cores and costs 2-4x. Measured on `lightning.qubit`, depth 2,
    s per Adam step (16-core WSL2 box):

        OMP_NUM_THREADS     batch 32    batch 128
        1                   0.444       1.059
        4                   0.369       0.728      <- use this
        16                  1.164       3.396
        unset (= 16 here)   0.750       2.907

    Depth 4 at OMP=4 is 0.681 (batch 32) / 1.992 (batch 128), i.e. 2.7x depth 2 - not the
    15x an earlier note claimed. Prefer small batches when steps are the scarce resource:
    batch 32 costs more per sample but delivers 2.7x more steps per second.

    For the five-fold run invert this - parallelise folds, not gates:
    `cross_val_predict(..., n_jobs=5)` with `OMP_NUM_THREADS=1`. Do not combine n_jobs=5
    with OMP=4; 20 threads on 16 cores puts you back in the thrash regime.

    Two things this class learned the hard way, both about the training budget rather
    than the circuit:

    **Count gradient steps, not epochs.** A 259-window training fold at `batch_size=128`
    is 3 steps per epoch; 30 epochs is 90 Adam steps, which at lr=0.01 moves 111
    parameters almost not at all. The loss descends convincingly and the model still
    predicts one class throughout. Budget by `epochs * ceil(n / batch_size)`.

    **Weight the classes.** The split is 59% ARR / 19% CHF / 22% NSR, so a model with
    informative probabilities can still argmax to ARR on every sample - macro-F1 0.2481,
    which is exactly what "always predict ARR" scores. `class_weight="balanced"` is the
    default here for that reason; unweighted cross-entropy on this prior is a trap.
    """

    def __init__(self, n_layers: int = 2, epochs: int = 60, batch_size: int = 128,
                 lr: float = 0.01, seed: int = 0, verbose: bool = False,
                 class_weight="balanced", eval_set=None, eval_every: int = 0):
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.class_weight = class_weight
        self.eval_set = eval_set
        self.eval_every = eval_every

    def _logits(self, z, W, b):
        return z @ W + b

    def fit(self, X, y):
        import pennylane as qml
        from pennylane import numpy as pnp

        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        n_c = len(self.classes_)
        self.n_qubits_ = X.shape[1]
        yi = np.searchsorted(self.classes_, y)
        Y = np.eye(n_c)[yi]
        if self.class_weight == "balanced":
            cw = len(y) / (n_c * np.bincount(yi, minlength=n_c))
        elif self.class_weight is None:
            cw = np.ones(n_c)
        else:
            cw = np.asarray([self.class_weight[c] for c in self.classes_], dtype=float)
        self.class_weight_ = cw
        sw = cw[yi]

        circuit = vqc_qnode(self.n_qubits_, self.n_layers)
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits_)
        w = pnp.array(rng.normal(0, 0.1, shape), requires_grad=True)
        W = pnp.array(rng.normal(0, 0.1, (self.n_qubits_, n_c)), requires_grad=True)
        b = pnp.array(np.zeros(n_c), requires_grad=True)

        def cost(w, W, b, xb, yb, wb):
            z = pnp.stack(circuit(xb, w)).T
            lg = self._logits(z, W, b)
            m = pnp.max(lg, axis=1, keepdims=True)
            logp = lg - m - pnp.log(pnp.sum(pnp.exp(lg - m), axis=1, keepdims=True))
            return -pnp.sum(wb * pnp.sum(yb * logp, axis=1)) / pnp.sum(wb)

        opt = qml.AdamOptimizer(self.lr)
        n = len(X)
        self.loss_ = []
        self.history_ = []
        self.n_steps_ = self.epochs * int(np.ceil(n / self.batch_size))
        if self.verbose:
            print(f"    {n} samples, {self.n_steps_} Adam steps, "
                  f"{int(np.prod(shape)) + W.size + n_c} parameters", flush=True)
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb = pnp.array(X[idx], requires_grad=False)
                yb = pnp.array(Y[idx], requires_grad=False)
                wb = pnp.array(sw[idx], requires_grad=False)
                (w, W, b, _, _, _), c = opt.step_and_cost(cost, w, W, b, xb, yb, wb)
                ep_loss += float(c) * len(idx)
            self.loss_.append(ep_loss / n)
            self.w_, self.W_, self.b_ = w, W, b
            self._circuit = circuit
            if self.eval_every and (ep + 1) % self.eval_every == 0:
                from sklearn.metrics import f1_score
                row = {"epoch": ep + 1,
                       "steps": (ep + 1) * int(np.ceil(n / self.batch_size)),
                       "loss": self.loss_[-1],
                       "train_f1": f1_score(y, self.predict(X), average="macro")}
                if self.eval_set is not None:
                    Xv, yv = self.eval_set
                    row["val_f1"] = f1_score(yv, self.predict(Xv), average="macro")
                    row["gap"] = row["train_f1"] - row["val_f1"]
                self.history_.append(row)
                if self.verbose:
                    print("    " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                             else f"{k}={v}" for k, v in row.items()),
                          flush=True)
            elif self.verbose and (ep % 10 == 0 or ep == self.epochs - 1):
                print(f"    epoch {ep:3d}  loss {self.loss_[-1]:.4f}", flush=True)

        self.w_, self.W_, self.b_ = w, W, b
        self._circuit = circuit
        return self

    def _qnode(self):
        """The circuit, rebuilt on demand.

        A QNode holds a device handle and does not pickle, so `__getstate__` drops it and
        this puts it back. Rebuilding is free — `vqc_qnode` only wires up a device — and
        it is what lets a fitted model survive `joblib.dump` or `save`/`load`.
        """
        c = getattr(self, "_circuit", None)
        if c is None:
            c = self._circuit = vqc_qnode(self.n_qubits_, self.n_layers)
        return c

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_circuit"}

    def decision_function(self, X):
        from pennylane import numpy as pnp
        check_is_fitted(self, "w_")
        X = np.asarray(X, dtype=float)
        circuit = self._qnode()
        out = []
        for s in range(0, len(X), 512):
            xb = pnp.array(X[s:s + 512], requires_grad=False)
            z = np.asarray(pnp.stack(circuit(xb, self.w_)).T, dtype=float)
            out.append(self._logits(z, np.asarray(self.W_), np.asarray(self.b_)))
        return np.vstack(out)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    # -- persistence -------------------------------------------------------------
    #
    # 111 trained numbers — 72 circuit parameters, 36 head weights, 3 biases — are the
    # entire product of a 12-minute fit, and until now they were discarded the moment
    # `cross_val_predict` finished. `save` writes them as a plain .npz that reloads
    # without PennyLane present and without unpickling anything.
    #
    # A saved model is *not* a classifier on raw features. It expects input that has
    # already been through the fold's fitted `MRMRSelector` and `TanhAngleScaler`, so
    # save the whole pipeline (`joblib.dump`) when you want end-to-end inference, and
    # this when you want the weights themselves.

    def save(self, path):
        """Write the trained parameters, the label order and the fit metadata to .npz."""
        check_is_fitted(self, "w_")
        np.savez_compressed(
            path,
            w=np.asarray(self.w_, dtype=float),
            W=np.asarray(self.W_, dtype=float),
            b=np.asarray(self.b_, dtype=float),
            classes=np.asarray(self.classes_).astype(str),
            class_weight=np.asarray(getattr(self, "class_weight_", []), dtype=float),
            loss=np.asarray(self.loss_, dtype=float),
            n_qubits=self.n_qubits_, n_layers=self.n_layers, epochs=self.epochs,
            batch_size=self.batch_size, lr=self.lr, seed=self.seed,
            n_steps=getattr(self, "n_steps_", 0))
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a ready-to-`predict` classifier from `save`. No training, no gradient."""
        f = np.load(path, allow_pickle=False)
        m = cls(n_layers=int(f["n_layers"]), epochs=int(f["epochs"]),
                batch_size=int(f["batch_size"]), lr=float(f["lr"]), seed=int(f["seed"]))
        m.w_, m.W_, m.b_ = f["w"], f["W"], f["b"]
        m.classes_ = f["classes"]
        m.n_qubits_ = int(f["n_qubits"])
        m.class_weight_ = f["class_weight"]
        m.loss_ = list(f["loss"])
        m.n_steps_ = int(f["n_steps"])
        return m
