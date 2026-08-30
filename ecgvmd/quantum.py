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
           "product_angle_kernel", "gram_matrix", "QuantumKernelSVC"]

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
        if self.embedding == "angle":
            self.qnode_ = None if self.exact_angle else angle_kernel_qnode(n_features)
            self.kernel_fn_ = product_angle_kernel if self.exact_angle else None
        elif self.embedding == "iqp":
            self.qnode_ = iqp_kernel_qnode(n_features, self.n_repeats)
            self.kernel_fn_ = None
        else:
            raise ValueError(f"unknown embedding {self.embedding!r}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.X_train_ = X
        self.n_qubits_ = X.shape[1]
        self._setup(self.n_qubits_)
        self.gram_ = gram_matrix(X, qnode=self.qnode_, chunk=self.chunk,
                                 kernel_fn=self.kernel_fn_)
        self.svc_ = SVC(kernel="precomputed", C=self.C,
                        class_weight=self.class_weight).fit(self.gram_, y)
        self.classes_ = self.svc_.classes_
        return self

    def _test_kernel(self, X):
        check_is_fitted(self, "svc_")
        return gram_matrix(np.asarray(X, dtype=float), self.X_train_,
                           qnode=self.qnode_, chunk=self.chunk,
                           kernel_fn=self.kernel_fn_)

    def predict(self, X):
        return self.svc_.predict(self._test_kernel(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_kernel(X))
