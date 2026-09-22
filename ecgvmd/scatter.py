"""Wavelet Scattering Transform (Mallat, 2012; Anden & Mallat, 2014).

A cascade of wavelet filters, moduli, and a lowpass average:

    S0 = x * phi
    S1 = |x * psi_a| * phi
    S2 = ||x * psi_a| * psi_b| * phi        (only where xi_b < xi_a)

VMD returns (B, K, N): K modes at full time resolution, which `features.py` then has to
reduce to 28 descriptors apiece. Scattering returns (B, P, N//T): P paths already
averaged. The coefficients ARE the features, which is why this module has no equivalent
of `mode_features`. The output is also invariant to time shifts up to T - the arbitrary
QRS phase that `seg_mode="fixed"` produces is exactly what VMD does not handle.

Two backends, one contract (:class:`ScatterResult`):

* `scatter_kymatio` - the reference. kymatio 0.3.0 owns the filter design, the padding
                      and the path pruning.
* `scatter_batch`   - the same cascade by hand, and the default. `scatter_check` runs
                      both over six configurations: BIT-IDENTICAL at the project's
                      settings and at three others, and below 1.5e-16 at the remaining
                      two, which is FFT operation order. The two are one specification
                      implemented twice, not two specifications.
* `_morlet_bank`    - analytic Morlet filters in the Fourier domain, plus the xi/sigma/j
                      schedule that decides how many of them there are.
* `_gauss_lowpass`  - the averaging filter, at an arbitrary scale T.

`scatter_features` turns a :class:`ScatterResult` into the `(X, names)` pair the rest of
the pipeline consumes, which is the only step VMD needs `features.py` for and scattering
does not.

J and T are separate knobs on purpose: J sets how many octaves the bank spans, T sets
how much time invariance the output has. Most write-ups tie them at T = 2**J, but they
answer different questions and the sweep over T at fixed J is worth having. kymatio
0.3.0 does take T as a kwarg, so the split survives into the reference backend.

**The (J, Q, N) support constraint.** Narrow filters are long in time: a bandwidth sigma
in cycles/sample has an effective support of ~4/(2*pi*sigma) samples. At Q=8 the filters
must be narrow to be distinct (adjacent centres differ by 2**(1/8) = 9%), so the
low-frequency end of a nominal J*Q bank does not fit inside a 500-sample window.
kymatio handles this by staying constant-Q only while the filters fit and switching to
constant-bandwidth spacing below that.

Measured, J=6 / Q=(8,1) / shape=500 / T=64:

    126 paths (1 / 38 / 87 by order) x 7 time bins, first order spanning 55.7 - 0.52 Hz

Note 38 first-order filters, not the nominal J*Q = 48. If you need finer resolution at
the bottom of the band, lengthen the window - lowering J raises the floor instead.

**Order 0 is SIGNED; orders 1 and 2 are not.** S0 is `x * phi`, a plain local average,
and `standardise` only forces the mean of the WHOLE window to zero - each half-second
bin is free to swing either way. Measured on the 1620-window set at J=6 / T=64: -1.12 to
+1.13, 50.6% negative, and both backends agree to the digit. (An earlier note here
claimed -0.20 to +0.14 and 57% negative. It does not reproduce at any J/T tried - the
range scales with T, from -1.86/+1.65 at T=32 to -0.04/+0.05 at T=500 - and the sign
split is 0.50 +/- 0.03 throughout. The conclusion below was always the point and is
unaffected.) Orders 1 and 2 are moduli, so they are strictly positive. Anything
downstream that takes a log must handle the order-0 row separately, or it gets NaN. It is
not a degenerate near-zero column that can be filtered out by variance - it carries real
signal, just not on a log scale, which is why `scatter_features` leaves it LINEAR rather
than dropping it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import ifft as _ifft

__all__ = ["ScatterResult", "scatter_kymatio", "scatter_batch", "scatter_features",
           "order1_envelopes",
           "scatter_check", "morlet_bank_table"]


# ---------------------------------------------------------------------------------
# The contract both backends return
# ---------------------------------------------------------------------------------
@dataclass
class ScatterResult:
    """Scattering coefficients plus the path table that makes them interpretable.

    Mirrors :class:`ecgvmd.vmd.VMDResult`: `xi` / `xi_hz` play the part that `omega` /
    `omega_hz` play there. There is no `iters` or `capped_fraction` - the cascade is a
    fixed sequence of convolutions and moduli, so there is nothing to converge and no
    unconverged-decomposition trap to guard against.

    Attributes
    ----------
    coeffs : (B, P, n_bins) float64. Time-averaged scattering coefficients.
    order  : (P,) int, 0 / 1 / 2. Which order each path belongs to.
    xi     : (P, 2) centre frequencies in CYCLES/SAMPLE, NaN where the path has no
             wavelet at that order. The order-0 row is NaN in both columns.
    sigma  : (P, 2) bandwidths in cycles/sample, same NaN convention.
    """

    coeffs: np.ndarray
    order: np.ndarray
    xi: np.ndarray
    sigma: np.ndarray
    fs: float = 128.0
    J: int = 6
    Q: tuple | int = (8, 1)
    T: int = 64
    seconds: float = 0.0

    # -- derived ---------------------------------------------------------------
    @property
    def xi_hz(self) -> np.ndarray:
        """Centre frequencies in Hz. (P, 2), NaN where the path has no such order."""
        return self.xi * self.fs

    @property
    def n_paths(self) -> int:
        return int(self.coeffs.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.coeffs.shape[-1])

    @property
    def bin_seconds(self) -> float:
        """Seconds of signal per output time bin - i.e. the invariance scale."""
        return self.T / self.fs

    def meta(self):
        """The path table as a DataFrame. Sorting and filtering this is how you find
        out what a selected feature actually was."""
        import pandas as pd
        hz = self.xi_hz
        return pd.DataFrame({"order": self.order,
                             "xi1_hz": hz[:, 0], "xi2_hz": hz[:, 1]})

    def __repr__(self) -> str:
        o = np.bincount(np.asarray(self.order), minlength=3)
        f1 = self.xi_hz[self.order == 1, 0]
        span = f"{f1.max():.1f}-{f1.min():.2f} Hz" if f1.size else "no order-1 paths"
        return (f"ScatterResult({self.coeffs.shape[0]} x {self.n_paths} paths x "
                f"{self.n_bins} bins | J={self.J} Q={self.Q} T={self.T} | {span} | "
                f"order 0/1/2 = {o[0]}/{o[1]}/{o[2]})")


# ---------------------------------------------------------------------------------
# Reference backend: kymatio
# ---------------------------------------------------------------------------------
#
# Import the 1D frontend DIRECTLY, not `from kymatio.numpy import Scattering1D`. The
# package's `numpy.py` eagerly imports the 2D and 3D frontends too, and the 3D one needs
# `scipy.special.sph_harm`, which was deprecated in scipy 1.15 and REMOVED in 1.17 -
# this project pins scipy 1.18.1, so that import raises. The line below is verbatim
# line 1 of kymatio/numpy.py, without the two that follow it.
_KYMATIO_1D = "kymatio.scattering1d.frontend.numpy_frontend"


def _hashable_q(Q):
    """lru_cache keys must hash; a list Q would not."""
    return Q if isinstance(Q, int) else tuple(Q)


@lru_cache(maxsize=8)
def _kymatio(N: int, J: int, Q, T: int, max_order: int):
    """Cached `Scattering1D`. Constructing one builds the entire filter bank, which is
    wasted work when sweeping J/Q/T in a loop. Imported lazily so that `import ecgvmd`
    still works without kymatio installed - same reason `quantum.py` defers pennylane.
    """
    from importlib import import_module
    S1D = import_module(_KYMATIO_1D).ScatteringNumPy1D
    return S1D(J=J, shape=N, Q=Q, T=T, max_order=max_order)


def _two_cols(a) -> np.ndarray:
    """Force a meta column-pair to `(P, 2)`, NaN-padded.

    kymatio sizes its meta by `max_order`, so at `max_order=1` it hands back `(P, 1)` and
    `ScatterResult.meta()` raises on `xi[:, 1]`. The documented contract here is `(P, 2)`
    at every order, so that a first-order-only ablation and the full transform can be
    compared row against row without a shape check at each use.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.shape[1] >= 2:
        return a[:, :2]
    return np.hstack([a, np.full((a.shape[0], 2 - a.shape[1]), np.nan)])


def scatter_kymatio(X, J: int = 6, Q=(8, 1), T: int = 64, max_order: int = 2,
                    fs: float = 128.0) -> ScatterResult:
    """Scatter a batch of windows. `(B, N)` -> :class:`ScatterResult`.

    Parameters
    ----------
    X   : (B, N) or (N,) windows. Z-scored per window is what the rest of the pipeline
          produces and what this expects - scattering is NOT scale-invariant, so the
          `standardise` step in `segment.py` is doing real work.
    J   : octaves spanned by the filter bank. Sets the LOW frequency edge.
    Q   : wavelets per octave, `(Q1, Q2)` or a single int. Q1 is frequency resolution;
          Q2=1 is right for the second order, which encodes broadband modulation.
    T   : averaging scale in samples. Sets time invariance AND the output bin count.
    max_order : 2 is standard. 1 is the cheap ablation - order 2 is most of the paths.

    Notes
    -----
    Do not pre-pad to a power of two. kymatio derives its own padding from the filter
    support, which is the thing that makes the low-frequency filters usable at all.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Qh = _hashable_q(Q)

    t0 = time.time()
    S = _kymatio(int(X.shape[-1]), int(J), Qh, int(T), int(max_order))
    C = np.asarray(S(X), dtype=np.float64)
    m = S.meta()

    return ScatterResult(
        coeffs=C,
        order=np.asarray(m["order"], dtype=int),
        xi=_two_cols(m["xi"]),
        sigma=_two_cols(m["sigma"]),
        fs=float(fs), J=int(J), Q=Qh, T=int(T),
        seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------------
# Step 1: the filters, in the Fourier domain
# ---------------------------------------------------------------------------------
#
# Every filter is built as its own DFT at the padded length, because that is the only
# place the cascade uses it: convolution is a product in Fourier, and a filter is only
# ever brought back to time to normalise it. The design follows Lostanlen's thesis
# (2017), which is also what kymatio 0.3.0 implements, so the two backends here are one
# specification written twice rather than two specifications.
#
# The three constants below are kymatio's defaults and are NOT free parameters of this
# project. They are named rather than inlined so that the formulas below read as the
# thesis writes them, and so that a future sweep has something to sweep.

SIGMA0 = 0.1              # lowpass width at T=1; sigma_low = SIGMA0 / T
ALPHA = 5.0               # aliasing tolerance: a filter is treated as reaching to
                          # xi + ALPHA * sigma, and may be subsampled below that
R_PSI = math.sqrt(0.5)    # adjacent filters cross at this fraction of their peak


def _adaptive_p(sigma: float, eps: float = 1e-7) -> int:
    """How many periods of the frequency axis a Gaussian of width `sigma` needs.

    `np.fft.fftfreq` covers [-0.5, 0.5). A filter wide enough to be non-negligible at
    the ends of that interval is discontinuous there, and the discontinuity shows up in
    time as ringing. Building it over [1-P, P) and folding back (`_periodize`) removes
    that, and P is the smallest number of periods for which the tails are below `eps`.
    """
    return int(math.ceil(math.sqrt(-2 * sigma ** 2 * math.log(eps)) + 1))


def _periodize(h_f: np.ndarray, nperiods: int = 1) -> np.ndarray:
    """Fold `nperiods` copies of a spectrum onto one, by averaging.

    Two uses, both the same identity. In filter design it is discretisation in time.
    In the cascade (`_subsample`) it is subsampling in time by `nperiods`. The mean
    rather than the sum is what keeps the DFT convention consistent with the shortened
    length.
    """
    return h_f.reshape(nperiods, h_f.shape[0] // nperiods).mean(axis=0)


def _morlet_ft(N: int, xi: float | None, sigma: float) -> np.ndarray:
    """DFT of length `N` of one Morlet wavelet at centre `xi`, width `sigma`.

    `xi=None` gives the Gaussian instead, which is how `_gauss_lowpass` is built - the
    lowpass is the same object with the carrier removed, not a separate design.

        psi(t) = g_sigma(t) * (exp(i xi t) - kappa)

    `kappa` is fixed by the requirement that the transform vanish at DC. Skipping it
    leaves the wavelet a DC leak, which would put the window's local mean into every
    first-order coefficient - and `standardise` only zeroes the mean of the whole
    window, not of each averaging bin (see the module docstring on order 0).

    Normalised to unit L1 norm IN TIME. That is what makes `|x * psi|` comparable
    across scales: under L2 the wide low-frequency filters would answer a given
    amplitude more strongly than the narrow ones, and every order-1 coefficient would
    carry the filter's scale as well as the signal's content.
    """
    P = min(_adaptive_p(sigma), 5)
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    # At P=1 the wide interval degenerates to one period, and fftfreq's [-0.5, 0.5)
    # ordering is the one that stays continuous across 0.
    freqs_low = np.fft.fftfreq(N) if P == 1 else freqs
    low = _periodize(np.exp(-freqs_low ** 2 / (2 * sigma ** 2)), 2 * P - 1)
    if xi is None:
        h = low
    else:
        gabor = _periodize(np.exp(-(freqs - xi) ** 2 / (2 * sigma ** 2)), 2 * P - 1)
        h = gabor - (gabor[0] / low[0]) * low
    return h / np.abs(_ifft(h)).sum()


def _gauss_lowpass(N: int, T: int, sigma0: float = SIGMA0) -> np.ndarray:
    """The averaging filter phi, at an arbitrary invariance scale `T`.

    `sigma_low = sigma0 / T`: the width in frequency is inversely proportional to the
    support in time, so doubling T halves the bandwidth and doubles the invariance.
    This is the one filter whose scale is set by T rather than by J, which is what
    keeps the two knobs independent (see the module docstring).
    """
    return _morlet_ft(N, None, sigma0 / T)


def _sigma_psi(xi: float, Q: int, r: float = R_PSI) -> float:
    """Bandwidth for a filter at `xi` in a family of `Q` per octave.

    Chosen so that adjacent filters cross at a fraction `r` of their peak: wider and
    the bank is redundant, narrower and it has gaps that no path can see into.
    """
    factor = 1.0 / 2.0 ** (1.0 / Q)
    return xi * ((1 - factor) / (1 + factor)) / math.sqrt(2 * math.log(1.0 / r))


def _bank_params(J: int, Q: int, alpha: float = ALPHA, r_psi: float = R_PSI,
                 sigma0: float = SIGMA0):
    """The `(xi, sigma, j)` schedule for a bank of `Q` filters per octave over `J`.

    This is where the module docstring's support constraint is actually enforced, and
    it is the reason a nominal J*Q = 48 bank comes out as 38 filters at J=6 / Q=8.

    Two regions, meeting at an elbow:

    * **constant-Q**, while `sigma` is still above `sigma_min = sigma0 / 2**J`: xi and
      sigma both fall geometrically by 2**(1/Q) per step, so Q = xi/sigma is fixed and
      each filter is a dilation of the last.
    * **constant-bandwidth**, below it: sigma is pinned at `sigma_min` and xi falls
      *arithmetically* in Q-1 equal steps of `elbow/Q`, ending at `elbow/Q` rather than
      at zero - a filter centred on DC would be the lowpass, which the cascade already
      has. Continuing the geometric progression here would ask for filters longer than
      the padded window; this keeps covering the bottom of the band with what fits.

    `j` is the largest dyadic subsampling each filter tolerates without aliasing, from
    its upper edge `xi + alpha*sigma`. The cascade uses it twice: to subsample after
    each convolution, and to decide which second-order paths exist at all (`j2 > j1`,
    i.e. only ever scatter a fast modulation onto a slower one).
    """
    sigma_min = sigma0 / 2.0 ** J
    xi_max = max(1.0 / (1.0 + 2.0 ** (3.0 / Q)), 0.35)
    sigma_max = _sigma_psi(xi_max, Q, r_psi)

    xis: list[float] = []
    sigmas: list[float] = []
    if sigma_max <= sigma_min:
        elbow = sigma_max                       # no constant-Q region fits at all
    else:
        xis, sigmas = [xi_max], [sigma_max]
        while sigmas[-1] > sigma_min * 2.0 ** (1.0 / Q):
            xis.append(xis[-1] / 2.0 ** (1.0 / Q))
            sigmas.append(sigmas[-1] / 2.0 ** (1.0 / Q))
        elbow = xis[-1]
    for q in range(1, Q):
        xis.append(elbow - q / Q * elbow)
        sigmas.append(sigma_min)

    js = [int(math.floor(-math.log2(min(xi + alpha * s, 0.5))) - 1)
          for xi, s in zip(xis, sigmas)]
    return xis, sigmas, js


def _morlet_bank(N: int, J: int, Q: int, alpha: float = ALPHA, r_psi: float = R_PSI,
                 sigma0: float = SIGMA0):
    """Analytic Morlet filters in the Fourier domain: `(psi, xis, sigmas, js)`.

    `psi` is `(n_filters, N)` real - the DFT of a Morlet is real because |psi| is
    symmetric about xi, and keeping it real halves the work in the cascade's products.
    Ordered from the highest centre frequency down, which is the order the path table
    and every `ScatterResult` row inherits.
    """
    xis, sigmas, js = _bank_params(J, Q, alpha, r_psi, sigma0)
    psi = np.stack([_morlet_ft(N, xi, s) for xi, s in zip(xis, sigmas)]) if xis \
        else np.zeros((0, N))
    return psi, xis, sigmas, js


def _temporal_support(h_f: np.ndarray, criterion_amplitude: float = 1e-3) -> int:
    """Half the effective time support of a filter given in Fourier.

    The smallest N for which truncating the filter to [-N, N] changes a convolution by
    less than `criterion_amplitude` in L-infinity, found from the L1 tail of the
    impulse response. The padding is three of these, so the wavelets never see the
    reflection seam.
    """
    h = _ifft(np.atleast_2d(h_f), axis=1)
    half = h.shape[1] // 2
    tail = np.fliplr(np.cumsum(np.fliplr(np.abs(h)[:, :half]), axis=1))
    worst = tail.max(axis=0)
    ok = np.flatnonzero(worst <= criterion_amplitude)
    return int(ok.min()) + 1 if ok.size else half


# ---------------------------------------------------------------------------------
# Step 2: the cascade
# ---------------------------------------------------------------------------------
#
# The plan is everything that depends only on (N, J, Q, T, max_order) - filters, padding,
# the subsampling schedule - and so can be built once and reused across a whole dataset.
# `scatter_batch` is then just arithmetic over it.
#
# Subsampling is the part worth reading twice. After convolving with a filter whose upper
# edge is 2**-(j+1), the result is bandlimited and can be decimated by 2**j for free; the
# cascade does exactly that, so a low-frequency path is computed on a short array. The
# schedule is bookkept so that every path arrives at the same final length: k1 + k1_J and
# k1 + k2 + k2_T both sum to log2_T, which is why the output is a dense (B, P, n_bins)
# array rather than a ragged list.


def _subsample(x: np.ndarray, k: int) -> np.ndarray:
    """Subsample a batch of spectra in time by `k`, i.e. periodize them in Fourier."""
    if k == 1:
        return x
    return x.reshape(x.shape[0], k, x.shape[-1] // k).mean(axis=1)


def _levels(h_f: np.ndarray, max_level: int) -> list:
    """`h_f` and its periodizations, so that `levels[k]` matches an input subsampled by
    `2**k`. A filter must be resampled to the resolution of whatever it multiplies."""
    out = [h_f]
    for level in range(1, max_level + 1):
        out.append(_periodize(h_f, 2 ** level))
    return out


@lru_cache(maxsize=8)
def _plan(N_input: int, J: int, Q, T: int, max_order: int) -> dict:
    """Filters, padding and border indices for one `(N, J, Q, T, max_order)`.

    Cached, like `_kymatio`, because building the bank dominates a single call and is
    pure waste when sweeping. The returned dict is shared between callers - treat it as
    read-only.
    """
    Q1, Q2 = (Q, 1) if isinstance(Q, int) else (Q[0], Q[1] if len(Q) > 1 else 1)
    log2_T = int(math.floor(math.log2(T)))

    # Pad by three lowpass supports, then up to a power of two - but never so far that
    # a reflection would have to repeat the signal, which is what the second term caps.
    min_to_pad = 3 * _temporal_support(_gauss_lowpass(N_input, T))
    J_max_support = int(math.floor(math.log2(3 * N_input - 2)))
    J_pad = min(int(math.ceil(math.log2(N_input + 2 * min_to_pad))), J_max_support)
    N_pad = 2 ** J_pad
    to_add = N_pad - N_input
    pad_left, pad_right = to_add // 2, to_add - to_add // 2

    # Where the true signal sits inside the padded array, at every subsampling. Rounding
    # up on both ends is deliberately conservative: better to drop a boundary sample than
    # to report one that is partly reflection.
    ind_start, ind_end = {0: pad_left}, {0: pad_left + N_input}
    for j in range(1, max(log2_T, J) + 1):
        ind_start[j] = (ind_start[j - 1] // 2) + (ind_start[j - 1] % 2)
        ind_end[j] = (ind_end[j - 1] // 2) + (ind_end[j - 1] % 2)

    psi1, xi1s, sigma1s, j1s = _morlet_bank(N_pad, J, Q1)
    psi2, xi2s, sigma2s, j2s = _morlet_bank(N_pad, J, Q2)

    # Second-order filters are applied to an already-subsampled order-1 modulus, so each
    # needs its own spectrum at every subsampling it can actually receive: the j1 values
    # of the order-1 filters it is allowed to follow.
    psi2_levels = []
    for n2, j2 in enumerate(j2s):
        feasible = [j1 for j1 in j1s if j2 > j1]
        psi2_levels.append(_levels(psi2[n2], max(feasible, default=0)))

    max_sub_phi = min(max(max(j1s, default=0), max(j2s, default=0)), log2_T)
    phi_levels = _levels(_gauss_lowpass(N_pad, T), max_sub_phi)

    return {"N_pad": N_pad, "pad_left": pad_left, "pad_right": pad_right,
            "ind_start": ind_start, "ind_end": ind_end, "log2_T": log2_T,
            "psi1": psi1, "xi1s": xi1s, "sigma1s": sigma1s, "j1s": j1s,
            "psi2_levels": psi2_levels, "xi2s": xi2s, "sigma2s": sigma2s, "j2s": j2s,
            "phi_levels": phi_levels}


def scatter_batch(X, J: int = 6, Q=(8, 1), T: int = 64, max_order: int = 2,
                  fs: float = 128.0) -> ScatterResult:
    """Scatter a batch of windows, by hand. `(B, N)` -> :class:`ScatterResult`.

    Same signature and same contract as `scatter_kymatio`, and the same numbers to
    float64 round-off (`scatter_check`). This is the default because the cascade is the
    thing being studied: the path pruning, the subsampling schedule and the L1
    normalisation are all visible here, and a question about any of them is answered by
    reading forty lines rather than by tracing four modules of someone else's package.

    Parameters
    ----------
    X   : (B, N) or (N,) windows. Z-scored per window is what the rest of the pipeline
          produces and what this expects - scattering is NOT scale-invariant, so the
          `standardise` step in `segment.py` is doing real work.
    J   : octaves spanned by the filter bank. Sets the LOW frequency edge.
    Q   : wavelets per octave, `(Q1, Q2)` or a single int. Q1 is frequency resolution;
          Q2=1 is right for the second order, which encodes broadband modulation.
    T   : averaging scale in samples. Sets time invariance AND the output bin count.
    max_order : 2 is standard. 1 is the cheap ablation - order 2 is most of the paths.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Qh = _hashable_q(Q)
    t0 = time.time()
    p = _plan(int(X.shape[-1]), int(J), Qh, int(T), int(max_order))
    log2_T, i0, i1 = p["log2_T"], p["ind_start"], p["ind_end"]
    phi = p["phi_levels"]

    # Reflect rather than zero-pad: a zero pad introduces a step at the seam, and a step
    # has energy at every frequency, which every filter in the bank would then report.
    xp = np.pad(X, ((0, 0), (p["pad_left"], p["pad_right"])), mode="reflect")
    U0 = np.fft.fft(xp)

    # One list per order, concatenated at the end. The output is grouped by order, not
    # by the tree walk below - so an order-2 path is appended to `out2` while its parent
    # order-1 path sits in `out1`, and the two only meet in the final stack. Getting this
    # wrong is silent: the shapes still match and every coefficient is still present, but
    # the path table no longer describes the rows.
    out1: list = []
    out2: list = []
    meta1: list = []
    meta2: list = []

    # --- order 0: the plain local average -------------------------------------
    S0 = _ifft(_subsample(U0 * phi[0], 2 ** log2_T)).real
    out0 = [S0[:, i0[log2_T]:i1[log2_T]]]
    meta0 = [(0, (np.nan, np.nan), (np.nan, np.nan))]

    for n1, (xi1, s1, j1) in enumerate(zip(p["xi1s"], p["sigma1s"], p["j1s"])):
        # --- order 1: |x * psi1| * phi -----------------------------------------
        k1 = min(j1, log2_T)
        U1 = np.abs(_ifft(_subsample(U0 * p["psi1"][n1], 2 ** k1)))
        U1_hat = np.fft.fft(U1)

        k1_J = max(log2_T - k1, 0)
        S1 = _ifft(_subsample(U1_hat * phi[k1], 2 ** k1_J)).real
        out1.append(S1[:, i0[k1 + k1_J]:i1[k1 + k1_J]])
        meta1.append((1, (xi1, np.nan), (s1, np.nan)))

        if max_order < 2:
            continue

        for n2, (xi2, s2, j2) in enumerate(zip(p["xi2s"], p["sigma2s"], p["j2s"])):
            # Only scatter onto a SLOWER filter. A second wavelet at or above the first
            # one's frequency sees nothing: the modulus of a bandpass response is a slow
            # envelope, so |U1 * psi2| would be numerically zero and the path is pruned
            # rather than computed and stored. This single condition is what keeps the
            # order-2 count at 87 instead of 38 * 38.
            if j2 <= j1:
                continue
            k2 = max(min(j2, log2_T) - k1, 0)
            U2 = np.abs(_ifft(_subsample(U1_hat * p["psi2_levels"][n2][k1], 2 ** k2)))

            k2_T = max(log2_T - k2 - k1, 0)
            S2 = _ifft(_subsample(np.fft.fft(U2) * phi[k1 + k2], 2 ** k2_T)).real
            out2.append(S2[:, i0[k1 + k2 + k2_T]:i1[k1 + k2 + k2_T]])
            meta2.append((2, (xi1, xi2), (s1, s2)))

    rows = meta0 + meta1 + meta2
    return ScatterResult(
        coeffs=np.stack(out0 + out1 + out2, axis=1),
        order=np.asarray([r[0] for r in rows], dtype=int),
        xi=np.asarray([r[1] for r in rows], dtype=float),
        sigma=np.asarray([r[2] for r in rows], dtype=float),
        fs=float(fs), J=int(J), Q=Qh, T=int(T),
        seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------------
# From coefficients to a design matrix
# ---------------------------------------------------------------------------------
#
# This is the whole of what `features.py` does for VMD, and it is four lines of it,
# because the coefficients already ARE the features. The only real decisions are the
# log and what to do about order 0.
#
# **Why log.** Scattering coefficients of orders 1 and 2 span several decades across the
# band (measured on this data: 6.7e-06 to 2.3e-01, four and a half orders of magnitude)
# and are strongly right-skewed within any one path. Anden & Mallat (2014) take the log
# for exactly this reason: it turns the multiplicative structure of the cascade into an
# additive one, and it is what makes a Euclidean distance between two coefficient vectors
# mean anything. It matters more here than in a tree-based pipeline, because
# `TanhAngleScaler` standardises and then saturates: on a raw skewed column almost every
# value lands in one flat tail of the tanh and the qubit stops encoding anything.
#
# **Order 0 is left LINEAR, not dropped.** log(S0) is NaN 57% of the time (module
# docstring), and dropping the row throws away the window's local mean trajectory, which
# on a z-scored window is baseline wander - a real discriminant between these classes.
# Keeping it linear alongside 125 log columns is the honest option and costs nothing: the
# scaler standardises each column separately anyway.


def scatter_features(res: ScatterResult, log: bool = True, reduce: str | None = None,
                     eps: float = 1e-12):
    """`ScatterResult` -> `(X, names)`, the pair the rest of the pipeline consumes.

    Parameters
    ----------
    res    : the result to flatten.
    log    : take `log(S + eps)` on orders 1 and 2, leaving order 0 linear. The default,
             for the reasons in the comment above. `log=False` is the ablation.
    reduce : `None` keeps every time bin, giving `P * n_bins` columns - the full
             transform, time-localised. `"mean"` averages each path over its bins,
             giving `P` columns: fully time-invariant, and much closer to VMD's feature
             count, which is the comparison that controls for dimension.
    eps    : floor inside the log. Coefficients are non-negative but can be exactly 0
             where a path is pruned to numerical nothing.

    Returns
    -------
    X     : (B, d) float32, matching the dtype `features.py` blocks use.
    names : list of str, e.g. `o1_14.0hz_t3` or `o2_14.0hz_1.1hz_mean`. The Hz values
            are the path's centre frequencies, so a selected feature can be read back
            off `res.meta()` without re-running anything.
    """
    C = np.asarray(res.coeffs, dtype=np.float64)          # (B, P, n_bins)
    if log:
        m = res.order > 0
        C = C.copy()
        C[:, m, :] = np.log(C[:, m, :] + eps)

    if reduce == "mean":
        C = C.mean(axis=-1)[:, :, None]
    elif reduce is not None:
        raise ValueError(f"reduce must be None or 'mean', got {reduce!r}")

    hz = res.xi_hz
    tags = ["mean"] if reduce == "mean" else [f"t{t}" for t in range(C.shape[-1])]
    names = []
    for pth in range(C.shape[1]):
        o = int(res.order[pth])
        if o == 0:
            stem = "o0_dc"
        elif o == 1:
            stem = f"o1_{hz[pth, 0]:.1f}hz"
        else:
            stem = f"o2_{hz[pth, 0]:.1f}hz_{hz[pth, 1]:.2f}hz"
        names += [f"{stem}_{t}" for t in tags]

    X = C.reshape(C.shape[0], -1)
    bad = int((~np.isfinite(X)).sum())
    if bad:                       # loud, like features.py - a silent fix hides a bug
        print(f"  scatter_features: {bad} non-finite values -> 0.0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), names


# ---------------------------------------------------------------------------------
# Cross-check and inspection
# ---------------------------------------------------------------------------------


def order1_envelopes(X, J: int = 6, Q=(8, 1), T: int = 64, fs: float = 128.0):
    """`|x * psi1|` at FULL time resolution: `(env, centre_hz)` with env `(B, n_bands, N)`.

    The intermediate `scatter_batch` computes and then destroys. Inside the cascade each
    order-1 envelope is subsampled by `2**k1` and convolved with `phi` to buy time
    invariance, and *that averaging* is what removes the within-band structure that
    entropy, TKEO and waveform-length descriptors read. Recovered here, an envelope is
    the same kind of object a VMD mode is: a band-limited component at full resolution,
    ready for `features.mode_features`.

    This exists because the front-end comparison in `architects/wst_vs_vmd_vqc.md` §7 was
    otherwise confounded - VMD was contributing 28 descriptors per mode while the
    scattering block contributed raw time-averaged coefficients and no descriptors at all.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    B, N = X.shape
    p = _plan(N, int(J), _hashable_q(Q), int(T), 2)
    xp = np.pad(X, ((0, 0), (p["pad_left"], p["pad_right"])), mode="reflect")
    U0 = np.fft.fft(xp)
    sl = slice(p["pad_left"], p["pad_left"] + N)
    env = np.empty((B, len(p["xi1s"]), N), dtype=np.float64)
    for n1 in range(len(p["xi1s"])):
        env[:, n1, :] = np.abs(_ifft(U0 * p["psi1"][n1]))[:, sl]
    return env, np.asarray(p["xi1s"], dtype=float) * float(fs)


def scatter_check(X=None, configs=None, verbose: bool = True) -> "object":
    """Assert `scatter_batch` reproduces `scatter_kymatio`, over several configurations.

    This is the test that licenses using the hand-rolled backend at all, and it is a
    function rather than a notebook cell so that it can be re-run after any edit to the
    filter design. Returns a DataFrame with one row per configuration.

    The tolerance is float64 round-off and nothing more. The two backends agree EXACTLY
    (0.0) at four of the six configurations, including the project's own, and below
    1.5e-16 at the other two - the difference is FFT operation order alone, so its exact
    value moves with the input and is not worth pinning. If a change here makes this
    print anything above ~1e-12, the designs have genuinely diverged: fix the design, do
    not widen the tolerance.
    """
    import pandas as pd
    if X is None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 500))
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    configs = configs or [(6, (8, 1), 64, 2), (6, (8, 1), 64, 1), (5, (8, 1), 32, 2),
                          (6, 8, 64, 2), (4, (4, 2), 16, 2), (7, (12, 1), 128, 2)]
    rows = []
    for J, Q, T, mo in configs:
        a = scatter_batch(X, J=J, Q=Q, T=T, max_order=mo)
        b = scatter_kymatio(X, J=J, Q=Q, T=T, max_order=mo)
        assert a.coeffs.shape == b.coeffs.shape, f"shape {a.coeffs.shape} vs {b.coeffs.shape}"
        assert np.array_equal(a.order, b.order), "path order differs"
        assert np.allclose(a.xi, b.xi, equal_nan=True), "xi differs"
        d = float(np.abs(a.coeffs - b.coeffs).max())
        assert d < 1e-12, f"coefficients differ by {d:.3e} at J={J} Q={Q} T={T}"
        rows.append({"J": J, "Q": str(Q), "T": T, "max_order": mo,
                     "paths": a.n_paths, "bins": a.n_bins, "max_abs_diff": d,
                     "batch_s": round(a.seconds, 3), "kymatio_s": round(b.seconds, 3)})
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
        print(f"\nworst disagreement over {len(rows)} configurations: "
              f"{df.max_abs_diff.max():.3e}")
    return df


def morlet_bank_table(J: int = 6, Q: int = 8, fs: float = 128.0, N: int = 500,
                      T: int = 64):
    """The filter bank as a table: centre frequency, bandwidth, subsampling, support.

    What the module docstring's support constraint looks like in numbers. `support` is
    the effective half-support in samples, measured at the padded length the cascade
    actually uses (1024 for a 500-sample window at T=64, not 512 - the padding is what
    makes the wide filters usable).

    Read the last two columns together. Through the constant-Q region the support grows
    with every step, and by n=30 it is 294 - a full support of 588, already past the
    500-sample window and living entirely on the padding. Then it stops: the
    constant-bandwidth region pins sigma at sigma_min, so those seven filters all share
    one envelope of 317 and only their carrier moves. That flat tail is the constraint
    being respected rather than hit - continuing geometrically would have asked for
    filters the padded length cannot hold.
    """
    import pandas as pd
    xis, sigmas, js = _bank_params(J, Q)
    N_pad = _plan(int(N), int(J), (Q, 1), int(T), 2)["N_pad"]
    sup = [_temporal_support(_morlet_ft(N_pad, xi, s)) for xi, s in zip(xis, sigmas)]
    return pd.DataFrame({"n": np.arange(len(xis)), "xi_hz": np.asarray(xis) * fs,
                         "sigma_hz": np.asarray(sigmas) * fs, "j": js,
                         "support": sup,
                         "region": ["constant-Q" if s > min(sigmas) * 1.001
                                    else "constant-bandwidth" for s in sigmas]})
