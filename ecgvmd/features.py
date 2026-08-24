"""IMF feature extraction - turning (B, K, N) modes into a design matrix.

Three families, kept separable so their contributions can be measured independently:

* `mode_features`   - 28 descriptors per intrinsic mode (28*K columns). The main event.
* `rr_features`     - 9 rhythm descriptors, from R-peaks detected on the QRS-band modes.
* `global_features` - 3 whole-decomposition descriptors (energy spread, residual).
* `raw_features`    - the CONTROL: the same descriptors on the undecomposed window,
                      plus classical Fourier band powers. This is what VMD has to beat.

Every function is vectorised over the batch axis; nothing loops over segments except
the R-peak detector, which cannot be.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import hilbert, welch, find_peaks
from scipy.stats import skew, kurtosis

from .config import CFG, EPS, Config
from .vmd import vmd_apply

__all__ = ["perm_entropy", "higuchi_fd", "mode_features", "rr_features",
           "global_features", "raw_features", "extract_features", "FeatureBundle",
           "MODE_FEATS", "RR_NAMES", "GLOBAL_NAMES"]


# ---------------------------------------------------------------------------------
# Complexity measures
# ---------------------------------------------------------------------------------
def perm_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> np.ndarray:
    """Normalised permutation entropy (Bandt & Pompe) along the last axis.

    Counts the relative frequency of the m! orderings of m consecutive samples. It is
    amplitude-blind and robust to noise, which is why it separates the high-frequency
    modes so well here.
    """
    *lead, N = x.shape
    n = N - (m - 1) * delay
    idx = np.arange(n)[:, None] + delay * np.arange(m)[None, :]
    order = np.argsort(x[..., idx], axis=-1, kind="stable")
    codes = (order * (m ** np.arange(m))).sum(-1).reshape(-1, n)
    nb = m ** m
    counts = np.stack([np.bincount(row, minlength=nb) for row in codes]) / n
    H = -(np.where(counts > 0, counts * np.log(counts + EPS), 0.0)).sum(-1)
    return (H / np.log(math.factorial(m))).reshape(lead)


def higuchi_fd(x: np.ndarray, kmax: int = 8) -> np.ndarray:
    """Higuchi fractal dimension along the last axis.

    Slope of log(curve length) against log(1/k); higher = more self-similar roughness.
    """
    *lead, N = x.shape
    xf = x.reshape(-1, N)
    ks = np.arange(1, kmax + 1)
    L = np.empty((xf.shape[0], kmax))
    for i, k in enumerate(ks):
        Lk = np.zeros(xf.shape[0])
        for m in range(k):
            idx = np.arange(m, N, k)
            if idx.size < 2:
                continue
            Lk += np.abs(np.diff(xf[:, idx], axis=1)).sum(1) * (N - 1) / ((idx.size - 1) * k * k)
        L[:, i] = Lk / k
    ylog = np.log(L + EPS)
    xlog = np.log(1.0 / ks)
    xc = xlog - xlog.mean()
    return ((xc @ (ylog - ylog.mean(1, keepdims=True)).T) / (xc ** 2).sum()).reshape(lead)


# ---------------------------------------------------------------------------------
# Per-mode descriptors
# ---------------------------------------------------------------------------------
MODE_FEATS = [
    # energy / amplitude
    "log_energy", "rel_energy", "std", "skew", "kurt", "ptp",
    # waveform shape
    "zcr", "wave_len", "tkeo", "shannon", "log_ent", "hjorth_mob", "hjorth_comp",
    # spectral
    "centre_hz", "spec_centroid", "spec_bandwidth", "spec_entropy", "peak_hz",
    "spec_flatness", "rolloff85",
    # Hilbert / instantaneous
    "env_mean", "env_std", "env_cv", "if_mean", "if_std",
    # complexity + relation to the parent signal
    "perm_entropy", "higuchi_fd", "corr_raw",
]


def mode_features(U: np.ndarray, omega_hz: np.ndarray, fs: float,
                  x: np.ndarray | None = None, prefix: str = "u"):
    """28 descriptors per mode.

    Parameters
    ----------
    U        : (B, K, N) modes.
    omega_hz : (B, K) centre frequencies in Hz (VMD's own estimate).
    x        : (B, N) the window the modes came from. Needed for `corr_raw`, the
               correlation between each mode and its parent signal. If omitted the
               column is filled with zeros rather than quietly meaning something else.

    Returns
    -------
    (mat, names) with `mat` of shape (B, 28*K), column-major over (feature, mode).
    """
    U = np.asarray(U, dtype=float)
    B, K, N = U.shape

    e = (U ** 2).sum(-1)
    rel = e / (e.sum(1, keepdims=True) + EPS)

    d1 = np.diff(U, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = U.var(-1) + EPS, d1.var(-1) + EPS, d2.var(-1) + EPS
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + EPS)

    pn = U ** 2 / ((U ** 2).sum(-1, keepdims=True) + EPS)

    P = np.abs(np.fft.rfft(U * np.hanning(N), axis=-1)) ** 2
    frq = np.fft.rfftfreq(N, 1 / fs)
    Pn = P / (P.sum(-1, keepdims=True) + EPS)
    centroid = Pn @ frq
    bandwidth = np.sqrt(np.maximum(
        (Pn * (frq[None, None, :] - centroid[..., None]) ** 2).sum(-1), 0))

    A = hilbert(U, axis=-1)
    env = np.abs(A)
    ifq = np.clip(np.diff(np.unwrap(np.angle(A), axis=-1), axis=-1) * fs / (2 * np.pi),
                  0, fs / 2)
    env_mean, env_std = env.mean(-1), env.std(-1)

    if x is None:
        corr = np.zeros((B, K))
    else:
        xs = np.asarray(x, dtype=float)[:, None, :N]
        xc = xs - xs.mean(-1, keepdims=True)
        uc = U - U.mean(-1, keepdims=True)
        corr = ((uc * xc).sum(-1)
                / (np.sqrt((uc ** 2).sum(-1) * (xc ** 2).sum(-1)) + EPS))

    cols = {
        "log_energy": np.log(e + EPS), "rel_energy": rel, "std": U.std(-1),
        "skew": skew(U, axis=-1), "kurt": kurtosis(U, axis=-1), "ptp": np.ptp(U, axis=-1),
        "zcr": np.diff(np.signbit(U), axis=-1).mean(-1),
        "wave_len": np.abs(d1).sum(-1),
        "tkeo": (U[..., 1:-1] ** 2 - U[..., :-2] * U[..., 2:]).mean(-1),
        "shannon": -(pn * np.log(pn + EPS)).sum(-1),
        "log_ent": np.log(U ** 2 + EPS).mean(-1),
        "hjorth_mob": mob, "hjorth_comp": comp,
        "centre_hz": np.asarray(omega_hz, dtype=float),
        "spec_centroid": centroid, "spec_bandwidth": bandwidth,
        "spec_entropy": -(Pn * np.log(Pn + EPS)).sum(-1) / np.log(Pn.shape[-1]),
        "peak_hz": frq[np.argmax(P, axis=-1)],
        "spec_flatness": np.exp(np.log(P + EPS).mean(-1)) / (P.mean(-1) + EPS),
        "rolloff85": frq[np.argmax(np.cumsum(Pn, axis=-1) >= 0.85, axis=-1)],
        "env_mean": env_mean, "env_std": env_std, "env_cv": env_std / (env_mean + EPS),
        "if_mean": ifq.mean(-1), "if_std": ifq.std(-1),
        "perm_entropy": perm_entropy(U, 3), "higuchi_fd": higuchi_fd(U, 8),
        "corr_raw": corr,
    }
    mat = np.stack([cols[f][:, k] for f in MODE_FEATS for k in range(K)], axis=1)
    names = [f"{prefix}{k + 1}_{f}" for f in MODE_FEATS for k in range(K)]
    return mat.astype(np.float32), names


# ---------------------------------------------------------------------------------
# Whole-decomposition descriptors
# ---------------------------------------------------------------------------------
GLOBAL_NAMES = ["mode_energy_entropy", "recon_norm_ratio", "recon_energy_frac"]


def global_features(U: np.ndarray, x: np.ndarray):
    """Three descriptors of the decomposition as a whole.

    `recon_norm_ratio` is ||x - sum u_k|| / ||x||; `recon_energy_frac` is its SQUARE,
    which is the share of *energy* left unexplained. Both are kept because the two are
    routinely confused - a norm ratio of 0.21 is 4.3% of the energy, not 21%.
    """
    U = np.asarray(U, dtype=float)
    x = np.asarray(x, dtype=float)[:, :U.shape[-1]]
    e = (U ** 2).sum(-1)
    p = e / (e.sum(1, keepdims=True) + EPS)
    ent = -(p * np.log2(p + EPS)).sum(1)
    r = np.linalg.norm(x - U.sum(1), axis=-1) / (np.linalg.norm(x, axis=-1) + EPS)
    return np.stack([ent, r, r ** 2], axis=1).astype(np.float32), list(GLOBAL_NAMES)


# ---------------------------------------------------------------------------------
# Rhythm descriptors from the QRS-band modes
# ---------------------------------------------------------------------------------
RR_NAMES = ["rr_rate_count", "rr_rate_mean", "rr_sdnn", "rr_cvnn",
            "rr_rmssd", "rr_pnn50", "rr_range", "rr_nbeats", "rr_amp_cv"]


def qrs_band(U: np.ndarray, omega_hz: np.ndarray, lo=5.0, hi=45.0) -> np.ndarray:
    """Sum only the modes whose centre frequency lands in the QRS band.

    This is VMD acting as an adaptive band-pass: the band edges are chosen by the data,
    not by a fixed filter design.
    """
    return (U * ((omega_hz >= lo) & (omega_hz <= hi))[..., None]).sum(-2)


def tk_envelope(x: np.ndarray, fs: float, win_s: float = 0.06) -> np.ndarray:
    """Smoothed Teager-Kaiser energy. Sharpens the QRS against everything else."""
    tk = np.empty_like(x)
    tk[:, 1:-1] = x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:]
    tk[:, 0], tk[:, -1] = tk[:, 1], tk[:, -2]
    tk = np.maximum(tk, 0.0)
    w = max(3, int(round(win_s * fs)) | 1)
    ker = np.hanning(w)
    ker /= ker.sum()
    pad = w // 2
    xp = np.pad(tk, ((0, 0), (pad, pad)), mode="edge")
    return np.apply_along_axis(lambda r: np.convolve(r, ker, mode="valid"), 1, xp)


def rr_features(U: np.ndarray, omega_hz: np.ndarray, fs: float,
                min_bpm: float = 30, max_bpm: float = 220):
    """Rate and variability, from R-peaks found on the reconstructed QRS band.

    Nine columns: beat count rate, mean RR rate, SDNN, CVNN, RMSSD, pNN50, RR range,
    beat count, and the coefficient of variation of the peak amplitudes.
    """
    q = qrs_band(U, omega_hz)
    env = tk_envelope(q, fs)
    dist = max(1, int(fs * 60 / max_bpm))
    dur = q.shape[1] / fs
    out = np.zeros((q.shape[0], len(RR_NAMES)))

    for i, e in enumerate(env):
        thr = 0.35 * np.percentile(e, 99)
        pk, _ = find_peaks(e, height=thr, distance=dist)
        n = len(pk)
        rate_count = n / dur * 60
        rr = np.diff(pk) / fs if n >= 3 else np.array([])
        rr = rr[(rr > 60 / max_bpm) & (rr < 60 / min_bpm)] if rr.size else rr
        amp_cv = e[pk].std() / (e[pk].mean() + EPS) if n >= 2 else 0.0
        if rr.size >= 2:
            d = np.diff(rr)
            out[i] = [rate_count, 60 / rr.mean(), rr.std(), rr.std() / (rr.mean() + EPS),
                      np.sqrt((d ** 2).mean()), (np.abs(d) > 0.05).mean(),
                      rr.max() - rr.min(), n, amp_cv]
        else:
            out[i] = [rate_count, rate_count, 0, 0, 0, 0, 0, n, amp_cv]
    return out.astype(np.float32), list(RR_NAMES)


# ---------------------------------------------------------------------------------
# Control features (no VMD)
# ---------------------------------------------------------------------------------
BANDS = [(0.0, 0.5), (0.5, 4.0), (4.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 64.0)]


def raw_features(X: np.ndarray, fs: float):
    """The same descriptors on the *undecomposed* window, plus fixed-band Fourier powers.

    Without this control there is no way to tell whether VMD is contributing anything
    beyond "we computed 28 statistics".
    """
    X = np.asarray(X, dtype=float)
    mat, names = mode_features(X[:, None, :], np.zeros((len(X), 1)), fs,
                               x=X, prefix="raw")
    names = [n.replace("raw1_", "raw_") for n in names]

    f, P = welch(X, fs=fs, nperseg=min(256, X.shape[1]), axis=-1)
    Pn = P / (P.sum(-1, keepdims=True) + EPS)
    bp = np.stack([Pn[:, (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], 1)
    bnames = [f"bp_{lo:g}_{hi:g}Hz" for lo, hi in BANDS]

    pairs = [(i, j) for i in range(len(BANDS)) for j in range(i + 1, len(BANDS))]
    ratios = np.log(np.stack([bp[:, i] / (bp[:, j] + EPS) for i, j in pairs], 1) + EPS)
    rnames = [f"bpr_{BANDS[i][0]:g}/{BANDS[j][0]:g}" for i, j in pairs]

    # rel_energy is identically 1 and centre_hz identically 0 for a single "mode";
    # corr_raw is identically 1. Drop the three degenerate columns.
    drop = ("rel_energy", "centre_hz", "corr_raw")
    keep = [n for n in names if not n.endswith(drop)]
    ki = [names.index(n) for n in keep]
    return (np.hstack([mat[:, ki], bp, ratios]).astype(np.float32),
            keep + bnames + rnames)


# ---------------------------------------------------------------------------------
# One streaming pass over the data
# ---------------------------------------------------------------------------------
class FeatureBundle:
    """The output of one extraction run: several named feature blocks over the same rows.

    `bundle["VMD modes + rhythm"]` returns `(matrix, names)`, which is what the
    evaluation code consumes. Blocks share row order with `y` and `groups`.
    """

    def __init__(self, blocks: dict, y, groups, cfg, iters=None, imfs=None):
        self.blocks = blocks
        self.y = np.asarray(y)
        self.groups = np.asarray(groups)
        self.cfg = cfg
        self.iters = None if iters is None else np.asarray(iters)
        self.imfs = imfs                      # (B, K, N) float32, only if explicitly kept

    def __getitem__(self, k):
        return self.blocks[k]

    def keys(self):
        return self.blocks.keys()

    @property
    def capped_fraction(self) -> float:
        if self.iters is None:
            return float("nan")
        return float((self.iters >= self.cfg.max_iter).mean())

    def describe(self) -> str:
        lines = [f"{len(self.y)} segments from {len(np.unique(self.groups))} records",
                 f"config: {self.cfg.summary()}"]
        if self.iters is not None:
            lines.append(f"VMD capped fraction: {100 * self.capped_fraction:.1f}%")
        for k, (M, n) in self.blocks.items():
            lines.append(f"  {k:26s} {M.shape[1]:4d} features")
        return "\n".join(lines)

    def save(self, path: str):
        """Write every block to a single .npz (plus names, labels, groups, config)."""
        import json
        payload = {"__y": self.y, "__groups": self.groups,
                   "__config": np.array(json.dumps(self.cfg.to_dict()))}
        if self.iters is not None:
            payload["__iters"] = self.iters
        if self.imfs is not None:
            payload["__imfs"] = self.imfs
        for k, (M, n) in self.blocks.items():
            payload[f"X::{k}"] = M
            payload[f"names::{k}"] = np.array(n, dtype=object)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str):
        import json
        from .config import Config
        z = np.load(path, allow_pickle=True)
        blocks = {}
        for key in z.files:
            if key.startswith("X::"):
                name = key[3:]
                blocks[name] = (z[key], list(z[f"names::{name}"]))
        cfg = Config(**json.loads(str(z["__config"])))
        return cls(blocks, z["__y"], z["__groups"], cfg,
                   iters=z["__iters"] if "__iters" in z.files else None,
                   imfs=z["__imfs"] if "__imfs" in z.files else None)


def extract_features(W: np.ndarray, y, groups, cfg: Config | None = None,
                     with_control: bool = True, keep_imfs: bool = False,
                     verbose: bool = True) -> FeatureBundle:
    """VMD every window and reduce it to features in one streaming pass.

    Parameters
    ----------
    W           : (B, N) z-scored windows.
    keep_imfs   : also retain the raw (B, K, N) mode tensor. That is the input a deep
                  model wants, but it is ~4 bytes * B * K * N - keep it off for the
                  full dataset unless you have the memory.

    Returns
    -------
    FeatureBundle with the blocks: control (no VMD), VMD modes, rhythm only,
    VMD modes + rhythm, VMD + rhythm + control.
    """
    cfg = cfg or CFG
    W = np.asarray(W, dtype=np.float64)
    holder: dict = {}
    kept: list = []
    cursor = {"i": 0}

    def per_chunk(U, omega_hz):
        n = U.shape[0]
        i0 = cursor["i"]
        xs = W[i0:i0 + n]
        cursor["i"] = i0 + n
        fm, nm = mode_features(U, omega_hz, cfg.fs, x=xs)
        fg, ng = global_features(U, xs)
        fr, nr = rr_features(U, omega_hz, cfg.fs)
        holder["mode_names"] = nm + ng
        holder["rr_names"] = nr
        if keep_imfs:
            kept.append(U.astype(np.float32))
        return np.hstack([fm, fg]), fr

    if verbose:
        print(f"VMD + features: {len(W)} windows, {cfg.summary()}")
    parts, iters = vmd_apply(W, cfg=cfg, fn=per_chunk, verbose=verbose)

    F_mode = np.vstack([p[0] for p in parts])
    F_rr = np.vstack([p[1] for p in parts])
    mode_names, rr_names = holder["mode_names"], holder["rr_names"]

    blocks = {
        "VMD modes": (F_mode, mode_names),
        "rhythm only": (F_rr, rr_names),
        "VMD modes + rhythm": (np.hstack([F_mode, F_rr]), mode_names + rr_names),
    }
    if with_control:
        F_raw, raw_names = raw_features(W, cfg.fs)
        blocks["control (no VMD)"] = (F_raw, raw_names)
        blocks["VMD + rhythm + control"] = (np.hstack([F_mode, F_rr, F_raw]),
                                            mode_names + rr_names + raw_names)

    # Sanitise once, loudly. A silent nan_to_num hides a real bug in a new feature.
    for k, (M, n) in blocks.items():
        bad = int((~np.isfinite(M)).sum())
        if bad:
            print(f"  {k}: {bad} non-finite values -> 0.0")
        blocks[k] = (np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), n)

    return FeatureBundle(blocks, y, groups, cfg, iters=iters,
                         imfs=np.concatenate(kept) if kept else None)
