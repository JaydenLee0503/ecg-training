"""Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014).

Two solvers, deliberately kept side by side:

* `vmd`        - the reference, one signal at a time. Slow, transparent, and the thing
                 every other implementation gets checked against.
* `vmd_batch`  - the same ADMM updates, vectorised over a batch and restricted to the
                 analytic (non-negative frequency) half of the spectrum. ~50x faster.

Both return a :class:`VMDResult`, which carries the iteration counts so that the
**capped fraction** - the share of signals that hit `max_iter` without meeting `tol` -
is impossible to forget about. That number matters: at low `alpha` the ADMM iteration
converges *more slowly*, and an unconverged decomposition is not the decomposition you
think you are analysing.

Measured on this dataset (K=8, seg_len=500, max_iter=500):

    alpha=2000 -> 33% capped | 200 -> 90% | 50 -> 100% | 5 -> 100%
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import CFG, EPS, Config

__all__ = ["VMDResult", "vmd", "vmd_batch", "vmd_apply"]


@dataclass
class VMDResult:
    """Modes plus the convergence bookkeeping that makes them interpretable."""

    modes: np.ndarray            # (K, N) from `vmd`, or (B, K, N) from `vmd_batch`
    omega: np.ndarray            # centre frequencies in CYCLES/SAMPLE, ascending
    iters: np.ndarray            # iterations actually used, per signal
    max_iter: int
    fs: float = 128.0
    omega_hist: np.ndarray | None = None   # (n_iter, K) trajectory; reference solver only
    seconds: float = 0.0

    @property
    def omega_hz(self) -> np.ndarray:
        """Centre frequencies in Hz."""
        return self.omega * self.fs

    @property
    def capped(self) -> np.ndarray:
        """Boolean mask of signals that exhausted `max_iter`."""
        return np.atleast_1d(self.iters) >= self.max_iter

    @property
    def capped_fraction(self) -> float:
        """Share of signals that never met `tol`. Anything above ~0.05 is a warning."""
        return float(self.capped.mean())

    @property
    def converged(self) -> bool:
        return self.capped_fraction == 0.0

    def reconstruction(self) -> np.ndarray:
        """Sum of the modes - what VMD claims the signal is."""
        return self.modes.sum(-2)

    def residual_norm_ratio(self, x: np.ndarray) -> np.ndarray:
        """||x - sum_k u_k|| / ||x||.

        This is a NORM ratio, not an energy fraction. The energy fraction is its
        SQUARE: a ratio of 0.21 means 4.3% of the energy is unexplained, not 21%.
        """
        x = np.asarray(x, dtype=float)
        x = x[..., :self.modes.shape[-1]]
        r = x - self.reconstruction()
        return (np.linalg.norm(r, axis=-1) / (np.linalg.norm(x, axis=-1) + EPS))

    def residual_energy_fraction(self, x: np.ndarray) -> np.ndarray:
        """The square of :meth:`residual_norm_ratio` - the honest "how much did we miss"."""
        return self.residual_norm_ratio(x) ** 2

    def report(self, label: str = "VMD") -> str:
        pct = 100 * self.capped_fraction
        n = np.atleast_1d(self.iters).size
        line = (f"{label}: {n} signal(s), median {int(np.median(self.iters))} iters, "
                f"{pct:.0f}% capped at max_iter={self.max_iter}")
        if pct > 5:
            line += ("\n  WARNING: those decompositions are UNCONVERGED. Raise max_iter "
                     "or alpha before drawing conclusions from them.")
        return line

    def __repr__(self) -> str:
        return (f"VMDResult(modes={self.modes.shape}, "
                f"capped={100 * self.capped_fraction:.0f}%, max_iter={self.max_iter})")


# ---------------------------------------------------------------------------------
# Reference solver
# ---------------------------------------------------------------------------------
def vmd(f, alpha=2000.0, tau=0.0, K=5, dc=False, init=1, tol=1e-7, max_iter=500,
        fs=128.0) -> VMDResult:
    """Decompose one 1-D signal into `K` band-limited modes.

    Parameters
    ----------
    f        : (N,) real signal. An odd trailing sample is dropped so N is even.
    alpha    : bandwidth penalty. Large -> narrow, well-separated modes.
    tau      : dual-ascent step. 0 disables the multiplier (tolerates noise).
    K        : number of modes.
    dc       : if True, mode 0 is pinned at omega=0 and absorbs baseline wander.
    init     : 1 = uniform omega spread (deterministic), 2 = log-random, 0 = zeros.
    tol      : convergence threshold on the summed relative mode update.
    max_iter : iteration cap. Reported back via `VMDResult.capped_fraction`.

    Returns
    -------
    VMDResult with `modes` (K, N) and `omega` (K,), both sorted by ascending frequency.
    """
    t0 = time.time()
    f = np.asarray(f, dtype=float).ravel()
    if f.size % 2:
        f = f[:-1]
    N = f.size
    h = N // 2

    # mirror extension: the FFT is circular, this keeps the wrap-around out of the modes
    f_mirr = np.concatenate([f[:h][::-1], f, f[-h:][::-1]])
    T = f_mirr.size
    freqs = (np.arange(1, T + 1) / T) - 0.5 - 1.0 / T      # cycles/sample in [-0.5, 0.5)

    # analytic spectrum: discard the negative half
    f_hat = np.fft.fftshift(np.fft.fft(f_mirr))
    f_hat[:T // 2] = 0.0
    pos = slice(T // 2, T)
    f_pos = freqs[pos]

    if init == 1:
        omega = (0.5 / K) * np.arange(K)
    elif init == 2:
        lo = 1.0 / N
        omega = np.sort(np.exp(np.log(lo) + (np.log(0.5) - np.log(lo)) * np.random.rand(K)))
    else:
        omega = np.zeros(K)
    if dc:
        omega[0] = 0.0

    u_hat = np.zeros((T, K), dtype=complex)
    lam = np.zeros(T, dtype=complex)
    total = np.zeros(T, dtype=complex)                     # running sum_k u_hat_k
    omega_hist = np.zeros((max_iter, K))
    eps_ = np.spacing(1.0)

    n = max_iter - 1
    for n in range(max_iter):
        u_diff = 0.0
        for k in range(K):                                 # Gauss-Seidel sweep over modes
            old = u_hat[:, k]
            # Wiener filter on the residual left by all the *other* modes
            num = f_hat - (total - old) - 0.5 * lam
            den = 1.0 + alpha * (freqs - omega[k]) ** 2
            new = num / den
            d = new - old
            u_diff += float((d * d.conj()).sum().real)
            total = total + d
            u_hat[:, k] = new
            if not (dc and k == 0):                        # centre of spectral mass
                p = np.abs(new[pos]) ** 2
                s = p.sum()
                if s > 0:
                    omega[k] = float(f_pos @ p / s)
        if tau > 0:
            lam = lam + tau * (total - f_hat)
        omega_hist[n] = omega
        if eps_ + u_diff / T <= tol:
            n += 1
            break
    n_iter = n if n else max_iter

    # rebuild the Hermitian spectrum and invert
    full = np.zeros((T, K), dtype=complex)
    full[T // 2:T, :] = u_hat[T // 2:T, :]
    full[np.arange(1, T // 2 + 1)[::-1], :] = np.conj(u_hat[T // 2:T, :])
    full[0, :] = np.conj(full[-1, :])

    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=0), axis=0)).T
    u = u[:, T // 4:3 * T // 4]                            # drop the mirrored halves

    order = np.argsort(omega)
    return VMDResult(modes=u[order], omega=omega[order], iters=np.array([n_iter]),
                     max_iter=max_iter, fs=fs, omega_hist=omega_hist[:n_iter][:, order],
                     seconds=time.time() - t0)


# ---------------------------------------------------------------------------------
# Batched solver
# ---------------------------------------------------------------------------------
def vmd_batch(X, alpha=2000.0, tau=0.0, K=5, dc=True, init=1, tol=1e-7, max_iter=500,
              fs=128.0) -> VMDResult:
    """Vectorised VMD over a batch of equal-length signals.

    Mathematically the same updates as :func:`vmd`, but every signal advances together
    and only the analytic half of the spectrum is stored. Rows that converge early are
    retired from the working set, so a batch costs roughly its slowest member.

    Parameters
    ----------
    X : (B, N) array of B real signals.

    Returns
    -------
    VMDResult with `modes` (B, K, N), `omega` (B, K), `iters` (B,).
    """
    t0 = time.time()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] % 2:
        X = X[:, :-1]
    B, N = X.shape
    h = N // 2

    Xm = np.concatenate([X[:, :h][:, ::-1], X, X[:, -h:][:, ::-1]], axis=1)
    T = Xm.shape[1]
    H = T // 2
    freqs = (np.arange(1, T + 1) / T) - 0.5 - 1.0 / T
    f_pos = freqs[H:T].copy()
    f_hat = np.fft.fftshift(np.fft.fft(Xm, axis=1), axes=1)[:, H:T].copy()

    if init == 1:
        omega = np.tile((0.5 / K) * np.arange(K), (B, 1))
    elif init == 2:
        lo = 1.0 / N
        omega = np.sort(np.exp(np.log(lo) + (np.log(0.5) - np.log(lo))
                               * np.random.rand(B, K)), axis=1)
    else:
        omega = np.zeros((B, K))
    if dc:
        omega[:, 0] = 0.0

    u_hat = np.zeros((B, H, K), dtype=complex)
    lam = np.zeros((B, H), dtype=complex)
    total = np.zeros((B, H), dtype=complex)

    active = np.arange(B)                 # original row index of each surviving buffer row
    alive = np.ones(B, dtype=bool)
    U_out = np.zeros((B, H, K), dtype=complex)
    O_out = np.zeros((B, K))
    iters = np.full(B, max_iter, dtype=int)
    eps_ = np.spacing(1.0)

    for n in range(max_iter):
        u_diff = np.zeros(u_hat.shape[0])
        for k in range(K):
            old = u_hat[:, :, k]
            num = f_hat - (total - old) - 0.5 * lam
            den = 1.0 + alpha * (f_pos[None, :] - omega[:, k, None]) ** 2
            new = num / den
            d = new - old
            u_diff += np.einsum("bt,bt->b", d, d.conj()).real
            total = total + d
            u_hat[:, :, k] = new
            if not (dc and k == 0):
                p = new.real ** 2 + new.imag ** 2
                s = p.sum(axis=1)
                omega[:, k] = np.where(s > 0, (p @ f_pos) / np.where(s > 0, s, 1.0),
                                       omega[:, k])
        if tau > 0:
            lam = lam + tau * (total - f_hat)

        u_diff = eps_ + 2.0 * u_diff / T          # x2 restores the mirrored negative half
        done = (u_diff <= tol) & alive
        if done.any():
            gi = active[done]
            U_out[gi], O_out[gi], iters[gi] = u_hat[done], omega[done], n + 1
            alive &= ~done
            if not alive.any():
                break
            if alive.mean() < 0.8:                # compact the working set
                s = alive
                active, u_hat, lam = active[s], u_hat[s], lam[s]
                total, omega, f_hat = total[s], omega[s], f_hat[s]
                alive = np.ones(int(s.sum()), dtype=bool)
    if alive.any():
        gi = active[alive]
        U_out[gi], O_out[gi] = u_hat[alive], omega[alive]

    full = np.zeros((B, T, K), dtype=complex)
    full[:, H:T, :] = U_out
    full[:, np.arange(1, H + 1)[::-1], :] = np.conj(U_out)
    full[:, 0, :] = np.conj(full[:, -1, :])

    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=1), axis=1))
    u = np.transpose(u, (0, 2, 1))[:, :, T // 4:3 * T // 4]

    order = np.argsort(O_out, axis=1)
    return VMDResult(modes=np.take_along_axis(u, order[:, :, None], axis=1),
                     omega=np.take_along_axis(O_out, order, axis=1),
                     iters=iters, max_iter=max_iter, fs=fs, seconds=time.time() - t0)


def vmd_apply(X, cfg: Config | None = None, chunk: int | None = None, fn=None,
              verbose: bool = False, warn_capped: bool = True):
    """Run :func:`vmd_batch` over `X` in cache-sized chunks.

    `fn(U, omega_hz)` is applied per chunk if given, which lets callers reduce each
    chunk to features immediately instead of materialising an (n_segments, K, N)
    array - for the full dataset that would be several hundred MB.

    Returns
    -------
    (parts, iters) where `parts` is the list of per-chunk outputs (the VMDResult
    itself when `fn is None`) and `iters` is the concatenated iteration counts.
    """
    cfg = cfg or CFG
    chunk = chunk or cfg.chunk
    out, all_iters, t0 = [], [], time.time()

    for i in range(0, len(X), chunk):
        res = vmd_batch(X[i:i + chunk], alpha=cfg.alpha, tau=cfg.tau, K=cfg.K,
                        dc=cfg.dc, init=cfg.init, tol=cfg.tol,
                        max_iter=cfg.max_iter, fs=cfg.fs)
        all_iters.append(res.iters)
        out.append(res if fn is None else fn(res.modes, res.omega_hz))
        if verbose and (i // chunk) % 25 == 0:
            done = min(i + chunk, len(X))
            print(f"  {done:6d}/{len(X)}  ({time.time() - t0:5.0f} s)", flush=True)

    iters = np.concatenate(all_iters) if all_iters else np.empty(0, dtype=int)
    frac = float((iters >= cfg.max_iter).mean()) if iters.size else 0.0
    if verbose:
        print(f"  done in {time.time() - t0:.0f} s  |  {100 * frac:.0f}% capped "
              f"at max_iter={cfg.max_iter}")
    if warn_capped and frac > 0.05:
        print(f"  WARNING: {100 * frac:.0f}% of segments hit max_iter={cfg.max_iter} "
              f"without reaching tol={cfg.tol:g} (alpha={cfg.alpha:g}).\n"
              f"           Lowering alpha makes this WORSE, not better. "
              f"Raise max_iter if these modes matter.")
    return out, iters
