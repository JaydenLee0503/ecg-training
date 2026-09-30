"""Diagnostic-only copy of the frozen CPU batched VMD with an observation callback.

The arithmetic and stopping rule are unchanged. Production modules are unmodified.
The callback observes (iteration, stopping statistic, mode centres) after each sweep.
"""
import time
import numpy as np
from ecgvmd.vmd import VMDResult

def traced_vmd_batch(X, alpha=2000.0, tau=0.0, K=5, dc=True, init=1, tol=1e-7, max_iter=500,
              fs=128.0, on_iteration=None) -> VMDResult:
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
        if on_iteration is not None:
            on_iteration(n+1, u_diff.copy(), omega.copy())
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

