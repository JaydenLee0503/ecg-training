"""Experimental FP64 CUDA VMD for a separate ACS engineering benchmark.

NumPy performs the same mirror/FFT/reconstruction as the frozen CPU solver.
One CUDA block advances one lead, in bounded 64-iteration launches. Descriptors
remain on the CPU. Nothing imports this module from the production extractor.
"""
from __future__ import annotations

import time
import numpy as np

from ecgvmd.vmd import VMDResult, vmd_batch
from ecgvmd.features import mode_features

CUDA_SOURCE = r'''
extern "C" __global__
void advance(const double2* f, const double* freq, double2* u,
             double2* total, double2* lam, double* omega, int* iters,
             int* finished, int H, int K, double alpha, double tau,
             int dc, double tol, int begin, int end) {
    const int b = blockIdx.x, t = threadIdx.x;
    if (finished[b]) return;
    __shared__ double om[32], power[256], weighted[256], change[256], delta;
    __shared__ int stop;
    if (t < K) om[t] = omega[b*K+t];
    if (t == 0) stop = 0;
    __syncthreads();
    for (int n = begin; n < end; ++n) {
        if (t == 0) delta = 0.0;
        __syncthreads();
        for (int k = 0; k < K; ++k) {
            double sp = 0.0, sw = 0.0, sd = 0.0;
            for (int j = t; j < H; j += blockDim.x) {
                const long ix = ((long)b*K+k)*H+j;
                const long bx = (long)b*H+j;
                const double2 old = u[ix], ft = f[bx], sm = total[bx], la = lam[bx];
                const double offset = freq[j]-om[k];
                const double den = 1.0+alpha*(offset*offset);
                const double re = (ft.x-(sm.x-old.x)-0.5*la.x)/den;
                const double im = (ft.y-(sm.y-old.y)-0.5*la.y)/den;
                const double dr = re-old.x, di = im-old.y;
                const double p = re*re+im*im;
                u[ix] = make_double2(re,im);
                total[bx] = make_double2(sm.x+dr,sm.y+di);
                sp += p; sw += p*freq[j]; sd += dr*dr+di*di;
            }
            power[t]=sp; weighted[t]=sw; change[t]=sd;
            __syncthreads();
            for (int stride=128; stride>0; stride/=2) {
                if (t<stride) {
                    power[t]+=power[t+stride];
                    weighted[t]+=weighted[t+stride];
                    change[t]+=change[t+stride];
                }
                __syncthreads();
            }
            if (t==0) {
                delta += change[0];
                if (!(dc && k==0) && power[0]>0.0) om[k]=weighted[0]/power[0];
            }
            __syncthreads();
        }
        if (tau>0.0) {
            for (int j=t; j<H; j+=blockDim.x) {
                const long bx=(long)b*H+j;
                lam[bx].x += tau*(total[bx].x-f[bx].x);
                lam[bx].y += tau*(total[bx].y-f[bx].y);
            }
        }
        if (t==0 && 2.2204460492503131e-16+delta/H<=tol) {
            iters[b]=n+1; finished[b]=1; stop=1;
        }
        __syncthreads();
        if (stop) break;
    }
    if (t<K) omega[b*K+t]=om[t];
}
'''

_kernel = None


def vmd_gpu_batch(X, alpha=2000.0, tau=0.0, K=8, dc=True, init=1,
                  tol=1e-7, max_iter=2000, fs=500.0):
    """Return CPU arrays; timing includes transfers and synchronized GPU work."""
    import cupy as cp
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(CUDA_SOURCE, 'advance',
                               options=('--std=c++11', '--fmad=false'))
    started = time.perf_counter()
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if (X.ndim != 2 or not len(X) or X.shape[1] < 4 or not np.isfinite(X).all()
            or not 1 <= K <= 32 or init not in (0, 1) or max_iter < 1):
        raise ValueError('Invalid input, K, initialization or iteration limit')
    if X.shape[1] % 2:
        X = X[:, :-1]
    B, N = X.shape
    h = N // 2
    mirrored = np.concatenate([X[:, :h][:, ::-1], X, X[:, -h:][:, ::-1]], axis=1)
    T = mirrored.shape[1]
    H = T // 2
    freq = ((np.arange(1, T+1)/T)-0.5-1.0/T)[H:T].copy()
    spectrum = np.fft.fftshift(np.fft.fft(mirrored, axis=1), axes=1)[:, H:T].copy()
    omega = np.tile((0.5/K)*np.arange(K), (B, 1)) if init == 1 else np.zeros((B, K))
    if dc:
        omega[:, 0] = 0.0
    f = cp.asarray(spectrum)
    freqs = cp.asarray(freq)
    u = cp.zeros((B, K, H), dtype=cp.complex128)
    total = cp.zeros((B, H), dtype=cp.complex128)
    lam = cp.zeros_like(total)
    om = cp.asarray(omega)
    iters = cp.full(B, max_iter, dtype=cp.int32)
    finished = cp.zeros(B, dtype=cp.int32)
    for begin in range(0, max_iter, 64):
        _kernel((B,), (256,), (f, freqs, u, total, lam, om, iters, finished,
                              np.int32(H), np.int32(K), np.float64(alpha),
                              np.float64(tau), np.int32(dc), np.float64(tol),
                              np.int32(begin), np.int32(min(begin+64, max_iter))))
    cp.cuda.get_current_stream().synchronize()
    positive = cp.asnumpy(u).transpose(0, 2, 1)
    omega, iterations = cp.asnumpy(om), cp.asnumpy(iters)
    full = np.zeros((B, T, K), dtype=np.complex128)
    full[:, H:T, :] = positive
    full[:, np.arange(1, H+1)[::-1], :] = positive.conj()
    full[:, 0, :] = full[:, -1, :].conj()
    modes = np.fft.ifft(np.fft.ifftshift(full, axes=1), axis=1).real
    modes = modes.transpose(0, 2, 1)[:, :, T//4:3*T//4]
    order = np.argsort(omega, axis=1)
    return VMDResult(np.take_along_axis(modes, order[:, :, None], axis=1),
                     np.take_along_axis(omega, order, axis=1), iterations,
                     max_iter, fs=fs, seconds=time.perf_counter()-started)


def extract_vmd(signals, protocol, *, backend, lead_batch):
    """Frozen retries/descriptors, with a separately declared execution batch."""
    if backend not in ('cpu', 'gpu') or lead_batch < 1:
        raise ValueError('Invalid backend or lead batch')
    X = np.asarray(signals, dtype=np.float64)
    v = protocol['vmd']
    kwargs = {k: v[k] for k in ('alpha', 'tau', 'K', 'dc', 'init', 'tol')}
    kwargs['fs'] = protocol['input']['fs']
    solver = vmd_batch if backend == 'cpu' else vmd_gpu_batch
    modes = np.empty((len(X), v['K'], X.shape[1]))
    omega = np.empty((len(X), v['K']))
    iterations = np.zeros(len(X), int)
    limits = np.zeros(len(X), int)
    attempts = []
    started = time.perf_counter()
    for start in range(0, len(X), lead_batch):
        remaining = np.arange(start, min(start+lead_batch, len(X)))
        for limit in v['iteration_limits']:
            result = solver(X[remaining], max_iter=limit, **kwargs)
            attempts.extend(dict(lead_index=int(i), limit=limit, iterations=int(n), capped=bool(c))
                            for i, n, c in zip(remaining, result.iters, result.capped))
            modes[remaining], omega[remaining] = result.modes, result.omega_hz
            iterations[remaining], limits[remaining] = result.iters, limit
            remaining = remaining[result.capped]
            if not len(remaining):
                break
        if len(remaining):
            raise RuntimeError(f'Unconverged leads after {limit}: {remaining.tolist()}')
    solver_seconds = time.perf_counter()-started
    parts=[]
    # Keep CPU descriptor groups identical to the frozen four-lead pipeline.
    for start in range(0, len(X), 4):
        features, names = mode_features(modes[start:start+4], omega[start:start+4],
                                        kwargs['fs'], x=X[start:start+4])
        parts.append(features)
    features = np.vstack(parts)
    if not np.isfinite(features).all() or not np.isfinite(modes).all():
        raise FloatingPointError('Nonfinite GPU benchmark result')
    return dict(features=features, names=names, modes=modes, omega_hz=omega,
                iterations=iterations, limits=limits, attempts=attempts,
                solver_seconds=solver_seconds, total_seconds=time.perf_counter()-started)
