"""Cutting the 512-second recordings into analysis windows.

Two strategies, both producing the same three arrays (windows, labels, groups):

* **fixed**  - a non-overlapping grid. Simple, uses all the data, but the QRS complex
               lands at an arbitrary phase in every window.
* **beat**   - windows centred on detected R-peaks, so morphology is aligned across
               samples. Costs an R-peak detector and throws away the inter-beat gaps.

`groups` is the record id and must be carried everywhere: splitting without it inflates
every score by ~12 macro-F1 points on this dataset (segments of one patient land on
both sides of the split).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .config import CFG, EPS, Config

__all__ = ["detect_r_peaks", "fixed_windows", "beat_windows", "segment", "standardise"]


def detect_r_peaks(sig: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """Pan-Tompkins style R-peak detection: bandpass -> derivative -> square -> integrate.

    Returns integer sample indices, sorted and unique. Returns an empty int array when
    nothing is found (`np.unique([])` is float64, which silently corrupts indexing).
    """
    sig = np.asarray(sig, dtype=float).ravel()
    ny = fs / 2
    b, a = butter(2, [5 / ny, min(20 / ny, 0.99)], btype="band")
    x = filtfilt(b, a, sig)

    sq = np.diff(x, prepend=x[0]) ** 2
    w = int(0.15 * fs) | 1
    integ = np.convolve(sq, np.ones(w) / w, mode="same")

    peaks, _ = find_peaks(integ, height=integ.mean() + 0.5 * integ.std(),
                          distance=int(0.25 * fs))
    if len(peaks) == 0:
        return np.empty(0, dtype=int)

    r = int(0.05 * fs)                          # snap to the local extremum of the band-passed signal
    snapped = [max(0, p - r) + int(np.argmax(np.abs(x[max(0, p - r):p + r + 1])))
               for p in peaks]
    return np.unique(snapped).astype(int)


def heart_rate(sig: np.ndarray, fs: float = 128.0,
               min_bpm: float = 30, max_bpm: float = 200) -> float:
    """Median heart rate in bpm, or NaN when too few beats are found."""
    rr = np.diff(detect_r_peaks(sig, fs)) / fs
    rr = rr[(rr > 60 / max_bpm) & (rr < 60 / min_bpm)]
    return float(60 / np.median(rr)) if rr.size >= 2 else float("nan")


def fixed_windows(data: np.ndarray, labels: np.ndarray, record_ids: np.ndarray,
                  seg_len: int = 500, n_per_record: int = 0, stride: int = 1,
                  seed: int = 0):
    """Chop each record into non-overlapping `seg_len` windows.

    `n_per_record > 0` keeps a random sample of that many windows per record, which is
    how the expensive experiments stay affordable without biasing toward the start of
    the recording.
    """
    rng = np.random.default_rng(seed)
    per_rec = data.shape[1] // seg_len

    W, y, g = [], [], []
    for i in range(data.shape[0]):
        idx = np.arange(per_rec)
        if n_per_record and n_per_record < per_rec:
            idx = np.sort(rng.choice(per_rec, n_per_record, replace=False))
        for j in idx:
            W.append(data[i, j * seg_len:(j + 1) * seg_len])
        y.extend([labels[i]] * len(idx))
        g.extend([record_ids[i]] * len(idx))

    W, y, g = np.asarray(W, dtype=np.float64), np.asarray(y), np.asarray(g)
    if stride > 1:
        W, y, g = W[::stride], y[::stride], g[::stride]
    return W, y, g


def beat_windows(data: np.ndarray, labels: np.ndarray, record_ids: np.ndarray,
                 fs: float = 128.0, seg_len: int = 500, n_per_record: int = 25,
                 stride: int = 1, seed: int = 0, min_beats: int | None = None):
    """Windows centred on detected R-peaks, so the QRS complex sits at the same phase.

    Records where fewer than `min_beats` usable peaks are found are skipped entirely -
    silently dropping a record is better than emitting windows built from noise, but
    the count is returned so the caller can see it happen.
    """
    rng = np.random.default_rng(seed)
    half = seg_len // 2
    min_beats = n_per_record if min_beats is None else min_beats

    W, y, g, skipped = [], [], [], []
    for i in range(data.shape[0]):
        sig = data[i]
        pk = detect_r_peaks(sig, fs)
        pk = pk[(pk > half) & (pk < len(sig) - half)]
        if len(pk) < min_beats:
            skipped.append(int(record_ids[i]))
            continue
        take = min(n_per_record, len(pk)) if n_per_record else len(pk)
        sel = np.sort(rng.choice(len(pk), take, replace=False))
        for b in sel:
            W.append(sig[pk[b] - half:pk[b] + half])
        y.extend([labels[i]] * take)
        g.extend([record_ids[i]] * take)

    W, y, g = np.asarray(W, dtype=np.float64), np.asarray(y), np.asarray(g)
    if stride > 1:
        W, y, g = W[::stride], y[::stride], g[::stride]
    if skipped:
        print(f"  beat_windows: skipped {len(skipped)} record(s) with < {min_beats} "
              f"usable R-peaks: {skipped}")
    return W, y, g


def standardise(W: np.ndarray) -> np.ndarray:
    """Z-score each window on its own.

    Per-window, not per-record: the classifier should key on morphology, not on the
    recording gain of whichever database the record came from.
    """
    mu = W.mean(-1, keepdims=True)
    sd = W.std(-1, keepdims=True)
    return (W - mu) / (sd + EPS)


def segment(ds, cfg: Config | None = None, zscore: bool = True):
    """Dispatch on `cfg.seg_mode` and return `(windows, labels, groups)`.

    `windows` is z-scored per window unless `zscore=False`.
    """
    cfg = cfg or CFG
    kw = dict(seg_len=cfg.seg_len, n_per_record=cfg.n_per_record,
              stride=cfg.seg_stride, seed=cfg.seed)
    if cfg.seg_mode == "fixed":
        W, y, g = fixed_windows(ds.data, ds.labels, ds.record_ids, **kw)
    else:
        kw["n_per_record"] = cfg.n_per_record or 25
        W, y, g = beat_windows(ds.data, ds.labels, ds.record_ids, fs=cfg.fs, **kw)
    return (standardise(W) if zscore else W), y, g
