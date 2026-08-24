"""Single source of truth for every tunable in the pipeline.

Everything downstream reads from a `Config` instance. Nothing hard-codes fs, K or
alpha, so a sweep is a matter of `replace(CFG, alpha=5.0)` rather than an edit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import hashlib
import json

EPS = 1e-12

CLASS_ORDER = ["ARR", "CHF", "NSR"]
CLASS_COLORS = {"ARR": "#c0392b", "CHF": "#e67e22", "NSR": "#2471a3"}


@dataclass(frozen=True)
class Config:
    # ---- signal -------------------------------------------------------------
    fs: float = 128.0        # Hz. Every record in ECGData.mat was resampled to 128 Hz.
    seg_len: int = 500       # samples per analysis window = 3.91 s

    # ---- segmentation -------------------------------------------------------
    # "fixed" = non-overlapping grid; "beat" = centred on detected R-peaks.
    seg_mode: str = "fixed"
    n_per_record: int = 0    # 0 = every window in the record; >0 = random sample of that many
    seg_stride: int = 1      # keep every n-th window (1 = all). Cheap way to shrink a run.

    # ---- VMD ----------------------------------------------------------------
    K: int = 8               # number of modes
    alpha: float = 2000.0    # bandwidth penalty. Higher -> narrower, better-separated modes.
    tau: float = 0.0         # dual-ascent step. 0 = noise-tolerant, no exact-reconstruction constraint.
    dc: bool = True          # pin mode 0 at omega=0 so it absorbs baseline wander
    init: int = 1            # 1 = uniform omega spread (deterministic)
    tol: float = 1e-7
    max_iter: int = 500      # see notebooks/01 - at low alpha this cap BINDS
    chunk: int = 64          # batch rows per vmd_batch call (cache-friendly)

    # ---- evaluation ---------------------------------------------------------
    n_folds: int = 5
    seed: int = 0

    # ---- housekeeping -------------------------------------------------------
    cache_dir: str = "cache"
    fast: bool = False       # True -> aggressive subsampling for a smoke run

    def __post_init__(self):
        if self.seg_mode not in ("fixed", "beat"):
            raise ValueError(f"seg_mode must be 'fixed' or 'beat', got {self.seg_mode!r}")
        if self.K < 1:
            raise ValueError("K must be >= 1")

    # -- derived ---------------------------------------------------------------
    @property
    def nyquist(self) -> float:
        return self.fs / 2

    @property
    def seg_seconds(self) -> float:
        return self.seg_len / self.fs

    def replace(self, **kw) -> "Config":
        """`cfg.replace(alpha=5.0)` -> a new frozen Config."""
        return replace(self, **kw)

    def signature(self, *extra: str) -> str:
        """Stable short key for cache filenames.

        Only the fields that change the numbers are included; cache_dir and the
        plotting-only fields are not, so moving the cache does not invalidate it.
        """
        rel = ("fs", "seg_len", "seg_mode", "n_per_record", "seg_stride",
               "K", "alpha", "tau", "dc", "init", "tol", "max_iter", "seed")
        d = {k: getattr(self, k) for k in rel}
        blob = json.dumps(d, sort_keys=True) + "|" + "|".join(map(str, extra))
        short = hashlib.sha1(blob.encode()).hexdigest()[:8]
        return (f"{self.seg_mode}_K{self.K}_a{self.alpha:g}_L{self.seg_len}"
                f"_it{self.max_iter}_{short}")

    def summary(self) -> str:
        return (f"fs={self.fs:g} Hz | window={self.seg_len} ({self.seg_seconds:.2f} s) | "
                f"seg_mode={self.seg_mode} | K={self.K} | alpha={self.alpha:g} | "
                f"dc={self.dc} | max_iter={self.max_iter}")

    def to_dict(self) -> dict:
        return asdict(self)


#: The default used when a caller passes `cfg=None`.
CFG = Config()

FAST = Config(fast=True, n_per_record=6, seg_stride=1)
