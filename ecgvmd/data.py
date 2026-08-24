"""Loading ECGData.mat, in a way that works identically in Colab and locally."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

_SEARCH = ("ECGData.mat", "data/ECGData.mat", "../ECGData.mat",
           "/content/ECGData.mat", "/content/drive/MyDrive/ECGData.mat",
           os.path.expanduser("~/ECGData.mat"))


def find_mat(path: str | None = None, prompt_upload: bool = True) -> str:
    """Return a usable path to ECGData.mat.

    Order: explicit `path` -> a short list of conventional locations -> (Colab only)
    an interactive upload prompt. Raises rather than returning something unusable.
    """
    cands = [path] if path else list(_SEARCH)
    for c in cands:
        if c and os.path.exists(c):
            return c

    if prompt_upload:
        try:                                   # Colab: let the user upload it
            from google.colab import files     # noqa: F401  (import IS the probe)
            print("ECGData.mat not found - select it in the upload dialog.")
            uploaded = files.upload()
            return next(iter(uploaded))
        except ImportError:
            pass

    raise FileNotFoundError(
        "ECGData.mat not found. Put it next to the notebook, or pass an explicit "
        f"path. Looked in: {', '.join(_SEARCH)}")


@dataclass
class ECGDataset:
    """The whole .mat file, unpacked.

    Attributes
    ----------
    data       : (n_records, n_samples) float64, one row per recording.
    labels     : (n_records,) str, one of ARR / CHF / NSR.
    record_ids : (n_records,) int. **The grouping variable that prevents patient leakage.**
    fs         : sampling rate in Hz (128 for this dataset; not stored in the file).
    path       : where it was loaded from.
    """
    data: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    fs: float
    path: str

    @property
    def n_records(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    def class_counts(self) -> dict:
        v, c = np.unique(self.labels, return_counts=True)
        return dict(zip(v.tolist(), c.tolist()))

    def __repr__(self) -> str:
        mins = self.n_samples / self.fs / 60
        return (f"ECGDataset({self.n_records} records x {self.n_samples} samples "
                f"= {mins:.1f} min each @ {self.fs:g} Hz, {self.class_counts()})")


def load_ecgdata(path: str | None = None, fs: float = 128.0) -> ECGDataset:
    """Load and unpack ECGData.mat.

    The MATLAB file holds a single struct with fields `Data` and `Labels`; the label
    cell array needs a double unwrap to reach the strings.

    NOTE: the records are stored **sorted by class** (96 ARR, then 30 CHF, then 36 NSR).
    Never take `data[:n]` as a "sample" - it will be all one class.
    """
    p = find_mat(path)
    mat = sio.loadmat(p)
    ecg = mat["ECGData"]

    data = np.asarray(ecg["Data"][0, 0], dtype=np.float64)
    raw_labels = ecg["Labels"][0, 0]
    labels = np.array([str(np.asarray(raw_labels[i][0]).ravel()[0])
                       for i in range(len(raw_labels))])

    if len(labels) != data.shape[0]:
        raise ValueError(f"{len(labels)} labels for {data.shape[0]} records")

    return ECGDataset(data=data, labels=labels,
                      record_ids=np.arange(data.shape[0]), fs=fs, path=p)


def stratified_record_sample(labels: np.ndarray, n_total: int, seed: int = 0) -> np.ndarray:
    """Pick ~`n_total` record indices, spread evenly across classes.

    Exists because the file is class-sorted: the naive `arange(n)` picks 100% ARR.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = max(1, n_total // len(classes))
    out = [rng.choice(np.where(labels == c)[0], min(per, int((labels == c).sum())),
                      replace=False) for c in classes]
    return np.sort(np.concatenate(out))
