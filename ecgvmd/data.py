"""Loading ECGData.mat, in a way that works identically in Colab and locally."""
from __future__ import annotations

import os
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

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
    data       : (n_records, n_samples) float64, one row per separated ECG lead.
    labels     : (n_records,) str, one of ARR / CHF / NSR.
    record_ids : (n_records,) int. Row identifiers, NOT patient identifiers.
    source_ids : (n_records,) str. Verified original two-lead recording identifiers.
    patient_ids: (n_records,) str. Verified patient identifiers for validation splits.
    fs         : sampling rate in Hz (128 for this dataset; not stored in the file).
    path       : where it was loaded from.
    """
    data: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    fs: float
    path: str
    source_ids: np.ndarray | None = None
    patient_ids: np.ndarray | None = None

    def group_ids(self, grouping: str = "patient") -> np.ndarray:
        """Patient groups by default; row groups only for explicit legacy reproduction."""
        if grouping == "row":
            return self.record_ids
        if grouping not in ("patient", "source"):
            raise ValueError("grouping must be 'patient', 'source', or 'row'")
        ids = self.patient_ids if grouping == "patient" else self.source_ids
        if ids is None:
            raise ValueError(f"Verified {grouping} IDs are missing; supply a verified subject map")
        return ids

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
        patients = len(np.unique(self.patient_ids)) if self.patient_ids is not None else 'unverified'
        return (f"ECGDataset({self.n_records} lead rows, {patients} patients x {self.n_samples} samples "
                f"= {mins:.1f} min each @ {self.fs:g} Hz, {self.class_counts()})")


def load_subject_map(data, labels, path: str | None = None):
    """Load source-verified IDs and check every waveform hash before using them.

    The checked-in map is recovered by scripts/verify_ecg_sources.py from two
    distant source excerpts per lead. Hashes prevent applying it to reordered or
    modified data. A row-order heuristic is never used as a fallback.
    """
    path = Path(path) if path is not None else Path(__file__).with_name("ecgdata_subjects.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(data):
        raise ValueError("Subject map row count does not match ECGData")
    sources, patients = [], []
    for i, (wave, label, row) in enumerate(zip(data, labels, rows)):
        digest = hashlib.sha256(np.asarray(wave, dtype='<f8').tobytes()).hexdigest()
        if int(row['row']) != i or row['label'] != str(label) or row['row_sha256'] != digest:
            raise ValueError(f"Subject map does not match ECGData row {i}; reverify source identities")
        if not row['source_id'] or not row['patient_id']:
            raise ValueError(f"Missing source/patient identity for row {i}")
        sources.append(row['source_id'])
        patients.append(row['patient_id'])
    for identities in (sources, patients):
        for identity in set(identities):
            if len(set(np.asarray(labels)[np.asarray(identities) == identity])) != 1:
                raise ValueError(f"Conflicting labels within group {identity}")
    return np.asarray(sources), np.asarray(patients)


def load_ecgdata(path: str | None = None, fs: float = 128.0,
                 subject_map: str | None = None) -> ECGDataset:
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

    source_ids, patient_ids = load_subject_map(data, labels, subject_map)
    return ECGDataset(data=data, labels=labels,
                      record_ids=np.arange(data.shape[0]), fs=fs, path=p,
                      source_ids=source_ids, patient_ids=patient_ids)


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
