"""Read-only ACS adapter and fit-partition-only streaming feature normalization."""
import csv
import hashlib
from pathlib import Path

import numpy as np

from .cepstral import CepstralConfig, cepstral_features

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = REPO / 'experiments/acs_omi_vmd_wst_vqc/results/omi_v1/splits.csv'


def validate_splits(rows):
    rows = list(rows)
    records, patients = {}, {}
    for original in rows:
        row = dict(original)
        rid, pid = row['record_id'], row['patient_id']
        part = row['partition']
        if not rid or not pid or rid in records:
            raise ValueError('Missing identity or duplicate record')
        if row['split'] != 'train' or part not in ('fit', 'validation'):
            raise ValueError('Official test or unknown partition is forbidden')
        if str(row['label']) not in ('0', '1'):
            raise ValueError('Expected binary OMI labels')
        if pid in patients and patients[pid] != part:
            raise ValueError('Patient leakage across fit and validation')
        if not row.get('waveform_sha256'):
            raise ValueError('Missing waveform identity')
        patients[pid] = part
        records[rid] = row
    for part in ('fit', 'validation'):
        if {str(r['label']) for r in records.values() if r['partition'] == part} != {'0', '1'}:
            raise ValueError('Both labels required in each partition')
    return records


def load_splits(path=DEFAULT_SPLITS, *, expected_sha256):
    """Pin the existing split rather than silently generating a new one."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Split checksum mismatch')
    with path.open(newline='') as stream:
        records = validate_splits(csv.DictReader(stream))
    return records


def extract_acs_record(record, splits, config=CepstralConfig()):
    """Accept a record from the existing verified ACS loader, never official test."""
    from experiments.acs_omi_vmd_wst_vqc.loader import LEADS

    info = record.info
    row = splits.get(info.record_id)
    if info.split != 'train' or row is None or row['partition'] not in ('fit', 'validation'):
        raise ValueError('Record is outside the reserved fit/validation cohort')
    if (row['patient_id'] != info.patient_id or int(row['label']) != info.label
            or row['waveform_sha256'] != record.decision.waveform_sha256):
        raise ValueError('Record identity, label, or waveform differs from frozen split')
    if (not record.decision.eligible or record.signal_mV is None
            or record.fs != config.fs or tuple(record.leads) != LEADS
            or record.signal_mV.shape != (5000, 12)):
        raise ValueError('Expected eligible native 500 Hz, 5000-sample, all-lead ACS ECG')
    features, metadata = cepstral_features(record.signal_mV, config)
    metadata.update(record_id=info.record_id, patient_id=info.patient_id,
                    label=info.label, partition=row['partition'], leads=list(record.leads),
                    waveform_sha256=record.decision.waveform_sha256)
    return features, metadata


class FitStandardizer:
    """Per lead/coefficient statistics across fit frames only; no test-time fitting."""
    def fit(self, samples, splits):
        count, mean, m2, seen = 0, None, None, set()
        for rid, features in samples:
            if rid in seen or rid not in splits or splits[rid]['partition'] != 'fit':
                raise ValueError('Only unique fit records may fit normalization')
            x = np.asarray(features, dtype=np.float64)
            if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
                raise ValueError('Invalid feature tensor')
            if mean is not None and x.shape[1:] != mean.shape:
                raise ValueError('Inconsistent feature dimensions')
            n, local_mean = len(x), x.mean(axis=0)
            local_m2 = ((x - local_mean) ** 2).sum(axis=0)
            if mean is None:
                mean, m2 = local_mean, local_m2
            else:
                delta = local_mean - mean
                m2 = m2 + local_m2 + delta ** 2 * count * n / (count + n)
                mean = mean + delta * n / (count + n)
            count += n
            seen.add(rid)
        if not count:
            raise ValueError('No fit samples')
        self.mean_ = mean
        self.scale_ = np.sqrt(m2 / count)
        self.scale_[self.scale_ < 1e-8] = 1.0
        self.record_ids_ = sorted(seen)
        self.frame_count_ = count
        return self

    def transform(self, features):
        if not hasattr(self, 'mean_'):
            raise ValueError('Fit normalization on fit patients first')
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != self.mean_.shape or not np.isfinite(x).all():
            raise ValueError('Invalid feature tensor')
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def state_dict(self):
        return dict(mean=self.mean_.tolist(), scale=self.scale_.tolist(),
                    record_ids=self.record_ids_, frame_count=self.frame_count_)
