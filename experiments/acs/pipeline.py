"""Matched ACS inputs, patient splits, front ends and train-only preprocessing.

The original ECGData experiment modules are unchanged. See the separate ACS
protocol for choices and limitations. No official test ECG is accepted here.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from .loader import LEADS, POLICY_VERSION
from ecgvmd.features import mode_features
from ecgvmd.quantum import TanhAngleScaler
from ecgvmd.scatter import scatter_batch, scatter_features, scatter_kymatio
from ecgvmd.select import MRMRSelector
from ecgvmd.vmd import vmd_batch


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def patient_split(rows, protocol):
    """Assign all eligible official-training records; preserve conflicting labels.

    Sorting by ID makes the result independent of incoming ledger order. Patients
    may have ECGs with different labels; never invent a patient-level diagnosis.
    """
    rows = sorted((dict(row) for row in rows), key=lambda row: row['record_id'])
    if not rows or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Need unique, nonempty record IDs')
    if any(r['split'] != 'train' or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Splitting accepts only labeled official-training ECGs')
    y = np.array([r['label'] for r in rows], dtype=int)
    patients = np.array([r['patient_id'] for r in rows])
    if set(y) != {0, 1}:
        raise ValueError('Both diagnosis classes are required')
    setting = protocol['split']
    cv = StratifiedGroupKFold(setting['n_splits'], shuffle=True, random_state=setting['seed'])
    fold = np.full(len(rows), -1, dtype=int)
    for k, (train, val) in enumerate(cv.split(np.zeros((len(y), 1)), y, patients)):
        if set(patients[train]) & set(patients[val]):
            raise ValueError('Patient crosses split boundary')
        if set(y[train]) != {0, 1} or set(y[val]) != {0, 1}:
            raise ValueError('Every partition must contain both classes')
        if np.any(fold[val] != -1):
            raise ValueError('Repeated validation assignment')
        fold[val] = k
    if np.any(fold < 0):
        raise ValueError('Incomplete split coverage')
    for row, k in zip(rows, fold):
        row.update(fold=int(k), partition='validation' if k == setting['validation_fold'] else 'fit')
    return rows


def engineering_ids(rows, protocol):
    candidates = [r['record_id'] for r in rows if r['partition'] == 'fit']
    return sorted(candidates, key=lambda rid: hashlib.sha256(
        (protocol['id'] + ':' + rid).encode()).digest())[:protocol['engineering_check']['records']]


def extract_record(record, protocol, *, reference=False, on_attempt=None):
    """Return both feature vectors from exactly the same unmodified ECG.

    Work in lead batches of four to bound VMD memory. Each failed convergence
    attempt is observable before a retry, including when the last attempt fails.
    """
    expected = protocol['input']
    if (record.info.split != 'train' or record.info.label not in (0, 1)
            or record.fs != expected['fs'] or tuple(record.leads) != tuple(expected['leads'])
            or tuple(record.leads) != LEADS or expected['record_policy'] != POLICY_VERSION):
        raise ValueError('Record does not match the declared labeled ACS training input')
    if not record.decision.eligible or record.signal_mV is None:
        raise ValueError('Excluded ECG cannot enter feature extraction')
    if record.signal_mV.shape != (expected['samples'], len(LEADS)):
        raise ValueError('Unexpected ECG shape')
    X = np.asarray(record.signal_mV.T, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite ECG input')
    started = time.perf_counter()
    v = protocol['vmd']
    kwargs = {k: v[k] for k in ('K', 'alpha', 'tau', 'dc', 'init', 'tol')}
    kwargs['fs'] = record.fs
    attempts = []
    pieces, final_iters, final_limits = [], [], []
    for start in range(0, len(LEADS), 4):
        signals = X[start:start + 4]
        modes = np.empty((len(signals), v['K'], X.shape[1]))
        omega = np.empty((len(signals), v['K']))
        iters, limits = np.zeros(len(signals), int), np.zeros(len(signals), int)
        remaining = np.arange(len(signals))
        for limit in v['iteration_limits']:
            res = vmd_batch(signals[remaining], max_iter=limit, **kwargs)
            for local, niter, capped in zip(remaining, res.iters, res.capped):
                attempt = dict(lead=LEADS[start + int(local)], limit=limit,
                               iterations=int(niter), capped=bool(capped))
                attempts.append(attempt)
                if on_attempt is not None:
                    on_attempt(attempt)
            modes[remaining], omega[remaining] = res.modes, res.omega_hz
            iters[remaining], limits[remaining] = res.iters, limit
            remaining = remaining[res.capped]
            if not len(remaining):
                break
        if len(remaining):
            raise RuntimeError(f'{record.info.record_id}: unconverged VMD leads after {limit} iterations')
        features, names = mode_features(modes, omega, record.fs, x=signals)
        if not np.isfinite(features).all():
            raise FloatingPointError('Nonfinite VMD features; no zero replacement is permitted')
        pieces.append(features)
        final_iters.extend(iters.tolist())
        final_limits.extend(limits.tolist())
    vmd_seconds = time.perf_counter() - started
    Xv = np.vstack(pieces).reshape(-1)
    vnames = [f'{lead}:{name}' for lead in LEADS for name in names]
    w = protocol['wst']
    kw = {k: w[k] for k in ('J', 'T', 'max_order')}
    kw.update(Q=tuple(w['Q']), fs=record.fs)
    started = time.perf_counter()
    scattering = scatter_batch(X, **kw)
    if not np.isfinite(scattering.coeffs).all() or np.any(scattering.coeffs[:, scattering.order > 0] < 0):
        raise FloatingPointError('Invalid scattering coefficients')
    features, names = scatter_features(scattering, log=w['log'], reduce=w['reduce'], eps=w['eps'])
    if not np.isfinite(features).all():
        raise FloatingPointError('Nonfinite WST features')
    Xw = features.reshape(-1)
    wnames = [f'{lead}:p{i}:{name}' for lead in LEADS for i, name in enumerate(names)]
    wst_seconds = time.perf_counter() - started
    error = None
    reference_seconds = None
    if reference:
        started = time.perf_counter()
        ref = scatter_kymatio(X, **kw)
        np.testing.assert_array_equal(scattering.order, ref.order)
        np.testing.assert_allclose(scattering.xi, ref.xi, atol=1e-14, rtol=0, equal_nan=True)
        error = float(np.max(np.abs(scattering.coeffs - ref.coeffs)))
        cfg = protocol['engineering_check']
        np.testing.assert_allclose(scattering.coeffs, ref.coeffs,
                                   atol=cfg['reference_atol'], rtol=cfg['reference_rtol'])
        reference_seconds = time.perf_counter() - started
    details = dict(record_id=record.info.record_id, patient_id=record.info.patient_id,
                   label=record.info.label, waveform_sha256=record.decision.waveform_sha256,
                   physical_input_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                   vmd_attempts=attempts, vmd_iterations=final_iters, vmd_limits=final_limits,
                   vmd_seconds=vmd_seconds, wst_seconds=wst_seconds,
                   reference_seconds=reference_seconds, reference_max_abs_error=error,
                   vmd_features=len(Xv), wst_features=len(Xw))
    return dict(vmd=Xv, wst=Xw, vmd_names=np.asarray(vnames),
                wst_names=np.asarray(wnames)), details


def fit_preprocessor(X, y, patients, partitions, protocol):
    """Fit mRMR and scaling on fit patients only; transform both partitions.

    This boundary is shared by the quantum and matched classical classifiers.
    """
    X, y = np.asarray(X), np.asarray(y)
    patients, partitions = np.asarray(patients), np.asarray(partitions)
    if X.ndim != 2 or not (len(X) == len(y) == len(patients) == len(partitions)):
        raise ValueError('Features and row metadata must align')
    if set(partitions) != {'fit', 'validation'} or not np.isfinite(X).all():
        raise ValueError('Need finite features and exactly fit/validation partitions')
    fit, val = np.flatnonzero(partitions == 'fit'), np.flatnonzero(partitions == 'validation')
    if set(patients[fit]) & set(patients[val]):
        raise ValueError('A patient crosses the preprocessing boundary')
    if set(y[fit]) != {0, 1} or set(y[val]) != {0, 1}:
        raise ValueError('Both diagnosis classes required in each partition')
    pipeline = make_pipeline(MRMRSelector(k=protocol['preprocessing']['k']), TanhAngleScaler())
    pipeline.fit(X[fit], y[fit])
    transformed = pipeline.transform(X)
    if transformed.shape[1] != protocol['preprocessing']['k'] or not np.isfinite(transformed).all():
        raise ValueError('Feature selection did not produce the declared finite feature budget')
    return pipeline, transformed, fit, val


def make_classifiers(protocol):
    """Unfitted, fixed-budget classifiers; no validation search or best-seed choice."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from ecgvmd.reupload import ReuploadVQC

    settings = {k: v for k, v in protocol['vqc'].items() if k not in ('class', 'seeds')}
    return {
        **{f'vqc_seed{seed}': ReuploadVQC(seed=seed, **settings) for seed in protocol['vqc']['seeds']},
        'logistic': LogisticRegression(**protocol['controls']['logistic']),
        'weighted_knn': KNeighborsClassifier(**protocol['controls']['weighted_knn']),
    }
