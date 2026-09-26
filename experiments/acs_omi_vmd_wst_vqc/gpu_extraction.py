"""Production orchestration around the unchanged, validated FP64 GPU solver.

GPU iteration batches contain up to 16 ECGs (192 leads). Each lead retries from
initialization. Attempt callbacks run before a retry or failure. WST and VMD
descriptors use the frozen CPU functions; results are yielded one ECG at a time.
"""
import hashlib
import time

import numpy as np

from .gpu_vmd import vmd_gpu_batch
from .loader import LEADS, POLICY_VERSION
from ecgvmd.features import mode_features
from ecgvmd.scatter import scatter_batch, scatter_features, scatter_kymatio


def extract_records(records, protocol, *, on_attempt, reference_ids=()):
    expected = protocol['input']
    if not records or len(records) > 16 or len({r.info.record_id for r in records}) != len(records):
        raise ValueError('Need 1–16 unique ECG records')
    for record in records:
        if (record.info.split != 'train' or record.info.label not in (0, 1)
                or record.fs != expected['fs'] or tuple(record.leads) != tuple(expected['leads'])
                or tuple(record.leads) != LEADS or expected['record_policy'] != POLICY_VERSION
                or not record.decision.eligible or record.signal_mV is None
                or record.signal_mV.shape != (expected['samples'], len(LEADS))):
            raise ValueError('Record does not match the frozen labeled development input')
    X = np.vstack([r.signal_mV.T for r in records]).astype(np.float64, copy=False)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite ECG input')
    started = time.perf_counter()
    v = protocol['vmd']
    kwargs = {k: v[k] for k in ('K', 'alpha', 'tau', 'dc', 'init', 'tol')}
    kwargs['fs'] = expected['fs']
    modes = np.empty((len(X), v['K'], X.shape[1]), dtype=np.float64)
    omega = np.empty((len(X), v['K']), dtype=np.float64)
    iterations, limits = np.zeros(len(X), int), np.zeros(len(X), int)
    attempts = [[] for _ in records]
    remaining = np.arange(len(X))
    for limit in v['iteration_limits']:
        result = vmd_gpu_batch(X[remaining], max_iter=limit, **kwargs)
        for index, count, capped in zip(remaining, result.iters, result.capped):
            i, lead = divmod(int(index), len(LEADS))
            attempt = dict(lead=LEADS[lead], limit=limit, iterations=int(count), capped=bool(capped))
            attempts[i].append(attempt)
            on_attempt(records[i].info.record_id, attempt)
        if not np.isfinite(result.modes).all() or not np.isfinite(result.omega_hz).all():
            raise FloatingPointError('Nonfinite VMD output; extraction stopped')
        modes[remaining], omega[remaining] = result.modes, result.omega_hz
        iterations[remaining], limits[remaining] = result.iters, limit
        remaining = remaining[result.capped]
        if not len(remaining):
            break
    if len(remaining):
        failed = [(records[int(i)//12].info.record_id, LEADS[int(i)%12]) for i in remaining]
        raise RuntimeError(f'Unconverged leads after {limit}: {failed}')
    pieces = []
    for start in range(0, len(X), 4):
        features, names = mode_features(modes[start:start+4], omega[start:start+4],
                                        kwargs['fs'], x=X[start:start+4])
        if not np.isfinite(features).all():
            raise FloatingPointError('Nonfinite VMD descriptors')
        pieces.append(features)
    features = np.vstack(pieces)
    vnames = np.asarray([f'{lead}:{name}' for lead in LEADS for name in names])
    vmd_batch_seconds = time.perf_counter() - started
    w = protocol['wst']
    kw = {k: w[k] for k in ('J', 'T', 'max_order')}
    kw.update(Q=tuple(w['Q']), fs=expected['fs'])
    for i, record in enumerate(records):
        start = time.perf_counter()
        selection = slice(i*12, (i+1)*12)
        signals = X[selection]
        scattering = scatter_batch(signals, **kw)
        if (not np.isfinite(scattering.coeffs).all()
                or np.any(scattering.coeffs[:, scattering.order > 0] < 0)):
            raise FloatingPointError('Invalid WST coefficients')
        wf, names = scatter_features(scattering, log=w['log'], reduce=w['reduce'], eps=w['eps'])
        if not np.isfinite(wf).all():
            raise FloatingPointError('Nonfinite WST features')
        wst_seconds = time.perf_counter() - start
        error, reference_seconds = None, None
        if record.info.record_id in reference_ids:
            start = time.perf_counter()
            ref = scatter_kymatio(signals, **kw)
            np.testing.assert_array_equal(scattering.order, ref.order)
            np.testing.assert_allclose(scattering.xi, ref.xi, atol=1e-14, rtol=0, equal_nan=True)
            error = float(np.max(np.abs(scattering.coeffs - ref.coeffs)))
            cfg = protocol['engineering_check']
            np.testing.assert_allclose(scattering.coeffs, ref.coeffs,
                                       atol=cfg['reference_atol'], rtol=cfg['reference_rtol'])
            reference_seconds = time.perf_counter() - start
        vectors = dict(vmd=features[selection].reshape(-1), wst=wf.reshape(-1), vmd_names=vnames,
                       wst_names=np.asarray([f'{lead}:p{j}:{name}' for lead in LEADS
                                             for j, name in enumerate(names)]))
        details = dict(record_id=record.info.record_id, patient_id=record.info.patient_id,
                       label=record.info.label, waveform_sha256=record.decision.waveform_sha256,
                       physical_input_sha256=hashlib.sha256(np.ascontiguousarray(signals).tobytes()).hexdigest(),
                       vmd_attempts=attempts[i], vmd_iterations=iterations[selection].tolist(),
                       vmd_limits=limits[selection].tolist(), vmd_batch_seconds=vmd_batch_seconds,
                       vmd_batch_records=len(records), wst_seconds=wst_seconds,
                       reference_seconds=reference_seconds, reference_max_abs_error=error,
                       vmd_features=len(vectors['vmd']), wst_features=len(vectors['wst']),
                       execution_origin='gpu_fp64_vmd_cpu_wst')
        yield vectors, details


def compare_cpu(vectors, details, old_vectors, old_details):
    """Allow benchmark-declared rounding tolerance; require exact diagnostics/WST."""
    np.testing.assert_allclose(vectors['vmd'], old_vectors['vmd'], atol=1e-6, rtol=1e-5)
    for key in ('wst', 'vmd_names', 'wst_names'):
        np.testing.assert_array_equal(vectors[key], old_vectors[key])
    for key in ('physical_input_sha256', 'vmd_iterations', 'vmd_limits'):
        if details[key] != old_details[key]:
            raise ValueError(f'CPU/GPU diagnostic mismatch: {key}')
    key = lambda a: (a['lead'], a['limit'], a['iterations'], a['capped'])
    if sorted(details['vmd_attempts'], key=key) != sorted(old_details['vmd_attempts'], key=key):
        raise ValueError('CPU/GPU retry attempt mismatch')
    return dict(passed=True, vmd_exact=bool(np.array_equal(vectors['vmd'], old_vectors['vmd'])),
                vmd_max_abs_error=float(np.max(np.abs(vectors['vmd']-old_vectors['vmd']))),
                wst_exact=True, input_and_convergence_equal=True)
