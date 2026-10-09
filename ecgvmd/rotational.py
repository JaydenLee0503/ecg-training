"""Declared EMD/Hilbert rotational descriptors; no data or classifier fitting.

This is an independent adaptation, not a verified reproduction of Alavi et al.
PyEMD's solver is used unchanged; subclass hooks only collect diagnostics.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.signal import hilbert

NAMES = tuple([f'f{i}_hz' for i in range(1, 5)] +
              [f'env{i}_mV' for i in range(1, 5)] +
              ['global_frequency_hz', 'global_envelope_mV', 'centroid_x_mV', 'centroid_y_mV'])


class RotationalError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def arc_centroid(z):
    """Exact arc-length centroid of an OPEN piecewise-linear complex path.

    Equivalent to the limit of uniform arc-length interpolation/averaging.
    Repeated vertices have zero weight. No artificial end-to-start segment.
    """
    z = np.asarray(z, dtype=np.complex128)
    if z.ndim != 1 or len(z) < 3 or not np.isfinite(z).all():
        raise RotationalError('Invalid complex trajectory')
    lengths = np.abs(np.diff(z))
    total = float(lengths.sum())
    if not np.isfinite(total) or total <= 0:
        raise RotationalError('Degenerate complex trajectory')
    centroid = np.sum((z[1:] + z[:-1]) * .5 * (lengths / total))
    return complex(centroid), total


def analytic_descriptors(modes, fs, settings):
    modes = np.asarray(modes, dtype=np.float64)
    if (modes.ndim != 2 or modes.shape[0] != 4 or not np.isfinite(modes).all()
            or not np.isfinite(fs) or fs <= 0):
        raise RotationalError('Need four finite components and positive sampling rate')
    trim = int(round(fs * settings['trim_seconds']))
    if trim < 1 or modes.shape[1] - 2 * trim < 3:
        raise RotationalError('Insufficient samples after endpoint trimming')
    z = hilbert(modes, axis=-1)
    envelope = np.abs(z)
    phase = np.unwrap(np.angle(z), axis=-1)
    frequency = np.gradient(phase, 1. / fs, axis=-1, edge_order=2) / (2 * np.pi)
    z, envelope, frequency = (a[:, trim:-trim] for a in (z, envelope, frequency))
    floor = np.maximum(settings['envelope_floor_mV'],
                       settings['envelope_floor_relative'] * envelope.max(1))
    valid = envelope > floor[:, None]
    fraction = valid.mean(1)
    diagnostics = dict(trim_samples_each_end=trim, retained_samples=z.shape[1],
                       valid_phase_fraction=fraction.tolist(), envelope_floor_mV=floor.tolist())
    if np.any(fraction < settings['minimum_valid_phase_fraction']):
        raise RotationalError('Insufficient nonzero-amplitude phase support', diagnostics)
    component_frequency = np.array([f[v].mean() for f, v in zip(frequency, valid)])
    energy = np.where(valid, envelope ** 2, 0.)
    denominator = energy.sum(0)
    global_valid = denominator > 0
    if not np.all(global_valid):
        raise RotationalError('Undefined global frequency', diagnostics)
    global_frequency = np.sum(energy * frequency, axis=0) / denominator
    global_envelope = np.sqrt(np.sum(envelope ** 2, axis=0))
    trajectory = z.sum(0)
    centroid, length = arc_centroid(trajectory)
    features = np.r_[component_frequency, envelope.mean(1), global_frequency.mean(),
                     global_envelope.mean(), centroid.real, centroid.imag]
    if features.shape != (12,) or not np.isfinite(features).all():
        raise RotationalError('Nonfinite descriptor', diagnostics)
    diagnostics.update(negative_frequency_fraction=[float((f[v] < 0).mean())
                        for f, v in zip(frequency, valid)],
                       frequency_quantiles_hz=[np.quantile(f[v], [.01, .5, .99]).tolist()
                        for f, v in zip(frequency, valid)],
                       arc_length_mV=length, time_centroid_mV=[float(trajectory.real.mean()),
                                                            float(trajectory.imag.mean())])
    return features, diagnostics, trajectory


def decompose(signal, settings):
    from PyEMD import EMD, __version__
    if __version__ != settings['version']:
        raise RotationalError(f'Expected PyEMD {settings["version"]}, got {__version__}')
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 4 or not np.isfinite(signal).all() or np.ptp(signal) == 0:
        raise RotationalError('Signal must be finite, nonconstant and one-dimensional')

    class ObservedEMD(EMD):
        def __init__(self):
            super().__init__(**settings['parameters'])
            self.sifts = 0
            self.capped = False
            self.trace = []
            self.logger = self

        def debug(self, message, *args):
            pass

        def info(self, message, *args):
            if message.startswith('Max iterations reached'):
                self.capped = True

        def extract_max_min_spline(self, *args):
            self.sifts += 1
            return super().extract_max_min_spline(*args)

        def end_condition(self, S, IMF):
            ended = super().end_condition(S, IMF)
            self.trace.append(dict(candidate=len(IMF), sifts=self.sifts,
                capped=self.capped, decomposition_end_condition=bool(ended),
                reason='iteration_cap' if self.capped else ('trend' if self.sifts == 0 else 'imf_criterion')))
            self.sifts, self.capped = 0, False
            return ended

    started = time.perf_counter()
    solver = ObservedEMD()
    try:
        solver.emd(signal, max_imf=settings['components'])
    except Exception as exc:
        raise RotationalError(f'EMD solver error: {exc}', dict(sifting=solver.trace,
            unfinished_sifts=solver.sifts, seconds=time.perf_counter()-started)) from exc
    modes, residual = solver.get_imfs_and_residue()
    details = dict(seconds=time.perf_counter() - started, components=len(modes),
                   sifting=solver.trace)
    if any(item['capped'] for item in solver.trace):
        raise RotationalError('EMD reached the declared iteration cap', details)
    if len(modes) != settings['components']:
        raise RotationalError('Fewer than four intrinsic mode functions; residual is not an IMF', details)
    if not np.isfinite(modes).all() or not np.isfinite(residual).all():
        raise RotationalError('Nonfinite decomposition', details)
    error = float(np.max(np.abs(signal - modes.sum(0) - residual)))
    details['reconstruction_max_abs_mV'] = error
    if error > settings['reconstruction_atol_mV'] + settings['reconstruction_rtol'] * np.max(np.abs(signal)):
        raise RotationalError('Decomposition reconstruction failed', details)
    return modes, residual, details


def extract(signal, fs, spec, *, retain=False):
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(fs) or fs <= 0 or len(signal) < round(fs * spec['minimum_seconds']):
        raise RotationalError('Signal shorter than declared minimum duration')
    started = time.perf_counter()
    modes, residual, details = decompose(signal, spec['emd'])
    try:
        values, analytic, trajectory = analytic_descriptors(modes, fs, spec['analytic'])
    except RotationalError as exc:
        exc.diagnostics = dict(decomposition=details, analytic=exc.diagnostics)
        raise
    details.update(analytic=analytic, total_seconds=time.perf_counter() - started)
    return values, details, (dict(modes=modes, residual=residual, trajectory=trajectory) if retain else None)
