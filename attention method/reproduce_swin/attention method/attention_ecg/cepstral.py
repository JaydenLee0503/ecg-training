"""Framewise LFCC/MFCC; input (samples, leads), output (frames, leads, cepstra)."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window


@dataclass(frozen=True)
class CepstralConfig:
    fs: int = 500
    frame_samples: int = 500
    hop_samples: int = 125
    n_fft: int = 512
    n_filters: int = 40
    n_ceps: int = 20
    fmin: float = 0.0
    fmax: float = 100.0
    spacing: str = 'linear'
    power_floor: float = 1e-12

    def __post_init__(self):
        integers = (self.fs, self.frame_samples, self.hop_samples, self.n_fft,
                    self.n_filters, self.n_ceps)
        if any(type(x) is not int or x <= 0 for x in integers):
            raise ValueError('Sizes and sampling rate must be positive integers')
        if not (2 <= self.frame_samples <= self.n_fft and
                self.hop_samples <= self.frame_samples and self.n_ceps <= self.n_filters):
            raise ValueError('Invalid frame, FFT, hop, or cepstral dimensions')
        if not (0 <= self.fmin < self.fmax <= self.fs / 2):
            raise ValueError('Invalid frequency limits')
        if self.spacing not in ('linear', 'mel'):
            raise ValueError('spacing must be linear or mel')
        if not np.isfinite(self.power_floor) or self.power_floor <= 0:
            raise ValueError('power_floor must be finite and positive')

    def to_dict(self):
        return asdict(self)


def filterbank(config):
    """Unit-sum triangular bands; no silent acceptance of empty FFT bands."""
    c = config
    if c.spacing == 'linear':
        edges = np.linspace(c.fmin, c.fmax, c.n_filters + 2)
    else:
        mel = lambda f: 2595 * np.log10(1 + f / 700)
        edges = 700 * (10 ** (np.linspace(mel(c.fmin), mel(c.fmax),
                                         c.n_filters + 2) / 2595) - 1)
    f = np.fft.rfftfreq(c.n_fft, 1 / c.fs)
    left = (f[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    right = (edges[2:, None] - f[None, :]) / np.diff(edges)[1:, None]
    bank = np.maximum(0, np.minimum(left, right))
    mass = bank.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError('Empty triangular filter: increase n_fft or reduce n_filters')
    return bank / mass


def cepstral_features(signal, config=CepstralConfig()):
    """No resampling, pre-emphasis, detrending, or amplitude normalization.

    Right-reflect-pad only if needed to cover the final sample. A periodic Hann
    window precedes a one-sided power spectral density. Apply triangular bands,
    natural log with a fixed floor, and orthonormal DCT-II, retaining c0.
    Returns float32 coefficients and explicit frame/padding metadata.
    """
    x = np.asarray(signal, dtype=np.float64)
    c = config
    if x.ndim != 2 or x.shape[0] < c.frame_samples or x.shape[1] < 1:
        raise ValueError('Expected (samples >= frame_samples, leads >= 1)')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite ECG input')
    n = x.shape[0]
    pad = (-(n - c.frame_samples)) % c.hop_samples
    if pad:
        x = np.pad(x, ((0, pad), (0, 0)), mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(x, c.frame_samples, axis=0)
    frames = frames[::c.hop_samples]  # time, lead, sample
    window = get_window('hann', c.frame_samples, fftbins=True)
    spectrum = np.fft.rfft(frames * window, n=c.n_fft, axis=-1)
    power = np.abs(spectrum) ** 2 / (c.fs * np.sum(window ** 2))
    power[..., 1:-1 if c.n_fft % 2 == 0 else None] *= 2
    energies = power @ filterbank(c).T
    coeffs = dct(np.log(np.maximum(energies, c.power_floor)),
                 type=2, norm='ortho', axis=-1)[..., :c.n_ceps]
    if not np.isfinite(coeffs).all():
        raise ValueError('Nonfinite cepstral features')
    starts = np.arange(len(frames)) * c.hop_samples
    return coeffs.astype(np.float32), {
        'config': c.to_dict(), 'input_samples': n, 'right_pad_samples': int(pad),
        'frame_start_samples': starts.tolist(),
        'frame_center_seconds': ((starts + (c.frame_samples - 1) / 2) / c.fs).tolist(),
        'axes': ['time', 'lead', 'cepstrum'],
    }
