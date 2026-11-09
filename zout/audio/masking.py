from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

N_FFT = 2048
HOP = 512


def _stft(x: np.ndarray, sr: int):
    _, _, Z = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros", padded=True)
    return Z


def _istft(Z: np.ndarray, sr: int, length: int) -> np.ndarray:
    _, y = istft(Z, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)
    y = y[:length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y.astype(np.float32)


def min_gain(strength: float, max_atten_db: float) -> float:
    cap = 0.0 if max_atten_db >= 60 else 10 ** (-max_atten_db / 20)
    return float(max(cap, (1 - strength) ** 2 * 0.35))


def apply_adaptive_mask(x: np.ndarray, y_est: np.ndarray, sr: int, strength: float,
                        max_atten_db: float, **_ignored) -> np.ndarray:
    """Blend a bounded fraction of the removed component back into the engine estimate.

    out = y_est + g * (x - y_est), g = min_gain(strength, max_atten_db).
    The estimate is preserved exactly (no re-masking artifacts); strength and the attenuation cap
    only set how much of the removed signal leaks back, i.e. the floor of the suppression.
    """
    n = min(len(x), len(y_est))
    x, y_est = x[:n], y_est[:n]
    g = min_gain(strength, max_atten_db)
    return (y_est + g * (x - y_est)).astype(np.float32)


def noise_suppression_db(x_in: np.ndarray, y_out: np.ndarray) -> float:
    """Reduction in 10th-percentile frame energy (noise-floor proxy), in dB (positive = quieter)."""
    def floor(a):
        h = 1024
        m = len(a) // h
        if m == 0:
            return -100.0
        e = 10 * np.log10(np.mean(a[: m * h].reshape(m, h) ** 2, axis=1) + 1e-12)
        return float(np.percentile(e, 10))
    return floor(x_in) - floor(y_out)
