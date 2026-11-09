from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..audio.masking import _istft, _stft, apply_adaptive_mask

Progress = Callable[[str, float], None]


def kl_nmf(V: np.ndarray, W: np.ndarray, H: np.ndarray, n_iter: int, update_w: np.ndarray | None = None):
    """Multiplicative-update KL-NMF. `update_w` is a bool mask over columns of W allowed to change."""
    eps = 1e-9
    V = V.astype(np.float32) + eps
    for _ in range(n_iter):
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.sum(0)[:, None] + eps)
        WH = W @ H + eps
        num = (V / WH) @ H.T
        den = H.sum(1)[None, :] + eps
        upd = W * (num / den)
        if update_w is None:
            W = upd
        else:
            W = np.where(update_w[None, :], upd, W)
    return W, H


class NoisyAcousticEngine:
    """NOISY reference: learn its noise profile, remove it from the main recording.

    Stage 1: noisereduce spectral gating with the reference as y_noise.
    Stage 2: supervised KL-NMF with W_noise trained on the reference (frozen) + free signal bases.
    Stage 3: adaptive mask shaping against the original (strength / max attenuation).
    """

    K_NOISE = 24
    K_SIG = 48

    def run(self, ref: np.ndarray, main: np.ndarray, sr: int, strength: float,
            max_atten_db: float, progress: Progress | None = None,
            polish: bool = False) -> np.ndarray:
        import noisereduce as nr
        rng = np.random.default_rng(0)
        p = progress or (lambda *_: None)

        p("Training noise profile (KL-NMF)", 0.0)
        R = np.abs(_stft(ref, sr)).astype(np.float32)
        if R.shape[1] > 800:  # cap training frames
            R = R[:, :: R.shape[1] // 800 + 1]
        Wn = rng.random((R.shape[0], self.K_NOISE), dtype=np.float32) + 0.1
        Hn = rng.random((self.K_NOISE, R.shape[1]), dtype=np.float32) + 0.1
        Wn, _ = kl_nmf(R, Wn, Hn, 80)
        Wn /= Wn.sum(0, keepdims=True) + 1e-9

        p("Spectral gate with reference profile", 0.25)
        thresh = float(2.6 - 1.9 * strength)
        prop = float(min(1.0, 0.25 + 0.75 * strength))
        gated = nr.reduce_noise(y=main, sr=sr, y_noise=ref, stationary=True,
                                prop_decrease=prop, n_std_thresh_stationary=thresh,
                                freq_mask_smooth_hz=350, n_fft=1024).astype(np.float32)

        p("Separating noise bases from recording", 0.5)
        Z = _stft(gated, sr)
        V = np.abs(Z).astype(np.float32)
        F, T = V.shape
        Ws = rng.random((F, self.K_SIG), dtype=np.float32) + 0.1
        W = np.concatenate([Wn, Ws], axis=1)
        H = rng.random((W.shape[1], T), dtype=np.float32) + 0.1
        free = np.concatenate([np.zeros(self.K_NOISE, bool), np.ones(self.K_SIG, bool)])
        W, H = kl_nmf(V, W, H, 50, update_w=free)
        Vn = W[:, : self.K_NOISE] @ H[: self.K_NOISE]
        Vs = W[:, self.K_NOISE:] @ H[self.K_NOISE:]
        beta = 0.2 + 1.8 * strength
        pw = 1.0 + strength
        mask = Vs ** pw / (Vs ** pw + (beta * Vn) ** pw + 1e-9)
        est = _istft(Z * mask, sr, len(main))

        if polish:
            p("DeepFilterNet3 polish", 0.8)
            try:
                from .deepfilter import DeepFilter
                est = DeepFilter(100.0 if max_atten_db >= 60 else max_atten_db).process(est, sr)
            except Exception:
                pass
        p("Shaping mask", 0.9)
        return apply_adaptive_mask(main, est, sr, strength, max_atten_db)
