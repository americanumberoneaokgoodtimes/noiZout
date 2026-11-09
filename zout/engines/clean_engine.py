from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..audio.io import prepare_clean_enrollment, resample
from ..audio.masking import N_FFT, _istft, _stft, apply_adaptive_mask

Progress = Callable[[str, float], None]
SR_TSE = 16000


class CleanVoiceEngine:
    """CLEAN reference: enroll the speaker with VoxLock, keep only matching voice in the main recording.

    Above 8 kHz (outside VoxLock's 16 kHz band) the original high band is retained, gated by the
    frame-wise speech-presence ratio measured in the extracted low band.
    """

    def __init__(self, precision: str = "int8", threads: int = 0):
        self.precision = precision
        self.threads = threads
        self._vl = None

    def _model(self):
        if self._vl is None:
            import voxlock
            self._vl = voxlock.VoxLock(precision=self.precision, threads=self.threads)
        return self._vl

    def run(self, ref: np.ndarray, ref_sr: int, main: np.ndarray, sr: int, strength: float,
            max_atten_db: float, progress: Progress | None = None,
            polish: bool = True) -> tuple[np.ndarray, dict]:
        p = progress or (lambda *_: None)
        info: dict = {}

        p("Conditioning clean reference", 0.0)
        ref16 = resample(ref, ref_sr, SR_TSE)
        enroll, voiced = prepare_clean_enrollment(ref16, SR_TSE)
        info["ref_voiced_sec"] = voiced
        if voiced < 0.5:
            info["warning"] = f"only {voiced:.2f}s of voice found in reference; results may be poor"

        p("Loading VoxLock", 0.1)
        vl = self._model()

        work = main
        if polish:
            # Broadband noise breaks target-speaker extraction, so denoise first.
            p("DeepFilterNet3 pre-denoise", 0.15)
            try:
                from .deepfilter import DeepFilter
                work = DeepFilter(100.0).process(main, sr)
            except Exception as e:  # polish is optional
                info["polish_error"] = str(e)

        p("Extracting matching voice", 0.4)
        main16 = resample(work, sr, SR_TSE)
        tse16 = np.asarray(vl.isolate(main16, sample=enroll), dtype=np.float32)

        p("Restoring full-band signal", 0.85)
        est = self._restore_fullband(work, tse16, sr)
        return apply_adaptive_mask(main, est, sr, strength, max_atten_db), info

    @staticmethod
    def _restore_fullband(main: np.ndarray, tse16: np.ndarray, sr: int) -> np.ndarray:
        if sr <= SR_TSE:
            y = resample(tse16, SR_TSE, sr)
            return y[: len(main)] if len(y) >= len(main) else np.pad(y, (0, len(main) - len(y)))
        lo = resample(tse16, SR_TSE, sr)
        lo = lo[: len(main)] if len(lo) >= len(main) else np.pad(lo, (0, len(main) - len(lo)))
        X, Y = _stft(main, sr), _stft(lo, sr)
        f = np.fft.rfftfreq(N_FFT, 1 / sr)
        band = f < 7000
        r = np.abs(Y[band]).sum(0) / (np.abs(X[band]).sum(0) + 1e-8)
        r = np.clip(uniform_filter1d(r, 5), 0, 1)
        hi = np.clip((f - 7000) / 1000, 0, 1)[:, None]  # crossfade 7-8 kHz
        out = Y * (1 - hi) + X * r[None, :] * hi
        return _istft(out, sr, len(main))
