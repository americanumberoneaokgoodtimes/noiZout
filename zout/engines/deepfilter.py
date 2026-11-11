from __future__ import annotations

"""DeepFilterNet speech-enhancement polish stage.

Backend priority (first available wins):
  1. deepfilternet_slim  — C runtime shipped by the deepfilternet-slim private package.
                           Highest quality; runs entirely on-device via libDF ctypes.
  2. deepfilter-stream   — ONNX graph packaged as a public PyPI wheel (pip install deepfilter-stream).
                           Same DeepFilterNet3 weights, slightly different latency handling; CPU-only.

If neither backend is installed every caller in noisy_engine / clean_engine catches ImportError and
silently skips the polish pass, so the pipeline still produces valid output.

Install instructions:
  - Public (recommended): pip install deepfilter-stream
  - Private C runtime:    see project docs for deepfilternet-slim distribution
"""

import ctypes as C

import numpy as np

from ..audio.io import resample

DF_SR = 48000

_INSTALL_HINT = (
    "No DeepFilterNet backend found.\n"
    "  Public (recommended):  pip install deepfilter-stream\n"
    "  Private C runtime:     see project docs for deepfilternet-slim\n"
)


# ---------------------------------------------------------------------------
# Backend 1: deepfilternet_slim (original ctypes / C DLL wrapper)
# ---------------------------------------------------------------------------

class _SlimBackend:
    """ctypes wrapper over the DeepFilterNet3 C runtime from deepfilternet-slim."""

    def __init__(self, atten_lim_db: float):
        from deepfilternet_slim import runtime_assets  # noqa: PLC0415
        a = runtime_assets()
        self._lib = C.CDLL(str(a.library_path))
        L = self._lib
        L.df_create.restype = C.c_void_p
        L.df_create.argtypes = [C.c_char_p, C.c_float, C.c_char_p]
        L.df_get_frame_length.restype = C.c_size_t
        L.df_get_frame_length.argtypes = [C.c_void_p]
        L.df_set_atten_lim.argtypes = [C.c_void_p, C.c_float]
        L.df_process_frame.restype = C.c_float
        L.df_process_frame.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_float)]
        L.df_free.argtypes = [C.c_void_p]
        self._model_path = str(a.model_path).encode()
        self.atten = float(atten_lim_db)

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        x48 = resample(x, sr, DF_SR)
        st = self._lib.df_create(self._model_path, C.c_float(self.atten), b"error")
        if not st:
            raise RuntimeError("deepfilternet_slim: failed to initialise model")
        try:
            hop = int(self._lib.df_get_frame_length(st))
            n = len(x48)
            pad = (-n) % hop
            buf = np.ascontiguousarray(
                np.concatenate([x48, np.zeros(pad + 4 * hop, np.float32)]), np.float32
            )
            out = np.zeros_like(buf)
            fp = C.POINTER(C.c_float)
            for i in range(0, len(buf), hop):
                self._lib.df_process_frame(
                    st, buf[i:i + hop].ctypes.data_as(fp), out[i:i + hop].ctypes.data_as(fp)
                )
        finally:
            self._lib.df_free(st)

        # align delay
        lag = _cross_correlate_lag(x48, out)
        y48 = out[lag:lag + len(x48)]
        if len(y48) < len(x48):
            y48 = np.pad(y48, (0, len(x48) - len(y48)))
        return resample(y48, DF_SR, sr)[: len(x)]


# ---------------------------------------------------------------------------
# Backend 2: deepfilter-stream (public ONNX wheel)
# ---------------------------------------------------------------------------

class _StreamBackend:
    """ONNX DeepFilterNet3 via the public deepfilter-stream package."""

    def __init__(self, atten_lim_db: float):
        from deepfilter_stream import Denoiser  # noqa: PLC0415
        self._denoiser = Denoiser(atten_lim_db=float(atten_lim_db))

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        # Denoiser.process() resamples internally to 48 kHz and returns
        # slightly fewer samples (latency buffer). We pad/trim to match input.
        y = self._denoiser.process(x.astype(np.float32), sr=sr)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        return y[: len(x)]


# ---------------------------------------------------------------------------
# Public class: tries _SlimBackend first, then _StreamBackend
# ---------------------------------------------------------------------------

class DeepFilter:
    """DeepFilterNet3 speech-enhancement polish pass.

    Automatically selects the best available backend:
      1. deepfilternet_slim (private C runtime — if installed)
      2. deepfilter-stream  (public ONNX wheel — pip install deepfilter-stream)

    Raises ImportError with install instructions if neither is available.
    """

    def __init__(self, atten_lim_db: float = 100.0):
        self._backend: _SlimBackend | _StreamBackend | None = None
        self._atten = float(atten_lim_db)

    def _get_backend(self) -> _SlimBackend | _StreamBackend:
        if self._backend is not None:
            return self._backend
        try:
            self._backend = _SlimBackend(self._atten)
            return self._backend
        except (ImportError, OSError, Exception):
            pass
        try:
            self._backend = _StreamBackend(self._atten)
            return self._backend
        except ImportError:
            raise ImportError(_INSTALL_HINT) from None

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        return self._get_backend().process(x, sr)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _cross_correlate_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 4000) -> int:
    """Estimate the sample delay between x and y via coarse+fine cross-correlation."""
    n = min(len(x), 10 * DF_SR)
    a = x[:n]
    b = y[:n + max_lag]
    best, lag = -1.0, 0
    for d in range(0, max_lag, 8):
        seg = b[d:d + n]
        m = min(len(seg), len(a))
        c = abs(float(np.dot(a[:m], seg[:m])))
        if c > best:
            best, lag = c, d
    for d in range(max(0, lag - 8), lag + 9):
        seg = b[d:d + n]
        m = min(len(seg), len(a))
        c = abs(float(np.dot(a[:m], seg[:m])))
        if c > best:
            best, lag = c, d
    return lag
