from __future__ import annotations

import ctypes as C

import numpy as np

from ..audio.io import resample

DF_SR = 48000


class DeepFilter:
    """ctypes wrapper over the DeepFilterNet3 C runtime shipped in `deepfilternet-slim`."""

    def __init__(self, atten_lim_db: float = 100.0):
        from deepfilternet_slim import runtime_assets
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
        self._path = str(a.model_path).encode()
        self.atten = float(atten_lim_db)

    def _run(self, x48: np.ndarray) -> np.ndarray:
        st = self._lib.df_create(self._path, C.c_float(self.atten), b"error")
        if not st:
            raise RuntimeError("DeepFilterNet failed to initialise")
        try:
            hop = int(self._lib.df_get_frame_length(st))
            n = len(x48)
            pad = (-n) % hop
            x = np.ascontiguousarray(np.concatenate([x48, np.zeros(pad + 4 * hop, np.float32)]), np.float32)
            y = np.zeros_like(x)
            fp = C.POINTER(C.c_float)
            for i in range(0, len(x), hop):
                self._lib.df_process_frame(st, x[i:i + hop].ctypes.data_as(fp), y[i:i + hop].ctypes.data_as(fp))
            return y
        finally:
            self._lib.df_free(st)

    @staticmethod
    def _delay(x: np.ndarray, y: np.ndarray, max_lag: int = 4000) -> int:
        n = min(len(x), 10 * DF_SR)
        a, b = x[:n], y[:n + max_lag]
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

    def process(self, x: np.ndarray, sr: int) -> np.ndarray:
        x48 = resample(x, sr, DF_SR)
        y = self._run(x48.astype(np.float32))
        d = self._delay(x48, y)
        y = y[d:d + len(x48)]
        if len(y) < len(x48):
            y = np.pad(y, (0, len(x48) - len(y)))
        return resample(y, DF_SR, sr)[: len(x)]
