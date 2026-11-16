from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..audio.io import prepare_clean_enrollment, resample
from ..audio.masking import N_FFT, _istft, _stft, apply_adaptive_mask
from .noisy_engine import MAX_SAMPLES_PER_CHUNK, _stream_disk_to_disk

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
            try:
                import voxlock
            except ImportError:
                raise ImportError(
                    "CLEAN mode requires the voxlock package — see project docs for installation instructions."
                ) from None
            self._vl = voxlock.VoxLock(precision=self.precision, threads=self.threads)
        return self._vl

    def run(self, ref: np.ndarray, ref_sr: int, main: np.ndarray, sr: int, strength: float,
            max_atten_db: float, progress: Progress | None = None,
            polish: bool = True,
            cancel_event: threading.Event | None = None) -> tuple[np.ndarray, dict]:
        p = progress or (lambda *_: None)
        info: dict = {}

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        p("Conditioning clean reference", 0.0)
        ref16 = resample(ref, ref_sr, SR_TSE)
        enroll, voiced = prepare_clean_enrollment(ref16, SR_TSE)
        del ref16
        info["ref_voiced_sec"] = voiced
        if voiced < 0.5:
            info["warning"] = f"only {voiced:.2f}s of voice found in reference; results may be poor"

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        p("Loading VoxLock", 0.1)
        vl = self._model()

        # Instantiate DeepFilter once — avoids re-loading ONNX model per call
        _polish_model: object = None
        if polish:
            try:
                from .deepfilter import DeepFilter
                _polish_model = DeepFilter(100.0)
            except Exception as e:
                info["polish_error"] = str(e)

        def _process_chunk(chunk: np.ndarray) -> np.ndarray:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")
            work = chunk
            if _polish_model is not None:
                p("DeepFilterNet3 pre-denoise", 0.15)
                try:
                    work = _polish_model.process(chunk, sr)
                except Exception as e:  # polish is optional
                    info["polish_error"] = str(e)

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            p("Extracting matching voice", 0.4)
            chunk16 = work if sr == SR_TSE else resample(work, sr, SR_TSE)
            tse16 = np.asarray(vl.isolate(chunk16, sample=enroll), dtype=np.float32)
            del chunk16

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            p("Restoring full-band signal", 0.85)
            est = self._restore_fullband(work, tse16, sr)
            del tse16
            if work is not chunk:
                del work
            res = apply_adaptive_mask(chunk, est, sr, strength, max_atten_db)
            del est
            return res

        return _process_chunk(main), info

    def run_streaming(
        self,
        ref: np.ndarray,
        ref_sr: int,
        main_path: Path | str,
        out_path: Path | str,
        sr: int,
        strength: float,
        max_atten_db: float,
        chunk_sec: float = 60.0,
        overlap_sec: float = 2.0,
        progress: Progress | None = None,
        polish: bool = True,
        start_sec: float | None = None,
        end_sec: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, dict]:
        p = progress or (lambda *_: None)
        info: dict = {}

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        p("Conditioning clean reference", 0.0)
        ref16 = resample(ref, ref_sr, SR_TSE)
        enroll, voiced = prepare_clean_enrollment(ref16, SR_TSE)
        info["ref_voiced_sec"] = voiced
        if voiced < 0.5:
            info["warning"] = f"only {voiced:.2f}s of voice found in reference; results may be poor"

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        p("Loading VoxLock", 0.1)
        vl = self._model()

        # Instantiate DeepFilter once — avoids re-loading ONNX model per chunk
        _polish_model: object = None
        if polish:
            try:
                from .deepfilter import DeepFilter
                _polish_model = DeepFilter(100.0)
            except Exception as e:
                info["polish_error"] = str(e)

        p("Speaker model ready — processing chunks", 0.12)

        def process_chunk(chunk: np.ndarray) -> np.ndarray:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")
            work = chunk
            if _polish_model is not None:
                try:
                    work = _polish_model.process(chunk, sr)
                except Exception as e:
                    info["polish_error"] = str(e)

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            chunk16 = work if sr == SR_TSE else resample(work, sr, SR_TSE)
            tse16 = np.asarray(vl.isolate(chunk16, sample=enroll), dtype=np.float32)
            est = self._restore_fullband(work, tse16, sr)
            return apply_adaptive_mask(chunk, est, sr, strength, max_atten_db)

        def _chunk_progress(msg: str, f: float) -> None:
            p(msg, 0.12 + f * 0.88)

        stats = _stream_disk_to_disk(
            main_path=main_path,
            out_path=out_path,
            sr=sr,
            process_chunk=process_chunk,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            start_sec=start_sec,
            end_sec=end_sec,
            progress=_chunk_progress,
            cancel_event=cancel_event,
        )
        info.update(stats)
        return Path(out_path), info


    @staticmethod
    def _restore_fullband(main: np.ndarray, tse16: np.ndarray, sr: int) -> np.ndarray:
        if sr <= SR_TSE:
            y = resample(tse16, SR_TSE, sr)
            return y[: len(main)] if len(y) >= len(main) else np.pad(y, (0, len(main) - len(y)))
        lo = resample(tse16, SR_TSE, sr)
        lo = lo[: len(main)] if len(lo) >= len(main) else np.pad(lo, (0, len(main) - len(lo)))
        X = _stft(main, sr)
        Y = _stft(lo, sr)
        del lo
        f = np.fft.rfftfreq(N_FFT, 1 / sr)
        band = f < 7000
        denom = np.abs(X[band]).sum(0) + 1e-8
        r = np.abs(Y[band]).sum(0) / denom
        del denom
        r = np.clip(uniform_filter1d(r, 5), 0, 1)
        hi = np.clip((f - 7000) / 1000, 0, 1)[:, None]  # crossfade 7-8 kHz
        out = Y * (1 - hi) + X * r[None, :] * hi
        del X, Y, r, hi
        y = _istft(out, sr, len(main))
        del out
        return y
