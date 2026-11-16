from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio.io import (
    NATIVE_EXTENSIONS,
    close_process_quietly,
    get_audio_info,
    local_peak_limit,
    open_ffmpeg_pcm_stream,
    read_pcm_frames,
)
from ..audio.masking import N_FFT, _istft, _stft, apply_adaptive_mask

Progress = Callable[[str, float], None]

MAX_SAMPLES_PER_CHUNK: int = 240_000  # Caps peak memory < 75MB (and < 150MB) across all sample rates (e.g. 5s @ 48kHz, 2.5s @ 96kHz, 15s @ 16kHz)


def _ensure_min_ref_length(ref: np.ndarray, min_len: int = N_FFT) -> np.ndarray:
    """Ensure noise reference array has at least `min_len` samples for full-bin STFT."""
    if len(ref) >= min_len:
        return ref
    if len(ref) == 0:
        return np.zeros(min_len, dtype=np.float32)
    reps = int(math.ceil(min_len / len(ref)))
    return np.tile(ref.astype(np.float32, copy=False), reps)[:min_len]


def _run_with_min_length(
    chunk: np.ndarray,
    min_len: int,
    fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Pad ultra-short trailing chunks (< N_FFT) cleanly so STFT/noisereduce cannot fail."""
    orig_len = len(chunk)
    if orig_len >= min_len:
        return fn(chunk)
    if orig_len == 0:
        return chunk
    pad_mode = "reflect" if orig_len > 1 else "constant"
    padded = np.pad(chunk, (0, min_len - orig_len), mode=pad_mode)
    return fn(padded)[:orig_len]


def kl_nmf(
    V: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    n_iter: int,
    update_w: np.ndarray | None = None,
    cancel_event: threading.Event | None = None,
):
    """Multiplicative-update KL-NMF. `update_w` is a bool mask over columns of W allowed to change."""
    if V.size == 0 or V.shape[1] == 0:
        return W, H
    eps = 1e-9
    V = V.astype(np.float32, copy=False) + eps
    # Pre-compute free-column indices once to avoid repeated np.where inside the loop
    free_cols = np.where(update_w)[0] if update_w is not None else None
    for i in range(n_iter):
        if cancel_event is not None and i % 10 == 0 and cancel_event.is_set():
            raise InterruptedError("Cancelled")
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.sum(0)[:, None] + eps)
        WH = W @ H + eps
        num = (V / WH) @ H.T
        den = H.sum(1) + eps  # (K,) — avoids 2D allocation
        if free_cols is None:
            W *= num / den[None, :]  # in-place: no temporary upd array
        else:
            W[:, free_cols] *= num[:, free_cols] / den[free_cols]  # update only free columns
    return W, H


def _collect_frame_energies(a: np.ndarray, collector: list[float], h: int = 1024) -> None:
    """Compute and append log frame energies for noise floor proxy calculation."""
    m = len(a) // h
    if m > 0:
        frames = a[: m * h].reshape(m, h)
        e = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
        collector.extend(e.tolist())


def _stream_disk_to_disk(
    main_path: Path | str,
    out_path: Path | str,
    sr: int,
    process_chunk: Callable[[np.ndarray], np.ndarray],
    chunk_sec: float = 60.0,
    overlap_sec: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    progress: Progress | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, float]:
    """Streams audio blocks disk-to-disk with 2-second raised-cosine overlap-add.

    Keeps peak memory bounded (< 85MB peak RAM) regardless of file duration.
    """
    p = progress or (lambda *_: None)
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Cancelled")

    info = get_audio_info(main_path)
    file_sr = info.samplerate
    total_frames = info.frames or int(round(info.duration * file_sr))
    is_native = Path(main_path).suffix.lower() in NATIVE_EXTENSIONS

    if start_sec is not None and start_sec >= info.duration:
        raise ValueError(f"start_sec exceeds audio duration ({start_sec} >= {info.duration})")
    if end_sec is not None and end_sec <= 0:
        raise ValueError("end_sec must be positive")
    if start_sec is not None and end_sec is not None and start_sec >= end_sec:
        raise ValueError(f"start_sec ({start_sec}) must be less than end_sec ({end_sec})")

    start_frame = int(round(start_sec * file_sr)) if start_sec is not None else 0
    end_frame = int(round(end_sec * file_sr)) if end_sec is not None else total_frames
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))
    target_frames = end_frame - start_frame
    if target_frames <= 0:
        raise ValueError("Sliced duration must be positive")

    out_p = Path(out_path)
    if out_p.suffix.lower() not in (".wav", ".flac", ".ogg"):
        out_p = out_p.with_suffix(".wav")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_24" if out_p.suffix.lower() == ".wav" else None

    max_chunk_sec = float(MAX_SAMPLES_PER_CHUNK) / float(file_sr)
    eff_chunk_sec = min(chunk_sec, max_chunk_sec)
    eff_overlap_sec = min(overlap_sec, eff_chunk_sec / 4.0)
    N_chunk = max(1024, int(round(eff_chunk_sec * file_sr)))
    N_overlap = int(round(eff_overlap_sec * file_sr)) if eff_overlap_sec > 0 else 0
    if N_overlap >= N_chunk:
        N_overlap = N_chunk // 4

    pcm_proc = None
    in_overlap: np.ndarray | None = None
    if not is_native:
        s_sec = start_frame / float(file_sr)
        e_sec = end_frame / float(file_sr)
        pcm_proc = open_ffmpeg_pcm_stream(main_path, sr=file_sr, channels=1, start_sec=s_sec, end_sec=e_sec)

    def read_input(num_f: int, start_f: int) -> np.ndarray:
        if is_native:
            data, _ = sf.read(str(main_path), start=start_f, frames=num_f, dtype="float32", always_2d=True)
            out = data.mean(axis=1) if data.ndim > 1 else data.reshape(-1)
        else:
            if in_overlap is not None and len(in_overlap) > 0:
                need = max(0, num_f - len(in_overlap))
                new_part = read_pcm_frames(pcm_proc.stdout, need, channels=1) if need > 0 else np.zeros(0, dtype=np.float32)
                out = np.concatenate([in_overlap, new_part]) if len(new_part) else in_overlap.copy()
            else:
                out = read_pcm_frames(pcm_proc.stdout, num_f, channels=1)
        if len(out) > num_f:
            return out[:num_f]
        if len(out) < num_f:
            return np.pad(out, (0, num_f - len(out)))
        return out

    in_frame_energies: list[float] = []
    out_frame_energies: list[float] = []
    in_sum_sq = 0.0
    out_sum_sq = 0.0
    total_out_samples = 0

    out_sf = None
    try:
        out_sf = sf.SoundFile(str(out_p), mode="w", samplerate=file_sr, channels=1, subtype=subtype)

        if target_frames <= N_chunk or N_overlap == 0:
            offset = start_frame
            total_chunks = max(1, math.ceil(target_frames / N_chunk))
            chunk_idx = 0
            while offset < end_frame:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                read_len = min(N_chunk, end_frame - offset)
                chunk_in = read_input(read_len, offset)
                in_sum_sq += float(np.sum(chunk_in.astype(np.float64) ** 2))
                _collect_frame_energies(chunk_in, in_frame_energies)

                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                y = local_peak_limit(process_chunk(chunk_in))
                if len(y) != len(chunk_in):
                    y = y[: len(chunk_in)] if len(y) > len(chunk_in) else np.pad(y, (0, len(chunk_in) - len(y)))

                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                out_sf.write(y)
                out_sum_sq += float(np.sum(y.astype(np.float64) ** 2))
                total_out_samples += len(y)
                _collect_frame_energies(y, out_frame_energies)

                offset += read_len
                chunk_idx += 1
                p(f"Processing chunk {chunk_idx}/{total_chunks}", min(0.95, chunk_idx / total_chunks))
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
        else:
            theta = np.pi * (np.arange(N_overlap, dtype=np.float32) + 0.5) / float(N_overlap)
            w_in = (0.5 * (1.0 - np.cos(theta))).astype(np.float32)
            w_out = (1.0 - w_in).astype(np.float32)

            hop = N_chunk - N_overlap
            total_chunks = max(1, math.ceil((target_frames - N_overlap) / hop))
            offset = start_frame
            chunk_idx = 0
            tail: np.ndarray | None = None

            while offset < end_frame:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")

                is_last = (offset + N_chunk >= end_frame)
                read_len = min(N_chunk, end_frame - offset)
                chunk_in = read_input(read_len, offset)
                if not is_native and not is_last:
                    in_overlap = chunk_in[hop:]
                else:
                    in_overlap = None

                in_slice = chunk_in if is_last else chunk_in[:hop]
                in_sum_sq += float(np.sum(in_slice.astype(np.float64) ** 2))
                _collect_frame_energies(in_slice, in_frame_energies)

                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                y = local_peak_limit(process_chunk(chunk_in))
                if len(y) != len(chunk_in):
                    y = y[: len(chunk_in)] if len(y) > len(chunk_in) else np.pad(y, (0, len(chunk_in) - len(y)))

                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")

                if tail is None:
                    to_write = y[:hop]
                    out_sf.write(to_write)
                    out_sum_sq += float(np.sum(to_write.astype(np.float64) ** 2))
                    total_out_samples += len(to_write)
                    _collect_frame_energies(to_write, out_frame_energies)
                    tail = y[hop:]
                    offset += hop
                elif is_last:
                    blend = local_peak_limit(tail * w_out + y[:N_overlap] * w_in)
                    out_sf.write(blend)
                    out_sum_sq += float(np.sum(blend.astype(np.float64) ** 2))
                    total_out_samples += len(blend)
                    _collect_frame_energies(blend, out_frame_energies)

                    remainder = y[N_overlap:]
                    out_sf.write(remainder)
                    out_sum_sq += float(np.sum(remainder.astype(np.float64) ** 2))
                    total_out_samples += len(remainder)
                    _collect_frame_energies(remainder, out_frame_energies)
                    tail = None
                    offset += read_len
                else:
                    blend = local_peak_limit(tail * w_out + y[:N_overlap] * w_in)
                    out_sf.write(blend)
                    out_sum_sq += float(np.sum(blend.astype(np.float64) ** 2))
                    total_out_samples += len(blend)
                    _collect_frame_energies(blend, out_frame_energies)

                    body = y[N_overlap:hop]
                    out_sf.write(body)
                    out_sum_sq += float(np.sum(body.astype(np.float64) ** 2))
                    total_out_samples += len(body)
                    _collect_frame_energies(body, out_frame_energies)

                    tail = y[hop:]
                    offset += hop

                chunk_idx += 1
                p(f"Processing chunk {chunk_idx}/{total_chunks}", min(0.95, chunk_idx / total_chunks))
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")

    finally:
        close_process_quietly(pcm_proc)
        if out_sf is not None and not out_sf.closed:
            out_sf.close()

    in_rms = float(20.0 * math.log10(math.sqrt(max(1e-12, in_sum_sq / max(1, target_frames)))))
    out_rms = float(20.0 * math.log10(math.sqrt(max(1e-12, out_sum_sq / max(1, total_out_samples)))))

    if in_frame_energies and out_frame_energies:
        floor_in = float(np.percentile(in_frame_energies, 10))
        floor_out = float(np.percentile(out_frame_energies, 10))
        suppression_db = float(max(0.0, floor_in - floor_out))
    else:
        suppression_db = float(max(0.0, in_rms - out_rms))

    return {
        "in_rms": in_rms,
        "out_rms": out_rms,
        "suppression_db": suppression_db,
        "samples_written": total_out_samples,
    }



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
            polish: bool = False,
            cancel_event: threading.Event | None = None) -> tuple[np.ndarray, dict]:
        try:
            import noisereduce as nr
        except ImportError:
            raise ImportError(
                "NOISY mode requires noisereduce — install it with: pip install noisereduce"
            ) from None
        rng = np.random.default_rng(0)
        p = progress or (lambda *_: None)
        info: dict = {}

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        p("Training noise profile (KL-NMF)", 0.0)
        ref = _ensure_min_ref_length(ref, N_FFT)
        R = np.abs(_stft(ref, sr)).astype(np.float32)
        if R.shape[1] > 800:  # cap training frames
            R = R[:, :: R.shape[1] // 800 + 1]
        Wn = rng.random((R.shape[0], self.K_NOISE), dtype=np.float32) + 0.1
        Hn = rng.random((self.K_NOISE, R.shape[1]), dtype=np.float32) + 0.1
        Wn, _ = kl_nmf(R, Wn, Hn, 40, cancel_event=cancel_event)
        Wn /= Wn.sum(0, keepdims=True) + 1e-9

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        thresh = float(2.6 - 1.9 * strength)
        prop = float(min(1.0, 0.25 + 0.75 * strength))
        beta = 0.2 + 1.8 * strength
        pw = 1.0 + strength
        free = np.concatenate([np.zeros(self.K_NOISE, bool), np.ones(self.K_SIG, bool)])

        # Pre-allocate W template: Wn columns frozen, Ws columns reset each chunk
        F_bins = Wn.shape[0]
        _W_template = np.empty((F_bins, self.K_NOISE + self.K_SIG), dtype=np.float32)
        _W_template[:, :self.K_NOISE] = Wn

        # Instantiate DeepFilter once — avoids re-loading ONNX model per call
        _polish_model: object = None
        if polish:
            try:
                from .deepfilter import DeepFilter
                _polish_model = DeepFilter(100.0 if max_atten_db >= 60 else max_atten_db)
            except Exception as e:
                info["polish_error"] = str(e)

        def _process_single(chunk: np.ndarray) -> np.ndarray:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")
            p("Spectral gate + noisereduce", 0.20)
            gated = nr.reduce_noise(y=chunk, sr=sr, y_noise=ref, stationary=True,
                                    prop_decrease=prop, n_std_thresh_stationary=thresh,
                                    freq_mask_smooth_hz=350, n_fft=1024,
                                    chunk_size=min(len(chunk), 100000)).astype(np.float32)

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            p("STFT decomposition", 0.40)
            Z = _stft(gated, sr)
            del gated
            V = np.abs(Z).astype(np.float32)
            F, T = V.shape
            W = _W_template.copy()
            W[:, self.K_NOISE:] = rng.random((F, self.K_SIG), dtype=np.float32) + 0.1
            H = rng.random((W.shape[1], T), dtype=np.float32) + 0.1
            p("KL-NMF source separation", 0.50)
            W, H = kl_nmf(V, W, H, 25, update_w=free, cancel_event=cancel_event)
            Vn = W[:, : self.K_NOISE] @ H[: self.K_NOISE]
            Vs = W[:, self.K_NOISE:] @ H[self.K_NOISE:]
            mask = Vs ** pw / (Vs ** pw + (beta * Vn) ** pw + 1e-9)
            Z *= mask
            del mask, Vn, Vs
            p("Reconstructing signal", 0.78)
            est = _istft(Z, sr, len(chunk))
            del Z, V, W, H

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            if _polish_model is not None:
                p("DeepFilterNet3 polish", 0.84)
                try:
                    est = _polish_model.process(est, sr)
                except Exception as e:
                    info["polish_error"] = str(e)
            p("Shaping adaptive mask", 0.92)
            return apply_adaptive_mask(chunk, est, sr, strength, max_atten_db)

        p("Noise profile ready", 0.10)
        return _run_with_min_length(main, N_FFT, _process_single), info

    def run_streaming(
        self,
        ref: np.ndarray,
        main_path: Path | str,
        out_path: Path | str,
        sr: int,
        strength: float,
        max_atten_db: float,
        chunk_sec: float = 60.0,
        overlap_sec: float = 2.0,
        progress: Progress | None = None,
        polish: bool = False,
        start_sec: float | None = None,
        end_sec: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, dict]:
        try:
            import noisereduce as nr
        except ImportError:
            raise ImportError(
                "NOISY mode requires noisereduce — install it with: pip install noisereduce"
            ) from None
        rng = np.random.default_rng(0)
        p = progress or (lambda *_: None)

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        p("Training noise profile (KL-NMF)", 0.0)
        ref = _ensure_min_ref_length(ref, N_FFT)
        R = np.abs(_stft(ref, sr)).astype(np.float32)
        if R.shape[1] > 800:
            R = R[:, :: R.shape[1] // 800 + 1]
        Wn = rng.random((R.shape[0], self.K_NOISE), dtype=np.float32) + 0.1
        Hn = rng.random((self.K_NOISE, R.shape[1]), dtype=np.float32) + 0.1
        Wn, _ = kl_nmf(R, Wn, Hn, 40, cancel_event=cancel_event)
        Wn /= Wn.sum(0, keepdims=True) + 1e-9

        thresh = float(2.6 - 1.9 * strength)
        prop = float(min(1.0, 0.25 + 0.75 * strength))
        beta = 0.2 + 1.8 * strength
        pw = 1.0 + strength
        free = np.concatenate([np.zeros(self.K_NOISE, bool), np.ones(self.K_SIG, bool)])

        # Pre-allocate W template: Wn columns frozen, Ws columns reset each chunk
        F_bins = Wn.shape[0]
        _W_template = np.empty((F_bins, self.K_NOISE + self.K_SIG), dtype=np.float32)
        _W_template[:, :self.K_NOISE] = Wn

        # Instantiate DeepFilter once — avoids re-loading ONNX model per chunk
        _polish_notes: dict = {}
        _polish_model: object = None
        if polish:
            try:
                from .deepfilter import DeepFilter
                _polish_model = DeepFilter(100.0 if max_atten_db >= 60 else max_atten_db)
            except Exception as e:
                _polish_notes["polish_error"] = str(e)

        p("Noise profile ready — processing chunks", 0.10)

        def process_chunk(chunk: np.ndarray) -> np.ndarray:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")
            gated = nr.reduce_noise(y=chunk, sr=sr, y_noise=ref, stationary=True,
                                    prop_decrease=prop, n_std_thresh_stationary=thresh,
                                    freq_mask_smooth_hz=350, n_fft=1024,
                                    chunk_size=min(len(chunk), 100000)).astype(np.float32)
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Cancelled")
            Z = _stft(gated, sr)
            del gated
            V = np.abs(Z).astype(np.float32)
            F, T = V.shape
            W = _W_template.copy()
            W[:, self.K_NOISE:] = rng.random((F, self.K_SIG), dtype=np.float32) + 0.1
            H = rng.random((W.shape[1], T), dtype=np.float32) + 0.1
            W, H = kl_nmf(V, W, H, 25, update_w=free, cancel_event=cancel_event)
            Vn = W[:, : self.K_NOISE] @ H[: self.K_NOISE]
            Vs = W[:, self.K_NOISE :] @ H[self.K_NOISE :]
            mask = Vs ** pw / (Vs ** pw + (beta * Vn) ** pw + 1e-9)
            Z *= mask
            del mask, Vn, Vs
            est = _istft(Z, sr, len(chunk))
            del Z, V, W, H
            if _polish_model is not None:
                try:
                    est = _polish_model.process(est, sr)
                except Exception as e:
                    _polish_notes["polish_error"] = str(e)
            return apply_adaptive_mask(chunk, est, sr, strength, max_atten_db)

        def _chunk_progress(msg: str, f: float) -> None:
            p(msg, 0.10 + f * 0.90)

        def _guarded_chunk(chunk: np.ndarray) -> np.ndarray:
            return _run_with_min_length(chunk, N_FFT, process_chunk)

        stats = _stream_disk_to_disk(
            main_path=main_path,
            out_path=out_path,
            sr=sr,
            process_chunk=_guarded_chunk,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            start_sec=start_sec,
            end_sec=end_sec,
            progress=_chunk_progress,
            cancel_event=cancel_event,
        )
        stats.update(_polish_notes)
        return Path(out_path), stats
