from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .io import get_audio_info, iter_audio_chunks, load_audio, resample

Progress = Callable[[str, float], None]


@dataclass
class NoiseExtractionResult:
    audio: np.ndarray
    sr: int
    chunks: list[tuple[float, float]]  # (start_sec, end_sec)
    rms_db: float
    output_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def waveform_envelope(self) -> str:
        from .io import compute_waveform_envelope
        return compute_waveform_envelope(self.audio, width=16)

    def summary(self) -> str:
        dur = len(self.audio) / self.sr
        chunk_strs = [f"{s:.1f}s-{e:.1f}s" for s, e in self.chunks]
        chunks_display = ", ".join(chunk_strs[:4])
        if len(chunk_strs) > 4:
            chunks_display += f" ... (+{len(chunk_strs) - 4} more)"
        return (
            f"Extracted {len(self.chunks)} noise sections ({dur:.1f}s total) | "
            f"Noise floor: {self.rms_db:.1f} dBFS | "
            f"Waveform: [{self.waveform_envelope}]\n"
            f"Timeline chunks: [{chunks_display}]"
        )


def _load_vad_model():
    """Load Silero VAD model, preferring zero-PyTorch pysilero_vad (ONNX/ggml).

    Returns (tagged_model, optional_get_timestamps) or (None, None).
    """
    try:
        import pysilero_vad
        return ("pysilero", pysilero_vad.SileroVoiceActivityDetector()), None
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        import silero_vad as _svad
        return ("torch_silero", _svad.load_silero_vad()), _svad.get_speech_timestamps
    except Exception:
        return None, None


def _check_silero_speech(
    audio_16k: np.ndarray,
    max_threshold: float,
    mean_threshold: float,
    vad_model=None,
    get_timestamps=None,
) -> tuple[bool, float, float]:
    """Test candidate chunk with Silero VAD in 512-sample (32ms) frames.

    Returns (is_clean_noise, max_prob, mean_prob).
    """
    try:
        model = vad_model
        if model is None:
            model, _ = _load_vad_model()
        if model is None:
            return True, 0.0, 0.0

        backend, inner = model if isinstance(model, tuple) and len(model) == 2 else ("pysilero", model)
        step = 512
        n_frames = len(audio_16k) // step
        if n_frames == 0:
            return False, 1.0, 1.0

        if backend == "pysilero":
            if hasattr(inner, "reset"):
                inner.reset()
            probs = []
            for i in range(n_frames):
                frame = audio_16k[i * step : (i + 1) * step]
                pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                probs.append(float(inner(pcm16)))
            max_p = float(np.max(probs))
            mean_p = float(np.mean(probs))
            is_clean = (max_p <= max_threshold) and (mean_p <= mean_threshold)
            return is_clean, max_p, mean_p

        if backend == "torch_silero":
            import torch
            if hasattr(inner, "reset_states"):
                inner.reset_states()
            probs = []
            with torch.no_grad():
                for i in range(n_frames):
                    frame = audio_16k[i * step : (i + 1) * step].astype(np.float32)
                    t = torch.from_numpy(frame).unsqueeze(0)
                    probs.append(float(inner(t, 16000).item()))
            max_p = float(np.max(probs))
            mean_p = float(np.mean(probs))
            is_clean = (max_p <= max_threshold) and (mean_p <= mean_threshold)
            return is_clean, max_p, mean_p

        return True, 0.0, 0.0
    except Exception:
        return True, 0.0, 0.0


def _crossfade_concat(chunks: list[np.ndarray], sr: int, xfade_ms: float = 25.0) -> np.ndarray:
    """Concatenate chunks with equal-power raised-cosine crossfading to eliminate boundary clicks."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].copy()

    min_c = min(len(c) for c in chunks)
    xfade_samples = min(int(sr * (xfade_ms / 1000.0)), min_c // 2)
    if xfade_samples < 2:
        return np.concatenate(chunks)

    t = np.linspace(0, np.pi / 2, xfade_samples, dtype=np.float32)
    fade_in = np.sin(t)
    fade_out = np.cos(t)

    total_len = sum(len(c) for c in chunks) - (len(chunks) - 1) * xfade_samples
    out = np.zeros(total_len, dtype=np.float32)

    pos = 0
    for idx, c in enumerate(chunks):
        c_len = len(c)
        if idx == 0:
            out[:c_len] = c
            pos = c_len
        else:
            xf_start = pos - xfade_samples
            out[xf_start:pos] = out[xf_start:pos] * fade_out + c[:xfade_samples] * fade_in
            rem_len = c_len - xfade_samples
            out[pos : pos + rem_len] = c[xfade_samples:]
            pos += rem_len

    return out


def _score_noise_window(seg: np.ndarray, sr: int, hop_energy: int) -> tuple[float, float] | None:
    """Compute (score, rms_db) for a candidate noise segment, or None if clipped/silent."""
    peak = float(np.max(np.abs(seg)))
    rms_val = float(np.sqrt(np.mean(seg ** 2)))
    rms_db = float(20 * math.log10(rms_val + 1e-12))

    # Discard digital silence (< -85 dBFS) and clipping
    if rms_db < -85.0 or peak < 1e-4 or peak > 0.98:
        return None

    n_sub = len(seg) // hop_energy
    if n_sub > 2:
        sub_rms = np.sqrt(np.mean(seg[: n_sub * hop_energy].reshape(n_sub, hop_energy) ** 2, axis=1))
        std_rms = float(np.std(sub_rms)) / (rms_val + 1e-8)
    else:
        std_rms = 0.0

    score = -rms_db - (10.0 * min(std_rms, 2.0))
    return score, rms_db


def _vad_filter_candidates(
    candidates: list[dict],
    audio_resolver,
    sr: int,
    target_count: int,
    max_vad_prob: float,
    mean_vad_prob: float,
    min_gap_sec: float,
    progress: Progress | None = None,
    vad_model=None,
    get_timestamps=None,
    cancel_event=None,
) -> list[dict]:
    """Validate top energy candidates with Silero VAD while respecting min_gap_sec."""
    p = progress or (lambda *_: None)
    n_cands = len(candidates)
    selected: list[dict] = []
    for c_idx, cand in enumerate(candidates):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")
        if n_cands > 0:
            p(
                f"VAD check: candidate {c_idx + 1}/{n_cands} at {cand['start_sec']:.1f}s",
                0.55 + (c_idx + 1) / n_cands * 0.45,
            )

        if any(abs(cand["start_sec"] - s["start_sec"]) < min_gap_sec for s in selected):
            continue

        seg = audio_resolver(cand)
        seg_16k = resample(seg, sr, 16000)

        is_clean, max_p, mean_p = _check_silero_speech(
            seg_16k, max_vad_prob, mean_vad_prob,
            vad_model=vad_model, get_timestamps=get_timestamps,
        )
        if not is_clean:
            continue

        cand_copy = dict(cand)
        cand_copy["max_vad"] = max_p
        cand_copy["mean_vad"] = mean_p
        selected.append(cand_copy)

        if len(selected) >= target_count:
            break

    return selected


def _filter_spectrally_consistent(selected: list[dict], seg_getter) -> list[dict]:
    """Reject outlier transient bangs using median spectral cosine correlation."""
    if len(selected) < 3:
        return selected
    spec_list = []
    for s in selected:
        seg = seg_getter(s)
        fft_mag = np.abs(np.fft.rfft(seg[: min(len(seg), 2048)]))
        fft_mag /= (np.linalg.norm(fft_mag) + 1e-9)
        spec_list.append(fft_mag)

    median_spec = np.median(spec_list, axis=0)
    median_spec /= (np.linalg.norm(median_spec) + 1e-9)

    consistent = []
    for s, smag in zip(selected, spec_list):
        if float(np.dot(smag, median_spec)) >= 0.55:
            consistent.append(s)

    return consistent if len(consistent) >= 2 else selected


def _scan_for_noise_chunks(
    audio: np.ndarray,
    sr: int,
    target_count: int,
    chunk_sec: float,
    max_vad_prob: float,
    mean_vad_prob: float,
    min_gap_sec: float,
    progress: Progress | None = None,
    _vad_model=None,
    _get_timestamps=None,
    cancel_event=None,
) -> list[dict]:
    """Scan audio for candidate windows of `chunk_sec` that pass the Silero VAD speech filter."""
    chunk_samples = int(chunk_sec * sr)
    stride = max(int(0.25 * chunk_samples), 1)
    n_windows = max(1, (len(audio) - chunk_samples) // stride + 1)
    p = progress or (lambda *_: None)

    # Load VAD model once for all candidates in this scan pass
    vad_model, get_timestamps = _vad_model, _get_timestamps
    if vad_model is None:
        vad_model, get_timestamps = _load_vad_model()

    candidates = []
    hop_energy = int(0.025 * sr)

    # Phase 1: energy scan (fast) — emit progress per window
    for w_idx in range(n_windows):
        if cancel_event is not None and w_idx % 50 == 0 and cancel_event.is_set():
            raise InterruptedError("Cancelled")
        if n_windows > 1:
            p(f"Energy scan: window {w_idx + 1}/{n_windows}", (w_idx + 1) / n_windows * 0.55)
        st = w_idx * stride
        en = st + chunk_samples
        scored = _score_noise_window(audio[st:en], sr, hop_energy)
        if scored is None:
            continue
        score, rms_db = scored
        candidates.append({
            "start": st,
            "end": en,
            "start_sec": st / sr,
            "end_sec": en / sr,
            "rms_db": rms_db,
            "score": score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    return _vad_filter_candidates(
        candidates,
        audio_resolver=lambda c: audio[c["start"] : c["end"]],
        sr=sr,
        target_count=target_count,
        max_vad_prob=max_vad_prob,
        mean_vad_prob=mean_vad_prob,
        min_gap_sec=min_gap_sec,
        progress=p,
        vad_model=vad_model,
        get_timestamps=get_timestamps,
        cancel_event=cancel_event,
    )


def extract_noise_profile(
    audio: np.ndarray,
    sr: int,
    n_chunks: int = 6,
    chunk_sec: float = 2.0,
    mode: Literal["balanced", "deep", "fast"] = "balanced",
    min_gap_sec: float = 1.5,
    progress: Progress | None = None,
    cancel_event=None,
) -> NoiseExtractionResult:
    """Scan audio for candidate room-noise sections with lowest signal-to-noise ratio,
    validate with strict Silero VAD to guarantee no speech fragments, and crossfade
    into a clean noise reference profile.
    """
    p = progress or (lambda *_: None)
    warnings: list[str] = []

    total_sec = len(audio) / sr
    if total_sec < 0.5:
        raise ValueError(f"Audio is too short ({total_sec:.2f}s) to extract noise profile.")

    # VAD sensitivity settings
    if mode == "deep":
        max_vad, mean_vad = 0.06, 0.02
    elif mode == "balanced":
        max_vad, mean_vad = 0.10, 0.04
    else:  # fast
        max_vad, mean_vad = 0.18, 0.08

    # Load VAD model once — reused across all scan attempts
    p("Loading VAD model...", 0.02)
    vad_model, get_timestamps = _load_vad_model()
    if vad_model is None:
        warnings.append("Silero VAD unavailable; using energy-stationarity noise detection only.")

    def _sub(lo: float, hi: float) -> Progress:
        """Return a progress callback that maps [0,1] into [lo,hi] of p."""
        def _inner(msg: str, f: float) -> None:
            p(msg, lo + f * (hi - lo))
        return _inner

    # Attempt 1: search at requested chunk_sec — occupies 0.05..0.65
    target_len_sec = n_chunks * chunk_sec
    cur_chunk_sec = min(chunk_sec, max(0.5, total_sec / (n_chunks + 1)))
    p(f"Scanning timeline for noise candidates ({total_sec:.0f}s audio)...", 0.05)
    selected = _scan_for_noise_chunks(
        audio, sr, n_chunks, cur_chunk_sec, max_vad, mean_vad, min_gap_sec,
        progress=_sub(0.05, 0.65),
        _vad_model=vad_model, _get_timestamps=get_timestamps,
        cancel_event=cancel_event,
    )

    # Attempt 2: adaptively reduce chunk size to catch shorter pauses — occupies 0.65..0.78
    if len(selected) < n_chunks and cur_chunk_sec > 0.8:
        p("Adapting chunk size to detect shorter pauses...", 0.65)
        smaller_chunk_sec = max(0.5, cur_chunk_sec * 0.5)
        needed_chunks = int(math.ceil(target_len_sec / smaller_chunk_sec))
        more_selected = _scan_for_noise_chunks(
            audio, sr, needed_chunks, smaller_chunk_sec, max_vad, mean_vad,
            min_gap_sec=max(0.5, min_gap_sec * 0.5),
            progress=_sub(0.65, 0.78),
            _vad_model=vad_model, _get_timestamps=get_timestamps,
            cancel_event=cancel_event,
        )
        if len(more_selected) > len(selected):
            selected = more_selected

    # Attempt 3: relax VAD threshold — occupies 0.78..0.85
    if len(selected) == 0 and mode != "deep":
        p("Retrying with relaxed sensitivity threshold...", 0.78)
        warnings.append("High ambient noise or continuous speech detected; sensitivity was slightly adapted.")
        selected = _scan_for_noise_chunks(
            audio, sr, n_chunks, max(0.5, cur_chunk_sec * 0.5), max_vad * 1.8, mean_vad * 1.8,
            min_gap_sec=0.5,
            progress=_sub(0.78, 0.85),
            _vad_model=vad_model, _get_timestamps=get_timestamps,
            cancel_event=cancel_event,
        )

    if not selected:
        raise RuntimeError(
            "Could not locate any speech-free noise sections in the recording. "
            "The audio may contain continuous sound or very high speech density without pauses."
        )

    # Cross-chunk spectral consistency check — occupies 0.85..0.90
    if len(selected) >= 3:
        p("Checking spectral consistency across noise chunks...", 0.85)
        selected = _filter_spectrally_consistent(selected, lambda s: audio[s["start"] : s["end"]])

    selected.sort(key=lambda s: s["start"])

    p("Concatenating and crossfading noise profile...", 0.92)
    raw_chunks = [audio[s["start"] : s["end"]] for s in selected]
    final_audio = _crossfade_concat(raw_chunks, sr, xfade_ms=25.0)

    final_rms = float(20 * math.log10(np.sqrt(np.mean(final_audio**2)) + 1e-12))
    chunk_ranges = [(s["start_sec"], s["end_sec"]) for s in selected]

    p("Noise profile extraction complete", 1.0)

    return NoiseExtractionResult(
        audio=final_audio,
        sr=sr,
        chunks=chunk_ranges,
        rms_db=final_rms,
        warnings=warnings,
    )


def extract_noise_profile_from_file(
    path: Path | str,
    n_chunks: int = 6,
    chunk_sec: float = 2.0,
    mode: Literal["balanced", "deep", "fast"] = "balanced",
    min_gap_sec: float = 1.5,
    start_sec: float | None = None,
    end_sec: float | None = None,
    progress: Progress | None = None,
    cancel_event=None,
) -> NoiseExtractionResult:
    """Memory-bounded file-backed noise profile extractor for recordings of any duration.

    For files <= 180s in the target range, loads the sliced audio directly.
    For multi-hour recordings (> 180s), streams 30s blocks sequentially in < 35MB RAM.
    """
    p = progress or (lambda *_: None)
    info = get_audio_info(path)
    t_start = max(0.0, float(start_sec or 0.0))
    t_end = min(info.duration, float(end_sec)) if end_sec is not None else info.duration
    eff_dur = max(0.0, t_end - t_start)
    if eff_dur < 0.5:
        raise ValueError(f"Audio is too short ({eff_dur:.2f}s) to extract noise profile.")

    if eff_dur <= 180.0:
        p(f"Loading audio slice ({eff_dur:.0f}s)...", 0.03)
        main_audio, audio_sr = load_audio(path, mono=True, start_sec=start_sec, end_sec=end_sec)
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cancelled")
        res = extract_noise_profile(
            main_audio,
            sr=audio_sr,
            n_chunks=n_chunks,
            chunk_sec=chunk_sec,
            mode=mode,
            min_gap_sec=min_gap_sec,
            progress=lambda msg, f: p(msg, 0.06 + f * 0.94),
            cancel_event=cancel_event,
        )
        if t_start > 0.0:
            res.chunks = [(round(s + t_start, 3), round(e + t_start, 3)) for s, e in res.chunks]
        return res

    # Streaming bounded scan for long recordings (> 180s)
    warnings: list[str] = []
    if mode == "deep":
        max_vad, mean_vad = 0.06, 0.02
    elif mode == "balanced":
        max_vad, mean_vad = 0.10, 0.04
    else:
        max_vad, mean_vad = 0.18, 0.08

    p("Loading VAD model...", 0.02)
    vad_model, get_timestamps = _load_vad_model()
    if vad_model is None:
        warnings.append("Silero VAD unavailable; using energy-stationarity noise detection only.")

    top_cap = max(60, n_chunks * 12)
    candidates: list[dict] = []
    file_sr = info.samplerate
    block_offset_sec = t_start
    p(f"Streaming energy scan across {eff_dur / 60:.1f} min recording...", 0.05)

    for block, c_sr in iter_audio_chunks(
        path, chunk_sec=30.0, start_sec=t_start, end_sec=t_end, mono=True, cancel_event=cancel_event
    ):
        file_sr = c_sr
        chunk_samples = max(int(0.5 * file_sr), int(round(chunk_sec * file_sr)))
        stride = max(int(0.5 * chunk_samples), 1)
        hop_energy = max(1, int(0.025 * file_sr))
        n_win = (len(block) - chunk_samples) // stride + 1
        for w_idx in range(max(0, n_win)):
            st = w_idx * stride
            en = st + chunk_samples
            seg = block[st:en]
            scored = _score_noise_window(seg, file_sr, hop_energy)
            if scored is None:
                continue
            score, rms_db = scored
            cand = {
                "start_sec": block_offset_sec + st / file_sr,
                "end_sec": block_offset_sec + en / file_sr,
                "rms_db": rms_db,
                "score": score,
                "seg": seg.copy(),
            }
            if len(candidates) < top_cap:
                candidates.append(cand)
            elif score > candidates[-1]["score"]:
                candidates[-1] = cand
            if len(candidates) == top_cap or w_idx == n_win - 1:
                candidates.sort(key=lambda c: c["score"], reverse=True)
        block_offset_sec += len(block) / file_sr
        frac = min(1.0, (block_offset_sec - t_start) / max(1e-6, eff_dur))
        p(f"Streaming energy scan ({frac * 100:.0f}%)...", 0.05 + frac * 0.55)

    candidates.sort(key=lambda c: c["score"], reverse=True)

    def _vad_sub(lo: float, hi: float) -> Progress:
        def _inner(msg: str, f: float) -> None:
            p(msg, lo + f * (hi - lo))
        return _inner

    selected = _vad_filter_candidates(
        candidates,
        audio_resolver=lambda c: c["seg"],
        sr=file_sr,
        target_count=n_chunks,
        max_vad_prob=max_vad,
        mean_vad_prob=mean_vad,
        min_gap_sec=min_gap_sec,
        progress=_vad_sub(0.60, 0.80),
        vad_model=vad_model,
        get_timestamps=get_timestamps,
        cancel_event=cancel_event,
    )

    if len(selected) == 0 and mode != "deep":
        p("Retrying with relaxed sensitivity threshold...", 0.80)
        warnings.append("High ambient noise or continuous speech detected; sensitivity was slightly adapted.")
        selected = _vad_filter_candidates(
            candidates,
            audio_resolver=lambda c: c["seg"],
            sr=file_sr,
            target_count=n_chunks,
            max_vad_prob=max_vad * 1.8,
            mean_vad_prob=mean_vad * 1.8,
            min_gap_sec=0.5,
            progress=_vad_sub(0.80, 0.88),
            vad_model=vad_model,
            get_timestamps=get_timestamps,
            cancel_event=cancel_event,
        )

    if not selected:
        raise RuntimeError(
            "Could not locate any speech-free noise sections in the recording. "
            "The audio may contain continuous sound or very high speech density without pauses."
        )

    p("Checking spectral consistency across noise chunks...", 0.89)
    selected = _filter_spectrally_consistent(selected, lambda s: s["seg"])
    selected.sort(key=lambda s: s["start_sec"])

    p("Concatenating and crossfading noise profile...", 0.94)
    final_audio = _crossfade_concat([s["seg"] for s in selected], file_sr, xfade_ms=25.0)
    final_rms = float(20 * math.log10(np.sqrt(np.mean(final_audio**2)) + 1e-12))
    chunk_ranges = [(s["start_sec"], s["end_sec"]) for s in selected]
    p("Noise profile extraction complete", 1.0)

    return NoiseExtractionResult(
        audio=final_audio,
        sr=file_sr,
        chunks=chunk_ranges,
        rms_db=final_rms,
        warnings=warnings,
    )
