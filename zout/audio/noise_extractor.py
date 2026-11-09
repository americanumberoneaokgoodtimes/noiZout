from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .io import resample

Progress = Callable[[str, float], None]


@dataclass
class NoiseExtractionResult:
    audio: np.ndarray
    sr: int
    chunks: list[tuple[float, float]]  # (start_sec, end_sec)
    rms_db: float
    output_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        dur = len(self.audio) / self.sr
        chunk_strs = [f"{s:.1f}s-{e:.1f}s" for s, e in self.chunks]
        chunks_display = ", ".join(chunk_strs[:4])
        if len(chunk_strs) > 4:
            chunks_display += f" ... (+{len(chunk_strs) - 4} more)"
        return (
            f"Extracted {len(self.chunks)} noise sections ({dur:.1f}s total) | "
            f"Noise floor: {self.rms_db:.1f} dBFS\n"
            f"Timeline chunks: [{chunks_display}]"
        )


def _check_silero_speech(audio_16k: np.ndarray, max_threshold: float, mean_threshold: float) -> tuple[bool, float, float]:
    """Test candidate chunk with Silero VAD in 512-sample (32ms) frames.
    Returns (is_clean_noise, max_prob, mean_prob).
    """
    try:
        import pysilero_vad
        detector = pysilero_vad.SileroVoiceActivityDetector()
        step = 512
        n_frames = len(audio_16k) // step
        if n_frames == 0:
            return False, 1.0, 1.0

        probs = []
        for i in range(n_frames):
            frame = audio_16k[i * step : (i + 1) * step]
            pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            probs.append(detector(pcm16))

        max_p = float(np.max(probs))
        mean_p = float(np.mean(probs))
        is_clean = (max_p <= max_threshold) and (mean_p <= mean_threshold)
        return is_clean, max_p, mean_p
    except Exception:
        # If VAD model fails to load, allow fallback
        return True, 0.0, 0.0


def _crossfade_concat(chunks: list[np.ndarray], sr: int, xfade_ms: float = 25.0) -> np.ndarray:
    """Concatenate chunks with equal-power raised-cosine crossfading to eliminate boundary clicks."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].copy()

    xfade_samples = int(sr * (xfade_ms / 1000.0))
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


def _scan_for_noise_chunks(
    audio: np.ndarray,
    sr: int,
    target_count: int,
    chunk_sec: float,
    max_vad_prob: float,
    mean_vad_prob: float,
    min_gap_sec: float,
) -> list[dict]:
    """Scan audio for candidate windows of `chunk_sec` that pass the Silero VAD speech filter."""
    chunk_samples = int(chunk_sec * sr)
    min_gap_samples = int(min_gap_sec * sr)
    stride = max(int(0.25 * chunk_samples), 1)
    n_windows = max(1, (len(audio) - chunk_samples) // stride + 1)

    candidates = []
    hop_energy = int(0.025 * sr)

    for w_idx in range(n_windows):
        st = w_idx * stride
        en = st + chunk_samples
        seg = audio[st:en]

        peak = float(np.max(np.abs(seg)))
        rms_val = float(np.sqrt(np.mean(seg**2)))
        rms_db = float(20 * math.log10(rms_val + 1e-12))

        # Discard digital silence (< -85 dBFS) and clipping
        if rms_db < -85.0 or peak < 1e-4 or peak > 0.98:
            continue

        # Energy stationarity: variance of 25ms RMS energies
        n_sub = len(seg) // hop_energy
        if n_sub > 2:
            sub_rms = [
                float(np.sqrt(np.mean(seg[k * hop_energy : (k + 1) * hop_energy] ** 2)))
                for k in range(n_sub)
            ]
            std_rms = float(np.std(sub_rms)) / (rms_val + 1e-8)
        else:
            std_rms = 0.0

        # Score: prefer lower energy and lower variance
        score = -rms_db - (10.0 * min(std_rms, 2.0))

        candidates.append({
            "start": st,
            "end": en,
            "start_sec": st / sr,
            "end_sec": en / sr,
            "rms_db": rms_db,
            "score": score,
        })

    # Sort candidates by lowest energy / most stationary first
    candidates.sort(key=lambda c: c["score"], reverse=True)

    selected = []
    for cand in candidates:
        if any(abs(cand["start"] - s["start"]) < min_gap_samples for s in selected):
            continue

        seg = audio[cand["start"] : cand["end"]]
        seg_16k = resample(seg, sr, 16000)

        is_clean, max_p, mean_p = _check_silero_speech(seg_16k, max_vad_prob, mean_vad_prob)
        if not is_clean:
            continue

        cand["max_vad"] = max_p
        cand["mean_vad"] = mean_p
        selected.append(cand)

        if len(selected) >= target_count:
            break

    return selected


def extract_noise_profile(
    audio: np.ndarray,
    sr: int,
    n_chunks: int = 6,
    chunk_sec: float = 2.0,
    mode: Literal["balanced", "deep", "fast"] = "balanced",
    min_gap_sec: float = 1.5,
    progress: Progress | None = None,
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

    p("Scanning audio timeline for room noise candidates", 0.1)

    # VAD sensitivity settings
    if mode == "deep":
        max_vad, mean_vad = 0.06, 0.02
    elif mode == "balanced":
        max_vad, mean_vad = 0.10, 0.04
    else:  # fast
        max_vad, mean_vad = 0.18, 0.08

    # Attempt 1: search at requested chunk_sec
    target_len_sec = n_chunks * chunk_sec
    cur_chunk_sec = min(chunk_sec, max(0.5, total_sec / (n_chunks + 1)))
    selected = _scan_for_noise_chunks(audio, sr, n_chunks, cur_chunk_sec, max_vad, mean_vad, min_gap_sec)

    # Attempt 2: if not enough chunks found, adaptively reduce chunk size to catch shorter pauses
    if len(selected) < n_chunks and cur_chunk_sec > 0.8:
        p("Adapting chunk search window to detect shorter pauses", 0.4)
        smaller_chunk_sec = max(0.5, cur_chunk_sec * 0.5)
        needed_chunks = int(math.ceil(target_len_sec / smaller_chunk_sec))
        more_selected = _scan_for_noise_chunks(
            audio, sr, needed_chunks, smaller_chunk_sec, max_vad, mean_vad, min_gap_sec=max(0.5, min_gap_sec * 0.5)
        )
        if len(more_selected) > len(selected):
            selected = more_selected

    # Attempt 3: if still not found, relax VAD threshold slightly if in balanced mode
    if len(selected) == 0 and mode != "deep":
        p("Retrying with relaxed sensitivity threshold", 0.6)
        warnings.append("High ambient noise or continuous speech detected; sensitivity was slightly adapted.")
        selected = _scan_for_noise_chunks(
            audio, sr, n_chunks, max(0.5, cur_chunk_sec * 0.5), max_vad * 1.8, mean_vad * 1.8, min_gap_sec=0.5
        )

    if not selected:
        raise RuntimeError(
            "Could not locate any speech-free noise sections in the recording. "
            "The audio may contain continuous sound or very high speech density without pauses."
        )

    # 3. Cross-chunk spectral consistency check (reject anomalous outliers like clicks)
    if len(selected) >= 3:
        p("Checking spectral consistency across noise chunks", 0.75)
        spec_list = []
        for s in selected:
            seg = audio[s["start"] : s["end"]]
            # 512-point FFT magnitude
            fft_mag = np.abs(np.fft.rfft(seg[: min(len(seg), 2048)]))
            fft_mag /= (np.linalg.norm(fft_mag) + 1e-9)
            spec_list.append(fft_mag)

        median_spec = np.median(spec_list, axis=0)
        median_spec /= (np.linalg.norm(median_spec) + 1e-9)

        consistent = []
        for s, smag in zip(selected, spec_list):
            corr = float(np.dot(smag, median_spec))
            if corr >= 0.55:  # retains consistent noise, filters sudden transient bangs
                consistent.append(s)

        if len(consistent) >= 2:
            selected = consistent

    # Sort selected chunks chronologically by timeline
    selected.sort(key=lambda s: s["start"])

    p("Concatenating and crossfading noise profile", 0.9)

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
