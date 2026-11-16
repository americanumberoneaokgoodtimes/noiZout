from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soxr

from .io import get_audio_info, iter_audio_chunks

def _iter_audio_chunks(
    path: Path,
    chunk_sec: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    mono: bool = True,
):
    return iter_audio_chunks(
        path, chunk_sec=chunk_sec, start_sec=start_sec, end_sec=end_sec, mono=mono, _get_info=get_audio_info
    )

if TYPE_CHECKING:
    from ..config import JobConfig


def sanitize_float(val: float | int | None, default: float = -120.0, ndigits: int = 4) -> float:
    """Sanitize float value ensuring finite number and rounding to ndigits."""
    if val is None:
        return default
    try:
        f = float(val)
        if not math.isfinite(f):
            return default
        return round(f, ndigits)
    except (TypeError, ValueError):
        return default


def _slice_array(
    audio: np.ndarray,
    sr: int | None,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[np.ndarray, int]:
    """Convert array to mono float32 and apply optional [start_sec, end_sec] bounds."""
    effective_sr = sr or 48000
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if start_sec is not None or end_sec is not None:
        s = max(0, int(round((start_sec or 0.0) * effective_sr)))
        e = min(len(mono), int(round(end_sec * effective_sr))) if end_sec is not None else len(mono)
        mono = mono[s:e] if s < e else mono[:0]
    return mono, effective_sr


def compute_true_peak(
    audio: np.ndarray | Path | str,
    sr: int | None = None,
    oversample: int = 4,
    chunk_sec: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> float:
    """Compute ITU-R BS.1770 true-peak in dBFS with 4x oversampling via soxr.

    Streams in chunks to guarantee memory usage remains < 15MB even for multi-hour audio.
    Computes oversampled true-peak per channel independently and takes max() across channels.
    Returns finite float in dBFS (clamped to -120.0 on silence or invalid signal).
    """
    global_peak_linear = 0.0
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if not path.is_file():
            return -120.0
        try:
            for chunk, file_sr in iter_audio_chunks(path, chunk_sec=chunk_sec, start_sec=start_sec, end_sec=end_sec, mono=False):
                if len(chunk) == 0:
                    continue
                if chunk.ndim == 1 or (chunk.ndim == 2 and chunk.shape[1] == 1):
                    ch_data = chunk.ravel()
                    resampled = soxr.resample(ch_data, file_sr, file_sr * oversample, quality="HQ")
                    chunk_peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
                    if chunk_peak > global_peak_linear:
                        global_peak_linear = chunk_peak
                else:
                    for c in range(chunk.shape[1]):
                        ch_data = chunk[:, c]
                        resampled = soxr.resample(ch_data, file_sr, file_sr * oversample, quality="HQ")
                        chunk_peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
                        if chunk_peak > global_peak_linear:
                            global_peak_linear = chunk_peak
            if global_peak_linear <= 1e-6:
                return -120.0
            return sanitize_float(20.0 * math.log10(global_peak_linear), default=-120.0, ndigits=2)
        except Exception:
            return -120.0

    if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
        return -120.0

    effective_sr = sr or 48000
    arr = audio
    if start_sec is not None or end_sec is not None:
        s = max(0, int(round((start_sec or 0.0) * effective_sr)))
        e = min(len(arr), int(round(end_sec * effective_sr))) if end_sec is not None else len(arr)
        arr = arr[s:e] if s < e else arr[:0]

    if len(arr) == 0:
        return -120.0

    chunk_frames = max(1024, int(chunk_sec * effective_sr))

    for i in range(0, len(arr), chunk_frames):
        chunk = arr[i : i + chunk_frames]
        if len(chunk) == 0:
            continue
        if chunk.ndim == 1 or (chunk.ndim == 2 and chunk.shape[1] == 1):
            ch_data = chunk.ravel()
            resampled = soxr.resample(ch_data, effective_sr, effective_sr * oversample, quality="HQ")
            chunk_peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
            if chunk_peak > global_peak_linear:
                global_peak_linear = chunk_peak
        else:
            for c in range(chunk.shape[1]):
                ch_data = chunk[:, c]
                resampled = soxr.resample(ch_data, effective_sr, effective_sr * oversample, quality="HQ")
                chunk_peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
                if chunk_peak > global_peak_linear:
                    global_peak_linear = chunk_peak

    if global_peak_linear <= 1e-6:
        return -120.0
    return sanitize_float(20.0 * math.log10(global_peak_linear), default=-120.0, ndigits=2)


def compute_rms(
    audio: np.ndarray | Path | str,
    sr: int | None = None,
    chunk_sec: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> float:
    """Compute RMS level in dBFS across all channels.

    Streams in chunks to guarantee memory usage remains < 15MB.
    Returns finite float in dBFS (clamped to -120.0 on silence).
    """
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if not path.is_file():
            return -120.0
        try:
            sum_sq = 0.0
            total_samples = 0
            for chunk, _ in iter_audio_chunks(path, chunk_sec=chunk_sec, start_sec=start_sec, end_sec=end_sec, mono=False):
                if len(chunk) == 0:
                    continue
                sum_sq += float(np.sum(chunk.astype(np.float64) ** 2))
                total_samples += chunk.size
            if total_samples == 0:
                return -120.0
            rms_lin = math.sqrt(sum_sq / total_samples)
            if rms_lin <= 1e-6:
                return -120.0
            return sanitize_float(20.0 * math.log10(rms_lin), default=-120.0, ndigits=2)
        except Exception:
            return -120.0

    if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
        return -120.0

    effective_sr = sr or 48000
    arr = audio
    if start_sec is not None or end_sec is not None:
        s = max(0, int(round((start_sec or 0.0) * effective_sr)))
        e = min(len(arr), int(round(end_sec * effective_sr))) if end_sec is not None else len(arr)
        arr = arr[s:e] if s < e else arr[:0]
    if arr.size == 0:
        return -120.0
    rms_lin = math.sqrt(float(np.mean(arr.astype(np.float64) ** 2)))
    if rms_lin <= 1e-6:
        return -120.0
    return sanitize_float(20.0 * math.log10(rms_lin), default=-120.0, ndigits=2)


def extract_vad_intervals(
    audio: np.ndarray | Path | str,
    sr: int | None = None,
    threshold: float = 0.15,
    min_speech_sec: float = 0.03,
    min_silence_sec: float = 0.35,
    chunk_sec: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[dict[str, float]]:
    """Extract continuous speech intervals using pysilero_vad (ONNX) in streaming passes.

    Falls back to torch + silero_vad if pysilero_vad is unavailable.
    """
    detector = None
    try:
        import pysilero_vad
        detector = pysilero_vad.SileroVoiceActivityDetector()
    except Exception:
        detector = None

    if detector is not None:
        step = 512
        frame_dur = step / 16000.0
        active_frames: list[bool] = []
        rem_16k = np.zeros(0, dtype=np.float32)

        def _process_16k_stream(chunk_16k: np.ndarray) -> None:
            nonlocal rem_16k
            buf = np.concatenate([rem_16k, chunk_16k])
            n_frames = len(buf) // step
            for i in range(n_frames):
                frame = buf[i * step : (i + 1) * step]
                pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                prob = float(detector(pcm))
                active_frames.append(prob >= threshold)
            rem_16k = buf[n_frames * step :]

        try:
            if isinstance(audio, (Path, str)):
                path = Path(audio)
                if not path.is_file():
                    return []
                for mono, c_sr in iter_audio_chunks(path, chunk_sec=chunk_sec, start_sec=start_sec, end_sec=end_sec):
                    if len(mono) == 0:
                        continue
                    c16 = (
                        soxr.resample(mono, c_sr, 16000, quality="HQ").astype(np.float32)
                        if c_sr != 16000
                        else mono.astype(np.float32, copy=False)
                    )
                    _process_16k_stream(c16)
            else:
                if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
                    return []
                mono, effective_sr = _slice_array(audio, sr or 16000, start_sec, end_sec)
                if len(mono) == 0:
                    return []
                chunk_frames = max(1024, int(chunk_sec * effective_sr))
                for i in range(0, len(mono), chunk_frames):
                    sub = mono[i : i + chunk_frames]
                    c16 = (
                        soxr.resample(sub, effective_sr, 16000, quality="HQ").astype(np.float32)
                        if effective_sr != 16000
                        else sub.astype(np.float32, copy=False)
                    )
                    _process_16k_stream(c16)

            if len(rem_16k) > 0:
                frame = np.pad(rem_16k, (0, step - len(rem_16k)))
                pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                prob = float(detector(pcm))
                active_frames.append(prob >= threshold)
        except Exception:
            return []

        if not active_frames or not any(active_frames):
            return []

        raw_intervals: list[dict[str, float]] = []
        in_speech = False
        start_frame = 0
        for idx, is_sp in enumerate(active_frames):
            if is_sp and not in_speech:
                in_speech = True
                start_frame = idx
            elif not is_sp and in_speech:
                in_speech = False
                raw_intervals.append({
                    "start": round(start_frame * frame_dur, 3),
                    "end": round(idx * frame_dur, 3),
                    "duration": round((idx - start_frame) * frame_dur, 3),
                })
        if in_speech:
            raw_intervals.append({
                "start": round(start_frame * frame_dur, 3),
                "end": round(len(active_frames) * frame_dur, 3),
                "duration": round((len(active_frames) - start_frame) * frame_dur, 3),
            })

        merged: list[dict[str, float]] = []
        for item in raw_intervals:
            if not merged:
                merged.append(dict(item))
            else:
                prev = merged[-1]
                if item["start"] - prev["end"] <= min_silence_sec:
                    prev["end"] = item["end"]
                    prev["duration"] = round(prev["end"] - prev["start"], 3)
                else:
                    merged.append(dict(item))

        return [i for i in merged if i["duration"] >= min_speech_sec]

    # Optional fallback if torch + silero_vad happens to be installed instead of pysilero_vad
    try:
        import torch
        import silero_vad as _svad
        model = _svad.load_silero_vad()
    except Exception:
        return []

    chunks_16k: list[np.ndarray] = []
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if not path.is_file():
            return []
        try:
            for mono, c_sr in _iter_audio_chunks(path, chunk_sec=chunk_sec, start_sec=start_sec, end_sec=end_sec):
                c16 = (
                    soxr.resample(mono, c_sr, 16000, quality="HQ").astype(np.float32)
                    if c_sr != 16000
                    else mono.astype(np.float32)
                )
                chunks_16k.append(c16)
        except Exception:
            return []
    else:
        if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
            return []
        mono, effective_sr = _slice_array(audio, sr or 16000, start_sec, end_sec)
        if len(mono) > 0:
            chunks_16k.append(
                soxr.resample(mono, effective_sr, 16000, quality="HQ").astype(np.float32)
                if effective_sr != 16000
                else mono.astype(np.float32)
            )

    if not chunks_16k:
        return []
    mono_16k = np.concatenate(chunks_16k)
    if len(mono_16k) == 0:
        return []

    try:
        audio_tensor = torch.from_numpy(mono_16k)
        raw = _svad.get_speech_timestamps(
            audio_tensor, model,
            return_seconds=True,
            threshold=threshold,
            min_speech_duration_ms=int(min_speech_sec * 1000),
            min_silence_duration_ms=int(min_silence_sec * 1000),
        )
    except Exception:
        return []

    intervals = []
    for seg in raw:
        start = round(float(seg["start"]), 3)
        end = round(float(seg["end"]), 3)
        dur = round(end - start, 3)
        if dur >= min_speech_sec:
            intervals.append({"start": start, "end": end, "duration": dur})
    return intervals


def compute_speech_summary(vad_intervals: list[dict[str, float]], total_duration_sec: float) -> dict[str, float]:
    """Compute summary statistics for speech and non-speech durations."""
    speech_sec = round(sum(i["duration"] for i in vad_intervals), 3)
    total_sec = round(max(0.0, total_duration_sec), 3)
    non_speech_sec = round(max(0.0, total_sec - speech_sec), 3)
    ratio = round(speech_sec / total_sec, 4) if total_sec > 0 else 0.0
    return {
        "speech_duration_sec": speech_sec,
        "non_speech_duration_sec": non_speech_sec,
        "total_duration_sec": total_sec,
        "speech_presence_ratio": ratio,
    }


def build_parameters_manifest(cfg: JobConfig) -> dict[str, Any]:
    """Build serializable dictionary of all active processing parameters."""
    role_str = cfg.role.value if hasattr(cfg.role, "value") else str(cfg.role)
    return {
        "role": role_str,
        "strength": float(cfg.strength),
        "max_atten_db": float(cfg.max_atten_db),
        "polish": bool(cfg.use_polish()),
        "start_sec": cfg.start_sec,
        "end_sec": cfg.end_sec,
        "chunk_duration": float(cfg.chunk_duration),
        "overlap_duration": float(cfg.overlap_duration),
        "reference": str(cfg.reference) if cfg.reference else None,
        "main": str(cfg.main) if cfg.main else None,
        "output": str(cfg.output or cfg.default_output()) if (cfg.output or cfg.default_output()) else None,
    }


def _get_audio_duration(
    audio: np.ndarray | Path | str | None,
    sr: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> float:
    """Helper to get audio duration in seconds safely."""
    if audio is None:
        return 0.0
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if path.is_file():
            try:
                dur = float(get_audio_info(path).duration)
                if start_sec is not None or end_sec is not None:
                    s = max(0.0, float(start_sec or 0.0))
                    e = min(dur, float(end_sec)) if end_sec is not None else dur
                    return max(0.0, e - s)
                return dur
            except Exception:
                return 0.0
        return 0.0
    if isinstance(audio, np.ndarray) and audio.size > 0:
        eff_sr = sr or 48000
        dur = float(len(audio) / eff_sr)
        if start_sec is not None or end_sec is not None:
            s = max(0.0, float(start_sec or 0.0))
            e = min(dur, float(end_sec)) if end_sec is not None else dur
            return max(0.0, e - s)
        return dur
    return 0.0


def generate_audit_manifest(
    cfg: JobConfig,
    output_path: Path | str | None = None,
    *args: Any,
    input_audio: np.ndarray | Path | str | None = None,
    output_audio: np.ndarray | Path | str | None = None,
    sr: int | None = None,
    suppression_db: float | None = None,
    in_rms: float | None = None,
    out_rms: float | None = None,
    audit_path: Path | str | None = None,
    **kwargs: Any,
) -> Path:
    """Generate RFC 8259 compliant forensic audit JSON metadata manifest.

    Args:
        cfg: JobConfig containing configuration parameters and optional audit paths.
        output_path: Output audio file path.
        input_audio: Input audio as array or file path.
        output_audio: Output audio as array or file path.
        sr: Audio sample rate.
        suppression_db: Precomputed noise suppression in dB.
        in_rms: Precomputed input RMS in dBFS.
        out_rms: Precomputed output RMS in dBFS.
        audit_path: Explicit override path for the manifest file.

    Returns:
        Path to the saved .json audit sidecar.
    """
    # Absorb positional arguments if supplied (e.g. in_rms, out_rms, supp_db)
    if len(args) >= 1 and in_rms is None and isinstance(args[0], (int, float)):
        in_rms = float(args[0])
    if len(args) >= 2 and out_rms is None and isinstance(args[1], (int, float)):
        out_rms = float(args[1])
    if len(args) >= 3 and suppression_db is None and isinstance(args[2], (int, float)):
        suppression_db = float(args[2])

    # Absorb alias keyword arguments
    if input_audio is None and "in_audio" in kwargs:
        input_audio = kwargs["in_audio"]
    if output_audio is None and "out_audio" in kwargs:
        output_audio = kwargs["out_audio"]
    if suppression_db is None and "suppression" in kwargs:
        suppression_db = kwargs["suppression"]

    # 1. Resolve target destination path
    out_target: Path | None = Path(output_path) if output_path is not None else None
    if out_target is None and cfg.output is not None:
        out_target = Path(cfg.output)
    if out_target is None:
        out_target = cfg.default_output()

    target_json: Path
    if audit_path is not None:
        target_json = Path(audit_path)
    elif cfg.audit_path is not None:
        target_json = Path(cfg.audit_path)
    elif cfg.default_audit_path() is not None:
        target_json = Path(cfg.default_audit_path())
    elif out_target is not None:
        target_json = out_target.with_name(out_target.stem + "_audit.json")
    else:
        target_json = Path("audit.json")

    target_json.parent.mkdir(parents=True, exist_ok=True)

    # 2. Resolve input and output sources for metrics
    in_src = input_audio if input_audio is not None else cfg.main
    out_src = output_audio if output_audio is not None else out_target
    # Slice file inputs to [cfg.start_sec, cfg.end_sec] (in-memory arrays from pipeline are already sliced)
    in_start = cfg.start_sec if isinstance(in_src, (Path, str)) else None
    in_end = cfg.end_sec if isinstance(in_src, (Path, str)) else None

    # 3. Compute True-Peak levels (4x oversampled with soxr)
    in_tp = compute_true_peak(in_src, sr=sr, start_sec=in_start, end_sec=in_end)
    out_tp = compute_true_peak(out_src, sr=sr)

    # 4. Compute RMS levels in dBFS
    if in_rms is None:
        in_rms = compute_rms(in_src, sr=sr, start_sec=in_start, end_sec=in_end)
    else:
        in_rms = sanitize_float(in_rms, default=-120.0, ndigits=2)

    if out_rms is None:
        out_rms = compute_rms(out_src, sr=sr)
    else:
        out_rms = sanitize_float(out_rms, default=-120.0, ndigits=2)

    # 5. Compute Noise Suppression dB
    if suppression_db is None:
        suppression_db = sanitize_float(in_rms - out_rms, default=0.0, ndigits=2)
    else:
        suppression_db = sanitize_float(suppression_db, default=0.0, ndigits=2)

    # 6. Continuous VAD speech intervals (analyzed on input audio — documents what speech
    #    was present in the original recording; falls back to output only if input was unreadable or empty)
    in_dur = _get_audio_duration(in_src, sr=sr, start_sec=in_start, end_sec=in_end)
    if in_src is not None and in_dur > 0:
        vad_intervals = extract_vad_intervals(in_src, sr=sr, start_sec=in_start, end_sec=in_end)
    elif out_src is not None:
        vad_intervals = extract_vad_intervals(out_src, sr=sr)
    else:
        vad_intervals = []

    # 7. Compute speech summary
    total_dur = _get_audio_duration(out_src, sr=sr) or _get_audio_duration(
        in_src, sr=sr, start_sec=in_start, end_sec=in_end
    )
    speech_summary = compute_speech_summary(vad_intervals, total_dur)

    # 8. Assemble manifest dictionary matching full schema
    manifest_data = {
        "version": "1.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "noise_reduction_db": suppression_db,
        "suppression_db": suppression_db,
        "true_peak": {
            "input": in_tp,
            "output": out_tp,
        },
        "true_peak_dbfs": {
            "input": in_tp,
            "output": out_tp,
        },
        "rms": {
            "input": in_rms,
            "output": out_rms,
        },
        "rms_dbfs": {
            "input": in_rms,
            "output": out_rms,
        },
        "vad_speech_timestamps": vad_intervals,
        "speech_summary": speech_summary,
        "parameters": build_parameters_manifest(cfg),
    }

    # 9. Strict UTF-8 RFC 8259 JSON serialization
    serialized = json.dumps(manifest_data, indent=2, ensure_ascii=False)
    target_json.write_text(serialized, encoding="utf-8")
    return target_json
