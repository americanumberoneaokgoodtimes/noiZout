from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import soxr

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


def compute_true_peak(
    audio: np.ndarray | Path | str,
    sr: int | None = None,
    oversample: int = 4,
    chunk_sec: float = 10.0,
) -> float:
    """Compute ITU-R BS.1770 true-peak in dBFS with 4x oversampling via soxr.

    Streams in chunks to guarantee memory usage remains < 15MB even for multi-hour audio.
    Returns finite float in dBFS (clamped to -120.0 on silence or invalid signal).
    """
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if not path.is_file():
            return -120.0
        try:
            with sf.SoundFile(str(path)) as snd:
                file_sr = snd.samplerate
                chunk_frames = max(1024, int(chunk_sec * file_sr))
                global_peak_linear = 0.0
                while True:
                    chunk = snd.read(chunk_frames, dtype="float32", always_2d=True)
                    if len(chunk) == 0:
                        break
                    mono = chunk.mean(axis=1)
                    resampled = soxr.resample(mono, file_sr, file_sr * oversample, quality="HQ")
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

    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    effective_sr = sr or 48000
    chunk_frames = max(1024, int(chunk_sec * effective_sr))
    global_peak_linear = 0.0

    for i in range(0, len(mono), chunk_frames):
        chunk = mono[i : i + chunk_frames]
        if len(chunk) == 0:
            continue
        resampled = soxr.resample(chunk, effective_sr, effective_sr * oversample, quality="HQ")
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
) -> float:
    """Compute RMS level in dBFS.

    Streams in chunks to guarantee memory usage remains < 15MB.
    Returns finite float in dBFS (clamped to -120.0 on silence).
    """
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if not path.is_file():
            return -120.0
        try:
            with sf.SoundFile(str(path)) as snd:
                file_sr = snd.samplerate
                chunk_frames = max(1024, int(chunk_sec * file_sr))
                sum_sq = 0.0
                total_samples = 0
                while True:
                    chunk = snd.read(chunk_frames, dtype="float32", always_2d=True)
                    if len(chunk) == 0:
                        break
                    mono = chunk.mean(axis=1)
                    sum_sq += float(np.sum(mono.astype(np.float64) ** 2))
                    total_samples += len(mono)
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

    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if len(mono) == 0:
        return -120.0
    sum_sq = float(np.sum(mono.astype(np.float64) ** 2))
    rms_lin = math.sqrt(sum_sq / len(mono))
    if rms_lin <= 1e-6:
        return -120.0
    return sanitize_float(20.0 * math.log10(rms_lin), default=-120.0, ndigits=2)


def extract_vad_intervals(
    audio: np.ndarray | Path | str,
    sr: int | None = None,
    threshold: float = 0.10,
    min_speech_sec: float = 0.03,
    min_silence_sec: float = 0.35,
    chunk_sec: float = 10.0,
) -> list[dict[str, float]]:
    """Extract continuous speech intervals using silero_vad.

    Args:
        audio: Audio waveform as 1D/2D array or file Path.
        sr: Sample rate if audio is array.
        threshold: Speech probability threshold (default 0.10; silero_vad discriminates
                   speech harmonics from noise/hum at this level while keeping false
                   positives near zero on broadband Gaussian noise).
        min_speech_sec: Minimum duration for a valid speech interval.
        min_silence_sec: Bridge silence gaps shorter than this duration.
        chunk_sec: Processing block size in seconds.

    Returns:
        List of dicts: [{"start": float, "end": float, "duration": float}]
    """
    try:
        import torch
        import silero_vad as _svad
        model = _svad.load_silero_vad()
    except Exception:
        return []

    def _get_mono_16k(arr: np.ndarray, arr_sr: int) -> np.ndarray:
        mono = arr if arr.ndim == 1 else arr.mean(axis=1)
        if arr_sr != 16000:
            mono = soxr.resample(mono, arr_sr, 16000, quality="HQ")
        return mono.astype(np.float32)

    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if not path.is_file():
            return []
        try:
            with sf.SoundFile(str(path)) as snd:
                file_sr = snd.samplerate
                data = snd.read(dtype="float32", always_2d=True)
            mono_16k = _get_mono_16k(data, file_sr)
        except Exception:
            return []
    else:
        if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
            return []
        effective_sr = sr or 16000
        mono_16k = _get_mono_16k(audio, effective_sr)

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


def _get_audio_duration(audio: np.ndarray | Path | str | None, sr: int | None = None) -> float:
    """Helper to get audio duration in seconds safely."""
    if audio is None:
        return 0.0
    if isinstance(audio, (Path, str)):
        path = Path(audio)
        if path.is_file():
            try:
                info = sf.info(str(path))
                return float(info.duration)
            except Exception:
                return 0.0
        return 0.0
    if isinstance(audio, np.ndarray) and audio.size > 0:
        eff_sr = sr or 48000
        mono_len = len(audio) if audio.ndim == 1 else len(audio)
        return float(mono_len / eff_sr)
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

    # 3. Compute True-Peak levels (4x oversampled with soxr)
    in_tp = compute_true_peak(in_src, sr=sr)
    out_tp = compute_true_peak(out_src, sr=sr)

    # 4. Compute RMS levels in dBFS
    if in_rms is None:
        in_rms = compute_rms(in_src, sr=sr)
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
    #    was present in the original recording; falls back to output if input unavailable)
    vad_target = in_src if in_src is not None else out_src
    vad_intervals = extract_vad_intervals(vad_target, sr=sr)

    # 7. Compute speech summary
    total_dur = _get_audio_duration(out_src, sr=sr) or _get_audio_duration(in_src, sr=sr)
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
