from __future__ import annotations

import functools
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from ..config import NATIVE_EXTENSIONS

GLYPHS = " ▂▃▄▅▆▇█"


@dataclass
class AudioInfo:
    duration: float
    samplerate: int
    channels: int

    def short(self) -> str:
        m, s = divmod(int(round(self.duration)), 60)
        return f"{m:02d}:{s:02d} | {self.samplerate / 1000:g}kHz | Ch{self.channels}"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (needed for this audio format)")
    return exe


def parse_time_spec(val: str | float | int | None) -> float | None:  # noqa: PYI041
    """Parse time specification into seconds as float.

    Accepts:
        - None or empty string -> None
        - int or float -> seconds (e.g. 45, 90.5)
        - str:
            - seconds: "45", "45.5"
            - mm:ss: "01:30", "1:30", "0:45.5"
            - hh:mm:ss: "01:15:30", "1:00:00"

    Raises:
        ValueError: if format is invalid, components are negative or out of bounds.
        TypeError: if type is unsupported.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        if not math.isfinite(v):
            raise ValueError(f"Time value must be finite: {val}")
        if v < 0 or math.copysign(1.0, v) < 0.0:
            raise ValueError(f"Time value cannot be negative: {val}")
        return v
    if not isinstance(val, str):
        raise TypeError(f"Expected str, float, int, or None, got: {type(val).__name__}")

    # Reject non-space whitespace like newlines, tabs, and null bytes
    if any(c in val for c in ("\n", "\r", "\t", "\x00")):
        raise ValueError(f"Malformed time spec: {val!r}")

    s = val.strip()
    if not s:
        raise ValueError("Empty time specification")

    if s.lower() in ("nan", "inf", "-inf", "+inf"):
        raise ValueError(f"Time value must be finite: {val}")

    if s.startswith("-"):
        raise ValueError(f"Time value cannot be negative: {val}")

    if ":" in s:
        if "e" in s.lower():
            raise ValueError(f"Scientific notation not permitted in clock format: {val}")
        parts = s.split(":")
        if any(p.strip().startswith("-") for p in parts):
            raise ValueError(f"Time components cannot be negative: {val}")
        if len(parts) == 2:
            try:
                m = float(parts[0])
                sec = float(parts[1])
            except ValueError:
                raise ValueError(f"Malformed time spec: {val}")
            if not (math.isfinite(m) and math.isfinite(sec)):
                raise ValueError(f"Time components must be finite: {val}")
            if m < 0 or sec < 0 or math.copysign(1.0, m) < 0.0 or math.copysign(1.0, sec) < 0.0:
                raise ValueError(f"Time components cannot be negative: {val}")
            if sec >= 60.0:
                raise ValueError(f"Seconds component must be < 60 in mm:ss format: {val}")
            return m * 60.0 + sec
        elif len(parts) == 3:
            try:
                h = float(parts[0])
                m = float(parts[1])
                sec = float(parts[2])
            except ValueError:
                raise ValueError(f"Malformed time spec: {val}")
            if not (math.isfinite(h) and math.isfinite(m) and math.isfinite(sec)):
                raise ValueError(f"Time components must be finite: {val}")
            if h < 0 or m < 0 or sec < 0 or math.copysign(1.0, h) < 0.0 or math.copysign(1.0, m) < 0.0 or math.copysign(1.0, sec) < 0.0:
                raise ValueError(f"Time components cannot be negative: {val}")
            if m >= 60.0:
                raise ValueError(f"Minutes component must be < 60 in hh:mm:ss format: {val}")
            if sec >= 60.0:
                raise ValueError(f"Seconds component must be < 60 in hh:mm:ss format: {val}")
            return h * 3600.0 + m * 60.0 + sec
        else:
            raise ValueError(f"Invalid time format (expected mm:ss or hh:mm:ss): {val}")
    else:
        try:
            v = float(s)
        except ValueError:
            raise ValueError(f"Malformed time spec: {val}")
        if not math.isfinite(v):
            raise ValueError(f"Time value must be finite: {val}")
        if v < 0 or math.copysign(1.0, v) < 0.0:
            raise ValueError(f"Time value cannot be negative: {val}")
        return v


def probe_audio(path: Path) -> AudioInfo | None:
    try:
        if path.suffix.lower() in NATIVE_EXTENSIONS:
            i = sf.info(str(path))
            return AudioInfo(i.duration, i.samplerate, i.channels)
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels:format=duration", "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=10, check=False).stdout
        d = dict(l.split("=", 1) for l in out.split() if "=" in l)
        return AudioInfo(float(d.get("duration", 0)), int(d.get("sample_rate", 0)), int(d.get("channels", 1)))
    except Exception:
        return None


def load_audio(
    path: Path | str,
    mono: bool = True,
    start_sec: float | str | None = None,
    end_sec: float | str | None = None,
) -> tuple[np.ndarray, int]:
    """Return float32 audio shaped (n,) if mono else (n, ch), and its sample rate.

    Supports native seeking via sf.read(start=..., stop=...) without decoding unneeded audio.
    """
    path = Path(path)
    start_s = parse_time_spec(start_sec)
    end_s = parse_time_spec(end_sec)

    if start_s is not None and start_s < 0:
        raise ValueError(f"start_sec cannot be negative: {start_s}")
    if end_s is not None and end_s < 0:
        raise ValueError(f"end_sec cannot be negative: {end_s}")
    if start_s is not None and end_s is not None and start_s >= end_s:
        raise ValueError(f"start_sec ({start_s}) must be strictly less than end_sec ({end_s})")

    if path.suffix.lower() in NATIVE_EXTENSIONS:
        if start_s is None and end_s is None:
            x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        else:
            info = sf.info(str(path))
            sr = info.samplerate
            total_frames = info.frames
            start_frame = max(0, round(start_s * sr)) if start_s is not None else 0
            stop_frame = min(total_frames, round(end_s * sr)) if end_s is not None else total_frames

            if start_frame >= stop_frame or start_frame >= total_frames:
                x = np.zeros((0, info.channels), dtype=np.float32)
            else:
                x, sr = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=True)
    else:
        sr = 48000
        cmd = [_ffmpeg(), "-v", "error"]
        if start_s is not None:
            cmd.extend(["-ss", f"{start_s:.6f}"])
        cmd.extend(["-i", str(path)])
        if end_s is not None:
            dur = end_s - (start_s or 0.0)
            cmd.extend(["-t", f"{dur:.6f}"])
        cmd.extend([
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(sr), "-ac", "2", "-"
        ])
        p = subprocess.run(cmd, capture_output=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode {path.name}: {p.stderr.decode(errors='replace')[:200]}")
        if len(p.stdout) == 0:
            x = np.zeros((0, 2), dtype=np.float32)
        else:
            valid_bytes = len(p.stdout) - (len(p.stdout) % 8)
            x = np.frombuffer(p.stdout[:valid_bytes], dtype=np.float32).reshape(-1, 2).copy()

    if mono:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def compute_waveform_envelope(audio: np.ndarray | Path | str, width: int = 16) -> str:
    """Compute an 8-level mini Unicode waveform envelope (' ▂▃▄▅▆▇█') with peak normalization.

    Args:
        audio: Audio waveform as 1D/2D numpy array, or a Path/str to an audio file.
        width: Number of characters to generate (default: 16).

    Returns:
        String of length `width` depicting the volume dynamics envelope.
    """
    if width <= 0:
        return ""
    if isinstance(audio, (Path, str)):
        return get_file_waveform_envelope(audio, width=width)
    if audio is None or len(audio) == 0:
        return " " * width

    n = len(audio)
    step = n / width
    window_energies = np.zeros(width, dtype=np.float32)

    for i in range(width):
        st = int(i * step)
        en = int((i + 1) * step)
        chunk = audio[st:en]
        if len(chunk) == 0:
            window_energies[i] = 0.0
        else:
            window_energies[i] = float(np.sqrt(np.mean(chunk ** 2)))

    peak = float(np.max(window_energies))
    if peak < 1e-6:
        return " " * width
    norm = window_energies / peak
    indices = np.clip(np.round(norm * 7).astype(int), 0, 7)
    return "".join(GLYPHS[i] for i in indices)


@functools.lru_cache(maxsize=256)
def _cached_file_waveform(path_str: str, mtime: float, width: int) -> str:
    path = Path(path_str)
    try:
        if path.suffix.lower() in NATIVE_EXTENSIONS:
            info = sf.info(str(path))
            if info.frames <= 0:
                return " " * width
            step = info.frames / width
            chunk_read = min(4096, max(1, int(step)))
            window_energies = np.zeros(width, dtype=np.float32)
            for i in range(width):
                st = int(i * step)
                data, _ = sf.read(str(path), start=st, frames=chunk_read, dtype="float32", always_2d=True)
                window_energies[i] = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
            peak = float(np.max(window_energies))
            if peak < 1e-6:
                return " " * width
            norm = window_energies / peak
            indices = np.clip(np.round(norm * 7).astype(int), 0, 7)
            return "".join(GLYPHS[i] for i in indices)
        else:
            # Non-native fallback: read first 60 seconds
            audio, _ = load_audio(path, mono=True, start_sec=0.0, end_sec=60.0)
            return compute_waveform_envelope(audio, width=width)
    except Exception:
        return " " * width


def get_file_waveform_envelope(path: Path | str, width: int = 16) -> str:
    """Cached preview envelope helper for audio files."""
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return " " * max(0, width)
    return _cached_file_waveform(str(p.resolve()), mtime, width)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)


def save_audio(path: Path, x: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in (".wav", ".flac", ".ogg"):
        path = path.with_suffix(".wav")
    sf.write(str(path), x, sr, subtype="PCM_24" if path.suffix == ".wav" else None)
    return path


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-10))


def peak_limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    ceil = 10 ** (ceiling_db / 20)
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (ceil / p) if p > ceil else x


def prepare_clean_enrollment(x: np.ndarray, sr: int, min_sec: float = 3.2) -> tuple[np.ndarray, float]:
    """Keep voiced frames of a clean clip; tile with crossfades up to min_sec.

    Returns (audio, voiced_seconds). VoxLock needs >= 3 s of the speaker.
    """
    hop = int(sr * 0.02)
    n = len(x) // hop
    if n == 0:
        return x, 0.0
    frames = x[: n * hop].reshape(n, hop)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    floor = np.percentile(e, 10)
    keep = e > max(floor + 10, -60)
    voiced = frames[keep].reshape(-1) if keep.any() else x
    voiced_sec = len(voiced) / sr
    need = int(min_sec * sr)
    if len(voiced) >= need:
        return voiced.astype(np.float32), voiced_sec
    xf = int(0.03 * sr)
    fade_in = np.linspace(0, 1, xf, dtype=np.float32)
    out = voiced.astype(np.float32).copy()
    if len(out) <= 2 * xf:
        out = np.pad(out, (0, 2 * xf + 1 - len(out)))
    while len(out) < need:
        nxt = voiced.astype(np.float32) if len(voiced) > 2 * xf else out[: 2 * xf + 1]
        out[-xf:] = out[-xf:] * fade_in[::-1] + nxt[:xf] * fade_in
        out = np.concatenate([out, nxt[xf:]])
    return out, voiced_sec
