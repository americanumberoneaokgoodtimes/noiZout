from __future__ import annotations

import functools
import math
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import soundfile as sf
import soxr
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from ..config import NATIVE_EXTENSIONS

GLYPHS = " ▂▃▄▅▆▇█"


def format_duration(seconds: float, *, rounded: bool = False) -> str:
    """Format duration as HH:MM:SS (>= 1h) or MM:SS (< 1h)."""
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    total = max(0, int(round(seconds)) if rounded else int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class AudioInfo:
    duration: float
    samplerate: int
    channels: int
    frames: int = 0

    def short(self) -> str:
        return f"{format_duration(self.duration, rounded=True)} | {self.samplerate / 1000:g}kHz | Ch{self.channels}"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (needed for this audio format)")
    return exe


def open_ffmpeg_pcm_stream(
    path: Path | str,
    sr: int = 48000,
    channels: int = 1,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> subprocess.Popen[bytes]:
    """Open a single sequential f32le PCM stdout pipe from ffmpeg."""
    cmd = [_ffmpeg(), "-v", "error"]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", f"{float(start_sec):.6f}"])
    cmd.extend(["-i", str(Path(path))])
    if end_sec is not None:
        dur = max(0.0, float(end_sec) - float(start_sec or 0.0))
        cmd.extend(["-t", f"{dur:.6f}"])
    cmd.extend([
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(int(sr)), "-ac", str(int(channels)), "-",
    ])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def close_process_quietly(proc: subprocess.Popen[bytes] | None) -> None:
    """Safely close pipes and terminate a subprocess without raising."""
    if proc is None:
        return
    for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    try:
        running = proc.poll() is None if hasattr(proc, "poll") else True
        if running:
            if hasattr(proc, "terminate"):
                proc.terminate()
            else:
                proc.kill()
            proc.wait(timeout=2.0)
        else:
            proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=1.0)
        except Exception:
            pass


def read_pcm_frames(
    stream,
    num_frames: int,
    channels: int = 1,
    mono: bool = True,
) -> np.ndarray:
    """Read up to ``num_frames`` of float32 PCM from a binary pipe."""
    if num_frames <= 0 or stream is None:
        return np.zeros((0, channels) if not mono and channels > 1 else 0, dtype=np.float32)
    sample_width = 4 * max(1, int(channels))
    target_bytes = int(num_frames) * sample_width
    chunks: list[bytes] = []
    remaining = target_bytes
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    raw = b"".join(chunks) if len(chunks) > 1 else (chunks[0] if chunks else b"")
    valid = len(raw) - (len(raw) % sample_width)
    if valid <= 0:
        return np.zeros((0, channels) if not mono and channels > 1 else 0, dtype=np.float32)
    arr = np.frombuffer(raw[:valid], dtype=np.float32).copy()
    if channels > 1:
        arr2d = arr.reshape(-1, channels)
        return arr2d.mean(axis=1) if mono else arr2d
    return arr


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
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        if path.suffix.lower() in NATIVE_EXTENSIONS:
            i = sf.info(str(path))
            return AudioInfo(i.duration, i.samplerate, i.channels, frames=i.frames)
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels,duration:format=duration", "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=10, check=False).stdout
        d = {}
        dur = 0.0
        for l in out.split():
            if "=" in l:
                k, v = l.split("=", 1)
                d[k] = v
                if k == "duration" and v.strip() not in ("N/A", ""):
                    try:
                        parsed_dur = float(v.strip())
                        if parsed_dur > 0:
                            dur = parsed_dur
                    except ValueError:
                        pass
        # load_audio decodes non-native formats via ffmpeg at 48 kHz (-ar 48000),
        # so report 48000 as the effective rate rather than the native codec rate.
        sr = 48000
        return AudioInfo(dur, sr, int(d.get("channels", 1)), frames=int(round(dur * sr)))
    except Exception:
        return None


def get_audio_info(path: Path | str) -> AudioInfo:
    """Return AudioInfo for both native soundfile formats and ffmpeg-backed formats."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"Audio file not found: {p}")
    if p.stat().st_size == 0:
        raise RuntimeError(f"Audio file is empty: {p}")
    if p.suffix.lower() in NATIVE_EXTENSIONS:
        i = sf.info(str(p))
        return AudioInfo(i.duration, i.samplerate, i.channels, frames=i.frames)
    info = probe_audio(p)
    if info is None or info.samplerate <= 0 or info.duration <= 0:
        raise RuntimeError(f"Failed to probe audio metadata for {p.name} (ensure ffprobe/ffmpeg is installed)")
    return info


def iter_audio_chunks(
    path: Path | str,
    chunk_sec: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    mono: bool = True,
    cancel_event: threading.Event | None = None,
    _get_info=None,
) -> Iterator[tuple[np.ndarray, int]]:
    """Stream float32 chunks within [start_sec, end_sec] using SoundFile or 1 ffmpeg pipe."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return
    if p.suffix.lower() in NATIVE_EXTENSIONS:
        with sf.SoundFile(str(p)) as snd:
            file_sr = snd.samplerate
            total_frames = snd.frames
            start_f = max(0, int(round((start_sec or 0.0) * file_sr)))
            end_f = min(total_frames, int(round(end_sec * file_sr))) if end_sec is not None else total_frames
            if start_f >= end_f:
                return
            snd.seek(start_f)
            chunk_frames = max(1024, int(chunk_sec * file_sr))
            pos = start_f
            while pos < end_f:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                chunk = snd.read(min(chunk_frames, end_f - pos), dtype="float32", always_2d=True)
                if len(chunk) == 0:
                    break
                pos += len(chunk)
                if mono:
                    yield chunk.mean(axis=1), file_sr
                else:
                    yield (chunk if chunk.shape[1] > 1 else chunk[:, 0]), file_sr
    else:
        info = (_get_info or get_audio_info)(p)
        file_sr = info.samplerate
        t_start = max(0.0, float(start_sec or 0.0))
        t_end = min(info.duration, float(end_sec)) if end_sec is not None else info.duration
        if t_start >= t_end:
            return
        channels = min(2, max(1, int(getattr(info, "channels", 2) or 1)))
        chunk_frames = max(1024, int(chunk_sec * file_sr))
        proc = open_ffmpeg_pcm_stream(p, sr=file_sr, channels=channels, start_sec=t_start, end_sec=t_end)
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                chunk = read_pcm_frames(proc.stdout, chunk_frames, channels=channels, mono=mono)
                if len(chunk) == 0:
                    break
                yield chunk, file_sr
        finally:
            close_process_quietly(proc)


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
            # Non-native fallback: sample first 15 seconds to keep picker scrolling responsive
            audio, _ = load_audio(path, mono=True, start_sec=0.0, end_sec=15.0)
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


WRITABLE_EXTENSIONS = (".wav", ".flac", ".ogg")


def resolve_writable_output_path(path: Path | str) -> Path:
    """Resolve a destination path to a SoundFile-writable audio extension."""
    p = Path(path)
    if p.suffix.lower() not in WRITABLE_EXTENSIONS:
        p = p.with_suffix(".wav")
    return p


def save_audio(path: Path, x: np.ndarray, sr: int) -> Path:
    path = resolve_writable_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_24" if path.suffix.lower() == ".wav" else None)
    return path


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-10))


def peak_limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    ceil = 10 ** (ceiling_db / 20)
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (ceil / p) if p > ceil else x


def local_peak_limit(
    x: np.ndarray,
    ceiling_db: float = -1.0,
    window_samples: int = 1024,
) -> np.ndarray:
    """Apply a smoothed local sliding-window peak limiter without attenuating the entire chunk.

    Returns ``x`` unmodified when ``max(abs(x)) <= ceiling``.
    """
    ceil = float(10 ** (ceiling_db / 20.0))
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= ceil:
        return x
    win = max(64, int(window_samples))
    local_env = maximum_filter1d(np.abs(x).astype(np.float32), size=win, mode="nearest")
    gain = np.where(local_env > ceil, ceil / np.maximum(local_env, 1e-12), 1.0).astype(np.float32)
    smooth_gain = uniform_filter1d(gain, size=max(16, win // 4), mode="nearest")
    effective_gain = np.minimum(gain, smooth_gain)
    limited = (x * effective_gain).astype(np.float32)
    return np.clip(limited, -ceil, ceil)


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
