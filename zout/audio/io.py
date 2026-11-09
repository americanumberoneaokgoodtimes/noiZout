from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from ..config import NATIVE_EXTENSIONS


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
            capture_output=True, text=True, timeout=10).stdout
        d = dict(l.split("=", 1) for l in out.split() if "=" in l)
        return AudioInfo(float(d.get("duration", 0)), int(d.get("sample_rate", 0)), int(d.get("channels", 1)))
    except Exception:
        return None


def load_audio(path: Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """Return float32 audio shaped (n,) if mono else (n, ch), and its sample rate."""
    path = Path(path)
    if path.suffix.lower() in NATIVE_EXTENSIONS:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    else:
        sr = 48000
        p = subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
             "-ar", str(sr), "-ac", "2", "-"], capture_output=True)
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError(f"ffmpeg failed to decode {path.name}: {p.stderr.decode(errors='replace')[:200]}")
        x = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2).copy()
    if mono:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


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
