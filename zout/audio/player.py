from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from .io import load_audio, resample


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    """Ensure audio array is 2D float32 stereo shaped (n, 2)."""
    if x.ndim == 1:
        return np.column_stack((x, x)).astype(np.float32)
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return np.column_stack((x[:, 0], x[:, 0])).astype(np.float32)
        elif x.shape[1] >= 2:
            return np.ascontiguousarray(x[:, :2], dtype=np.float32)
    raise ValueError(f"Unsupported audio shape: {x.shape}")


def _calc_dbfs(val: float, floor: float = -80.0, ceiling: float = 0.0) -> float:
    """Convert linear amplitude [0, 1+] to dBFS bounded within [floor, ceiling]."""
    if not math.isfinite(val):
        if math.isinf(val) and val > 0:
            return ceiling
        return floor
    if val <= 1e-4:  # <= -80 dBFS
        return floor
    db = float(20.0 * np.log10(val))
    return min(ceiling, max(floor, db))


def _calc_meters(block: np.ndarray) -> tuple[float, float, float, float]:
    """Compute instantaneous stereo RMS and Peak dBFS for a block of (N, 2) samples."""
    if len(block) == 0:
        return -80.0, -80.0, -80.0, -80.0
    if not np.all(np.isfinite(block)):
        block = np.nan_to_num(block, nan=0.0, posinf=1.0, neginf=-1.0)
    l, r = block[:, 0], block[:, 1]
    rms_l = float(np.sqrt(np.mean(np.square(l, dtype=np.float64))))
    rms_r = float(np.sqrt(np.mean(np.square(r, dtype=np.float64))))
    peak_l = float(np.max(np.abs(l)))
    peak_r = float(np.max(np.abs(r)))
    return (_calc_dbfs(rms_l), _calc_dbfs(rms_r), _calc_dbfs(peak_l), _calc_dbfs(peak_r))


class _PlaybackFinished(Exception):
    """Raised when playback reaches EOF to cleanly terminate HeadlessRunner or audio streams."""


class HeadlessRunner:
    """Simulates real-time audio playback when no hardware audio output device is available."""

    def __init__(self, callback, sr: int, blocksize: int = 1024, finished_callback=None):
        self._callback = callback
        self._sr = max(1, sr)
        self._blocksize = blocksize
        self._finished_callback = finished_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        outdata = np.zeros((self._blocksize, 2), dtype=np.float32)
        interval = self._blocksize / self._sr
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                self._callback(outdata, self._blocksize, None, None)
            except (_PlaybackFinished, Exception):
                break

            next_tick += interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                if self._stop_event.wait(timeout=sleep_time):
                    break
            else:
                next_tick = time.perf_counter()

        if self._finished_callback and not self._stop_event.is_set():
            try:
                self._finished_callback()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)


class AudioPlayer:
    """Continuous audio player with instantaneous A/B hot-swap and stereo VU metering."""

    def __init__(self, force_headless: bool = False):
        self._lock = threading.Lock()
        self._stream = None
        self._headless_runner: HeadlessRunner | None = None
        self._force_headless = force_headless
        self._is_headless = False

        self._audio_a: np.ndarray | None = None
        self._audio_b: np.ndarray | None = None
        self._sr: int = 44100
        self._total_frames: int = 0
        self._sample_index: int = 0
        self._active_track: str = "B"  # "A" or "B"
        self._is_playing: bool = False
        self._meter_levels: tuple[tuple[float, float], tuple[float, float]] = (
            (-80.0, -80.0),
            (-80.0, -80.0),
        )

        self._orig_path: Path | None = None
        self._proc_path: Path | None = None
        self._target_name: str | None = None
        self._start_time: float = 0.0
        self._duration: float = 0.0

    def play_pair(
        self,
        original_path: Path | str,
        processed_path: Path | str,
        start_with: str = "B",
        orig_start_sec: float | None = None,
        orig_end_sec: float | None = None,
        max_duration_sec: float | None = None,
    ) -> bool:
        """Load and start playback of an A/B pair. Starts on track A or B.

        Supports optional original-track time slicing (to match range-sliced output)
        and ``max_duration_sec`` preview capping for multi-hour recordings.
        """
        self.stop()
        p_orig = Path(original_path)
        p_proc = Path(processed_path)
        if not p_orig.is_file() or not p_proc.is_file():
            return False

        try:
            eff_orig_end = orig_end_sec
            eff_proc_end = max_duration_sec
            if max_duration_sec is not None:
                cap_orig_end = (orig_start_sec or 0.0) + max_duration_sec
                eff_orig_end = min(eff_orig_end, cap_orig_end) if eff_orig_end is not None else cap_orig_end

            if p_orig == p_proc:
                a_orig, sr_orig = load_audio(p_orig, mono=False, start_sec=orig_start_sec, end_sec=eff_orig_end)
                a_proc, sr_proc = a_orig.copy(), sr_orig
            else:
                a_orig, sr_orig = load_audio(p_orig, mono=False, start_sec=orig_start_sec, end_sec=eff_orig_end)
                a_proc, sr_proc = load_audio(p_proc, mono=False, start_sec=None, end_sec=eff_proc_end)

            if len(a_orig) == 0 or len(a_proc) == 0:
                return False

            a_orig = _ensure_stereo(a_orig)
            a_proc = _ensure_stereo(a_proc)

            target_sr = sr_proc
            if sr_orig != target_sr:
                a_orig = resample(a_orig, sr_orig, target_sr)
            if sr_proc != target_sr:
                a_proc = resample(a_proc, sr_proc, target_sr)

            max_len = max(len(a_orig), len(a_proc))
            if len(a_orig) < max_len:
                a_orig = np.pad(a_orig, ((0, max_len - len(a_orig)), (0, 0)))
            if len(a_proc) < max_len:
                a_proc = np.pad(a_proc, ((0, max_len - len(a_proc)), (0, 0)))

            start_track = start_with.upper() if start_with.upper() in ("A", "B") else "B"

            with self._lock:
                self._audio_a = a_orig
                self._audio_b = a_proc
                self._sr = target_sr
                self._total_frames = max_len
                self._sample_index = 0
                self._active_track = start_track
                self._orig_path = p_orig
                self._proc_path = p_proc
                self._duration = max_len / target_sr
                self._start_time = time.time()
                self._target_name = f"A/B Pair ({start_track})"
                self._meter_levels = ((-80.0, -80.0), (-80.0, -80.0))
                self._is_playing = True

            started = False
            if not self._force_headless:
                try:
                    import sounddevice as sd

                    self._stream = sd.OutputStream(
                        samplerate=self._sr,
                        channels=2,
                        dtype="float32",
                        blocksize=1024,
                        callback=self._audio_callback,
                        finished_callback=self._stream_finished,
                    )
                    self._stream.start()
                    self._is_headless = False
                    started = True
                except Exception:
                    self._stream = None

            if not started:
                self._is_headless = True
                self._headless_runner = HeadlessRunner(
                    callback=self._audio_callback,
                    sr=self._sr,
                    blocksize=1024,
                    finished_callback=self._stream_finished,
                )
                self._headless_runner.start()

            return True
        except Exception:
            self.stop()
            return False

    def play(
        self,
        path: Path | str,
        target_name: str = "Audio",
        start_sec: float | None = None,
        end_sec: float | None = None,
        max_duration_sec: float | None = None,
    ) -> bool:
        """Backward compatible playback wrapper for single files."""
        ok = self.play_pair(
            path, path, start_with="A",
            orig_start_sec=start_sec, orig_end_sec=end_sec, max_duration_sec=max_duration_sec,
        )
        if ok:
            with self._lock:
                self._target_name = target_name
        return ok

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        stop_after = False
        with self._lock:
            if not self._is_playing:
                outdata.fill(0)
                self._meter_levels = ((-80.0, -80.0), (-80.0, -80.0))
                stop_after = True
            elif self._audio_a is None or self._audio_b is None:
                outdata.fill(0)
                self._is_playing = False
                self._meter_levels = ((-80.0, -80.0), (-80.0, -80.0))
                stop_after = True
            else:
                audio = self._audio_a if self._active_track == "A" else self._audio_b
                start = self._sample_index
                total = len(audio)
                if start >= total:
                    outdata.fill(0)
                    self._is_playing = False
                    self._meter_levels = ((-80.0, -80.0), (-80.0, -80.0))
                    stop_after = True
                else:
                    end = start + frames
                    if end <= total:
                        block = audio[start:end]
                        outdata[:] = block
                        self._sample_index = end
                    else:
                        valid = total - start
                        block = audio[start:total]
                        outdata[:valid] = block
                        outdata[valid:].fill(0)
                        self._sample_index = total
                        self._is_playing = False
                        stop_after = True

                    rms_l, rms_r, peak_l, peak_r = _calc_meters(block)
                    self._meter_levels = ((rms_l, rms_r), (peak_l, peak_r))

        if stop_after:
            with self._lock:
                self._is_playing = False
                self._headless_runner = None
            try:
                import sounddevice as sd

                raise sd.CallbackStop
            except ImportError:
                raise _PlaybackFinished

    def _stream_finished(self) -> None:
        with self._lock:
            self._is_playing = False
            self._meter_levels = ((-80.0, -80.0), (-80.0, -80.0))
            self._headless_runner = None

    def toggle_ab(self) -> str:
        """Seamlessly toggle between Track A (original) and B (processed) without touching sample_index."""
        with self._lock:
            self._active_track = "A" if self._active_track == "B" else "B"
            if self._target_name and self._target_name.startswith("A/B Pair"):
                self._target_name = f"A/B Pair ({self._active_track})"
            return self._active_track

    def get_current_track(self) -> str:
        """Return the active track identifier ('A' or 'B')."""
        with self._lock:
            return self._active_track

    def get_meter_levels(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return instantaneous ((rms_l, rms_r), (peak_l, peak_r)) in dBFS."""
        with self._lock:
            return self._meter_levels

    def is_playing(self) -> bool:
        """Check if audio playback is currently active."""
        with self._lock:
            if not self._is_playing:
                return False
            if self._stream is not None:
                try:
                    if not self._stream.active:
                        self._is_playing = False
                except Exception:
                    self._is_playing = False
            elif self._headless_runner is not None and (
                self._headless_runner._stop_event.is_set()
                or (
                    self._headless_runner._thread is not None
                    and not self._headless_runner._thread.is_alive()
                )
            ):
                self._is_playing = False
            return self._is_playing

    def stop(self) -> None:
        """Stop playback and release stream resources safely outside the lock."""
        stream_to_stop = None
        runner_to_stop = None
        with self._lock:
            self._is_playing = False
            self._meter_levels = ((-80.0, -80.0), (-80.0, -80.0))
            self._target_name = None
            stream_to_stop = self._stream
            runner_to_stop = self._headless_runner
            self._stream = None
            self._headless_runner = None

        if stream_to_stop is not None:
            try:
                stream_to_stop.stop()
                stream_to_stop.close()
            except Exception:
                pass
        if runner_to_stop is not None:
            try:
                runner_to_stop.stop()
            except Exception:
                pass

    def status_text(self) -> str:
        """Return playback status string for TUI display."""
        if not self.is_playing():
            return ""
        with self._lock:
            dur = self._duration
            if self._audio_a is not None and self._sr > 0 and self._sample_index > 0:
                raw_elapsed = self._sample_index / float(self._sr)
            else:
                raw_elapsed = time.time() - self._start_time
            elapsed = min(dur, max(0.0, raw_elapsed)) if dur > 0 else max(0.0, raw_elapsed)
            if self._orig_path and self._proc_path and self._orig_path != self._proc_path:
                trk = self._active_track
                desc = "Original Raw" if trk == "A" else "Processed Output"
                return (
                    f"▶ Playing [{trk}: {desc}] ({elapsed:.1f}s / {dur:.1f}s) - "
                    "[Tab] A/B Swap | [Space] Stop"
                )
            name = self._target_name or "Audio"
            return f"▶ Playing {name} ({elapsed:.1f}s / {dur:.1f}s) - Press Space or P/p to stop"

    @property
    def sample_index(self) -> int:
        with self._lock:
            return self._sample_index

    @property
    def duration(self) -> float:
        with self._lock:
            return self._duration


# Global singleton instance for the application
player = AudioPlayer()
