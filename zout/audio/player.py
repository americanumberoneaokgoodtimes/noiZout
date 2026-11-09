from __future__ import annotations

import threading
import time
from pathlib import Path

from .io import load_audio


class AudioPlayer:
    """Non-blocking audio player using sounddevice."""

    def __init__(self):
        self._target_name: str | None = None
        self._start_time: float = 0.0
        self._duration: float = 0.0
        self._lock = threading.Lock()

    def play(self, path: Path | str, target_name: str = "Audio") -> bool:
        """Load and start playback in background. Stops any currently playing audio."""
        self.stop()
        path = Path(path)
        if not path.is_file():
            return False

        try:
            import sounddevice as sd
            audio, sr = load_audio(path, mono=True)
            if len(audio) == 0:
                return False

            with self._lock:
                self._target_name = target_name
                self._start_time = time.time()
                self._duration = len(audio) / sr

            sd.play(audio, sr)
            return True
        except Exception:
            self.stop()
            return False

    def stop(self) -> None:
        """Stop current playback if any."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        with self._lock:
            self._target_name = None
            self._start_time = 0.0
            self._duration = 0.0

    def is_playing(self) -> bool:
        try:
            import sounddevice as sd
            stream = sd.get_stream()
            active = stream.active if stream else False
            if not active:
                with self._lock:
                    self._target_name = None
            return active
        except Exception:
            return False

    def status_text(self) -> str:
        """Return a string indicating what is currently playing, or empty string."""
        if not self.is_playing():
            return ""
        with self._lock:
            elapsed = time.time() - self._start_time
            name = self._target_name or "Audio"
            dur = self._duration
            return f"▶ Playing {name} ({elapsed:.1f}s / {dur:.1f}s) - Press Space or P/p to stop"


# Global singleton instance for the application
player = AudioPlayer()
