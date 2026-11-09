from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".aiff", ".aif",
    ".mp3", ".m4a", ".aac", ".opus", ".wma", ".webm", ".mp4",
}
# Formats libsndfile reads natively; the rest go through ffmpeg.
NATIVE_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


class ReferenceRole(str, Enum):
    NOISY = "NOISY"
    CLEAN = "CLEAN"


@dataclass
class JobConfig:
    reference: Path | None = None
    role: ReferenceRole = ReferenceRole.NOISY
    main: Path | None = None
    strength: float = 0.75
    max_atten_db: float = 24.0  # >= 60 means uncapped
    output: Path | None = None
    polish: bool | None = None  # DeepFilterNet3 speech denoise; None = auto (on)

    def use_polish(self) -> bool:
        return self.polish if self.polish is not None else True

    def default_output(self) -> Path:
        assert self.main is not None
        return self.main.with_name(self.main.stem + "_zout.wav")

    def default_noise_output(self) -> Path:
        assert self.main is not None
        return self.main.with_name(self.main.stem + "_noise_profile.wav")

    def ready(self) -> bool:
        return self.reference is not None and self.main is not None
