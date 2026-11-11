from __future__ import annotations

import threading
from dataclasses import InitVar, dataclass
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
    # --- Existing fields (order preserved for backward compatibility) ---
    reference: Path | None = None
    role: ReferenceRole = ReferenceRole.NOISY
    main: Path | None = None
    strength: float = 0.75
    max_atten_db: float = 24.0  # >= 60 means uncapped
    output: Path | None = None
    polish: bool | None = None  # DeepFilterNet3 speech denoise; None = auto (on)

    # --- Additions for M1, M2, M3 ---
    start_sec: float | None = None
    end_sec: float | None = None
    chunk_duration: float = 60.0
    overlap_duration: float = 2.0
    audit: bool = False
    audit_path: Path | None = None
    cancel_event: threading.Event | None = None
    batch_dir: Path | None = None
    out_dir: Path | None = None

    # Pass allow_invalid_range=True to suppress the start_sec < end_sec check so
    # run_job can receive the config and emit the validation error at pipeline level.
    allow_invalid_range: InitVar[bool] = False

    def __post_init__(self, allow_invalid_range: bool = False) -> None:
        if isinstance(self.start_sec, str):
            from .audio.io import parse_time_spec
            self.start_sec = parse_time_spec(self.start_sec)
        elif self.start_sec is not None:
            self.start_sec = float(self.start_sec)

        if isinstance(self.end_sec, str):
            from .audio.io import parse_time_spec
            self.end_sec = parse_time_spec(self.end_sec)
        elif self.end_sec is not None:
            self.end_sec = float(self.end_sec)

        if self.reference is not None and isinstance(self.reference, str):
            self.reference = Path(self.reference)
        if self.main is not None and isinstance(self.main, str):
            self.main = Path(self.main)
        if self.output is not None and isinstance(self.output, str):
            self.output = Path(self.output)
        if self.batch_dir is not None and isinstance(self.batch_dir, str):
            self.batch_dir = Path(self.batch_dir)
        if self.out_dir is not None and isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        if self.audit_path is not None and isinstance(self.audit_path, str):
            self.audit_path = Path(self.audit_path)
        if isinstance(self.role, str) and not isinstance(self.role, ReferenceRole):
            self.role = ReferenceRole(self.role.upper())

        # Range and integrity validations
        if self.start_sec is not None and self.start_sec < 0:
            raise ValueError(f"start_sec cannot be negative: {self.start_sec}")
        if self.end_sec is not None and self.end_sec < 0:
            raise ValueError(f"end_sec cannot be negative: {self.end_sec}")
        if not allow_invalid_range and self.start_sec is not None and self.end_sec is not None and self.start_sec >= self.end_sec:
            raise ValueError(f"start_sec ({self.start_sec}) must be strictly less than end_sec ({self.end_sec})")
        if self.chunk_duration <= 0:
            raise ValueError(f"chunk_duration must be positive: {self.chunk_duration}")
        if self.overlap_duration < 0:
            raise ValueError(f"overlap_duration cannot be negative: {self.overlap_duration}")
        if self.overlap_duration >= self.chunk_duration:
            raise ValueError(
                f"overlap_duration ({self.overlap_duration}) must be strictly less than chunk_duration ({self.chunk_duration})"
            )

    def use_polish(self) -> bool:
        return self.polish if self.polish is not None else True

    def default_output(self) -> Path | None:
        if self.main is None:
            if self.out_dir is not None:
                return self.out_dir / "output_zout.wav"
            if self.batch_dir is not None:
                return self.batch_dir / "processed" / "output_zout.wav"
            return None
        target_dir = self.out_dir if self.out_dir is not None else self.main.parent
        return target_dir / (self.main.stem + "_zout.wav")

    def default_noise_output(self) -> Path | None:
        if self.main is None:
            if self.out_dir is not None:
                return self.out_dir / "noise_profile.wav"
            if self.batch_dir is not None:
                return self.batch_dir / "processed" / "noise_profile.wav"
            return None
        target_dir = self.out_dir if self.out_dir is not None else self.main.parent
        return target_dir / (self.main.stem + "_noise_profile.wav")

    def default_audit_path(self) -> Path | None:
        if self.audit_path is not None:
            return self.audit_path
        if self.output is not None:
            return self.output.with_name(self.output.stem + "_audit.json")
        if self.main is not None:
            out = self.default_output()
            return out.with_name(out.stem + "_audit.json") if out else None
        target_dir = self.out_dir if self.out_dir is not None else (self.batch_dir / "processed" if self.batch_dir else None)
        if target_dir is not None:
            return target_dir / "audit.json"
        return None

    def ready(self) -> bool:
        return self.reference is not None and (self.main is not None or self.batch_dir is not None)

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()
