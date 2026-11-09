from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .audio.io import load_audio, peak_limit, resample, rms_db, save_audio
from .audio.masking import noise_suppression_db
from .config import JobConfig, ReferenceRole
from .engines.clean_engine import CleanVoiceEngine
from .engines.noisy_engine import NoisyAcousticEngine

Progress = Callable[[str, float], None]


@dataclass
class Result:
    output: Path
    seconds: float
    in_rms: float
    out_rms: float
    suppression_db: float
    notes: dict = field(default_factory=dict)


def run_job(cfg: JobConfig, progress: Progress | None = None) -> Result:
    if not cfg.ready():
        raise ValueError("reference and main files are required")
    p = progress or (lambda *_: None)
    t0 = time.time()

    p("Loading audio", 0.0)
    ref, ref_sr = load_audio(cfg.reference)
    main, sr = load_audio(cfg.main)
    if len(main) < sr // 4:
        raise ValueError("main recording is too short")

    # scale sub-progress of engines into [0.05, 0.95]
    def sub(msg: str, f: float):
        p(msg, 0.05 + 0.9 * f)

    notes: dict = {}
    if cfg.role is ReferenceRole.NOISY:
        out = NoisyAcousticEngine().run(resample(ref, ref_sr, sr), main, sr,
                                        cfg.strength, cfg.max_atten_db, sub, cfg.use_polish())
    else:
        out, notes = CleanVoiceEngine().run(ref, ref_sr, main, sr,
                                            cfg.strength, cfg.max_atten_db, sub, cfg.use_polish())

    p("Writing output", 0.97)
    out = peak_limit(out)
    path = save_audio(cfg.output or cfg.default_output(), out, sr)
    p("Done", 1.0)
    return Result(path, time.time() - t0, rms_db(main), rms_db(out),
                  noise_suppression_db(main, out), notes)
