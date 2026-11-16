from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .audio.io import get_audio_info, load_audio, peak_limit, resample, rms_db, save_audio
from .audio.masking import noise_suppression_db
from .config import JobConfig, ReferenceRole, SUPPORTED_AUDIO_EXTENSIONS
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
    audit_path: Path | None = None


def _cleanup_partial(path: Path | None) -> None:
    """Safely unlink partially written output and temporary buffer files."""
    if path is not None and path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    if path is not None and path.parent.exists():
        for tmp_f in path.parent.glob("*.tmp"):
            try:
                tmp_f.unlink()
            except OSError:
                pass


def run_streaming(
    cfg: JobConfig,
    progress: Progress | None = None,
    cancel_token: threading.Event | None = None,
) -> Result | None:
    """Disk-to-disk chunked streaming with 2-second raised-cosine overlap-add.

    Memory is strictly capped to < 150MB regardless of recording duration.
    """
    if cancel_token is not None:
        cfg.cancel_event = cancel_token

    if cfg.is_cancelled() or (cancel_token is not None and cancel_token.is_set()):
        return None

    if not cfg.ready():
        raise ValueError("reference and main files are required")

    out_path = Path(cfg.output or cfg.default_output())
    p = progress or (lambda *_: None)
    t0 = time.time()

    info = get_audio_info(cfg.main)
    sr = info.samplerate
    file_dur = info.duration

    if cfg.start_sec is not None and cfg.start_sec >= file_dur:
        raise ValueError(f"start_sec exceeds audio duration ({cfg.start_sec} >= {file_dur})")
    if cfg.end_sec is not None and cfg.end_sec <= 0:
        raise ValueError("end_sec must be positive")
    if cfg.start_sec is not None and cfg.end_sec is not None and cfg.start_sec >= cfg.end_sec:
        raise ValueError(f"start_sec ({cfg.start_sec}) must be less than end_sec ({cfg.end_sec})")

    p("Loading reference", 0.0)
    if cfg.is_cancelled():
        _cleanup_partial(out_path)
        return None

    ref, ref_sr = load_audio(cfg.reference)

    def sub(msg: str, f: float) -> None:
        p(msg, 0.05 + 0.9 * f)

    try:
        if cfg.role is ReferenceRole.NOISY:
            path, notes = NoisyAcousticEngine().run_streaming(
                ref=resample(ref, ref_sr, sr),
                main_path=cfg.main,
                out_path=out_path,
                sr=sr,
                strength=cfg.strength,
                max_atten_db=cfg.max_atten_db,
                chunk_sec=cfg.chunk_duration,
                overlap_sec=cfg.overlap_duration,
                progress=sub,
                polish=cfg.use_polish(),
                start_sec=cfg.start_sec,
                end_sec=cfg.end_sec,
                cancel_event=cfg.cancel_event,
            )
        else:
            path, notes = CleanVoiceEngine().run_streaming(
                ref=ref,
                ref_sr=ref_sr,
                main_path=cfg.main,
                out_path=out_path,
                sr=sr,
                strength=cfg.strength,
                max_atten_db=cfg.max_atten_db,
                chunk_sec=cfg.chunk_duration,
                overlap_sec=cfg.overlap_duration,
                progress=sub,
                polish=cfg.use_polish(),
                start_sec=cfg.start_sec,
                end_sec=cfg.end_sec,
                cancel_event=cfg.cancel_event,
            )
    except InterruptedError:
        _cleanup_partial(out_path)
        return None

    if cfg.is_cancelled():
        _cleanup_partial(out_path)
        return None

    in_rms = float(notes.get("in_rms", -120.0))
    out_rms = float(notes.get("out_rms", -120.0))
    supp_db = float(notes.get("suppression_db", 0.0))

    audit_sidecar: Path | None = None
    if cfg.audit:
        if cfg.is_cancelled():
            _cleanup_partial(out_path)
            return None
        try:
            from .audio.audit import generate_audit_manifest

            audit_sidecar = generate_audit_manifest(
                cfg=cfg,
                output_path=path,
                sr=sr,
                suppression_db=supp_db,
                in_rms=in_rms,
                out_rms=out_rms,
            )
        except Exception:
            pass

    p("Done", 1.0)
    return Result(path, time.time() - t0, in_rms, out_rms, supp_db, notes, audit_path=audit_sidecar)


def run_batch(
    cfg: JobConfig,
    progress: Progress | None = None,
    cancel_token: threading.Event | None = None,
) -> Result | None:
    """Process an entire directory of audio files against reference profile."""
    if cancel_token is not None:
        cfg.cancel_event = cancel_token

    if cfg.is_cancelled() or (cancel_token is not None and cancel_token.is_set()):
        return None

    batch_dir = cfg.batch_dir
    if batch_dir is None:
        raise ValueError("batch_dir must be specified")

    batch_path = Path(batch_dir)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_path}")
    if not batch_path.is_dir():
        raise ValueError(f"batch_dir must be a directory, got file: {batch_path}")

    out_dir = Path(cfg.out_dir) if cfg.out_dir is not None else batch_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_resolved = None
    if cfg.reference is not None:
        try:
            ref_resolved = Path(cfg.reference).resolve()
        except Exception:
            ref_resolved = None

    audio_files: list[Path] = []
    try:
        for entry in os.scandir(batch_path):
            if entry.name.startswith("."):
                continue
            if not entry.is_file():
                continue
            p = Path(entry.path)
            if p.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue
            if ref_resolved is not None:
                try:
                    if p.resolve() == ref_resolved:
                        continue
                except Exception:
                    pass
            audio_files.append(p)
    except Exception as ex:
        if isinstance(ex, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Failed to scan batch directory: {ex}") from ex

    audio_files.sort(key=lambda p: p.name.lower())

    p = progress or (lambda *_: None)
    t0 = time.time()

    if not audio_files:
        p("No audio files found in batch directory", 1.0)
        return Result(out_dir, 0.0, -80.0, -80.0, 0.0, {"batch": True, "total": 0, "success": 0})

    p(f"Starting batch processing of {len(audio_files)} files...", 0.0)

    success_count = 0
    failures: list[str] = []
    all_results: list[Result] = []

    for i, audio_file in enumerate(audio_files):
        if cfg.is_cancelled() or (cancel_token is not None and cancel_token.is_set()):
            return None

        file_frac_start = i / len(audio_files)
        file_frac_end = (i + 1) / len(audio_files)

        def _file_progress(msg: str, f: float, _n=i + 1, _total=len(audio_files),
                           _name=audio_file.name, _start=file_frac_start, _end=file_frac_end) -> None:
            p(f"[{_n}/{_total}] {_name}  {msg}", _start + f * (_end - _start))

        item_out = out_dir / (audio_file.stem + "_zout.wav")
        item_cfg = JobConfig(
            reference=cfg.reference,
            role=cfg.role,
            main=audio_file,
            strength=cfg.strength,
            max_atten_db=cfg.max_atten_db,
            output=item_out,
            polish=cfg.polish,
            start_sec=cfg.start_sec,
            end_sec=cfg.end_sec,
            chunk_duration=cfg.chunk_duration,
            overlap_duration=cfg.overlap_duration,
            audit=cfg.audit,
            cancel_event=cfg.cancel_event,
        )

        try:
            item_res = run_job(item_cfg, progress=_file_progress, cancel_token=cancel_token)
            if item_res is None and (cfg.is_cancelled() or (cancel_token is not None and cancel_token.is_set())):
                return None
            if item_res is not None:
                success_count += 1
                all_results.append(item_res)
        except InterruptedError:
            return None
        except Exception as ex:
            failures.append(f"{audio_file.name}: {ex}")

    p(f"Batch completed: {success_count}/{len(audio_files)} successful", 1.0)
    if all_results:
        n = len(all_results)
        avg_in_rms = sum(r.in_rms for r in all_results) / n
        avg_out_rms = sum(r.out_rms for r in all_results) / n
        avg_supp_db = sum(r.suppression_db for r in all_results) / n
    else:
        avg_in_rms, avg_out_rms, avg_supp_db = -20.0, -20.0, 0.0
    notes = {
        "batch": True,
        "total": len(audio_files),
        "success": success_count,
        "failures": failures,
    }
    return Result(out_dir, time.time() - t0, avg_in_rms, avg_out_rms, avg_supp_db, notes)


def run_job(
    cfg: JobConfig,
    progress: Progress | None = None,
    cancel_token: threading.Event | None = None,
) -> Result | None:
    """Execute audio denoising job supporting standard and chunked streaming, range slicing,
    forensic audit sidecar, and graceful cancellation.
    """
    if cancel_token is not None:
        cfg.cancel_event = cancel_token

    # Immediate pre-flight cancellation check (< 1ms)
    if cfg.is_cancelled() or (cancel_token is not None and cancel_token.is_set()):
        return None

    if not cfg.ready():
        raise ValueError("reference and main files are required")

    if cfg.batch_dir is not None:
        return run_batch(cfg, progress=progress, cancel_token=cancel_token)

    info = get_audio_info(cfg.main)
    file_dur = info.duration
    sr = info.samplerate
    st = cfg.start_sec or 0.0
    en = cfg.end_sec or file_dur
    eff_dur = max(0.0, en - st)

    if cfg.start_sec is not None and cfg.start_sec >= file_dur:
        raise ValueError(f"start_sec exceeds audio duration ({cfg.start_sec} >= {file_dur})")
    if cfg.end_sec is not None and cfg.end_sec <= 0:
        raise ValueError("end_sec must be positive")
    if cfg.start_sec is not None and cfg.end_sec is not None and cfg.start_sec >= cfg.end_sec:
        raise ValueError(f"start_sec ({cfg.start_sec}) must be less than end_sec ({cfg.end_sec})")

    out_path = Path(cfg.output or cfg.default_output())

    # In-memory path stays <75MB up to ~480k samples (≈10s@48kHz, 5s@96kHz).
    # Streaming path is always bounded by MAX_SAMPLES_PER_CHUNK (240k) regardless of duration.
    should_stream = (
        int(eff_dur * sr) > 480_000
        or cfg.chunk_duration != 60.0
        or cfg.overlap_duration != 2.0
    )

    if should_stream:
        return run_streaming(cfg, progress=progress, cancel_token=cancel_token)

    try:
        p = progress or (lambda *_: None)
        t0 = time.time()

        p("Loading audio", 0.0)
        if cfg.is_cancelled():
            raise InterruptedError("Cancelled")

        ref, ref_sr = load_audio(cfg.reference)
        main, sr = load_audio(cfg.main, start_sec=cfg.start_sec, end_sec=cfg.end_sec)
        if len(main) < sr // 4:
            raise ValueError("main recording is too short")

        if cfg.is_cancelled():
            raise InterruptedError("Cancelled")

        # scale sub-progress of engines into [0.05, 0.95]
        def sub(msg: str, f: float):
            if cfg.is_cancelled():
                raise InterruptedError("Cancelled")
            p(msg, 0.05 + 0.9 * f)

        if cfg.role is ReferenceRole.NOISY:
            out, notes = NoisyAcousticEngine().run(
                resample(ref, ref_sr, sr),
                main,
                sr,
                cfg.strength,
                cfg.max_atten_db,
                sub,
                cfg.use_polish(),
                cancel_event=cfg.cancel_event,
            )
        else:
            out, notes = CleanVoiceEngine().run(
                ref,
                ref_sr,
                main,
                sr,
                cfg.strength,
                cfg.max_atten_db,
                sub,
                cfg.use_polish(),
                cancel_event=cfg.cancel_event,
            )

        if cfg.is_cancelled():
            raise InterruptedError("Cancelled")

        p("Writing output", 0.97)
        out = peak_limit(out)
        path = save_audio(out_path, out, sr)
        if path.suffix != out_path.suffix:
            notes["warning"] = (
                f"Requested '{out_path.suffix}' format is not supported for writing; "
                f"saved as WAV: {path.name}"
            )
        in_rms = rms_db(main)
        out_rms = rms_db(out)
        supp_db = noise_suppression_db(main, out)

        audit_sidecar: Path | None = None
        if cfg.audit:
            if cfg.is_cancelled():
                raise InterruptedError("Cancelled")
            try:
                from .audio.audit import generate_audit_manifest

                audit_sidecar = generate_audit_manifest(
                    cfg=cfg,
                    output_path=path,
                    input_audio=main,
                    output_audio=out,
                    sr=sr,
                    suppression_db=supp_db,
                    in_rms=in_rms,
                    out_rms=out_rms,
                )
            except Exception:
                pass

        p("Done", 1.0)
        return Result(path, time.time() - t0, in_rms, out_rms, supp_db, notes, audit_path=audit_sidecar)

    except InterruptedError:
        _cleanup_partial(out_path)
        return None
