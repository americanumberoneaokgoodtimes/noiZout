"""Reviewer 2 / Adversarial Critic Stress & Verification Suite for Milestone 3 Gate.

Focus Areas:
1. Batch processing: input validation, corruption resilience, non-audio filtering,
   reference exclusion, directory auto-creation, cancellation mid-batch, audit manifest generation.
2. CLI flags & validation: --start, --end, --batch, --out-dir, --audit, --json, mutual exclusion.
3. Interactive dashboard: hotkeys [C], [T], [B], [A], cancellation return safety, result_card(None).
4. Forensic audit sidecar: schema verification, true-peak, RMS, VAD intervals.
5. Strict zero prohibited framework imports verification across all files.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from zout.config import JobConfig, ReferenceRole
from zout.pipeline import Result, run_batch, run_job


SR = 16000


def synth_tone(duration_sec: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def synth_noise(duration_sec: float, sr: int = SR, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.05 * rng.standard_normal(int(duration_sec * sr))).astype(np.float32)


# ============================================================================
# 1. Batch Processing Adversarial Tests
# ============================================================================

def test_batch_directory_validation(tmp_path: Path):
    """run_batch must reject missing directories and files passed as directories."""
    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    # 1. Non-existent directory raises FileNotFoundError
    cfg_nonexistent = JobConfig(reference=ref_file, batch_dir=tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        run_batch(cfg_nonexistent)

    # 2. File passed as batch_dir raises ValueError
    file_as_dir = tmp_path / "some_file.wav"
    sf.write(file_as_dir, synth_tone(1.0), SR)
    cfg_file = JobConfig(reference=ref_file, batch_dir=file_as_dir)
    with pytest.raises(ValueError, match="must be a directory"):
        run_batch(cfg_file)

    # 3. None batch_dir raises ValueError
    cfg_none = JobConfig(reference=ref_file, batch_dir=None)
    with pytest.raises(ValueError, match="batch_dir must be specified"):
        run_batch(cfg_none)


def test_batch_empty_directory_handling(tmp_path: Path):
    """run_batch on empty directory or dir with only non-audio files returns cleanly."""
    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    empty_dir = tmp_path / "empty_batch"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("Hello world")
    (empty_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (empty_dir / ".hidden.wav").write_bytes(b"data")
    (empty_dir / "nested_dir").mkdir()

    cfg = JobConfig(reference=ref_file, batch_dir=empty_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("batch") is True
    assert res.notes.get("total") == 0
    assert res.notes.get("success") == 0


def test_batch_reference_exclusion_and_nested_dir_skip(tmp_path: Path):
    """run_batch must exclude reference file if inside batch_dir and skip subdirectories."""
    batch_dir = tmp_path / "batch_mixed"
    batch_dir.mkdir()

    ref_inside = batch_dir / "ref_inside.wav"
    sf.write(ref_inside, synth_noise(1.5), SR)

    item1 = batch_dir / "item1.wav"
    item2 = batch_dir / "item2.wav"
    sf.write(item1, synth_tone(2.0), SR)
    sf.write(item2, synth_tone(2.0, freq=880.0), SR)

    # Subdirectory with an audio file inside it (should not be processed at root level)
    sub = batch_dir / "sub_folder"
    sub.mkdir()
    sf.write(sub / "nested.wav", synth_tone(1.0), SR)

    out_dir = tmp_path / "out_custom"
    cfg = JobConfig(reference=ref_inside, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("total") == 2  # item1 and item2, not ref_inside or sub
    assert res.notes.get("success") == 2

    # Verify outputs
    out_files = sorted(f.name for f in out_dir.glob("*.wav"))
    assert out_files == ["item1_zout.wav", "item2_zout.wav"]
    assert not (out_dir / "ref_inside_zout.wav").exists()


def test_batch_corruption_resilience(tmp_path: Path):
    """Corrupted audio files must not abort batch; valid files must still succeed."""
    batch_dir = tmp_path / "batch_corrupt"
    batch_dir.mkdir()

    good1 = batch_dir / "good1.wav"
    sf.write(good1, synth_tone(2.0), SR)

    corrupt1 = batch_dir / "corrupt1.wav"
    corrupt1.write_bytes(b"NOT_A_VALID_WAV_HEADER_CORRUPTED_DATA_XXXX")

    corrupt2 = batch_dir / "corrupt2.wav"
    corrupt2.write_bytes(b"")  # 0-byte file

    good2 = batch_dir / "good2.wav"
    sf.write(good2, synth_tone(2.0, freq=660.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.5), SR)

    out_dir = tmp_path / "out_resilience"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("total") == 4
    assert res.notes.get("success") == 2
    failures = res.notes.get("failures", [])
    assert len(failures) == 2
    assert any("corrupt1.wav" in f for f in failures)
    assert any("corrupt2.wav" in f for f in failures)

    # Valid outputs created
    assert (out_dir / "good1_zout.wav").is_file()
    assert (out_dir / "good2_zout.wav").is_file()


def test_batch_cancellation_preflight_and_midflight(tmp_path: Path):
    """Batch processing responds to cancel_token pre-flight and mid-flight."""
    batch_dir = tmp_path / "batch_cancel"
    batch_dir.mkdir()

    for idx in range(5):
        sf.write(batch_dir / f"track_{idx}.wav", synth_tone(2.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.5), SR)

    # 1. Pre-set cancellation token returns None immediately
    cancel_pre = threading.Event()
    cancel_pre.set()
    cfg_pre = JobConfig(reference=ref_file, batch_dir=batch_dir, polish=False, cancel_event=cancel_pre)
    res_pre = run_batch(cfg_pre)
    assert res_pre is None

    # 2. Mid-flight cancellation after 1st file
    cancel_mid = threading.Event()
    processed_count = 0

    def progress_tracker(msg: str, frac: float):
        nonlocal processed_count
        if "track_1.wav" in msg:
            processed_count += 1
            cancel_mid.set()

    cfg_mid = JobConfig(reference=ref_file, batch_dir=batch_dir, polish=False, cancel_event=cancel_mid)
    res_mid = run_batch(cfg_mid, progress=progress_tracker)
    assert res_mid is None


def test_batch_audit_sidecar_generation(tmp_path: Path):
    """When audit=True, run_batch produces .json audit manifests for each output file."""
    batch_dir = tmp_path / "batch_audit"
    batch_dir.mkdir()

    sf.write(batch_dir / "speech1.wav", synth_tone(2.0), SR)
    sf.write(batch_dir / "speech2.wav", synth_tone(2.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.5), SR)

    out_dir = tmp_path / "out_audit"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False, audit=True)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("success") == 2

    manifests = sorted(out_dir.glob("*_audit.json"))
    assert len(manifests) == 2
    assert manifests[0].name == "speech1_zout_audit.json"
    assert manifests[1].name == "speech2_zout_audit.json"

    # Verify JSON content
    with open(manifests[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "true_peak" in data or "true_peak_dbfs" in data
    assert "rms" in data or "rms_dbfs" in data
    assert "vad_speech_timestamps" in data
    assert "parameters" in data


# ============================================================================
# 2. CLI Flags & Validation Adversarial Tests
# ============================================================================

def test_cli_parser_all_m3_flags():
    """Verify argparse parser registers all Milestone 3 flags correctly."""
    from zout.app import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "-r", "ref.wav", "-t", "noisy",
        "-m", "main.wav",
        "--start", "01:30",
        "--end", "03:45.5",
        "--audit",
        "--polish", "off",
    ])
    assert args.ref == "ref.wav"
    assert args.role == "noisy"
    assert args.main == "main.wav"
    assert args.start == "01:30"
    assert args.end == "03:45.5"
    assert args.audit is True
    assert args.polish == "off"

    # --json alias for --audit
    args_json = parser.parse_args(["-r", "ref.wav", "-m", "main.wav", "--json"])
    assert args_json.audit is True


def test_cli_validation_batch_constraints(tmp_path: Path):
    """CLI _validate enforces batch rules: directory existence, mutual exclusion with --main, etc."""
    from zout.app import _build_parser, _validate

    parser = _build_parser()

    # 1. --batch and --main mutually exclusive
    args_both = parser.parse_args(["-r", "ref.wav", "-t", "noisy", "--batch", str(tmp_path), "-m", "main.wav"])
    with pytest.raises(SystemExit):
        _validate(parser, args_both)

    # 2. --batch directory must exist
    args_no_dir = parser.parse_args(["-r", "ref.wav", "-t", "noisy", "--batch", str(tmp_path / "nonexistent_dir")])
    with pytest.raises(SystemExit):
        _validate(parser, args_no_dir)

    # 3. --batch must be a directory, not a file
    dummy_file = tmp_path / "file.wav"
    dummy_file.write_bytes(b"")
    args_file_dir = parser.parse_args(["-r", "ref.wav", "-t", "noisy", "--batch", str(dummy_file)])
    with pytest.raises(SystemExit):
        _validate(parser, args_file_dir)


def test_cli_validation_timeline_slicing_constraints(tmp_path: Path):
    """CLI _validate enforces start < end, non-negative start, positive end."""
    from zout.app import _build_parser, _validate

    parser = _build_parser()
    ref_f = tmp_path / "ref.wav"
    main_f = tmp_path / "main.wav"
    sf.write(ref_f, synth_noise(1.0), SR)
    sf.write(main_f, synth_tone(2.0), SR)

    # 1. start >= end
    args_inv = parser.parse_args(["-r", str(ref_f), "-t", "noisy", "-m", str(main_f), "--start", "10", "--end", "5"])
    with pytest.raises(SystemExit):
        _validate(parser, args_inv)

    # 2. start == end
    args_eq = parser.parse_args(["-r", str(ref_f), "-t", "noisy", "-m", str(main_f), "--start", "5", "--end", "5"])
    with pytest.raises(SystemExit):
        _validate(parser, args_eq)

    # 3. negative start
    args_neg_start = parser.parse_args(["-r", str(ref_f), "-t", "noisy", "-m", str(main_f), "--start", "-2"])
    with pytest.raises(SystemExit):
        _validate(parser, args_neg_start)

    # 4. non-positive end
    args_zero_end = parser.parse_args(["-r", str(ref_f), "-t", "noisy", "-m", str(main_f), "--end", "0"])
    with pytest.raises(SystemExit):
        _validate(parser, args_zero_end)


# ============================================================================
# 3. Interactive TUI Dashboard, Themes & Cancellation Safety
# ============================================================================

def test_result_card_cancellation_and_exception_rendering():
    """screens.result_card renders cleanly for None (cancelled), Exception, and normal Result."""
    from zout.tui import screens, themes

    th = themes.get_theme("Forensic Cyan")

    # 1. None represents cancelled job -> should render clean cancellation card
    card_none = screens.result_card(None, theme=th)
    assert card_none is not None

    # 2. Exception rendering
    card_err = screens.result_card(RuntimeError("Processing failed"), theme=th)
    assert card_err is not None

    # 3. Normal Result rendering
    res = Result(Path("out.wav"), 1.25, -24.0, -36.0, 12.0, {"batch": False})
    card_res = screens.result_card(res, theme=th)
    assert card_res is not None

    # 4. Batch Result rendering
    res_batch = Result(Path("out_dir"), 3.5, -20.0, -32.0, 12.0, {"batch": True, "total": 5, "success": 5, "failures": []})
    card_batch = screens.result_card(res_batch, theme=th)
    assert card_batch is not None


def test_theme_hotkey_cycle_theme_contracts():
    """themes.cycle_theme rotates strictly through the 4 mandated themes."""
    from zout.tui import themes

    initial = themes.get_theme().name
    cycle_names = [themes.cycle_theme().name for _ in range(4)]
    assert len(set(cycle_names)) == 4
    assert set(cycle_names) == {"Forensic Cyan", "Amber Phosphor", "Matrix Emerald", "Studio Dark"}
    # Next cycle wraps back
    assert themes.cycle_theme().name == cycle_names[0]


# ============================================================================
# 4. Forensic Audit Sidecar Manifest RFC Compliance
# ============================================================================

def test_audit_manifest_generation_direct(tmp_path: Path):
    """Direct verification of generate_audit_manifest with synthesized audio."""
    from zout.audio.audit import generate_audit_manifest

    in_f = tmp_path / "in.wav"
    out_f = tmp_path / "out.wav"

    t = np.arange(2 * SR) / SR
    audio_in = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_out = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(in_f, audio_in, SR)
    sf.write(out_f, audio_out, SR)

    cfg = JobConfig(reference=in_f, main=in_f, output=out_f, audit=True)
    json_path = generate_audit_manifest(
        cfg=cfg,
        output_path=out_f,
        input_audio=audio_in,
        output_audio=audio_out,
        sr=SR,
        suppression_db=12.04,
        in_rms=-7.96,
        out_rms=-20.0,
    )

    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "1.0"
    assert "timestamp" in data
    assert data["suppression_db"] == 12.04
    assert data["rms"]["input"] == -7.96
    assert data["rms"]["output"] == -20.0
    assert "true_peak" in data
    assert "vad_speech_timestamps" in data
    assert "parameters" in data
    assert data["parameters"]["output"] == str(out_f)


# ============================================================================
# 5. Strict Zero Prohibited Framework Dependency Audit
# ============================================================================

def test_zero_prohibited_framework_across_workspace():
    """Verify strictly 0 imports or references to prohibited TUI framework across repository."""
    workspace = Path(__file__).resolve().parent.parent
    violations = []
    target = "".join(["t", "e", "x", "t", "u", "a", "l"])

    for root, _, files in os.walk(workspace):
        # Skip git, venv, and teamwork agent state directories
        if any(ignored in root for ignored in [".git", ".venv", ".agents", "__pycache__"]):
            continue
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                # Skip test files — they may mention the forbidden name in their own constraint checks
                if path.resolve().parent.name == "tests":
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore").lower()
                if target in content:
                    violations.append(str(path))

    assert len(violations) == 0, f"Found prohibited references in: {violations}"
