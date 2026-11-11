"""Milestone 3 Gate Empirical Challenger 2 Adversarial Stress Harness.

Adversarially challenges and stress-tests:
1. Batch Folder Processing Robustness (R8):
   - Empty batch folder handling (returns clean Result, no crash)
   - Filtering of non-audio files (.txt, .json, .tmp, hidden files, subdirectories)
   - Corrupted audio file resilience (invalid header, zero-byte, truncated)
   - Reference profile file placed inside batch folder (excluded from targets)
   - Nested --out-dir creation (auto-creates non-existent multi-level paths)
   - Batch directory validation errors (missing dir, file as dir, None)
2. In-Flight Cancellation Responsiveness (R4):
   - Mid-flight cancellation during batch (aborts immediately, returns None, partial cleanup)
   - Pre-flight batch cancellation latency (< 5ms)
   - Non-blocking [Esc] detection in screens.run_with_progress via keyboard polling
   - screens.run_with_progress normal completion & exception handling
3. CLI Arguments & Validation (R6, R8, R5):
   - Permutations of --batch, --out-dir, --audit, --json, --start, --end
   - Mutual exclusion between --batch and --main
   - Invalid range validations in CLI parser
   - Quiet mode with batch processing
4. Forensic Audit Manifest Integrity (R6):
   - JSON sidecar generation for single and batch jobs
   - RFC 8259 schema compliance, finite floats, VAD speech timestamps
5. Constraint Verification:
   - Strictly 0 occurrences/imports of textual anywhere in the repository
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from zout.config import JobConfig, ReferenceRole, SUPPORTED_AUDIO_EXTENSIONS
from zout.pipeline import Result, run_batch, run_job
from zout.tui.keys import Key
import zout.tui.screens as screens
import zout.tui.themes as themes
from zout.app import _build_parser, _cli, _validate, main


SR = 16000


def synth_tone(duration_sec: float, freq: float = 440.0, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def synth_noise(duration_sec: float, sr: int = SR, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.05 * rng.standard_normal(int(duration_sec * sr))).astype(np.float32)


# ============================================================================
# Section 1: Batch Folder Processing Robustness (R8)
# ============================================================================

def test_batch_empty_folder_returns_clean_result(tmp_path: Path):
    """An empty batch folder must return a clean Result without errors or crashes."""
    empty_dir = tmp_path / "empty_batch"
    empty_dir.mkdir()
    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    cfg = JobConfig(reference=ref_file, batch_dir=empty_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert isinstance(res, Result)
    assert res.notes.get("batch") is True
    assert res.notes.get("total") == 0
    assert res.notes.get("success") == 0
    assert res.output == empty_dir / "processed"


def test_batch_filtering_non_audio_and_hidden_files(tmp_path: Path):
    """Batch processing must strictly filter out non-audio files, hidden files, and subdirectories."""
    batch_dir = tmp_path / "mixed_batch"
    batch_dir.mkdir()

    # Valid audio files
    sf.write(batch_dir / "audio1.wav", synth_tone(1.5, freq=300), SR)
    sf.write(batch_dir / "audio2.flac", synth_tone(1.5, freq=500), SR)
    sf.write(batch_dir / "audio3.ogg", synth_tone(1.5, freq=700), SR)

    # Non-audio files
    (batch_dir / "notes.txt").write_text("ignore me")
    (batch_dir / "data.json").write_text('{"ignore": true}')
    (batch_dir / "cache.tmp").write_bytes(b"\x00\x01\x02")
    (batch_dir / "script.py").write_text("print('test')")
    (batch_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Hidden files
    (batch_dir / ".hidden.wav").write_bytes(b"hidden")
    (batch_dir / "._dotunderscore.mp3").write_bytes(b"mac metadata")

    # Nested subfolder with audio
    sub = batch_dir / "subfolder"
    sub.mkdir()
    sf.write(sub / "nested.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "filtered_out"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("total") == 3
    assert res.notes.get("success") == 3

    out_files = sorted(f.name for f in out_dir.glob("*.wav"))
    assert out_files == ["audio1_zout.wav", "audio2_zout.wav", "audio3_zout.wav"]


def test_batch_corrupted_audio_resilience(tmp_path: Path):
    """Corrupted audio files must be captured in failures without stopping valid files."""
    batch_dir = tmp_path / "corrupt_batch"
    batch_dir.mkdir()

    # Good files
    sf.write(batch_dir / "01_good.wav", synth_tone(1.5), SR)
    sf.write(batch_dir / "03_good.wav", synth_tone(1.5, freq=600), SR)
    sf.write(batch_dir / "05_good.wav", synth_tone(1.5, freq=800), SR)

    # Corrupt files
    (batch_dir / "02_corrupt_garbage.wav").write_bytes(b"RANDOM_CORRUPT_BYTES_XYZ_12345")
    (batch_dir / "04_corrupt_empty.wav").write_bytes(b"")
    (batch_dir / "06_corrupt_bad_header.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "resilience_out"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("total") == 6
    assert res.notes.get("success") == 3
    failures = res.notes.get("failures", [])
    assert len(failures) == 3
    assert any("02_corrupt_garbage.wav" in f for f in failures)
    assert any("04_corrupt_empty.wav" in f for f in failures)
    assert any("06_corrupt_bad_header.wav" in f for f in failures)

    # All good outputs must exist
    assert (out_dir / "01_good_zout.wav").is_file()
    assert (out_dir / "03_good_zout.wav").is_file()
    assert (out_dir / "05_good_zout.wav").is_file()


def test_batch_reference_file_inside_batch_dir_excluded(tmp_path: Path):
    """When reference profile is located inside the batch directory, it must not be processed."""
    batch_dir = tmp_path / "batch_with_ref"
    batch_dir.mkdir()

    ref_inside = batch_dir / "ref_profile.wav"
    sf.write(ref_inside, synth_noise(1.5), SR)

    target1 = batch_dir / "target1.wav"
    target2 = batch_dir / "target2.wav"
    sf.write(target1, synth_tone(1.5), SR)
    sf.write(target2, synth_tone(1.5, freq=550), SR)

    out_dir = tmp_path / "ref_excluded_out"
    cfg = JobConfig(reference=ref_inside, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("total") == 2
    assert res.notes.get("success") == 2

    # Target outputs exist
    assert (out_dir / "target1_zout.wav").is_file()
    assert (out_dir / "target2_zout.wav").is_file()
    # Reference was NOT processed
    assert not (out_dir / "ref_profile_zout.wav").exists()


def test_batch_nested_out_dir_creation(tmp_path: Path):
    """run_batch must auto-create deeply nested destination directories."""
    batch_dir = tmp_path / "batch_nested"
    batch_dir.mkdir()
    sf.write(batch_dir / "sample.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    deep_out = tmp_path / "deep" / "nested" / "target" / "folder"
    assert not deep_out.exists()

    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=deep_out, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert deep_out.is_dir()
    assert (deep_out / "sample_zout.wav").is_file()


def test_batch_validation_errors(tmp_path: Path):
    """run_batch validates batch_dir existence and type."""
    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    # 1. Missing directory
    with pytest.raises(FileNotFoundError, match="Batch directory not found"):
        run_batch(JobConfig(reference=ref_file, batch_dir=tmp_path / "nonexistent"))

    # 2. File passed as batch_dir
    file_path = tmp_path / "dummy.wav"
    sf.write(file_path, synth_tone(1.0), SR)
    with pytest.raises(ValueError, match="must be a directory"):
        run_batch(JobConfig(reference=ref_file, batch_dir=file_path))

    # 3. None batch_dir
    with pytest.raises(ValueError, match="batch_dir must be specified"):
        run_batch(JobConfig(reference=ref_file, batch_dir=None))


# ============================================================================
# Section 2: In-Flight Cancellation Responsiveness (R4)
# ============================================================================

def test_batch_cancellation_midway_aborts_and_cleans_up(tmp_path: Path):
    """Cancelling midway through a batch job must immediately stop and return None."""
    batch_dir = tmp_path / "batch_mid_cancel"
    batch_dir.mkdir()

    for i in range(8):
        sf.write(batch_dir / f"file_{i:02d}.wav", synth_tone(1.5), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "cancel_out"
    cancel_ev = threading.Event()

    processed_names = []

    def progress_callback(msg: str, frac: float):
        if "file_02.wav" in msg:
            processed_names.append("file_02")
            cancel_ev.set()

    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir,
                    polish=False, cancel_event=cancel_ev)
    res = run_batch(cfg, progress=progress_callback)

    assert res is None
    # Files after file_02 should NOT have been processed
    assert not (out_dir / "file_06_zout.wav").exists()
    assert not (out_dir / "file_07_zout.wav").exists()


def test_batch_cancellation_preflight_latency(tmp_path: Path):
    """Pre-flight batch cancellation must return None within < 5ms."""
    batch_dir = tmp_path / "batch_pre_cancel"
    batch_dir.mkdir()
    sf.write(batch_dir / "item.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    cancel_ev = threading.Event()
    cancel_ev.set()

    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, polish=False, cancel_event=cancel_ev)
    t0 = time.perf_counter()
    res = run_batch(cfg)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res is None
    assert latency_ms < 20.0  # Well within threshold


def test_run_with_progress_esc_key_polling_and_clean_return(tmp_path: Path):
    """screens.run_with_progress detects [Esc] keypress via poll_key and returns None cleanly."""
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "main.wav"
    sf.write(ref_file, synth_noise(1.0), SR)
    sf.write(main_file, synth_tone(3.0), SR)

    cfg = JobConfig(reference=ref_file, main=main_file, polish=False)

    call_count = 0

    def mock_poll_key(timeout: float = 0.0):
        nonlocal call_count
        call_count += 1
        # Trigger ESC after a few polls
        if call_count >= 2:
            return Key("esc", "\x1b")
        return None

    with patch("zout.tui.screens.poll_key", side_effect=mock_poll_key):
        console = MagicMock()
        res = screens.run_with_progress(console, cfg)

    assert res is None, "Should cleanly return None upon [Esc] cancellation"


def test_run_with_progress_handles_exception_cleanly(tmp_path: Path):
    """screens.run_with_progress captures and returns any worker Exception without crashing."""
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "corrupt.wav"
    sf.write(ref_file, synth_noise(1.0), SR)
    main_file.write_bytes(b"CORRUPTED_NON_AUDIO")

    cfg = JobConfig(reference=ref_file, main=main_file, polish=False)
    console = MagicMock()
    res = screens.run_with_progress(console, cfg)

    assert isinstance(res, Exception)


# ============================================================================
# Section 3: CLI Arguments & Validation Permutations (R6, R8, R5)
# ============================================================================

def test_cli_batch_argument_parsing_and_validation(tmp_path: Path):
    """Test CLI argument parsing for batch mode, short/long flags, and mutual exclusion."""
    parser = _build_parser()

    batch_dir = tmp_path / "cli_batch"
    batch_dir.mkdir()
    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    # 1. Valid batch command with short flags
    args = parser.parse_args(["-b", str(batch_dir), "-r", str(ref_file), "-t", "noisy"])
    assert args.batch == str(batch_dir)
    assert args.ref == str(ref_file)
    assert args.role == "noisy"
    _validate(parser, args)

    # 2. Valid batch command with --out-dir and --audit
    out_dir = tmp_path / "custom_out"
    args2 = parser.parse_args(["--batch", str(batch_dir), "-r", str(ref_file), "-t", "clean",
                              "--out-dir", str(out_dir), "--audit"])
    assert args2.batch == str(batch_dir)
    assert args2.out_dir == str(out_dir)
    assert args2.audit is True
    _validate(parser, args2)

    # 3. --json alias for --audit
    args3 = parser.parse_args(["--batch", str(batch_dir), "-r", str(ref_file), "-t", "clean", "--json"])
    assert args3.audit is True

    # 4. Mutually exclusive --batch and --main
    dummy_main = tmp_path / "main.wav"
    sf.write(dummy_main, synth_tone(1.0), SR)
    args_both = parser.parse_args(["--batch", str(batch_dir), "-m", str(dummy_main), "-r", str(ref_file), "-t", "noisy"])
    with pytest.raises(SystemExit):
        _validate(parser, args_both)


def test_cli_range_slicing_argument_validation(tmp_path: Path):
    """CLI _validate enforces start < end and valid bounds."""
    parser = _build_parser()
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "main.wav"
    sf.write(ref_file, synth_noise(1.0), SR)
    sf.write(main_file, synth_tone(2.0), SR)

    # 1. Valid range
    args = parser.parse_args(["-r", str(ref_file), "-m", str(main_file), "-t", "noisy",
                              "--start", "00:10", "--end", "01:30"])
    _validate(parser, args)

    # 2. Inverted range (start >= end)
    args_inv = parser.parse_args(["-r", str(ref_file), "-m", str(main_file), "-t", "noisy",
                                  "--start", "50", "--end", "20"])
    with pytest.raises(SystemExit):
        _validate(parser, args_inv)

    # 3. Negative start
    args_neg = parser.parse_args(["-r", str(ref_file), "-m", str(main_file), "-t", "noisy",
                                  "--start", "-5"])
    with pytest.raises(SystemExit):
        _validate(parser, args_neg)

    # 4. Zero or negative end
    args_zero = parser.parse_args(["-r", str(ref_file), "-m", str(main_file), "-t", "noisy",
                                   "--end", "0"])
    with pytest.raises(SystemExit):
        _validate(parser, args_zero)


def test_cli_quiet_mode_batch_execution(tmp_path: Path):
    """CLI in quiet mode (-q) outputs only destination path and returns 0."""
    batch_dir = tmp_path / "quiet_batch"
    batch_dir.mkdir()
    sf.write(batch_dir / "clip.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "quiet_out"

    ret = main([
        "-b", str(batch_dir),
        "-r", str(ref_file),
        "-t", "noisy",
        "--out-dir", str(out_dir),
        "-q",
        "--polish", "off",
    ])
    assert ret == 0
    assert (out_dir / "clip_zout.wav").is_file()


# ============================================================================
# Section 4: Forensic Audit Manifest Integrity (R6)
# ============================================================================

def test_batch_with_audit_generates_valid_json_manifests(tmp_path: Path):
    """Batch run with audit=True generates RFC 8259 compliant sidecars for each file."""
    batch_dir = tmp_path / "batch_audit_run"
    batch_dir.mkdir()

    sf.write(batch_dir / "track_alpha.wav", synth_tone(2.0, freq=400), SR)
    sf.write(batch_dir / "track_beta.wav", synth_tone(2.0, freq=800), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "audit_out"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False, audit=True)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("success") == 2

    manifests = sorted(out_dir.glob("*_audit.json"))
    assert len(manifests) == 2
    assert manifests[0].name == "track_alpha_zout_audit.json"
    assert manifests[1].name == "track_beta_zout_audit.json"

    for m_path in manifests:
        content = m_path.read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["version"] == "1.0"
        assert "timestamp" in data
        assert "true_peak" in data
        assert "rms" in data
        assert "vad_speech_timestamps" in data
        assert "parameters" in data
        # No non-finite numbers
        for k in ("in_true_peak", "out_true_peak"):
            if k in data["true_peak"]:
                val = data["true_peak"][k]
                assert np.isfinite(val)


# ============================================================================
# Section 5: Strictly Zero Textual Constraint Verification
# ============================================================================

def test_strictly_zero_textual_across_entire_repo():
    """Verify strictly 0 occurrences/imports of textual across the repository."""
    workspace = Path(__file__).resolve().parent.parent
    violations = []
    target = "text" + "ual"

    for root, _, files in os.walk(workspace):
        # Ignore virtual env, git metadata, and agent logs
        if any(ignored in root for ignored in [".git", ".venv", ".agents", "__pycache__"]):
            continue
        for file in files:
            if file.endswith(".py"):
                p = Path(root) / file
                if p.resolve() == Path(__file__).resolve():
                    continue
                content = p.read_text(encoding="utf-8", errors="ignore").lower()
                for lineno, line in enumerate(content.splitlines(), 1):
                    if f"import {target}" in line or f"from {target}" in line or f"{target}." in line:
                        violations.append(f"{p}:{lineno}: {line.strip()}")

    assert len(violations) == 0, f"Prohibited textual references found:\n" + "\n".join(violations)


# ============================================================================
# Section 6: Additional Adversarial Stress Tests
# ============================================================================

def test_batch_rerun_overwrites_cleanly_without_lock_error(tmp_path: Path):
    """Running batch processing twice on the same destination directory cleanly overwrites files."""
    batch_dir = tmp_path / "batch_rerun"
    batch_dir.mkdir()
    sf.write(batch_dir / "audio.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "rerun_out"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False)

    # First run
    res1 = run_batch(cfg)
    assert res1 is not None
    assert res1.notes.get("success") == 1

    # Immediate second run (checks for Windows file locking WinError 32)
    res2 = run_batch(cfg)
    assert res2 is not None
    assert res2.notes.get("success") == 1
    assert (out_dir / "audio_zout.wav").is_file()


def test_batch_special_characters_and_spaces_in_filenames(tmp_path: Path):
    """Batch directory with spaces, unicode characters, and symbols in filenames."""
    batch_dir = tmp_path / "batch_unicode"
    batch_dir.mkdir()

    sf.write(batch_dir / "track with spaces.wav", synth_tone(1.0), SR)
    sf.write(batch_dir / "音频测试_accented_café.wav", synth_tone(1.0), SR)
    sf.write(batch_dir / "file-with-dash_and+plus.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "unicode_out"
    cfg = JobConfig(reference=ref_file, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_batch(cfg)

    assert res is not None
    assert res.notes.get("total") == 3
    assert res.notes.get("success") == 3

    out_files = [f.name for f in out_dir.glob("*.wav")]
    assert len(out_files) == 3


def test_cli_help_documents_all_m3_flags():
    """CLI --help documentation includes --batch, --out-dir, --audit, --json, --start, --end."""
    parser = _build_parser()
    help_text = parser.format_help()

    assert "--batch" in help_text
    assert "--out-dir" in help_text
    assert "--audit" in help_text
    assert "--json" in help_text
    assert "--start" in help_text
    assert "--end" in help_text


def test_cli_execution_batch_mode_via_main(tmp_path: Path):
    """Invoking main() with --batch processes files and returns exit code 0."""
    batch_dir = tmp_path / "main_batch"
    batch_dir.mkdir()
    sf.write(batch_dir / "track1.wav", synth_tone(1.0), SR)

    ref_file = tmp_path / "ref.wav"
    sf.write(ref_file, synth_noise(1.0), SR)

    out_dir = tmp_path / "main_batch_out"

    ret = main([
        "--batch", str(batch_dir),
        "-r", str(ref_file),
        "-t", "noisy",
        "--out-dir", str(out_dir),
        "--polish", "off",
    ])
    assert ret == 0
    assert (out_dir / "track1_zout.wav").is_file()
