"""Milestone 3 Gate Empirical Challenger 1 Stress Harness.

Adversarially tests:
1. Theme cycling, thread safety, 500+ rapid cycles, case-insensitivity, Rich style validity.
2. VU meter dBFS accuracy, RMS/Peak bounds, silence, quiet (-40 dBFS), clip indicator (>= -0.5 dBFS), stereo isolation, NaN/Inf handling.
3. Waveform envelope glyph scaling, step/impulse dynamics, Dirac peak isolation, zero/NaN resilience.
4. Timeline range slicing: parse_time_spec boundaries, clock format validations, exact sample slicing length, inverted ranges.
5. Zero-forbidden-library constraint audit across all files.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from rich.console import Console
from rich.style import Style

import zout.tui.themes as themes
from zout.audio.io import (
    GLYPHS,
    compute_waveform_envelope,
    get_file_waveform_envelope,
    load_audio,
    parse_time_spec,
    save_audio,
)
from zout.audio.player import AudioPlayer, _calc_dbfs, _calc_meters
from zout.config import JobConfig, ReferenceRole
from zout.tui.screens import prompt_timeline_range, render_playback_card, render_stereo_vu_meter


# ============================================================================
# 1. THEME CONSISTENCY & RAPID CYCLING STRESS TESTS
# ============================================================================

def test_theme_sequential_cycle_500_times():
    """Verify rapid sequential cycling 500 times preserves order and wraparound."""
    themes.reset()
    expected_order = [
        "Forensic Cyan",
        "Amber Phosphor",
        "Matrix Emerald",
        "Studio Dark",
    ]
    assert themes.get_theme().name == expected_order[0]

    for cycle in range(500):
        next_theme = themes.cycle_theme()
        expected = expected_order[(cycle + 1) % 4]
        assert next_theme.name == expected, f"Cycle {cycle+1}: expected {expected}, got {next_theme.name}"


def test_theme_thread_safe_concurrent_cycling():
    """Verify concurrent cycling across 10 threads (5,000 total cycles) without corruption or deadlocks."""
    themes.reset()
    manager = themes.ThemeManager()
    errors = []

    def worker(num_cycles: int):
        try:
            for _ in range(num_cycles):
                t = manager.cycle_theme()
                assert t.name in [th.name for th in themes.THEMES]
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(500,)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Worker thread deadlocked during concurrent cycling"

    assert len(errors) == 0, f"Concurrent cycling raised errors: {errors}"


def test_theme_lookup_case_insensitivity_and_fallbacks():
    """Verify case-insensitivity, whitespace tolerance, and fallback for unknown names."""
    themes.reset()
    assert themes.get_theme("forensic cyan").name == "Forensic Cyan"
    assert themes.get_theme("  AMBER PHOSPHOR  ").name == "Amber Phosphor"
    assert themes.get_theme("mAtRiX eMeRaLd").name == "Matrix Emerald"
    assert themes.get_theme("STUDIO DARK\t").name == "Studio Dark"

    # Fallbacks
    assert themes.get_theme("NonExistentTheme").name == "Forensic Cyan"
    assert themes.get_theme("").name == "Forensic Cyan"
    assert themes.get_theme(None).name == "Forensic Cyan"


def test_theme_all_rich_styles_are_valid():
    """Ensure every single style string in every theme parses cleanly in Rich."""
    for theme in themes.THEMES:
        style_attrs = [
            "primary", "secondary", "accent", "key", "border", "title",
            "muted", "status", "success", "warning", "error",
            "bar_fill", "bar_empty", "meter_low", "meter_mid",
            "meter_high", "meter_peak",
        ]
        for attr in style_attrs:
            val = getattr(theme, attr)
            assert val, f"Theme {theme.name} missing value for {attr}"
            try:
                Style.parse(val)
            except Exception as e:
                pytest.fail(f"Theme {theme.name} has invalid Rich style for '{attr}' ('{val}'): {e}")


def test_theme_cycling_memory_leak():
    """Check tracemalloc memory delta over 10,000 rapid cycles."""
    themes.reset()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    for _ in range(10000):
        themes.cycle_theme()

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_diff_kb = sum(stat.size_diff for stat in stats) / 1024.0
    # Memory growth should be negligible (< 100 KB across 10,000 cycles)
    assert total_diff_kb < 100.0, f"Potential memory leak detected: {total_diff_kb:.2f} KB growth"


# ============================================================================
# 2. VU METER & STEREO LEVEL ACCURACY TESTS
# ============================================================================

def test_vu_meter_silence_handling():
    """Test silence produces -80 dBFS and does not trigger clip warning."""
    zero_block = np.zeros((1024, 2), dtype=np.float32)
    rms_l, rms_r, peak_l, peak_r = _calc_meters(zero_block)

    assert rms_l <= -80.0
    assert rms_r <= -80.0
    assert peak_l <= -80.0
    assert peak_r <= -80.0

    table = render_stereo_vu_meter(((rms_l, rms_r), (peak_l, peak_r)), width=24)
    console = Console(record=True, width=100)
    console.print(table)
    rendered = console.export_text()

    assert "-∞ dB" in rendered or "-80.0 dB" in rendered
    assert "[CLIP]" not in rendered


def test_vu_meter_quiet_signal_accuracy():
    """Test -40 dBFS sine wave computes accurately within +/- 0.5 dB."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Sine wave with RMS = -40 dBFS. Peak = RMS * sqrt(2)
    rms_target_linear = 10.0 ** (-40.0 / 20.0)
    amp = rms_target_linear * np.sqrt(2.0)
    sine = (amp * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    stereo = np.column_stack([sine, sine])

    rms_l, rms_r, peak_l, peak_r = _calc_meters(stereo)

    assert abs(rms_l - (-40.0)) < 0.2, f"Expected RMS ~ -40 dB, got {rms_l}"
    assert abs(rms_r - (-40.0)) < 0.2, f"Expected RMS ~ -40 dB, got {rms_r}"
    expected_peak = -40.0 + 20.0 * np.log10(np.sqrt(2.0))  # ~ -36.99 dBFS
    assert abs(peak_l - expected_peak) < 0.2, f"Expected peak ~ {expected_peak}, got {peak_l}"
    assert abs(peak_r - expected_peak) < 0.2, f"Expected peak ~ {expected_peak}, got {peak_r}"


def test_vu_meter_clipping_detection():
    """Test full-scale and clipped signals trigger [CLIP] indicator."""
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)

    # 1. Full scale 0 dBFS sine (Peak = 1.0 = 0 dBFS >= -0.5 dBFS)
    sine_0db = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    stereo_0db = np.column_stack([sine_0db, sine_0db])
    rms_l, rms_r, peak_l, peak_r = _calc_meters(stereo_0db)

    assert abs(peak_l - 0.0) < 0.1
    table_0db = render_stereo_vu_meter(((rms_l, rms_r), (peak_l, peak_r)), width=24)
    c0 = Console(record=True, width=100)
    c0.print(table_0db)
    assert "[CLIP]" in c0.export_text(), "0 dBFS peak should trigger [CLIP]"

    # 2. Oversaturated signal (+6 dBFS)
    sine_clip = (2.0 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    stereo_clip = np.column_stack([sine_clip, sine_clip])
    rms_l2, rms_r2, peak_l2, peak_r2 = _calc_meters(stereo_clip)

    assert peak_l2 == 0.0, f"Peak should be bounded at ceiling 0.0 dBFS, got {peak_l2}"
    table_clip = render_stereo_vu_meter(((rms_l2, rms_r2), (peak_l2, peak_r2)), width=24)
    c_clip = Console(record=True, width=100)
    c_clip.print(table_clip)
    assert "[CLIP]" in c_clip.export_text(), "Clipped signal must trigger [CLIP]"


def test_vu_meter_stereo_channel_independence():
    """Verify left and right channels are strictly independent without crosstalk."""
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    zero = np.zeros_like(sine)

    # Left only
    left_only = np.column_stack([sine, zero])
    rms_l, rms_r, peak_l, peak_r = _calc_meters(left_only)
    assert rms_l > -10.0 and rms_r <= -80.0
    assert peak_l > -10.0 and peak_r <= -80.0

    # Right only
    right_only = np.column_stack([zero, sine])
    rms_l, rms_r, peak_l, peak_r = _calc_meters(right_only)
    assert rms_l <= -80.0 and rms_r > -10.0
    assert peak_l <= -80.0 and peak_r > -10.0


def test_vu_meter_nan_inf_robustness():
    """Verify audio blocks containing NaN and Inf do not crash meters or rendering."""
    nan_block = np.array([
        [float("nan"), float("inf")],
        [float("-inf"), 0.5],
        [0.0, 0.0],
    ], dtype=np.float32)

    rms_l, rms_r, peak_l, peak_r = _calc_meters(nan_block)
    assert math.isfinite(rms_l) and math.isfinite(rms_r)
    assert math.isfinite(peak_l) and math.isfinite(peak_r)

    # Rendering test across all themes with NaN levels
    for theme in themes.THEMES:
        t = render_stereo_vu_meter(((rms_l, rms_r), (peak_l, peak_r)), theme=theme)
        c = Console(record=True, width=100)
        c.print(t)
        assert len(c.export_text()) > 0


# ============================================================================
# 3. WAVEFORM ENVELOPE DYNAMICS TESTS
# ============================================================================

def test_waveform_envelope_silence_produces_spaces():
    """Silence should produce all empty spaces."""
    silence = np.zeros(44100, dtype=np.float32)
    env = compute_waveform_envelope(silence, width=16)
    assert env == " " * 16, f"Expected 16 spaces, got: {repr(env)}"


def test_waveform_envelope_dirac_impulse():
    """A single Dirac impulse should produce peak block only in its bucket."""
    audio = np.zeros(16000, dtype=np.float32)
    # Put impulse at 50%
    audio[8000] = 1.0
    env = compute_waveform_envelope(audio, width=16)
    assert len(env) == 16
    assert env[8] == GLYPHS[-1], f"Expected highest glyph '{GLYPHS[-1]}' at index 8, got '{env[8]}'"
    for i in range(16):
        if i != 8:
            assert env[i] == GLYPHS[0] or env[i] == " ", f"Index {i} should be low/empty, got '{env[i]}'"


def test_waveform_envelope_linear_ramp_monotonicity():
    """A ramp signal from 0 to 1 should have monotonically non-decreasing glyphs."""
    audio = np.linspace(0.0, 1.0, 32000, dtype=np.float32)
    env = compute_waveform_envelope(audio, width=16)
    glyph_map = {g: i for i, g in enumerate(GLYPHS)}
    glyph_map[" "] = 0
    indices = [glyph_map[c] for c in env]

    assert len(indices) == 16
    assert indices[0] <= indices[1]
    assert indices[-1] == 7  # Maximum glyph at the peak
    # Monotonicity check
    for i in range(len(indices) - 1):
        assert indices[i] <= indices[i + 1], f"Ramp envelope not monotonic: {indices}"


def test_waveform_envelope_cached_file_helper(tmp_path: Path):
    """Test get_file_waveform_envelope on disk audio files."""
    wav_path = tmp_path / "test_ramp.wav"
    sr = 22050
    audio = np.linspace(0.0, 0.8, sr * 2, dtype=np.float32)
    save_audio(wav_path, audio, sr)

    env1 = get_file_waveform_envelope(wav_path, width=12)
    assert len(env1) == 12
    assert env1[-1] == GLYPHS[-1]

    # Test cache hit (same mtime)
    env2 = get_file_waveform_envelope(wav_path, width=12)
    assert env1 == env2

    # Test non-existent file returns spaces
    env_missing = get_file_waveform_envelope(tmp_path / "non_existent.wav", width=12)
    assert env_missing == " " * 12


# ============================================================================
# 4. TIMELINE RANGE SLICING ADVERSARIAL TESTS
# ============================================================================

def test_parse_time_spec_valid_formats():
    """Verify various valid time specifications."""
    assert parse_time_spec(0) == 0.0
    assert parse_time_spec(45) == 45.0
    assert parse_time_spec(12.345) == 12.345
    assert parse_time_spec("0") == 0.0
    assert parse_time_spec("90") == 90.0
    assert parse_time_spec("01:30") == 90.0
    assert parse_time_spec("00:00:45") == 45.0
    assert parse_time_spec("01:02:03") == 3723.0
    assert parse_time_spec("00:00:00.5") == 0.5
    assert parse_time_spec(None) is None


def test_parse_time_spec_invalid_and_adversarial():
    """Verify rejection of invalid, negative, or malformed time formats."""
    with pytest.fail if False else pytest.raises(ValueError):
        parse_time_spec("-1")

    with pytest.raises(ValueError):
        parse_time_spec("-00:05")

    with pytest.raises(ValueError):
        parse_time_spec("01:60")  # Seconds >= 60 in clock format

    with pytest.raises(ValueError):
        parse_time_spec("01:60:00")  # Minutes >= 60 in clock format

    with pytest.raises(ValueError):
        parse_time_spec("1:2:3:4")  # Too many parts

    with pytest.raises(ValueError):
        parse_time_spec("abc")

    with pytest.raises(ValueError):
        parse_time_spec("1e5:00")  # Scientific notation in clock format

    with pytest.raises(ValueError):
        parse_time_spec("nan")

    with pytest.raises(ValueError):
        parse_time_spec("inf")

    with pytest.raises(ValueError):
        parse_time_spec("")

    with pytest.raises(TypeError):
        parse_time_spec([123])


def test_audio_slicing_exact_duration_and_boundaries(tmp_path: Path):
    """Verify load_audio slices exact sample counts without drift."""
    wav_path = tmp_path / "10sec.wav"
    sr = 16000
    duration = 10.0
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    save_audio(wav_path, audio, sr)

    # 1. Exact slice [2.0, 7.0] -> 5.0 seconds = 80,000 samples
    sliced, sr_out = load_audio(wav_path, start_sec=2.0, end_sec=7.0)
    assert sr_out == sr
    assert len(sliced) == 5 * sr

    # 2. Fractional slice [0.5, 3.25] -> 2.75 seconds = 44,000 samples
    sliced_frac, _ = load_audio(wav_path, start_sec=0.5, end_sec=3.25)
    assert len(sliced_frac) == int(2.75 * sr)

    # 3. Start at 0, end at duration
    full, _ = load_audio(wav_path, start_sec=0.0, end_sec=10.0)
    assert len(full) == total_samples

    # 4. Out of bounds clamp: end > duration
    clamped, _ = load_audio(wav_path, start_sec=8.0, end_sec=15.0)
    assert len(clamped) == 2 * sr  # Only 2.0s left in file

    # 5. Inverted range: start >= end should fail or be empty
    with pytest.raises(ValueError):
        load_audio(wav_path, start_sec=7.0, end_sec=2.0)


def test_cli_timeline_range_validation_adversarial(tmp_path: Path):
    """Verify CLI argument validation catches inverted, negative, or malformed start/end."""
    from zout.app import _build_parser, _validate

    dummy_ref = tmp_path / "ref.wav"
    dummy_main = tmp_path / "main.wav"
    save_audio(dummy_ref, np.zeros(16000, dtype=np.float32), 16000)
    save_audio(dummy_main, np.zeros(16000, dtype=np.float32), 16000)

    ap = _build_parser()

    # Valid range
    args = ap.parse_args(["-r", str(dummy_ref), "-m", str(dummy_main), "--role", "noisy", "--start", "00:10", "--end", "00:20"])
    _validate(ap, args)  # Must not raise

    # Inverted range: start >= end
    args_inv = ap.parse_args(["-r", str(dummy_ref), "-m", str(dummy_main), "--role", "noisy", "--start", "00:30", "--end", "00:10"])
    with pytest.raises(SystemExit):
        _validate(ap, args_inv)

    # Identical start and end: start == end
    args_equal = ap.parse_args(["-r", str(dummy_ref), "-m", str(dummy_main), "--role", "noisy", "--start", "15", "--end", "15"])
    with pytest.raises(SystemExit):
        _validate(ap, args_equal)

    # Negative start
    args_neg = ap.parse_args(["-r", str(dummy_ref), "-m", str(dummy_main), "--role", "noisy", "--start", "-5", "--end", "10"])
    with pytest.raises(SystemExit):
        _validate(ap, args_neg)

    # Zero or negative end
    args_zero_end = ap.parse_args(["-r", str(dummy_ref), "-m", str(dummy_main), "--role", "noisy", "--end", "0"])
    with pytest.raises(SystemExit):
        _validate(ap, args_zero_end)

    # Malformed time format
    args_bad = ap.parse_args(["-r", str(dummy_ref), "-m", str(dummy_main), "--role", "noisy", "--start", "invalid_time"])
    with pytest.raises(SystemExit):
        _validate(ap, args_bad)


def test_render_playback_card_across_all_themes_and_states(tmp_path: Path):
    """Stress test render_playback_card across all 4 themes in playing, paused, and hot-swap states."""
    p_orig = tmp_path / "orig.wav"
    p_proc = tmp_path / "proc.wav"
    sr = 22050
    audio = np.sin(np.linspace(0, 10, sr * 2)).astype(np.float32)
    save_audio(p_orig, audio, sr)
    save_audio(p_proc, audio * 0.5, sr)

    player = AudioPlayer(force_headless=True)
    player.play_pair(p_orig, p_proc, start_with="B")

    try:
        for theme in themes.THEMES:
            # When on Track B
            assert player.get_current_track() == "B"
            console_b = Console(record=True, width=100)
            card_b = render_playback_card(player, title="Testing Track B", theme=theme)
            console_b.print(card_b)
            text_b = console_b.export_text()
            assert "TRACK B" in text_b
            assert "Elapsed:" in text_b

            # Toggle to Track A
            player.toggle_ab()
            assert player.get_current_track() == "A"
            console_a = Console(record=True, width=100)
            card_a = render_playback_card(player, title="Testing Track A", theme=theme)
            console_a.print(card_a)
            text_a = console_a.export_text()
            assert "TRACK A" in text_a

            # Toggle back to B for next theme iteration
            player.toggle_ab()
            assert player.get_current_track() == "B"
    finally:
        player.stop()


# ============================================================================
# 5. ZERO FORBIDDEN FRAMEWORK AUDIT
# ============================================================================

def test_zero_forbidden_framework_audit():
    """Ensure zero occurrences or imports of the forbidden framework across all project python files."""
    repo_root = Path(__file__).resolve().parent.parent
    violations = []
    target_str = "".join(["t", "e", "x", "t", "u", "a", "l"])

    for root, dirs, files in os.walk(repo_root):
        # Skip virtualenvs and git metadata
        if any(ignored in root for ignored in [".venv", ".git", "__pycache__", ".agents"]):
            continue
        for f in files:
            if f.endswith(".py"):
                full_path = Path(root) / f
                if full_path.resolve() == Path(__file__).resolve():
                    continue
                content = full_path.read_text(encoding="utf-8", errors="ignore").lower()
                if target_str in content:
                    violations.append(str(full_path))

    assert len(violations) == 0, f"Found violations in: {violations}"
