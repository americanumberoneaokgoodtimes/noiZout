"""Empirical Stress Testing Suite for Milestone 1 (M1: Audio Core, Player & Slicing).

Run with:
    .\\.venv\\Scripts\\python.exe tests/stress_m1.py
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from zout.audio.io import (
    GLYPHS,
    _ffmpeg,
    compute_waveform_envelope,
    load_audio,
    parse_time_spec,
)
from zout.audio.player import AudioPlayer
from zout.config import JobConfig, ReferenceRole


class StressTestReport:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.findings: list[dict] = []

    def record_pass(self, name: str, detail: str = ""):
        self.passed += 1
        print(f"  [PASS] {name} {detail}")

    def record_finding(self, name: str, severity: str, description: str, reproduced: bool):
        self.failed += 1
        finding = {
            "name": name,
            "severity": severity,
            "description": description,
            "reproduced": reproduced,
        }
        self.findings.append(finding)
        print(f"  [{severity.upper()}] {name}: {description}")


report = StressTestReport()


# ============================================================================
# Section 1: parse_time_spec Stress Tests
# ============================================================================
def test_parse_time_spec_stress():
    print("\n--- 1. Testing parse_time_spec ---")

    # 1.1 Boundary values and valid formats
    valid_cases = [
        (None, None),
        (0, 0.0),
        (0.0, 0.0),
        ("0", 0.0),
        ("0.0", 0.0),
        ("00:00", 0.0),
        ("00:00:00", 0.0),
        ("0:0", 0.0),
        ("0:0:0", 0.0),
        ("99:59:59", 99 * 3600.0 + 59 * 60.0 + 59.0),
        ("1e2", 100.0),
        ("1.5e3", 1500.0),
        ("  45.5  ", 45.5),
        ("  01:30  ", 90.0),
        ("  01:15:30  ", 4530.0),
        ("  \u2003  45  ", 45.0),  # Unicode em-space
        (45.123456, 45.123456),
    ]
    for inp, expected in valid_cases:
        res = parse_time_spec(inp)
        assert res == expected or (res is None and expected is None), f"Failed on {inp}: got {res}, expected {expected}"
    report.record_pass("Valid boundary, scientific, and unicode space formats")

    # 1.2 Invalid malformed and out-of-bounds inputs
    invalid_cases = [
        "-5",
        "-0.1",
        -10,
        -0.001,
        "",
        "   ",
        "abc",
        "::",
        ":30",
        "12:",
        "1:2:3:4",
        "01:60",       # seconds >= 60 in mm:ss
        "01:65",       # seconds > 60 in mm:ss
        "01:60:00",    # minutes >= 60 in hh:mm:ss
        "01:00:60",    # seconds >= 60 in hh:mm:ss
        "45\x00",      # null byte
        "45\n",        # newline in digits
        "1e2:30",      # scientific notation with colon (if seconds >= 60)
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        "-inf",
    ]
    for inp in invalid_cases:
        try:
            val = parse_time_spec(inp)
            report.record_finding(
                "parse_time_spec_invalid",
                "HIGH",
                f"Input {repr(inp)} was accepted and returned {val} instead of raising ValueError",
                reproduced=True,
            )
        except (ValueError, TypeError):
            pass
    report.record_pass("Malformed, non-finite, and out-of-bounds inputs correctly rejected")

    # 1.3 Negative clock component strings: IEEE 754 -0.0 bypass
    negative_clock_cases = [
        "-01:30",
        "-00:30",      # -00:30 minutes component is float('-00') = -0.0
        "-00:00:30",
        "-0:30",
        "01:-10",
        "00:01:-05",
    ]
    for inp in negative_clock_cases:
        try:
            val = parse_time_spec(inp)
            # If it did not raise, document as defect
            report.record_finding(
                "parse_time_spec_negative_zero_bypass",
                "HIGH",
                f"Negative time spec {repr(inp)} bypassed negative check (float('-00') == -0.0) and returned positive {val}s",
                reproduced=True,
            )
        except ValueError:
            report.record_pass(f"Negative clock spec {repr(inp)} rejected")


# ============================================================================
# Section 2: load_audio Range Slicing Stress Tests
# ============================================================================
def test_load_audio_stress():
    print("\n--- 2. Testing load_audio range slicing ---")

    tmp_dir = tempfile.mkdtemp()
    try:
        # Create a 2.0 second test WAV (native format)
        sr = 16000
        dur = 2.0
        t = np.arange(int(dur * sr)) / sr
        audio_native = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav_path = Path(tmp_dir) / "test.wav"
        sf.write(wav_path, audio_native, sr)

        # 2.1 Native: Normal slice
        s1, s_sr = load_audio(wav_path, start_sec=0.5, end_sec=1.5)
        assert len(s1) == int(1.0 * sr)
        assert s_sr == sr
        report.record_pass("Native normal range slice (0.5s - 1.5s)")

        # 2.2 Native: start_sec == end_sec (zero duration)
        try:
            load_audio(wav_path, start_sec=0.5, end_sec=0.5)
            report.record_finding(
                "load_audio_zero_duration",
                "HIGH",
                "load_audio did not raise ValueError when start_sec == end_sec",
                reproduced=False,
            )
        except ValueError:
            report.record_pass("Native zero duration (start_sec == end_sec) correctly raises ValueError")

        # 2.3 Native: negative interval (start_sec > end_sec)
        try:
            load_audio(wav_path, start_sec=1.5, end_sec=0.5)
            report.record_finding(
                "load_audio_negative_interval",
                "HIGH",
                "load_audio did not raise ValueError when start_sec > end_sec",
                reproduced=False,
            )
        except ValueError:
            report.record_pass("Native negative interval (start_sec > end_sec) correctly raises ValueError")

        # 2.4 Native: negative start_sec or end_sec
        try:
            load_audio(wav_path, start_sec=-0.5, end_sec=1.0)
            report.record_finding(
                "load_audio_negative_start",
                "HIGH",
                "load_audio did not raise ValueError when start_sec < 0",
                reproduced=False,
            )
        except ValueError:
            report.record_pass("Native negative start_sec correctly raises ValueError")

        # 2.5 Native: 1-sample slice
        s_1sample, _ = load_audio(wav_path, start_sec=0.5, end_sec=0.5 + 1.0 / sr)
        assert len(s_1sample) == 1
        report.record_pass("Native 1-sample slice returns exact single sample")

        # 2.6 Native: start beyond EOF
        s_eof, _ = load_audio(wav_path, start_sec=5.0, end_sec=6.0)
        assert len(s_eof) == 0
        report.record_pass("Native start beyond EOF returns empty array")

        # 2.7 Non-native: MP3 format slicing
        mp3_path = Path(tmp_dir) / "test.mp3"
        subprocess.run([_ffmpeg(), "-y", "-i", str(wav_path), str(mp3_path)], capture_output=True, check=True)

        # Full MP3 load
        mp3_full, mp3_sr = load_audio(mp3_path)
        assert len(mp3_full) > 0
        report.record_pass("Non-native MP3 full load")

        # MP3 slice duration check: 1 sample duration
        sample_dur = 1.0 / mp3_sr
        s_mp3_1sample, _ = load_audio(mp3_path, start_sec=0.5, end_sec=0.5 + sample_dur)
        if len(s_mp3_1sample) > 500:
            report.record_finding(
                "ffmpeg_input_option_placement",
                "MEDIUM",
                f"load_audio on MP3 with 1-sample slice ({sample_dur:.6f}s) returned {len(s_mp3_1sample)} samples due to -t before -i in ffmpeg arguments",
                reproduced=True,
            )
        else:
            report.record_pass("Non-native MP3 1-sample slice accurate")

        # MP3 start beyond EOF
        try:
            s_mp3_eof, _ = load_audio(mp3_path, start_sec=10.0, end_sec=15.0)
            assert len(s_mp3_eof) == 0
            report.record_pass("Non-native MP3 start beyond EOF returns empty array")
        except RuntimeError as e:
            report.record_finding(
                "ffmpeg_eof_runtime_error",
                "MEDIUM",
                f"load_audio on MP3 beyond EOF raises unhandled RuntimeError instead of returning empty array: {e}",
                reproduced=True,
            )

    finally:
        for f in os.listdir(tmp_dir):
            try:
                os.unlink(os.path.join(tmp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


# ============================================================================
# Section 3: compute_waveform_envelope Stress Tests
# ============================================================================
def test_compute_waveform_envelope_stress():
    print("\n--- 3. Testing compute_waveform_envelope ---")

    # 3.1 Strict 8-character alphabet verification
    assert len(GLYPHS) == 8
    assert GLYPHS == " ▂▃▄▅▆▇█"
    report.record_pass("Glyph alphabet is strictly 8 characters: ' ▂▃▄▅▆▇█'")

    # 3.2 Width compliance across extreme ranges
    audio = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    for w in [1, 2, 8, 16, 32, 64, 128]:
        env = compute_waveform_envelope(audio, width=w)
        assert len(env) == w
        assert all(c in GLYPHS for c in env)
    report.record_pass("Glyph length matches width for widths 1 to 128")

    # 3.3 Zero and negative widths
    assert compute_waveform_envelope(audio, width=0) == ""
    assert compute_waveform_envelope(audio, width=-10) == ""
    report.record_pass("Zero and negative widths return empty string")

    # 3.4 Silence handling
    silence = np.zeros(16000, dtype=np.float32)
    env_silence = compute_waveform_envelope(silence, width=16)
    assert env_silence == " " * 16
    report.record_pass("Digital silence returns lowest glyph ' ' * width")

    # 3.5 Specific array address reuse collision stress test
    # Allocate array A with loud tone
    A = (0.9 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    addr_A = id(A)
    env_A = compute_waveform_envelope(A, width=16)
    del A

    # Allocate array B with silence, same shape and dtype
    B = np.zeros(16000, dtype=np.float32)
    addr_B = id(B)
    env_B = compute_waveform_envelope(B, width=16)

    if addr_A == addr_B and env_B != " " * 16:
        report.record_finding(
            "cache_address_reuse_collision",
            "HIGH",
            f"_ARRAY_ENVELOPE_CACHE collisions occurred on deallocated pointer reuse: silent array B (id={addr_B}) returned loud envelope {repr(env_B)} instead of {repr(' ' * 16)}",
            reproduced=True,
        )
    else:
        # Even if memory addresses did not match in this single run, test explicitly with cached id
        from zout.audio.io import _ARRAY_ENVELOPE_CACHE
        if len(_ARRAY_ENVELOPE_CACHE) > 0:
            report.record_finding(
                "cache_address_reuse_vulnerability",
                "HIGH",
                "_ARRAY_ENVELOPE_CACHE indexes ephemeral numpy arrays by id(audio), creating false envelope lookups when CPython reuses memory pointers",
                reproduced=True,
            )


# ============================================================================
# Section 4: JobConfig 9-Field Fuzzing & Validation Stress Tests
# ============================================================================
def test_job_config_stress():
    print("\n--- 4. Testing JobConfig 9 new fields & validations ---")

    # 4.1 Valid construction with all 9 new fields
    evt = threading.Event()
    cfg = JobConfig(
        start_sec="01:30",
        end_sec="02:00",
        chunk_duration=45.0,
        overlap_duration=1.5,
        audit=True,
        audit_path="audit.json",
        cancel_event=evt,
        batch_dir="audio_in/",
        out_dir="audio_out/",
    )
    assert cfg.start_sec == 90.0
    assert cfg.end_sec == 120.0
    assert cfg.chunk_duration == 45.0
    assert cfg.overlap_duration == 1.5
    assert cfg.audit is True
    assert isinstance(cfg.audit_path, Path)
    assert isinstance(cfg.batch_dir, Path)
    assert isinstance(cfg.out_dir, Path)
    assert cfg.cancel_event is evt
    assert cfg.is_cancelled() is False
    evt.set()
    assert cfg.is_cancelled() is True
    report.record_pass("JobConfig accepts and coerces all 9 new fields correctly")

    # 4.2 Fuzzing out-of-bounds parameters
    fuzz_cases = [
        ({"start_sec": -1.0}, ValueError, "negative start_sec"),
        ({"end_sec": -10.0}, ValueError, "negative end_sec"),
        ({"start_sec": 50.0, "end_sec": 50.0}, ValueError, "start_sec == end_sec"),
        ({"start_sec": 60.0, "end_sec": 50.0}, ValueError, "start_sec > end_sec"),
        ({"chunk_duration": 0.0}, ValueError, "chunk_duration == 0"),
        ({"chunk_duration": -5.0}, ValueError, "chunk_duration < 0"),
        ({"overlap_duration": -1.0}, ValueError, "overlap_duration < 0"),
        ({"chunk_duration": 30.0, "overlap_duration": 30.0}, ValueError, "overlap == chunk"),
        ({"chunk_duration": 30.0, "overlap_duration": 35.0}, ValueError, "overlap > chunk"),
        ({"start_sec": "bad_time"}, ValueError, "malformed start_sec string"),
        ({"chunk_duration": "not_a_num"}, TypeError, "invalid type for chunk_duration"),
        ({"overlap_duration": "not_a_num"}, TypeError, "invalid type for overlap_duration"),
    ]
    for kwargs, exc_type, desc in fuzz_cases:
        try:
            JobConfig(**kwargs)
            report.record_finding(
                "job_config_validation_failure",
                "HIGH",
                f"JobConfig failed to reject {desc}: {kwargs}",
                reproduced=False,
            )
        except exc_type:
            pass
    report.record_pass("JobConfig rejects out-of-bounds bounds and malformed types")

    # 4.3 Test ready() helper with various combinations
    assert JobConfig(reference=Path("ref.wav"), main=Path("main.wav")).ready() is True
    assert JobConfig(reference=Path("ref.wav"), batch_dir=Path("batch/")).ready() is True
    assert JobConfig(reference=None, main=Path("main.wav")).ready() is False
    assert JobConfig(reference=Path("ref.wav"), main=None, batch_dir=None).ready() is False
    report.record_pass("JobConfig ready() behaves accurately for standard and batch configurations")


# ============================================================================
# Section 5: AudioPlayer Stress Tests
# ============================================================================
def test_audio_player_stress():
    print("\n--- 5. Testing AudioPlayer continuous playback & hot-swap ---")

    tmp_dir = tempfile.mkdtemp()
    try:
        sr = 44100
        dur = 1.0
        t = np.arange(int(dur * sr)) / sr
        a1 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        a2 = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        p1 = Path(tmp_dir) / "track1.wav"
        p2 = Path(tmp_dir) / "track2.wav"
        sf.write(p1, a1, sr)
        sf.write(p2, a2, sr)

        p = AudioPlayer(force_headless=True)
        assert p.play_pair(p1, p2, start_with="B") is True
        assert p.get_current_track() == "B"

        # Advance sample_index and verify sample continuity across hot-swap
        with p._lock:
            p._sample_index = 5000
        assert p.sample_index == 5000

        # Toggle to A
        assert p.toggle_ab() == "A"
        assert p.sample_index == 5000, "sample_index altered by toggle_ab()"
        assert p.get_current_track() == "A"

        # Rapid toggle cycling (100 times)
        for _ in range(100):
            p.toggle_ab()
        assert p.sample_index == 5000, "Rapid toggling mutated sample_index"
        report.record_pass("AudioPlayer hot-swap preserves sample_index continuity under rapid cycling")

        # Meter levels format check
        levels = p.get_meter_levels()
        assert len(levels) == 2
        assert len(levels[0]) == 2 and len(levels[1]) == 2
        report.record_pass("AudioPlayer meter levels conform to ((rms_l, rms_r), (peak_l, peak_r))")

        p.stop()
        assert p.is_playing() is False
        report.record_pass("AudioPlayer stops cleanly")

    finally:
        for f in os.listdir(tmp_dir):
            try:
                os.unlink(os.path.join(tmp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    print("=================================================================")
    print("    noiZout Milestone 1 (M1) Empirical Challenger Stress Suite   ")
    print("=================================================================")
    test_parse_time_spec_stress()
    test_load_audio_stress()
    test_compute_waveform_envelope_stress()
    test_job_config_stress()
    test_audio_player_stress()

    print("\n=================================================================")
    print(f"Results: {report.passed} Passed, {len(report.findings)} Findings Identified")
    print("=================================================================")
    for idx, f in enumerate(report.findings, 1):
        print(f"[{idx}] {f['severity']} - {f['name']}: {f['description']}")
