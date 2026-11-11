"""Tier 5 Adversarial Coverage Hardening Suite for noiZout.

White-box stress tests, deep boundary edge cases, concurrency validation,
numerical stability tests, and architectural invariant checks across:
1. Audio Engine Boundaries & Chunking (zout/engines/)
2. Audio Player Concurrency & Hot-Swap (zout/audio/player.py)
3. Forensic Audit Sidecar & Metrics (zout/audio/audit.py)
4. TUI & CLI Permutations & Fault-Tolerance (zout/tui/, zout/app.py)
5. Zero Textual Constraint Enforcement
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from zout.app import _build_parser, _validate
from zout.audio.audit import (
    compute_rms,
    compute_speech_summary,
    compute_true_peak,
    extract_vad_intervals,
    generate_audit_manifest,
    sanitize_float,
)
from zout.audio.io import (
    compute_waveform_envelope,
    load_audio,
    parse_time_spec,
    peak_limit,
    resample,
    save_audio,
)
from zout.audio.masking import apply_adaptive_mask, min_gain
from zout.audio.player import AudioPlayer, _calc_dbfs, _calc_meters, _ensure_stereo
from zout.config import JobConfig, ReferenceRole
from zout.engines.noisy_engine import NoisyAcousticEngine
from zout.pipeline import Result, run_batch, run_job, run_streaming
from zout.tui import themes
from zout.tui.keys import Key, poll_key, read_key
from zout.tui.screens import render_stereo_vu_meter


# ==============================================================================
# Helper fixtures and generator utilities
# ==============================================================================


def _generate_synthetic_tone(
    duration_sec: float,
    sr: int = 16000,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate pure sine tone."""
    n_samples = int(round(duration_sec * sr))
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _generate_speech_proxy(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Generate synthetic speech-like harmonic bursts for VAD detection."""
    n_samples = int(round(duration_sec * sr))
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    carrier = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1760 * t)
    )
    modulator = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    return (carrier * modulator).astype(np.float32)


# ==============================================================================
# Section 1: Audio Engine Boundaries & Chunking (zout/engines/)
# ==============================================================================


class TestTier5AudioEngineBoundaries:
    """Adversarial testing of audio engines under exact sample boundaries,
    odd sample lengths, non-standard sample rates, and extreme attenuation.
    """

    @pytest.mark.parametrize(
        "sample_count",
        [239999, 240000, 240001, 479999, 480000, 480001],
    )
    def test_exact_chunk_boundaries_sample_preservation(self, tmp_path: Path, sample_count: int):
        """Verify audio processing across exact boundary transitions (240k and 480k).

        Ensures exact sample count output, zero NaNs, and zero Infs.
        """
        sr = 16000
        ref_path = tmp_path / f"ref_boundary_{sample_count}.wav"
        main_path = tmp_path / f"main_boundary_{sample_count}.wav"
        out_path = tmp_path / f"out_boundary_{sample_count}.wav"

        sf.write(str(ref_path), np.random.randn(sr * 2).astype(np.float32) * 0.1, sr)
        audio_in = (np.random.randn(sample_count).astype(np.float32) * 0.2)
        sf.write(str(main_path), audio_in, sr)

        cfg = JobConfig(
            reference=ref_path,
            main=main_path,
            output=out_path,
            role=ReferenceRole.NOISY,
            strength=0.75,
            max_atten_db=24.0,
        )
        res = run_job(cfg)
        assert res is not None, "Pipeline failed to produce result at boundary"
        assert res.output.is_file(), f"Output file missing for length {sample_count}"

        out_data, out_sr = sf.read(str(res.output))
        assert out_sr == sr
        assert len(out_data) == sample_count, (
            f"Sample count mismatch: input {sample_count} != output {len(out_data)}"
        )
        assert np.all(np.isfinite(out_data)), "Non-finite values (NaN/Inf) found in output"

    @pytest.mark.parametrize("odd_length", [240003, 240005, 360001])
    def test_odd_sample_length_chunked_streaming(self, tmp_path: Path, odd_length: int):
        """Verify chunked overlap-add streaming with odd sample lengths.

        Ensures the final remainder slice (remainder = y[N_overlap:]) is correctly
        handled and does not cause length truncation or off-by-one errors.
        """
        sr = 16000
        ref_path = tmp_path / f"ref_odd_{odd_length}.wav"
        main_path = tmp_path / f"main_odd_{odd_length}.wav"
        out_path = tmp_path / f"out_odd_{odd_length}.wav"

        sf.write(str(ref_path), np.random.randn(sr * 2).astype(np.float32) * 0.1, sr)
        audio_in = np.sin(np.linspace(0, 100, odd_length, dtype=np.float32)) * 0.3
        sf.write(str(main_path), audio_in, sr)

        cfg = JobConfig(
            reference=ref_path,
            main=main_path,
            output=out_path,
            role=ReferenceRole.NOISY,
        )
        res = run_job(cfg)
        assert res is not None
        out_data, _ = sf.read(str(res.output))
        assert len(out_data) == odd_length, (
            f"Odd length mismatch: expected {odd_length}, got {len(out_data)}"
        )
        assert not np.isnan(out_data).any()

    @pytest.mark.parametrize("sr", [11025, 22050, 32000, 88200])
    def test_non_standard_sample_rates(self, tmp_path: Path, sr: int):
        """Verify audio engine execution across non-standard and high sample rates."""
        ref_path = tmp_path / f"ref_sr_{sr}.wav"
        main_path = tmp_path / f"main_sr_{sr}.wav"
        out_path = tmp_path / f"out_sr_{sr}.wav"

        dur = 2.0
        n_samples = int(dur * sr)
        sf.write(str(ref_path), np.random.randn(n_samples).astype(np.float32) * 0.1, sr)
        sf.write(str(main_path), np.random.randn(n_samples).astype(np.float32) * 0.2, sr)

        cfg = JobConfig(
            reference=ref_path,
            main=main_path,
            output=out_path,
            role=ReferenceRole.NOISY,
        )
        res = run_job(cfg)
        assert res is not None
        out_data, out_sr = sf.read(str(res.output))
        assert out_sr == sr, f"Sample rate was not preserved: expected {sr}, got {out_sr}"
        assert len(out_data) == n_samples
        assert np.all(np.isfinite(out_data))

    def test_extreme_attenuation_and_strength_boundaries(self):
        """Verify min_gain and adaptive mask behavior at extreme attenuation caps."""
        # Case A: strength = 1.0 and max_atten_db >= 60.0 -> min_gain must be exactly 0.0
        g_uncapped = min_gain(strength=1.0, max_atten_db=60.0)
        assert g_uncapped == 0.0, f"Expected 0.0 for uncapped suppression, got {g_uncapped}"

        # Case B: strength = 0.0 and max_atten_db = 6.0 -> capped to min attenuation
        g_min_atten = min_gain(strength=0.0, max_atten_db=6.0)
        expected_floor = 10 ** (-6.0 / 20.0)  # ~0.501
        assert abs(g_min_atten - expected_floor) < 1e-4

        # Case C: apply_adaptive_mask with g = 0.0 returns purely y_est
        x = np.ones(256, dtype=np.float32)
        y_est = np.zeros(256, dtype=np.float32)
        masked = apply_adaptive_mask(x, y_est, 16000, strength=1.0, max_atten_db=60.0)
        assert np.allclose(masked, 0.0), "Mask did not eliminate signal when uncapped"

    def test_sub_minimum_recording_length_guard(self, tmp_path: Path):
        """Verify that audio recordings below minimum duration (< sr // 4) are rejected safely."""
        sr = 16000
        ref_path = tmp_path / "ref_short.wav"
        main_path = tmp_path / "main_too_short.wav"
        sf.write(str(ref_path), np.random.randn(sr * 2).astype(np.float32), sr)

        # 100 samples is well below sr // 4 (4000 samples)
        sf.write(str(main_path), np.random.randn(100).astype(np.float32), sr)

        cfg = JobConfig(reference=ref_path, main=main_path, role=ReferenceRole.NOISY)
        with pytest.raises(ValueError, match="main recording is too short"):
            run_job(cfg)


# ==============================================================================
# Section 2: Audio Player Concurrency & Hot-Swap (zout/audio/player.py)
# ==============================================================================


class TestTier5AudioPlayerConcurrencyAndMeters:
    """Adversarial stress-testing of AudioPlayer under heavy concurrency,
    subnormal meter inputs, EOF transitions, and continuous playback hot-swaps.
    """

    def test_concurrent_hot_swap_hammering(self, tmp_path: Path):
        """Verify thread-safety of toggle_ab under rapid concurrent multi-threaded hammering."""
        sr = 16000
        p_raw = tmp_path / "raw_concur.wav"
        p_proc = tmp_path / "proc_concur.wav"

        # 4 seconds of distinct tones
        sf.write(str(p_raw), _generate_synthetic_tone(4.0, sr=sr, freq=440.0), sr)
        sf.write(str(p_proc), _generate_synthetic_tone(4.0, sr=sr, freq=880.0), sr)

        player = AudioPlayer(force_headless=True)
        try:
            started = player.play_pair(p_raw, p_proc, start_with="B")
            assert started is True
            assert player.is_playing()

            thread_errors: list[Exception] = []
            num_threads = 12
            iterations = 40

            def hammer_worker():
                try:
                    for _ in range(iterations):
                        trk = player.toggle_ab()
                        assert trk in ("A", "B"), f"Invalid track identifier: {trk}"
                        current = player.get_current_track()
                        assert current in ("A", "B")
                        levels = player.get_meter_levels()
                        assert len(levels) == 2
                        (rms_l, rms_r), (peak_l, peak_r) = levels
                        assert math.isfinite(rms_l) and math.isfinite(rms_r)
                        assert math.isfinite(peak_l) and math.isfinite(peak_r)
                        time.sleep(0.002)
                except Exception as ex:
                    thread_errors.append(ex)

            threads = [threading.Thread(target=hammer_worker) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(thread_errors) == 0, f"Thread safety violated: {thread_errors}"
        finally:
            player.stop()

    def test_audio_player_eof_lifecycle_and_meter_reset(self, tmp_path: Path):
        """Verify player behavior upon reaching EOF: clean stop, sample preservation, meter reset."""
        sr = 16000
        p_short = tmp_path / "eof_test.wav"
        # 0.08s of audio = 1280 samples
        sf.write(str(p_short), np.random.randn(1280).astype(np.float32) * 0.3, sr)

        player = AudioPlayer(force_headless=True)
        try:
            ok = player.play_pair(p_short, p_short, start_with="A")
            assert ok is True
            assert player.is_playing()

            # Wait for runner to consume 1280 samples and reach EOF
            time.sleep(0.35)

            assert not player.is_playing(), "Player did not transition to stopped at EOF"
            assert player.sample_index >= 1280, f"Unexpected sample_index at EOF: {player.sample_index}"
            meters = player.get_meter_levels()
            assert meters == ((-80.0, -80.0), (-80.0, -80.0)), "Meters did not reset to -80 dBFS at EOF"

            # Verify toggle still works safely in idle state
            next_trk = player.toggle_ab()
            assert next_trk == "B"
        finally:
            player.stop()

    def test_meter_calculation_numerical_extremes(self):
        """Verify _calc_meters and _calc_dbfs against NaNs, Infs, zeros, and subnormals."""
        # 1. Pure silence (all zeros)
        silence = np.zeros((1024, 2), dtype=np.float32)
        rms_l, rms_r, peak_l, peak_r = _calc_meters(silence)
        assert (rms_l, rms_r, peak_l, peak_r) == (-80.0, -80.0, -80.0, -80.0)

        # 2. Full scale (all +1.0)
        full_scale = np.ones((1024, 2), dtype=np.float32)
        rms_l, rms_r, peak_l, peak_r = _calc_meters(full_scale)
        assert (rms_l, rms_r, peak_l, peak_r) == (0.0, 0.0, 0.0, 0.0)

        # 3. NaNs in input block (must not raise or produce NaN)
        nan_block = np.full((1024, 2), np.nan, dtype=np.float32)
        rms_l, rms_r, peak_l, peak_r = _calc_meters(nan_block)
        assert all(math.isfinite(v) for v in (rms_l, rms_r, peak_l, peak_r))
        assert (rms_l, rms_r, peak_l, peak_r) == (-80.0, -80.0, -80.0, -80.0)

        # 4. Infs in input block (must cap at 0.0 dBFS ceiling)
        inf_block = np.full((1024, 2), np.inf, dtype=np.float32)
        rms_l, rms_r, peak_l, peak_r = _calc_meters(inf_block)
        assert all(math.isfinite(v) for v in (rms_l, rms_r, peak_l, peak_r))
        assert (rms_l, rms_r, peak_l, peak_r) == (0.0, 0.0, 0.0, 0.0)

        # 5. Denormal / Subnormal numbers (e.g. 1e-45)
        denorm_block = np.full((1024, 2), 1e-45, dtype=np.float32)
        rms_l, rms_r, peak_l, peak_r = _calc_meters(denorm_block)
        assert (rms_l, rms_r, peak_l, peak_r) == (-80.0, -80.0, -80.0, -80.0)

        # 6. Empty block (0, 2)
        empty_block = np.zeros((0, 2), dtype=np.float32)
        rms_l, rms_r, peak_l, peak_r = _calc_meters(empty_block)
        assert (rms_l, rms_r, peak_l, peak_r) == (-80.0, -80.0, -80.0, -80.0)

    def test_ensure_stereo_shape_permutations(self):
        """Verify _ensure_stereo handles 1D, 2D mono, and multichannel cleanly."""
        # 1D array -> stereo (N, 2)
        a1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        s1 = _ensure_stereo(a1)
        assert s1.shape == (3, 2)
        assert np.array_equal(s1[:, 0], s1[:, 1])

        # 2D (N, 1) -> stereo (N, 2)
        a2 = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        s2 = _ensure_stereo(a2)
        assert s2.shape == (3, 2)

        # 2D (N, 6) surround -> clamped to first 2 channels (N, 2)
        a6 = np.random.randn(100, 6).astype(np.float32)
        s6 = _ensure_stereo(a6)
        assert s6.shape == (100, 2)
        assert np.array_equal(s6[:, 0], a6[:, 0])
        assert np.array_equal(s6[:, 1], a6[:, 1])

        # 3D invalid shape -> ValueError
        with pytest.raises(ValueError, match="Unsupported audio shape"):
            _ensure_stereo(np.zeros((10, 2, 2)))


# ==============================================================================
# Section 3: Forensic Audit Sidecar & Metrics (zout/audio/audit.py)
# ==============================================================================


class TestTier5ForensicAuditManifest:
    """Adversarial stress-testing of forensic JSON audit manifests,
    true-peak ITU-R BS.1770 computation, multichannel VAD, and bit-depth resilience.
    """

    def test_multichannel_and_bit_depth_audit_metrics(self, tmp_path: Path):
        """Verify audit metrics on 6-channel surround sound, 16-bit PCM, and 32-bit float."""
        sr = 48000
        n_samples = sr * 2

        # 1. 6-Channel 32-bit float
        path_6ch = tmp_path / "surround_6ch.wav"
        data_6ch = np.random.randn(n_samples, 6).astype(np.float32) * 0.15
        sf.write(str(path_6ch), data_6ch, sr, subtype="FLOAT")

        tp_6ch = compute_true_peak(path_6ch)
        rms_6ch = compute_rms(path_6ch)
        assert isinstance(tp_6ch, float) and -60.0 <= tp_6ch <= 10.0
        assert isinstance(rms_6ch, float) and -60.0 <= rms_6ch <= 10.0

        # 2. 16-bit PCM stereo
        path_16bit = tmp_path / "pcm_16bit.wav"
        data_16bit = (np.random.randn(n_samples, 2) * 0.1).astype(np.float32)
        sf.write(str(path_16bit), data_16bit, sr, subtype="PCM_16")

        tp_16 = compute_true_peak(path_16bit)
        rms_16 = compute_rms(path_16bit)
        assert math.isfinite(tp_16) and math.isfinite(rms_16)

        # 3. 24-bit PCM mono
        path_24bit = tmp_path / "pcm_24bit.wav"
        data_24bit = (np.random.randn(n_samples) * 0.1).astype(np.float32)
        sf.write(str(path_24bit), data_24bit, sr, subtype="PCM_24")

        tp_24 = compute_true_peak(path_24bit)
        rms_24 = compute_rms(path_24bit)
        assert math.isfinite(tp_24) and math.isfinite(rms_24)

    def test_vad_acoustic_hum_rejection_vs_speech_detection(self):
        """Verify that pure acoustic hum (120 Hz) yields 0 VAD intervals, while speech proxy is detected."""
        sr = 16000
        dur = 2.0

        # Case A: Pure 120Hz acoustic powerline hum at loud amplitude
        hum = _generate_synthetic_tone(dur, sr=sr, freq=120.0, amplitude=0.8)
        vad_hum = extract_vad_intervals(hum, sr=sr)
        assert len(vad_hum) == 0, f"Expected 0 speech intervals for pure hum, got {len(vad_hum)}"

        # Case B: Synthetic harmonic speech proxy
        speech = _generate_speech_proxy(dur, sr=sr)
        vad_speech = extract_vad_intervals(speech, sr=sr, threshold=0.15)
        # Should identify speech presence
        assert len(vad_speech) >= 1, "Failed to detect synthetic speech intervals"
        assert vad_speech[0]["duration"] >= 0.03

    def test_audit_manifest_strict_json_rfc8259_schema(self, tmp_path: Path):
        """Verify generated audit manifest matches all required schema keys and RFC 8259 compliance."""
        sr = 16000
        ref_path = tmp_path / "audit_schema_ref.wav"
        main_path = tmp_path / "audit_schema_main.wav"
        out_path = tmp_path / "audit_schema_out.wav"
        audit_json_path = tmp_path / "audit_custom.json"

        sf.write(str(ref_path), np.random.randn(sr * 2).astype(np.float32) * 0.1, sr)
        sf.write(str(main_path), _generate_speech_proxy(2.0, sr=sr), sr)

        cfg = JobConfig(
            reference=ref_path,
            main=main_path,
            output=out_path,
            role=ReferenceRole.NOISY,
            audit=True,
            audit_path=audit_json_path,
        )

        manifest_file = generate_audit_manifest(
            cfg=cfg,
            output_path=out_path,
            input_audio=main_path,
            output_audio=main_path,
            sr=sr,
            suppression_db=14.5,
            in_rms=-18.2,
            out_rms=-32.7,
            audit_path=audit_json_path,
        )

        assert manifest_file.is_file()
        content = manifest_file.read_text(encoding="utf-8")
        parsed = json.loads(content)

        # Mandatory schema checks
        required_keys = [
            "version",
            "timestamp",
            "noise_reduction_db",
            "suppression_db",
            "true_peak",
            "true_peak_dbfs",
            "rms",
            "rms_dbfs",
            "vad_speech_timestamps",
            "speech_summary",
            "parameters",
        ]
        for k in required_keys:
            assert k in parsed, f"Missing required manifest key: {k}"

        assert parsed["noise_reduction_db"] == 14.5
        assert parsed["true_peak"]["input"] is not None
        assert parsed["true_peak"]["output"] is not None
        assert parsed["parameters"]["role"] in ("NOISY", "noisy")
        assert isinstance(parsed["vad_speech_timestamps"], list)
        assert "speech_duration_sec" in parsed["speech_summary"]


# ==============================================================================
# Section 4: TUI & CLI Permutations & Fault-Tolerance (zout/tui/, zout/app.py)
# ==============================================================================


class TestTier5TuiAndCliPermutations:
    """Adversarial testing of CLI argument parsing permutations,
    batch directory fault tolerance, key events, and theme concurrency.
    """

    def test_cli_unbalanced_and_zero_start_time_permutations(self):
        """Verify CLI permits --start without --end, --end without --start, and --start 00:00:00."""
        ap = _build_parser()

        # 1. --start only
        args1 = ap.parse_args(["-r", "dummy.wav", "-m", "dummy.wav", "-t", "noisy", "--start", "00:30"])
        assert args1.start == "00:30"
        assert args1.end is None

        # 2. --end only
        args2 = ap.parse_args(["-r", "dummy.wav", "-m", "dummy.wav", "-t", "noisy", "--end", "02:15.5"])
        assert args2.end == "02:15.5"
        assert args2.start is None

        # 3. --start 00:00:00
        args3 = ap.parse_args(["-r", "dummy.wav", "-m", "dummy.wav", "-t", "noisy", "--start", "00:00:00"])
        assert args3.start == "00:00:00"

        # 4. Invalid --start > --end raises SystemExit in validate
        args4 = ap.parse_args(["-r", "dummy.wav", "-m", "dummy.wav", "-t", "noisy", "--start", "10", "--end", "5"])
        with pytest.raises(SystemExit):
            _validate(ap, args4)

        # 5. Invalid clock format minute >= 60 raises SystemExit
        args5 = ap.parse_args(["-r", "dummy.wav", "-m", "dummy.wav", "-t", "noisy", "--start", "01:60:00"])
        with pytest.raises(SystemExit):
            _validate(ap, args5)

    def test_cli_batch_mode_mutual_exclusion_and_validation(self):
        """Verify CLI batch options validate mutual exclusion and directory requirements."""
        ap = _build_parser()

        # Batch and main are mutually exclusive
        args_both = ap.parse_args(["-r", "dummy.wav", "-m", "dummy.wav", "-t", "noisy", "--batch", "."])
        with pytest.raises(SystemExit):
            _validate(ap, args_both)

        # Batch without ref and role raises error during validation
        args_no_ref = ap.parse_args(["--batch", "."])
        with pytest.raises(SystemExit):
            _validate(ap, args_no_ref)

    def test_batch_pipeline_resilience_corrupted_files_and_self_ref(self, tmp_path: Path):
        """Verify run_batch processes valid files, logs corrupt files, and skips the reference file."""
        sr = 16000
        batch_dir = tmp_path / "batch_input"
        batch_dir.mkdir()
        out_dir = tmp_path / "batch_output"

        # Reference audio inside batch directory
        ref_file = batch_dir / "reference_noise.wav"
        sf.write(str(ref_file), np.random.randn(sr * 2).astype(np.float32), sr)

        # Valid audio item
        item1 = batch_dir / "valid_voice.wav"
        sf.write(str(item1), np.random.randn(sr * 2).astype(np.float32), sr)

        # Corrupt 0-byte file
        corrupt = batch_dir / "corrupted_empty.wav"
        corrupt.write_bytes(b"")

        # Non-audio file
        readme = batch_dir / "notes.txt"
        readme.write_text("ignore this file", encoding="utf-8")

        cfg = JobConfig(
            reference=ref_file,
            batch_dir=batch_dir,
            out_dir=out_dir,
            role=ReferenceRole.NOISY,
        )

        res = run_batch(cfg)
        assert res is not None
        notes = res.notes
        assert notes.get("batch") is True
        assert notes.get("success") == 1
        assert notes.get("total") == 2  # valid_voice + corrupted_empty (reference excluded)
        assert len(notes.get("failures", [])) == 1

        # Check output directory contains processed file
        out_files = list(out_dir.glob("*.wav"))
        assert len(out_files) == 1
        assert out_files[0].name == "valid_voice_zout.wav"

    def test_theme_manager_rapid_concurrent_cycling(self):
        """Verify thread safety and memory stability of ThemeManager under rapid cycling."""
        errors: list[Exception] = []

        def cycle_worker():
            try:
                for _ in range(50):
                    t = themes.cycle_theme()
                    assert t.name in [
                        "Forensic Cyan",
                        "Amber Phosphor",
                        "Matrix Emerald",
                        "Studio Dark",
                    ]
                    # Lookup check
                    cur = themes.get_theme(t.name)
                    assert cur.name == t.name
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cycle_worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0, f"ThemeManager thread errors: {errors}"
        themes.reset()

    def test_stereo_vu_meter_table_generation_robustness(self):
        """Verify render_stereo_vu_meter renders cleanly without crashing on extreme levels."""
        # Level extremes: clip, normal, silence
        clip_levels = ((0.0, 0.0), (0.0, 0.0))
        table_clip = render_stereo_vu_meter(clip_levels, width=20)
        assert table_clip is not None

        silent_levels = ((-80.0, -80.0), (-80.0, -80.0))
        table_silence = render_stereo_vu_meter(silent_levels, width=20)
        assert table_silence is not None

        asymmetric_levels = ((-12.0, -45.0), (-3.0, -28.0))
        table_asym = render_stereo_vu_meter(asymmetric_levels, width=24)
        assert table_asym is not None


# ==============================================================================
# Section 5: Strictly Zero Textual Constraint Verification
# ==============================================================================


class TestTier5ZeroTextualEnforcement:
    """Adversarial verification of the strictly zero Textual library constraint.
    Inspects all Python files via AST and confirms absence of textual dependencies.
    """

    def test_strictly_zero_textual_imports_across_codebase(self):
        """Perform AST parse across all .py files in zout/ and ensure 0 textual imports."""
        repo_root = Path(__file__).resolve().parent.parent
        zout_dir = repo_root / "zout"
        assert zout_dir.is_dir(), "zout source directory not found"

        violations: list[str] = []

        for py_path in zout_dir.rglob("*.py"):
            code = py_path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(code, filename=str(py_path))
            except SyntaxError as e:
                violations.append(f"Syntax error parsing {py_path}: {e}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "textual" or alias.name.startswith("textual."):
                            violations.append(f"{py_path.name}:{node.lineno} -> import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod == "textual" or mod.startswith("textual."):
                        violations.append(f"{py_path.name}:{node.lineno} -> from {mod} import ...")

        assert len(violations) == 0, (
            "Violations of strictly zero Textual constraint detected in zout source:\n"
            + "\n".join(violations)
        )

    def test_pyproject_toml_strictly_excludes_textual(self):
        """Verify pyproject.toml does not declare textual in dependencies."""
        repo_root = Path(__file__).resolve().parent.parent
        pyproject_path = repo_root / "pyproject.toml"
        assert pyproject_path.is_file(), "pyproject.toml not found"

        content = pyproject_path.read_text(encoding="utf-8").lower()
        # Parse lines in dependencies section
        in_deps = False
        dep_lines = []
        for line in content.splitlines():
            if "dependencies" in line:
                in_deps = True
            elif in_deps and line.startswith("["):
                in_deps = False
            elif in_deps:
                dep_lines.append(line)

        for dl in dep_lines:
            assert "textual" not in dl, f"textual found in pyproject.toml dependencies: {dl}"
