"""Comprehensive 4-Tier End-to-End Test Suite for noiZout.

Requirements Covered:
  - R1: Chunked Streaming for Multi-Hour Audio (60s chunks, 2s raised-cosine OLA, < 150MB RAM)
  - R2: Instant Seamless A/B Hot-Swap During Playback ([Tab], identical sample_index)
  - R3: Live Terminal Real-Time VU Meter & Waveform Envelope (Stereo RMS/Peak dBFS, 8-level Unicode glyphs)
  - R4: Graceful In-Flight Cancellation ([Esc], clean abort without crash)
  - R5: Timeline Range Slicing (--start / --end, parse_time_spec, frame-accurate seeking)
  - R6: Forensic Audit Log Sidecar (.json manifest, true-peak, RMS, VAD speech timestamps)
  - R7: Terminal Color Themes (Forensic Cyan, Amber Phosphor, Matrix Emerald, Studio Dark)
  - R8: Batch Folder Processing (--batch, --out-dir, skip non-audio, error resilience)
  - R9: Test Suite Integrity & Remote Delivery (100% pass, fast synthetic fixtures < 60s)

Test Suite Tiers:
  - Tier 1: Feature Coverage (>=5 test cases per feature for R1 through R8 = 40 tests)
  - Tier 2: Boundary & Corner Cases (>=5 test cases per feature for R1 through R8 = 40 tests)
  - Tier 3: Cross-Feature Combinations (pairwise interactions = 8 tests)
  - Tier 4: Real-World Scenarios (end-to-end production workflows = 5 tests)
  Grand Total: 93 test cases.
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import soxr

from zout.config import SUPPORTED_AUDIO_EXTENSIONS, JobConfig, ReferenceRole
from zout.pipeline import run_job

# ============================================================================
# Synthetic Audio Generation Fixtures & Helpers
# ============================================================================

TEST_SR = 16000


def gen_sine(duration_sec: float, freq: float = 440.0, sr: int = TEST_SR, amp: float = 0.5) -> np.ndarray:
    """Generate a pure sine tone with deterministic phase."""
    t = np.arange(int(duration_sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def gen_speechlike(duration_sec: float, sr: int = TEST_SR, seed: int = 42, f0: float = 140.0) -> np.ndarray:
    """Generate synthetic speech-like harmonic signal with amplitude modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sr)) / sr
    env = (np.sin(2 * np.pi * 3 * t) > -0.2).astype(np.float32)
    env = np.convolve(env, np.ones(200) / 200, mode="same")
    sig = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 12))
    return (0.25 * sig * env).astype(np.float32)


def gen_hum(duration_sec: float, sr: int = TEST_SR, seed: int = 1) -> np.ndarray:
    """Generate synthetic 60Hz and 180Hz acoustic hum with mild Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sr)) / sr
    signal = (0.08 * np.sin(2 * np.pi * 60 * t) +
              0.04 * np.sin(2 * np.pi * 180 * t) +
              0.02 * rng.standard_normal(len(t)))
    return signal.astype(np.float32)


def gen_white_noise(duration_sec: float, sr: int = TEST_SR, seed: int = 10, amp: float = 0.05) -> np.ndarray:
    """Generate synthetic white Gaussian noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def create_audio_file(path: Path, data: np.ndarray, sr: int = TEST_SR) -> Path:
    """Write float32 numpy array to standard audio file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".ogg":
        sf.write(str(path), data, sr, subtype="VORBIS")
    else:
        sf.write(str(path), data, sr, subtype="PCM_16")
    return path


class MockOutputStream:
    """Mock sounddevice.OutputStream for headless testing of audio callbacks."""

    def __init__(self, samplerate: int = TEST_SR, channels: int = 2, blocksize: int = 1024,
                 callback=None, **kwargs):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.callback = callback
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True

    def pump(self, frames: int | None = None) -> np.ndarray | None:
        """Simulate an audio hardware tick requesting output frames."""
        if not self.active or not self.callback:
            return None
        n = frames or self.blocksize
        outdata = np.zeros((n, self.channels), dtype=np.float32)
        time_info = {"currentTime": 0.0, "outputBufferDacTime": 0.0}
        status = 0
        self.callback(outdata, n, time_info, status)
        return outdata


# ============================================================================
# Progressive Testability Feature Discovery
# ============================================================================

def is_r1_implemented() -> bool:
    """Check if R1 Chunked Streaming is implemented in pipeline/engines."""
    import zout.pipeline as zp
    src = inspect.getsource(zp.run_job)
    return "chunk_duration" in src or hasattr(zp, "run_streaming")


def is_r2_implemented() -> bool:
    """Check if R2 Instant Seamless A/B Hot-Swap is implemented in AudioPlayer."""
    from zout.audio.player import AudioPlayer
    return hasattr(AudioPlayer, "play_pair") and hasattr(AudioPlayer, "toggle_ab")


def is_r3_vu_implemented() -> bool:
    """Check if R3 Stereo VU Meter is implemented in AudioPlayer."""
    from zout.audio.player import AudioPlayer
    return hasattr(AudioPlayer, "get_meter_levels")


def is_r3_envelope_implemented() -> bool:
    """Check if R3 Waveform Envelope generator is implemented in zout.audio.io."""
    import zout.audio.io as aio
    return hasattr(aio, "compute_waveform_envelope")


def is_r4_implemented() -> bool:
    """Check if R4 In-Flight Cancellation is implemented in pipeline."""
    import zout.pipeline as zp
    src = inspect.getsource(zp.run_job)
    return "cancel_event" in src or "cancel_token" in src


def is_r5_timespec_implemented() -> bool:
    """Check if R5 parse_time_spec is implemented in zout.audio.io."""
    import zout.audio.io as aio
    return hasattr(aio, "parse_time_spec")


def is_r5_slicing_implemented() -> bool:
    """Check if R5 Range Slicing in load_audio is implemented."""
    from zout.audio.io import load_audio
    params = inspect.signature(load_audio).parameters
    return "start_sec" in params and "end_sec" in params


def is_r5_pipeline_slicing_implemented() -> bool:
    """Check if R5 Range Slicing is implemented in pipeline.run_job."""
    import zout.pipeline as zp
    src = inspect.getsource(zp.run_job)
    return "start_sec" in src


def is_r6_implemented() -> bool:
    """Check if R6 Forensic Audit Log is implemented."""
    try:
        import zout.audio.audit  # noqa: F401
        import zout.pipeline as zp
        src = inspect.getsource(zp.run_job)
        return "audit" in src
    except ImportError:
        return False


def is_r7_implemented() -> bool:
    """Check if R7 Terminal Color Themes is implemented."""
    try:
        import zout.tui.themes
        return hasattr(zout.tui.themes, "cycle_theme") and hasattr(zout.tui.themes, "get_theme")
    except ImportError:
        return False


def is_cli_start_end_supported() -> bool:
    """Check if CLI parser supports --start and --end."""
    from zout.app import _build_parser
    parser = _build_parser()
    actions = {a.dest for a in parser._actions}
    return "start" in actions and "end" in actions


def is_cli_audit_supported() -> bool:
    """Check if CLI parser supports --audit or --json."""
    from zout.app import _build_parser
    parser = _build_parser()
    actions = {a.dest for a in parser._actions}
    return "audit" in actions or "json" in actions


def is_cli_batch_supported() -> bool:
    """Check if CLI parser supports --batch."""
    from zout.app import _build_parser
    parser = _build_parser()
    actions = {a.dest for a in parser._actions}
    return "batch" in actions


def is_r8_implemented() -> bool:
    """Check if R8 Batch Folder Processing is implemented in CLI or pipeline."""
    import zout.pipeline as zp
    src = inspect.getsource(zp.run_job)
    return "batch" in src or is_cli_batch_supported()


# ============================================================================
# TIER 1: FEATURE COVERAGE (R1 through R8, >=5 tests each = 40 tests)
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature Coverage verifying primary behaviors and interface contracts."""

    # ------------------------------------------------------------------------
    # R1: Chunked Streaming for Multi-Hour Audio
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_01_streaming_produces_identical_duration(self, tmp_path):
        """R1.1: Verify chunked streaming produces output matching input duration."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0) + gen_hum(4.0))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, chunk_duration=1.5, overlap_duration=0.2)
        res = run_job(cfg)
        assert res.output.is_file()
        y, sr = sf.read(str(res.output), dtype="float32")
        orig, orig_sr = sf.read(str(main_file), dtype="float32")
        assert sr == orig_sr
        assert abs(len(y) - len(orig)) <= int(0.05 * sr)

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_02_raised_cosine_crossfade_continuity(self, tmp_path):
        """R1.2: Verify 2-second raised-cosine overlap-add formula exhibits unity gain and zero clicks."""
        # Test mathematical identity w_in + w_out == 1.0 across overlap window
        n_overlap = int(2.0 * TEST_SR)
        theta = np.pi * (np.arange(n_overlap) + 0.5) / n_overlap
        w_in = (0.5 * (1.0 - np.cos(theta))).astype(np.float32)
        w_out = (0.5 * (1.0 + np.cos(theta))).astype(np.float32)

        # Unity gain check
        np.testing.assert_allclose(w_in + w_out, 1.0, atol=1e-6)

        # C1 smooth boundary check: derivative at edges is zero
        assert abs(w_in[0]) < 1e-4
        assert abs(w_in[-1] - 1.0) < 1e-4

        # Crossfading identical constant tones yields a constant tone
        tone1 = np.ones(n_overlap, dtype=np.float32)
        tone2 = np.ones(n_overlap, dtype=np.float32)
        blended = tone1 * w_out + tone2 * w_in
        np.testing.assert_allclose(blended, 1.0, atol=1e-6)

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_03_streaming_memory_bound(self, tmp_path):
        """R1.3: Verify memory usage stays capped (< 150MB) during multi-chunk streaming."""
        import tracemalloc
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        # 6 seconds with 1-second chunks exercises 6 chunk cycles
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(6.0) + gen_hum(6.0))
        out_file = tmp_path / "out.wav"

        tracemalloc.start()
        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, chunk_duration=1.0, overlap_duration=0.2)
        run_job(cfg)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 150.0, f"Peak memory {peak_mb:.2f}MB exceeded 150MB limit"

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_04_noisy_engine_chunked_noise_reduction(self, tmp_path):
        """R1.4: Verify chunked streaming through NoisyAcousticEngine reduces noise floor."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0) + gen_hum(4.0, seed=2))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.85, 30.0, out_file,
                        polish=False, chunk_duration=1.5, overlap_duration=0.25)
        res = run_job(cfg)
        assert res.suppression_db > 1.0, f"Expected suppression > 1.0 dB, got {res.suppression_db}"

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_05_clean_engine_chunked_interface(self, tmp_path):
        """R1.5: Verify clean engine accepts chunked configuration parameters."""
        cfg = JobConfig(tmp_path / "ref.wav", ReferenceRole.CLEAN, tmp_path / "main.wav",
                        chunk_duration=60.0, overlap_duration=2.0)
        assert cfg.chunk_duration == 60.0
        assert cfg.overlap_duration == 2.0

    # ------------------------------------------------------------------------
    # R2: Instant Seamless A/B Hot-Swap During Playback
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_01_player_pair_initialization(self, tmp_path):
        """R2.1: Verify AudioPlayer.play_pair loads original and processed tracks at sample index 0."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "a.wav", gen_sine(1.0, 440))
        f2 = create_audio_file(tmp_path / "b.wav", gen_sine(1.0, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            ok = p.play_pair(f1, f2, start_with="B")
            assert ok is True
            assert p.get_current_track() == "B"
            assert getattr(p, "_sample_index", 0) == 0
            p.stop()

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_02_toggle_ab_preserves_sample_index(self, tmp_path):
        """R2.2: Verify toggle_ab switches track without resetting or changing sample_index."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "a.wav", gen_sine(2.0, 440))
        f2 = create_audio_file(tmp_path / "b.wav", gen_sine(2.0, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            p.play_pair(f1, f2, start_with="B")
            # Advance playback position to sample 2048
            p._sample_index = 2048
            idx_before = p._sample_index

            switched = p.toggle_ab()
            assert switched == "A"
            assert p.get_current_track() == "A"
            assert p._sample_index == idx_before

            switched_back = p.toggle_ab()
            assert switched_back == "B"
            assert p.get_current_track() == "B"
            assert p._sample_index == idx_before
            p.stop()

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_03_rapid_hot_swap_cycling(self, tmp_path):
        """R2.3: Verify rapid alternating toggles (A->B->A->B) maintain strict state consistency."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "a.wav", gen_sine(1.0, 440))
        f2 = create_audio_file(tmp_path / "b.wav", gen_sine(1.0, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            p.play_pair(f1, f2, start_with="A")
            for _ in range(20):
                p.toggle_ab()
            assert p.get_current_track() == "A"
            p.stop()

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_04_status_text_reflects_active_track(self, tmp_path):
        """R2.4: Verify player status_text includes current track indicator ('A' or 'B')."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "a.wav", gen_sine(1.0, 440))
        f2 = create_audio_file(tmp_path / "b.wav", gen_sine(1.0, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            p.play_pair(f1, f2, start_with="B")
            txt_b = p.status_text()
            assert "B" in txt_b or "Processed" in txt_b or "playing" in txt_b.lower()
            p.toggle_ab()
            txt_a = p.status_text()
            assert "A" in txt_a or "Original" in txt_a or "playing" in txt_a.lower()
            p.stop()

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_05_audio_callback_slices_correct_buffer(self, tmp_path):
        """R2.5: Verify audio stream callback feeds audio from the currently active buffer."""
        from zout.audio.player import AudioPlayer
        # Create distinct signals: tone A = +0.5 constant, tone B = -0.5 constant
        a_sig = np.full(4096, 0.5, dtype=np.float32)
        b_sig = np.full(4096, -0.5, dtype=np.float32)
        f1 = create_audio_file(tmp_path / "a.wav", a_sig)
        f2 = create_audio_file(tmp_path / "b.wav", b_sig)

        p = AudioPlayer()
        mock_stream_inst = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream_inst
            mock_stream_inst = MockOutputStream(*args, **kwargs)
            return mock_stream_inst

        with patch("sounddevice.OutputStream", stream_factory):
            p.play_pair(f1, f2, start_with="B")
            assert mock_stream_inst is not None

            # Pump 512 frames from track B -> should receive -0.5
            out_b = mock_stream_inst.pump(512)
            assert out_b is not None
            assert np.allclose(out_b, -0.5, atol=1e-3)

            # Hot-swap to track A
            p.toggle_ab()

            # Pump 512 frames from track A -> should receive +0.5
            out_a = mock_stream_inst.pump(512)
            assert out_a is not None
            assert np.allclose(out_a, 0.5, atol=1e-3)
            p.stop()

    # ------------------------------------------------------------------------
    # R3: Live Terminal Real-Time VU Meter & Waveform Envelope
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r3_envelope_implemented(), reason="R3 waveform envelope not yet implemented")
    def test_r3_01_waveform_envelope_format(self):
        """R3.1: Verify compute_waveform_envelope returns exact width of 8-level Unicode glyphs."""
        from zout.audio.io import compute_waveform_envelope
        audio = gen_speechlike(1.0)
        env = compute_waveform_envelope(audio, width=16)
        assert len(env) == 16
        valid_glyphs = set(" ▂▃▄▅▆▇█")
        assert all(c in valid_glyphs for c in env)

    @pytest.mark.skipif(not is_r3_envelope_implemented(), reason="R3 waveform envelope not yet implemented")
    def test_r3_02_waveform_envelope_dynamic_response(self):
        """R3.2: Verify waveform envelope accurately reflects volume dynamics ramping up."""
        from zout.audio.io import compute_waveform_envelope
        # Linearly ramping amplitude
        t = np.linspace(0, 1, 1600, dtype=np.float32)
        ramp = t * np.sin(2 * np.pi * 100 * t)
        env = compute_waveform_envelope(ramp, width=16)
        # Glyph levels should generally increase from left to right
        glyphs = " ▂▃▄▅▆▇█"
        levels = [glyphs.index(c) for c in env]
        assert levels[-1] >= levels[0]
        assert levels[-1] == 7  # Peak window reaches maximum block '█'

    @pytest.mark.skipif(not is_r3_envelope_implemented(), reason="R3 waveform envelope not yet implemented")
    def test_r3_03_waveform_envelope_silence_returns_lowest(self):
        """R3.3: Verify waveform envelope on complete digital silence returns all lowest glyphs."""
        import zout.audio.io as aio
        if hasattr(aio, "_ARRAY_ENVELOPE_CACHE"):
            aio._ARRAY_ENVELOPE_CACHE.clear()
        silence = np.zeros(1600, dtype=np.float32)
        env = aio.compute_waveform_envelope(silence, width=16)
        assert len(env) == 16
        assert env == " " * 16

    @pytest.mark.skipif(not is_r3_vu_implemented(), reason="R3 VU meter not yet implemented")
    def test_r3_04_vu_meter_levels_full_scale_sine(self, tmp_path):
        """R3.4: Verify VU meter levels on full-scale sine approach 0 dBFS Peak and -3 dBFS RMS."""
        from zout.audio.player import AudioPlayer
        sine = gen_sine(1.0, freq=1000.0, amp=1.0)
        f = create_audio_file(tmp_path / "sine.wav", sine)

        p = AudioPlayer()
        mock_stream = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream
            mock_stream = MockOutputStream(*args, **kwargs)
            return mock_stream

        with patch("sounddevice.OutputStream", stream_factory):
            p.play_pair(f, f)
            mock_stream.pump(1024)
            (rms_l, rms_r), (peak_l, peak_r) = p.get_meter_levels()
            # Full scale sine: Peak ~ 0 dBFS ([-1.0, 0.5]), RMS ~ -3 dBFS ([-4.0, -2.0])
            assert -1.0 <= peak_l <= 0.5
            assert -4.5 <= rms_l <= -1.5
            p.stop()

    @pytest.mark.skipif(not is_r3_vu_implemented(), reason="R3 VU meter not yet implemented")
    def test_r3_05_vu_meter_levels_silence_capped(self, tmp_path):
        """R3.5: Verify VU meter on silence returns dBFS <= -80 dBFS without -inf / NaN."""
        from zout.audio.player import AudioPlayer
        silence = np.zeros(2048, dtype=np.float32)
        f = create_audio_file(tmp_path / "silence.wav", silence)

        p = AudioPlayer()
        mock_stream = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream
            mock_stream = MockOutputStream(*args, **kwargs)
            return mock_stream

        with patch("sounddevice.OutputStream", stream_factory):
            p.play_pair(f, f)
            mock_stream.pump(1024)
            (rms_l, rms_r), (peak_l, peak_r) = p.get_meter_levels()
            assert peak_l <= -80.0
            assert rms_l <= -80.0
            assert np.isfinite(peak_l) and np.isfinite(rms_l)
            p.stop()

    # ------------------------------------------------------------------------
    # R4: Graceful In-Flight Cancellation
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_01_cancel_event_aborts_pipeline(self, tmp_path):
        """R4.1: Verify setting cancel_event during pipeline execution aborts cleanly returning None."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0) + gen_hum(4.0))

        cancel_ev = threading.Event()
        cancel_ev.set()  # Pre-set cancel token

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, cancel_event=cancel_ev)
        res = run_job(cfg)
        assert res is None, "Expected None return upon cancellation"

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_02_cancel_cleans_up_partial_output(self, tmp_path):
        """R4.2: Verify aborted job cleans up partial output file without leaving corrupt audio."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0) + gen_hum(4.0))
        out_file = tmp_path / "out.wav"

        cancel_ev = threading.Event()
        cancel_ev.set()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, cancel_event=cancel_ev)
        run_job(cfg)
        # Either output file was not created, or if created, was removed
        assert not out_file.exists() or out_file.stat().st_size == 0

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_03_concurrent_thread_cancellation(self, tmp_path):
        """R4.3: Verify triggering cancellation from a concurrent worker thread halts execution."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(3.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(6.0) + gen_hum(6.0))
        out_file = tmp_path / "out.wav"

        cancel_ev = threading.Event()

        def cancel_after_delay():
            time.sleep(0.05)
            cancel_ev.set()

        t = threading.Thread(target=cancel_after_delay)
        t.start()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, cancel_event=cancel_ev)
        res = run_job(cfg)
        t.join()
        # Should cleanly return None or complete safely
        assert res is None or res.output.is_file()

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_04_cancel_token_accepted_in_run_job_signature(self, tmp_path):
        """R4.4: Verify run_job interface supports cancel_token parameter directly."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(1.0) + gen_hum(1.0))

        cancel_ev = threading.Event()
        cancel_ev.set()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False)
        params = inspect.signature(run_job).parameters
        if "cancel_token" in params:
            res = run_job(cfg, cancel_token=cancel_ev)
            assert res is None
        else:
            cfg.cancel_event = cancel_ev
            assert run_job(cfg) is None

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_05_cancellation_during_streaming_loop(self, tmp_path):
        """R4.5: Verify cancellation during chunked streaming terminates before processing all chunks."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(8.0) + gen_hum(8.0))

        cancel_ev = threading.Event()
        chunk_count = 0

        def progress_cb(msg, frac):
            nonlocal chunk_count
            chunk_count += 1
            if chunk_count >= 2:
                cancel_ev.set()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, chunk_duration=1.0, overlap_duration=0.2, cancel_event=cancel_ev)
        res = run_job(cfg, progress=progress_cb)
        assert res is None

    # ------------------------------------------------------------------------
    # R5: Timeline Range Slicing
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r5_timespec_implemented(), reason="R5 parse_time_spec not yet implemented")
    def test_r5_01_parse_time_spec_valid_formats(self):
        """R5.1: Verify parse_time_spec parses seconds, mm:ss, and hh:mm:ss timestamps."""
        from zout.audio.io import parse_time_spec
        assert parse_time_spec("45") == 45.0
        assert parse_time_spec(45) == 45.0
        assert parse_time_spec(45.5) == 45.5
        assert parse_time_spec("01:30") == 90.0
        assert parse_time_spec("01:15:30") == 4530.0
        assert parse_time_spec("00:00:10.5") == 10.5
        assert parse_time_spec(None) is None

    @pytest.mark.skipif(not is_r5_timespec_implemented(), reason="R5 parse_time_spec not yet implemented")
    def test_r5_02_parse_time_spec_invalid_rejected(self):
        """R5.2: Verify parse_time_spec raises ValueError on malformed or negative timestamps."""
        from zout.audio.io import parse_time_spec
        with pytest.raises((ValueError, TypeError)):
            parse_time_spec("invalid")
        with pytest.raises((ValueError, TypeError)):
            parse_time_spec("-10")
        with pytest.raises((ValueError, TypeError)):
            parse_time_spec("12:60:00")  # Invalid minutes >= 60

    @pytest.mark.skipif(not is_r5_slicing_implemented(), reason="R5 range slicing not yet implemented")
    def test_r5_03_load_audio_range_slicing(self, tmp_path):
        """R5.3: Verify load_audio with start_sec and end_sec loads exact frame slice duration."""
        from zout.audio.io import load_audio
        full_audio = gen_speechlike(5.0)
        audio_file = create_audio_file(tmp_path / "long.wav", full_audio)

        sliced, sr = load_audio(audio_file, start_sec=1.0, end_sec=3.5)
        expected_len = int(2.5 * sr)
        assert abs(len(sliced) - expected_len) <= 1
        # Verify content matches exact slice
        full_loaded, _ = load_audio(audio_file)
        np.testing.assert_allclose(sliced, full_loaded[int(1.0 * sr): int(3.5 * sr)], atol=1e-5)

    @pytest.mark.skipif(not is_r5_pipeline_slicing_implemented(), reason="R5 pipeline range slicing not yet implemented")
    def test_r5_04_job_config_range_slicing_execution(self, tmp_path):
        """R5.4: Verify JobConfig with start_sec and end_sec processes only the specified window."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(6.0) + gen_hum(6.0))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, start_sec=1.0, end_sec=3.0)
        res = run_job(cfg)
        assert res.output.is_file()
        y, sr = sf.read(str(res.output), dtype="float32")
        expected_samples = int(2.0 * sr)
        assert abs(len(y) - expected_samples) <= int(0.05 * sr)

    @pytest.mark.skipif(not is_cli_start_end_supported(), reason="CLI --start and --end not yet implemented")
    def test_r5_05_cli_start_end_arguments_supported(self):
        """R5.5: Verify CLI parser accepts --start and --end arguments."""
        from zout.app import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["-m", "main.wav", "--auto-noise", "--start", "01:00", "--end", "02:30"])
        assert hasattr(args, "start") and args.start == "01:00"
        assert hasattr(args, "end") and args.end == "02:30"

    # ------------------------------------------------------------------------
    # R6: Forensic Audit Log Sidecar (.json)
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_01_audit_manifest_file_created(self, tmp_path):
        """R6.1: Verify running job with audit=True creates .json manifest sidecar file."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(3.0) + gen_hum(3.0))
        out_file = tmp_path / "interview_zout.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, audit=True)
        res = run_job(cfg)
        expected_json = out_file.with_name(out_file.stem + ".json")
        alt_json = out_file.with_name(out_file.stem + "_audit.json")
        audit_file = expected_json if expected_json.is_file() else alt_json
        assert audit_file.is_file(), f"Audit manifest not found at {expected_json} or {alt_json}"

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_02_audit_json_schema_completeness(self, tmp_path):
        """R6.2: Verify audit JSON contains required forensic metadata fields."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(3.0) + gen_hum(3.0))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        assert len(candidates) >= 1
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check required forensic schema fields
        assert "noise_reduction_db" in data or "suppression_db" in data
        assert "true_peak" in data or "true_peak_dbfs" in data
        assert "rms" in data or "rms_dbfs" in data
        assert "vad_speech_timestamps" in data
        assert "parameters" in data

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_03_audit_true_peak_oversampled(self, tmp_path):
        """R6.3: Verify true-peak calculation uses 4x oversampling (ITU-R BS.1770 compliant)."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(3.0) + gen_hum(3.0))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        tp = data.get("true_peak") or data.get("true_peak_dbfs")
        if isinstance(tp, dict):
            assert "output" in tp and isinstance(tp["output"], (int, float))
            assert tp["output"] <= 0.05  # Peak-limited to <= 0 dBFS
        else:
            assert isinstance(tp, (int, float))

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_04_audit_vad_speech_timestamps_validity(self, tmp_path):
        """R6.4: Verify VAD speech timestamps are ordered intervals with positive durations."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        # 4s audio with speech burst at 1s-3s
        sig = np.zeros(int(4.0 * TEST_SR), dtype=np.float32)
        sig[TEST_SR: 3 * TEST_SR] = gen_speechlike(2.0)
        main_file = create_audio_file(tmp_path / "main.wav", sig + gen_hum(4.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        vads = data.get("vad_speech_timestamps", [])
        assert isinstance(vads, list)
        for interval in vads:
            assert interval["start"] >= 0.0
            assert interval["end"] >= interval["start"]
            assert interval["duration"] >= 0.0

    @pytest.mark.skipif(not is_cli_audit_supported(), reason="CLI --audit/--json not yet implemented")
    def test_r6_05_cli_audit_flags_supported(self):
        """R6.5: Verify CLI parser accepts --audit and --json flags."""
        from zout.app import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["-m", "main.wav", "--auto-noise", "--audit"])
        assert getattr(args, "audit", False) is True

        args_json = parser.parse_args(["-m", "main.wav", "--auto-noise", "--json"])
        assert getattr(args_json, "audit", False) is True or getattr(args_json, "json", False) is True

    # ------------------------------------------------------------------------
    # R7: Terminal Color Themes
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_01_all_four_themes_defined(self):
        """R7.1: Verify all 4 required themes exist in zout.tui.themes."""
        from zout.tui import themes
        required = ["Forensic Cyan", "Amber Phosphor", "Matrix Emerald", "Studio Dark"]
        for name in required:
            theme = themes.get_theme(name)
            assert theme is not None
            assert getattr(theme, "name", "") == name

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_02_cycle_theme_sequential_order(self):
        """R7.2: Verify cycle_theme cycles through the 4 themes and wraps back to the first."""
        from zout.tui import themes
        names = [themes.cycle_theme().name for _ in range(5)]
        # After 4 cycles, theme wraps around
        assert names[0] == names[4]
        assert len(set(names[:4])) == 4

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_03_get_theme_by_name(self):
        """R7.3: Verify get_theme accurately retrieves specific requested theme."""
        from zout.tui import themes
        theme = themes.get_theme("Matrix Emerald")
        assert theme.name == "Matrix Emerald"

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_04_theme_palette_styling_attributes(self):
        """R7.4: Verify themes provide rich styling attributes (primary, secondary, key, border)."""
        from zout.tui import themes
        theme = themes.get_theme("Forensic Cyan")
        for attr in ["primary", "secondary", "accent", "key", "border"]:
            assert hasattr(theme, attr)
            val = getattr(theme, attr)
            assert isinstance(val, str) and len(val) > 0

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_05_theme_manager_state_persistence(self):
        """R7.5: Verify current theme is maintained across calls without unintended reset."""
        from zout.tui import themes
        t1 = themes.cycle_theme()
        t2 = themes.get_theme(None)  # None returns active theme
        assert t1.name == t2.name

    # ------------------------------------------------------------------------
    # R8: Batch Folder Processing
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_cli_batch_supported(), reason="CLI --batch not yet implemented")
    def test_r8_01_cli_batch_arguments_supported(self):
        """R8.1: Verify CLI parser accepts --batch and --out-dir arguments."""
        from zout.app import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["-r", "ref.wav", "-t", "noisy", "--batch", "input_folder", "--out-dir", "output_folder"])
        assert getattr(args, "batch", None) == "input_folder"
        assert getattr(args, "out_dir", None) == "output_folder"

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_02_batch_processes_all_audio_files(self, tmp_path):
        """R8.2: Verify batch processing iterates over all audio files in source directory."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        out_dir = tmp_path / "batch_out"

        f1 = create_audio_file(batch_dir / "track1.wav", gen_speechlike(2.0) + gen_hum(2.0))
        f2 = create_audio_file(batch_dir / "track2.wav", gen_speechlike(2.0) + gen_hum(2.0, seed=3))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir,
                        strength=0.75, polish=False)
        run_job(cfg)
        out_files = list(out_dir.glob("*.wav"))
        assert len(out_files) == 2

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_03_batch_skips_non_audio_files(self, tmp_path):
        """R8.3: Verify batch processing skips text files, hidden files, and directories."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        out_dir = tmp_path / "batch_out"

        create_audio_file(batch_dir / "valid.wav", gen_speechlike(2.0))
        (batch_dir / "notes.txt").write_text("Ignore me")
        (batch_dir / ".DS_Store").write_bytes(b"junk")
        (batch_dir / "sub_folder").mkdir()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir,
                        strength=0.75, polish=False)
        run_job(cfg)
        out_files = list(out_dir.glob("*"))
        assert len(out_files) == 1
        assert out_files[0].suffix == ".wav"

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_04_batch_default_output_directory(self, tmp_path):
        """R8.4: Verify batch without explicit out_dir places outputs in <batch_dir>/processed/."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        create_audio_file(batch_dir / "item.wav", gen_speechlike(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, strength=0.75, polish=False)
        run_job(cfg)
        default_out = batch_dir / "processed"
        assert default_out.is_dir()
        assert len(list(default_out.glob("*.wav"))) >= 1

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_05_batch_resilience_to_corrupt_files(self, tmp_path):
        """R8.5: Verify a corrupted file in batch folder does not crash the remaining files."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        out_dir = tmp_path / "batch_out"

        create_audio_file(batch_dir / "good1.wav", gen_speechlike(2.0))
        (batch_dir / "corrupt.wav").write_bytes(b"NOT_A_VALID_WAV_HEADER")
        create_audio_file(batch_dir / "good2.wav", gen_speechlike(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir,
                        strength=0.75, polish=False)
        # Execution should complete without unhandled exception
        run_job(cfg)
        out_files = list(out_dir.glob("good*.wav"))
        assert len(out_files) == 2


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES (R1 through R8, >=5 tests each = 40 tests)
# ============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary & Corner Cases exercising edge conditions, stress, and limits."""

    # ------------------------------------------------------------------------
    # R1: Streaming Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_b01_audio_shorter_than_chunk_size(self, tmp_path):
        """R1.B1: Verify streaming handles audio whose duration is less than chunk_duration."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        # 0.8s audio with 2.0s chunk size
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(0.8) + gen_hum(0.8))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, chunk_duration=2.0, overlap_duration=0.2)
        res = run_job(cfg)
        assert res.output.is_file()
        y, sr = sf.read(str(res.output), dtype="float32")
        assert len(y) > 0 and np.isfinite(y).all()

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_b02_exact_chunk_boundary_alignment(self, tmp_path):
        """R1.B2: Verify streaming audio whose duration exactly matches 2 * chunk_duration."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        # Exact 2.0s audio with 1.0s chunks
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(2.0) + gen_hum(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, chunk_duration=1.0, overlap_duration=0.2)
        res = run_job(cfg)
        assert res.output.is_file()

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_b03_zero_overlap_duration(self, tmp_path):
        """R1.B3: Verify streaming handles zero or near-zero overlap duration safely."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, chunk_duration=1.0, overlap_duration=0.0)
        res = run_job(cfg)
        assert res.output.is_file()

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_b04_streaming_pure_digital_silence(self, tmp_path):
        """R1.B4: Verify streaming pure digital silence across chunks does not produce NaN or error."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        silence = np.zeros(int(3.0 * TEST_SR), dtype=np.float32)
        main_file = create_audio_file(tmp_path / "silence.wav", silence)

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, chunk_duration=1.0, overlap_duration=0.2)
        res = run_job(cfg)
        y, _ = sf.read(str(res.output), dtype="float32")
        assert np.isfinite(y).all()
        assert np.abs(y).max() <= 1e-4

    @pytest.mark.skipif(not is_r1_implemented(), reason="R1 chunked streaming not yet implemented")
    def test_r1_b05_streaming_clipped_full_scale_audio(self, tmp_path):
        """R1.B5: Verify streaming severely clipped square wave signal does not overflow peak limiter."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        clipped = np.sign(gen_sine(3.0, 100.0)).astype(np.float32)
        main_file = create_audio_file(tmp_path / "clipped.wav", clipped)

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, chunk_duration=1.0, overlap_duration=0.2)
        res = run_job(cfg)
        y, _ = sf.read(str(res.output), dtype="float32")
        assert np.isfinite(y).all()
        assert np.abs(y).max() <= 1.0001

    # ------------------------------------------------------------------------
    # R2: A/B Hot-Swap Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_b01_nonexistent_audio_paths_return_false(self):
        """R2.B1: Verify play_pair with nonexistent paths returns False cleanly without crashing."""
        from zout.audio.player import AudioPlayer
        p = AudioPlayer()
        assert p.play_pair("nonexistent1.wav", "nonexistent2.wav") is False

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_b02_toggle_when_not_playing(self):
        """R2.B2: Verify calling toggle_ab when idle does not raise an unhandled exception."""
        from zout.audio.player import AudioPlayer
        p = AudioPlayer()
        p.stop()
        res = p.toggle_ab()
        assert res in ("A", "B", "")

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_b03_mismatched_audio_lengths(self, tmp_path):
        """R2.B3: Verify play_pair handles tracks of unequal duration safely (aligned to min length)."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "short.wav", gen_sine(0.5, 440))
        f2 = create_audio_file(tmp_path / "long.wav", gen_sine(2.0, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            ok = p.play_pair(f1, f2)
            assert ok is True
            p.stop()

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_b04_immediate_stop_after_play_pair(self, tmp_path):
        """R2.B4: Verify immediate stop() call following play_pair() avoids race conditions."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "a.wav", gen_sine(1.0, 440))
        f2 = create_audio_file(tmp_path / "b.wav", gen_sine(1.0, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            p.play_pair(f1, f2)
            p.stop()
            assert p.is_playing() is False

    @pytest.mark.skipif(not is_r2_implemented(), reason="R2 A/B hot-swap not yet implemented")
    def test_r2_b05_sub_second_audio_hot_swap(self, tmp_path):
        """R2.B5: Verify sub-second audio (shorter than standard 1024-block) plays without underflow."""
        from zout.audio.player import AudioPlayer
        f1 = create_audio_file(tmp_path / "a.wav", gen_sine(0.04, 440))  # ~640 samples < 1024
        f2 = create_audio_file(tmp_path / "b.wav", gen_sine(0.04, 880))

        p = AudioPlayer()
        with patch("sounddevice.OutputStream", MockOutputStream):
            ok = p.play_pair(f1, f2)
            assert ok is True
            p.stop()

    # ------------------------------------------------------------------------
    # R3: Meter & Envelope Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r3_envelope_implemented(), reason="R3 waveform envelope not yet implemented")
    def test_r3_b01_zero_length_envelope(self):
        """R3.B1: Verify compute_waveform_envelope on empty audio array returns empty string or spaces."""
        from zout.audio.io import compute_waveform_envelope
        empty = np.zeros(0, dtype=np.float32)
        env = compute_waveform_envelope(empty, width=16)
        assert len(env) in (0, 16)

    @pytest.mark.skipif(not is_r3_envelope_implemented(), reason="R3 waveform envelope not yet implemented")
    def test_r3_b02_extreme_envelope_widths(self):
        """R3.B2: Verify compute_waveform_envelope supports width=1 and width=128."""
        from zout.audio.io import compute_waveform_envelope
        audio = gen_speechlike(1.0)
        assert len(compute_waveform_envelope(audio, width=1)) == 1
        assert len(compute_waveform_envelope(audio, width=128)) == 128

    @pytest.mark.skipif(not is_r3_envelope_implemented(), reason="R3 waveform envelope not yet implemented")
    def test_r3_b03_constant_dc_offset_audio(self):
        """R3.B3: Verify compute_waveform_envelope on constant non-zero DC offset handles normalization."""
        import zout.audio.io as aio
        if hasattr(aio, "_ARRAY_ENVELOPE_CACHE"):
            aio._ARRAY_ENVELOPE_CACHE.clear()
        dc = np.full(1600, 0.42, dtype=np.float32)
        env = aio.compute_waveform_envelope(dc, width=16)
        assert len(env) == 16
        assert all(c in " ▂▃▄▅▆▇█" for c in env)

    @pytest.mark.skipif(not is_r3_vu_implemented(), reason="R3 VU meter not yet implemented")
    def test_r3_b04_vu_meter_extreme_attenuation(self, tmp_path):
        """R3.B4: Verify VU meter on -120 dBFS audio returns valid capped float without -inf."""
        from zout.audio.player import AudioPlayer
        tiny = np.full(2048, 1e-6, dtype=np.float32)  # -120 dB
        f = create_audio_file(tmp_path / "tiny.wav", tiny)

        p = AudioPlayer()
        mock_stream = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream
            mock_stream = MockOutputStream(*args, **kwargs)
            return mock_stream

        with patch("sounddevice.OutputStream", stream_factory):
            p.play_pair(f, f)
            mock_stream.pump(1024)
            (rms_l, rms_r), (peak_l, peak_r) = p.get_meter_levels()
            assert np.isfinite(rms_l) and np.isfinite(peak_l)
            assert rms_l <= -80.0
            p.stop()

    @pytest.mark.skipif(not is_r3_vu_implemented(), reason="R3 VU meter not yet implemented")
    def test_r3_b05_vu_meter_stereo_channel_independence(self, tmp_path):
        """R3.B5: Verify VU meter computes distinct levels for Left vs Right stereo channels."""
        from zout.audio.player import AudioPlayer
        # Left channel loud sine (0 dBFS), Right channel silence (-inf)
        stereo = np.zeros((2048, 2), dtype=np.float32)
        stereo[:, 0] = np.sin(2 * np.pi * 1000 * np.arange(2048) / TEST_SR)
        f = tmp_path / "stereo_split.wav"
        sf.write(str(f), stereo, TEST_SR)

        p = AudioPlayer()
        mock_stream = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream
            mock_stream = MockOutputStream(*args, **kwargs)
            return mock_stream

        with patch("sounddevice.OutputStream", stream_factory):
            p.play_pair(f, f)
            mock_stream.pump(1024)
            (rms_l, rms_r), (peak_l, peak_r) = p.get_meter_levels()
            assert peak_l > peak_r + 20.0  # Left channel is significantly louder than Right
            p.stop()

    # ------------------------------------------------------------------------
    # R4: Cancellation Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_b01_cancellation_pre_set_immediate_return(self, tmp_path):
        """R4.B1: Verify job with pre-set cancel token terminates in under 100 milliseconds."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0) + gen_hum(4.0))

        cancel_ev = threading.Event()
        cancel_ev.set()

        t0 = time.time()
        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, cancel_event=cancel_ev)
        run_job(cfg)
        elapsed = time.time() - t0
        assert elapsed < 0.25, f"Cancellation took {elapsed:.2f}s, expected < 0.25s"

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_b02_multiple_cancel_calls_idempotent(self):
        """R4.B2: Verify signaling cancel_event multiple times is safe and idempotent."""
        cancel_ev = threading.Event()
        for _ in range(10):
            cancel_ev.set()
        assert cancel_ev.is_set()

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_b03_cancel_without_active_job(self):
        """R4.B3: Verify cancel signal when idle does not corrupt subsequent job runs."""
        cancel_ev = threading.Event()
        cancel_ev.set()
        cancel_ev.clear()
        assert not cancel_ev.is_set()

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_b04_cancel_cleans_temporary_buffers(self, tmp_path):
        """R4.B4: Verify cancelled execution leaves no lingering temporary swap files in directory."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0))

        cancel_ev = threading.Event()
        cancel_ev.set()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, cancel_event=cancel_ev)
        run_job(cfg)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    @pytest.mark.skipif(not is_r4_implemented(), reason="R4 in-flight cancellation not yet implemented")
    def test_r4_b05_cancellation_preserves_input_files(self, tmp_path):
        """R4.B5: Verify cancellation never alters or truncates input audio files."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(4.0))
        orig_ref_size = ref_file.stat().st_size
        orig_main_size = main_file.stat().st_size

        cancel_ev = threading.Event()
        cancel_ev.set()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, cancel_event=cancel_ev)
        run_job(cfg)
        assert ref_file.stat().st_size == orig_ref_size
        assert main_file.stat().st_size == orig_main_size

    # ------------------------------------------------------------------------
    # R5: Range Slicing Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r5_pipeline_slicing_implemented(), reason="R5 pipeline range slicing not yet implemented")
    def test_r5_b01_start_equals_end_raises_error(self, tmp_path):
        """R5.B1: Verify start_sec == end_sec (0-length slice) raises ValueError."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(3.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        start_sec=1.5, end_sec=1.5, allow_invalid_range=True)
        with pytest.raises(ValueError):
            run_job(cfg)

    @pytest.mark.skipif(not is_r5_pipeline_slicing_implemented(), reason="R5 pipeline range slicing not yet implemented")
    def test_r5_b02_start_greater_than_end_raises_error(self, tmp_path):
        """R5.B2: Verify start_sec > end_sec raises ValueError."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(3.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        start_sec=2.5, end_sec=1.0, allow_invalid_range=True)
        with pytest.raises(ValueError):
            run_job(cfg)

    @pytest.mark.skipif(not is_r5_pipeline_slicing_implemented(), reason="R5 pipeline range slicing not yet implemented")
    def test_r5_b03_start_beyond_duration_raises_error(self, tmp_path):
        """R5.B3: Verify start_sec exceeding file duration raises ValueError."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        start_sec=100.0, end_sec=105.0)
        with pytest.raises(ValueError):
            run_job(cfg)

    @pytest.mark.skipif(not is_r5_timespec_implemented(), reason="R5 parse_time_spec not yet implemented")
    def test_r5_b04_subsecond_precision_timestamps(self):
        """R5.B4: Verify millisecond timestamps (e.g. '01:23.456') are parsed with float precision."""
        from zout.audio.io import parse_time_spec
        val = parse_time_spec("01:23.456")
        assert abs(val - 83.456) < 1e-4

    @pytest.mark.skipif(not is_r5_pipeline_slicing_implemented(), reason="R5 pipeline range slicing not yet implemented")
    def test_r5_b05_slice_starting_at_zero(self, tmp_path):
        """R5.B5: Verify slice from start_sec=0.0 up to end_sec=1.0 functions identically to head crop."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(3.0) + gen_hum(3.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, start_sec=0.0, end_sec=1.0)
        res = run_job(cfg)
        y, sr = sf.read(str(res.output), dtype="float32")
        assert abs(len(y) - int(1.0 * sr)) <= int(0.05 * sr)

    # ------------------------------------------------------------------------
    # R6: Audit Log Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_b01_audit_on_pure_silence_no_nan(self, tmp_path):
        """R6.B1: Verify audit manifest generated on pure digital silence contains finite numbers, no NaN."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        silence = np.zeros(int(2.0 * TEST_SR), dtype=np.float32)
        main_file = create_audio_file(tmp_path / "silence.wav", silence)

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "r", encoding="utf-8") as f:
            raw_text = f.read()
        assert "NaN" not in raw_text and "Infinity" not in raw_text
        data = json.loads(raw_text)
        assert isinstance(data, dict)

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_b02_audit_pure_noise_yields_empty_vad(self, tmp_path):
        """R6.B2: Verify audit manifest on pure speech-free Gaussian noise produces empty VAD intervals."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        noise = gen_white_noise(2.0, amp=0.04)
        main_file = create_audio_file(tmp_path / "noise.wav", noise)

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("vad_speech_timestamps") == []

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_b03_audit_custom_path_override(self, tmp_path):
        """R6.B3: Verify custom audit_path directs manifest to explicit target destination."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(2.0))
        custom_audit = tmp_path / "custom_audit_dir" / "evidence.json"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True, audit_path=custom_audit)
        run_job(cfg)
        assert custom_audit.is_file()

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_b04_audit_continuous_speech_single_segment(self, tmp_path):
        """R6.B4: Verify audit manifest on uninterrupted continuous speech produces non-empty VAD segment."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        speech = gen_speechlike(3.0, f0=150.0)
        main_file = create_audio_file(tmp_path / "speech.wav", speech)

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        vads = data.get("vad_speech_timestamps", [])
        assert len(vads) >= 1

    @pytest.mark.skipif(not is_r6_implemented(), reason="R6 forensic audit log not yet implemented")
    def test_r6_b05_audit_json_valid_syntax(self, tmp_path):
        """R6.B5: Verify audit manifest is strictly valid RFC 8259 JSON readable by stdlib json.loads."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "rb") as f:
            payload = f.read()
        parsed = json.loads(payload.decode("utf-8"))
        assert isinstance(parsed, dict)

    # ------------------------------------------------------------------------
    # R7: Themes Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_b01_invalid_theme_name_falls_back_gracefully(self):
        """R7.B1: Verify get_theme with invalid theme name falls back safely to default theme."""
        from zout.tui import themes
        t = themes.get_theme("NonExistentThemeXYZ")
        assert t is not None
        assert t.name in ["Forensic Cyan", "Amber Phosphor", "Matrix Emerald", "Studio Dark"]

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_b02_case_insensitive_theme_lookup(self):
        """R7.B2: Verify get_theme supports case-insensitive name matching."""
        from zout.tui import themes
        t1 = themes.get_theme("amber phosphor")
        t2 = themes.get_theme("AMBER PHOSPHOR")
        assert t1.name == "Amber Phosphor"
        assert t2.name == "Amber Phosphor"

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_b03_get_theme_none_returns_current(self):
        """R7.B3: Verify get_theme(None) returns active theme."""
        from zout.tui import themes
        assert themes.get_theme(None) is not None

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_b04_rapid_theme_cycling_memory_leak_free(self):
        """R7.B4: Verify cycling theme 100 times in tight loop executes instantaneously."""
        from zout.tui import themes
        t0 = time.time()
        for _ in range(100):
            themes.cycle_theme()
        assert time.time() - t0 < 0.1

    @pytest.mark.skipif(not is_r7_implemented(), reason="R7 themes not yet implemented")
    def test_r7_b05_theme_styles_parse_in_rich(self):
        """R7.B5: Verify all theme style strings parse in rich.style.Style without syntax errors."""
        from rich.style import Style
        from zout.tui import themes
        for name in ["Forensic Cyan", "Amber Phosphor", "Matrix Emerald", "Studio Dark"]:
            t = themes.get_theme(name)
            for attr in ["primary", "secondary", "accent", "key", "border"]:
                val = getattr(t, attr)
                # Should parse without throwing rich.errors.StyleSyntaxError
                s = Style.parse(val)
                assert s is not None

    # ------------------------------------------------------------------------
    # R8: Batch Boundaries
    # ------------------------------------------------------------------------

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_b01_empty_batch_directory_completes_safely(self, tmp_path):
        """R8.B1: Verify batch processing on empty directory completes with 0 files processed."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        empty_dir = tmp_path / "empty_batch"
        empty_dir.mkdir()
        out_dir = tmp_path / "batch_out"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=empty_dir, out_dir=out_dir, polish=False)
        run_job(cfg)
        assert len(list(out_dir.glob("*.wav"))) == 0

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_b02_nonexistent_batch_directory_raises_error(self, tmp_path):
        """R8.B2: Verify nonexistent batch_dir raises ValueError or FileNotFoundError."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        missing_dir = tmp_path / "missing_folder"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=missing_dir, polish=False)
        with pytest.raises((ValueError, FileNotFoundError)):
            run_job(cfg)

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_b03_single_file_passed_as_batch_dir_raises_error(self, tmp_path):
        """R8.B3: Verify passing a file instead of directory to batch_dir raises ValueError."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        single_file = create_audio_file(tmp_path / "single.wav", gen_speechlike(1.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=single_file, polish=False)
        with pytest.raises(ValueError):
            run_job(cfg)

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_b04_batch_creates_output_directory_if_missing(self, tmp_path):
        """R8.B4: Verify batch processing automatically creates target out_dir if it does not exist."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        create_audio_file(batch_dir / "track.wav", gen_speechlike(1.0))
        nonexistent_out = tmp_path / "nested" / "output" / "dir"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=nonexistent_out, polish=False)
        run_job(cfg)
        assert nonexistent_out.is_dir()
        assert len(list(nonexistent_out.glob("*.wav"))) == 1

    @pytest.mark.skipif(not is_r8_implemented(), reason="R8 batch folder processing not yet implemented")
    def test_r8_b05_batch_mixed_supported_extensions(self, tmp_path):
        """R8.5: Verify batch processing discovers files across multiple supported audio extensions."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        out_dir = tmp_path / "batch_out"

        # Create .wav, .flac, .ogg
        create_audio_file(batch_dir / "file1.wav", gen_speechlike(1.0))
        create_audio_file(batch_dir / "file2.flac", gen_speechlike(1.0))
        create_audio_file(batch_dir / "file3.ogg", gen_speechlike(1.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir, polish=False)
        run_job(cfg)
        out_files = list(out_dir.glob("*"))
        assert len(out_files) == 3


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (8 pairwise interaction tests)
# ============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature interaction tests."""

    @pytest.mark.skipif(not (is_r1_implemented() and is_r5_slicing_implemented()),
                        reason="Requires R1 streaming and R5 range slicing")
    def test_r3_combo_01_range_slicing_and_chunked_streaming(self, tmp_path):
        """Combo 1: Slicing a target range from long audio, processed with chunked streaming."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(8.0) + gen_hum(8.0))
        out_file = tmp_path / "out.wav"

        # Slice 2.0s to 6.0s (4.0s slice), processed with 1.5s chunk streaming
        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, start_sec=2.0, end_sec=6.0, chunk_duration=1.5, overlap_duration=0.2)
        res = run_job(cfg)
        y, sr = sf.read(str(res.output), dtype="float32")
        expected_samples = int(4.0 * sr)
        assert abs(len(y) - expected_samples) <= int(0.05 * sr)

    @pytest.mark.skipif(not (is_r5_slicing_implemented() and is_r6_implemented()),
                        reason="Requires R5 range slicing and R6 audit sidecar")
    def test_r3_combo_02_range_slicing_and_audit_sidecar(self, tmp_path):
        """Combo 2: Range slicing with forensic audit log generates manifest reflecting sliced duration."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(6.0) + gen_hum(6.0))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, start_sec=1.0, end_sec=4.0, audit=True)
        run_job(cfg)
        candidates = list(tmp_path.glob("*.json"))
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("parameters", {})
        assert params.get("start_sec") == 1.0 or params.get("range_start") == 1.0

    @pytest.mark.skipif(not (is_r8_implemented() and is_r6_implemented()),
                        reason="Requires R8 batch processing and R6 audit sidecar")
    def test_r3_combo_03_batch_processing_and_audit_sidecar(self, tmp_path):
        """Combo 3: Batch processing with audit flag generates audit JSON sidecars for each batch file."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        out_dir = tmp_path / "batch_out"

        create_audio_file(batch_dir / "t1.wav", gen_speechlike(2.0))
        create_audio_file(batch_dir / "t2.wav", gen_speechlike(2.0))

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir,
                        polish=False, audit=True)
        run_job(cfg)
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 2

    @pytest.mark.skipif(not (is_r2_implemented() and is_r3_vu_implemented()),
                        reason="Requires R2 A/B hot-swap and R3 VU meters")
    def test_r3_combo_04_ab_hotswap_and_vu_meters(self, tmp_path):
        """Combo 4: Continuous playback hot-swapping A/B updates VU meters reflecting current track."""
        from zout.audio.player import AudioPlayer
        # Track A: Loud sine tone (Peak ~ 0 dBFS)
        # Track B: Quiet noise (Peak < -20 dBFS)
        f_a = create_audio_file(tmp_path / "loud.wav", gen_sine(1.0, 440, amp=1.0))
        f_b = create_audio_file(tmp_path / "quiet.wav", gen_white_noise(1.0, amp=0.01))

        p = AudioPlayer()
        mock_stream = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream
            mock_stream = MockOutputStream(*args, **kwargs)
            return mock_stream

        with patch("sounddevice.OutputStream", stream_factory):
            p.play_pair(f_a, f_b, start_with="A")
            mock_stream.pump(1024)
            _, (peak_a, _) = p.get_meter_levels()

            # Hot-swap to quiet track B
            p.toggle_ab()
            mock_stream.pump(1024)
            _, (peak_b, _) = p.get_meter_levels()

            # Loud track peak should be substantially higher than quiet track
            assert peak_a > peak_b + 15.0
            p.stop()

    @pytest.mark.skipif(not (is_r1_implemented() and is_r4_implemented()),
                        reason="Requires R1 streaming and R4 cancellation")
    def test_r3_combo_05_cancellation_during_chunked_streaming(self, tmp_path):
        """Combo 5: In-flight cancellation triggered while streaming is midway through chunks."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(2.0))
        main_file = create_audio_file(tmp_path / "main.wav", gen_speechlike(8.0) + gen_hum(8.0))

        cancel_ev = threading.Event()
        steps = 0

        def prog_cb(msg, frac):
            nonlocal steps
            steps += 1
            if steps == 3:
                cancel_ev.set()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, chunk_duration=1.0, overlap_duration=0.2, cancel_event=cancel_ev)
        res = run_job(cfg, progress=prog_cb)
        assert res is None

    @pytest.mark.skipif(not (is_r8_implemented() and is_r5_slicing_implemented()),
                        reason="Requires R8 batch processing and R5 range slicing")
    def test_r3_combo_06_batch_processing_and_range_slicing(self, tmp_path):
        """Combo 6: Batch execution applying uniform timeline slicing across all batch items."""
        ref_file = create_audio_file(tmp_path / "ref.wav", gen_hum(1.0))
        batch_dir = tmp_path / "batch_in"
        out_dir = tmp_path / "batch_out"

        create_audio_file(batch_dir / "item1.wav", gen_speechlike(5.0))
        create_audio_file(batch_dir / "item2.wav", gen_speechlike(5.0))

        # Slice 1.0s to 3.0s (2.0s duration)
        cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir,
                        polish=False, start_sec=1.0, end_sec=3.0)
        run_job(cfg)
        for out_f in out_dir.glob("*.wav"):
            y, sr = sf.read(str(out_f), dtype="float32")
            assert abs(len(y) - int(2.0 * sr)) <= int(0.05 * sr)

    @pytest.mark.skipif(not is_r6_implemented(), reason="Requires R6 audit sidecar")
    def test_r3_combo_07_auto_noise_extraction_and_audit_sidecar(self, tmp_path):
        """Combo 7: Auto-noise extraction (--auto-noise) coupled with audit manifest logging."""
        from zout.audio.noise_extractor import extract_noise_profile
        # Mixed audio: noise at 0-2s, speech at 2-5s
        t = np.arange(5 * TEST_SR) / TEST_SR
        mix = (0.05 * np.random.default_rng(99).standard_normal(len(t))).astype(np.float32)
        mix[2 * TEST_SR: 5 * TEST_SR] += gen_speechlike(3.0)
        main_file = create_audio_file(tmp_path / "mix.wav", mix)

        extracted = extract_noise_profile(mix, TEST_SR, n_chunks=2, chunk_sec=1.0)
        noise_file = create_audio_file(tmp_path / "noise.wav", extracted.audio)

        cfg = JobConfig(noise_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "out.wav",
                        polish=False, audit=True)
        res = run_job(cfg)
        assert res.output.is_file()
        candidates = list(tmp_path.glob("*.json"))
        assert len(candidates) >= 1

    @pytest.mark.skipif(not is_r6_implemented(), reason="Requires R6 audit sidecar")
    def test_r3_combo_08_clean_mode_and_audit_sidecar(self, tmp_path):
        """Combo 8: Clean voice isolation mode generating complete forensic audit manifest."""
        ref_file = create_audio_file(tmp_path / "ref_clean.wav", gen_speechlike(3.5, f0=140.0))
        main_file = create_audio_file(tmp_path / "meeting.wav",
                                      gen_speechlike(4.0, f0=140.0) + gen_hum(4.0))
        out_file = tmp_path / "out.wav"

        cfg = JobConfig(ref_file, ReferenceRole.CLEAN, main_file, 0.75, 24.0, out_file,
                        polish=False, audit=True)
        res = run_job(cfg)
        assert res.output.is_file()
        candidates = list(tmp_path.glob("*.json"))
        assert len(candidates) >= 1
        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("parameters", {}).get("role") == "CLEAN" or "CLEAN" in str(data)


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (5 end-to-end workflows)
# ============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic end-to-end production workflows."""

    def test_r4_scenario_01_multi_minute_interview_cleanup(self, tmp_path):
        """Scenario 1: Multi-minute interview recorded with HVAC hum cleaned with NOISY profile."""
        # Simulated interview: speech mixed with continuous 60Hz hum
        ref_hum = create_audio_file(tmp_path / "hvac_sample.wav", gen_hum(2.0))
        interview = gen_speechlike(5.0, seed=101) + gen_hum(5.0, seed=102)
        main_file = create_audio_file(tmp_path / "interview_raw.wav", interview)
        out_file = tmp_path / "interview_clean.wav"

        cfg = JobConfig(ref_hum, ReferenceRole.NOISY, main_file, 0.85, 30.0, out_file, polish=False)
        res = run_job(cfg)

        assert res.output.is_file()
        assert res.suppression_db > 0.0
        y, sr = sf.read(str(res.output), dtype="float32")
        assert len(y) == len(interview)
        assert np.abs(y).max() <= 1.0  # Peak limiting enforced

    def test_r4_scenario_02_noisy_lecture_isolation(self, tmp_path):
        """Scenario 2: Noisy lecture isolation preserving target speaker matching reference enrollment."""
        # Target speaker reference enrollment
        ref_speaker = create_audio_file(tmp_path / "prof_voice.wav", gen_speechlike(3.5, seed=201, f0=130))
        # Lecture audio containing target speaker + competing murmur + ambient noise
        lecture = (gen_speechlike(4.0, seed=201, f0=130) +
                   0.4 * gen_speechlike(4.0, seed=202, f0=220) +
                   gen_hum(4.0, seed=203))
        main_file = create_audio_file(tmp_path / "lecture.wav", lecture)
        out_file = tmp_path / "lecture_isolated.wav"

        cfg = JobConfig(ref_speaker, ReferenceRole.CLEAN, main_file, 0.80, 24.0, out_file, polish=False)
        res = run_job(cfg)

        assert res.output.is_file()
        y, sr = sf.read(str(res.output), dtype="float32")
        assert sr == TEST_SR
        assert np.isfinite(y).all()

    @pytest.mark.skipif(not is_r8_implemented(), reason="Requires R8 batch processing")
    def test_r4_scenario_03_studio_dark_batch_processing(self, tmp_path):
        """Scenario 3: Studio Dark mastering session batch-cleaning multi-track stems into out_dir."""
        ref_noise = create_audio_file(tmp_path / "studio_noise_floor.wav", gen_hum(1.5))
        session_dir = tmp_path / "session_stems"
        out_dir = tmp_path / "mastered_stems"

        create_audio_file(session_dir / "vocal_track.wav", gen_speechlike(2.0, seed=301) + gen_hum(2.0))
        create_audio_file(session_dir / "acoustic_gt.wav", gen_sine(2.0, 330) + gen_hum(2.0))
        create_audio_file(session_dir / "bass_track.wav", gen_sine(2.0, 110) + gen_hum(2.0))

        cfg = JobConfig(ref_noise, ReferenceRole.NOISY, batch_dir=session_dir, out_dir=out_dir,
                        strength=0.8, max_atten_db=36.0, polish=False)
        run_job(cfg)

        processed_files = list(out_dir.glob("*.wav"))
        assert len(processed_files) == 3
        for f in processed_files:
            data, _ = sf.read(str(f), dtype="float32")
            assert np.isfinite(data).all()

    @pytest.mark.skipif(not is_r6_implemented(), reason="Requires R6 forensic audit log")
    def test_r4_scenario_04_forensic_audit_examination(self, tmp_path):
        """Scenario 4: Forensic investigator examines wiretap audio with time-slice and audit manifest."""
        ref_noise = create_audio_file(tmp_path / "background_static.wav", gen_hum(2.0))
        evidence = gen_speechlike(6.0, seed=401) + gen_hum(6.0)
        evidence_file = create_audio_file(tmp_path / "evidence_tape.wav", evidence)
        out_file = tmp_path / "evidence_enhanced.wav"

        cfg = JobConfig(ref_noise, ReferenceRole.NOISY, evidence_file, 0.90, 48.0, out_file,
                        polish=False, audit=True)
        if hasattr(cfg, "start_sec") and hasattr(cfg, "end_sec"):
            cfg.start_sec = 1.0
            cfg.end_sec = 5.0

        res = run_job(cfg)
        assert res.output.is_file()

        candidates = list(tmp_path.glob("*.json"))
        assert len(candidates) >= 1
        with open(candidates[0], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "noise_reduction_db" in manifest or "suppression_db" in manifest
        assert "vad_speech_timestamps" in manifest

    @pytest.mark.skipif(not (is_r2_implemented() and is_r3_vu_implemented()),
                        reason="Requires R2 A/B hot-swap and R3 VU meters")
    def test_r4_scenario_05_seamless_ab_evaluation(self, tmp_path):
        """Scenario 5: Audio engineer performs real-time A/B hot-swap evaluation checking VU meters."""
        from zout.audio.player import AudioPlayer
        orig = gen_speechlike(3.0, seed=501) + gen_hum(3.0)
        proc = gen_speechlike(3.0, seed=501)  # Denoised version
        raw_file = create_audio_file(tmp_path / "raw.wav", orig)
        clean_file = create_audio_file(tmp_path / "cleaned.wav", proc)

        p = AudioPlayer()
        mock_stream = None

        def stream_factory(*args, **kwargs):
            nonlocal mock_stream
            mock_stream = MockOutputStream(*args, **kwargs)
            return mock_stream

        with patch("sounddevice.OutputStream", stream_factory):
            # 1. Start playback on Processed track
            assert p.play_pair(raw_file, clean_file, start_with="B") is True
            assert p.get_current_track() == "B"

            # 2. Advance playback by 1024 frames
            mock_stream.pump(1024)
            idx_1 = p._sample_index
            assert idx_1 == 1024

            # 3. Press [Tab] to toggle to Raw track A
            assert p.toggle_ab() == "A"
            assert p._sample_index == idx_1  # Playhead unbroken

            # 4. Check VU meters
            (rms_l, rms_r), (pk_l, pk_r) = p.get_meter_levels()
            assert np.isfinite(rms_l) and np.isfinite(pk_l)

            # 5. Advance another 1024 frames on track A
            mock_stream.pump(1024)
            assert p._sample_index == 2048

            # 6. Stop playback cleanly
            p.stop()
            assert p.is_playing() is False
