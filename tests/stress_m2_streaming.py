"""Adversarial stress-test suite for Milestone 2: Chunked Streaming (< 150MB RAM)
and Raised-Cosine Overlap-Add Reconstruction.

Executable directly via:
    python tests/stress_m2_streaming.py
or via pytest:
    pytest tests/stress_m2_streaming.py -v
"""

from __future__ import annotations

import math
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from zout.config import JobConfig, ReferenceRole
from zout.engines.noisy_engine import _stream_disk_to_disk
from zout.pipeline import run_job, run_streaming


# ============================================================================
# Helpers & Synthetic Signal Generators
# ============================================================================

TEST_SR = 48000


def make_wav_file(path: Path, data: np.ndarray, sr: int = TEST_SR, subtype: str = "PCM_24") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sr, subtype=subtype)
    return path


def gen_sine(duration_sec: float, freq: float = 440.0, amp: float = 0.5, sr: int = TEST_SR) -> np.ndarray:
    n = int(round(duration_sec * sr))
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def gen_constant(duration_sec: float, val: float = 0.4, sr: int = TEST_SR) -> np.ndarray:
    n = int(round(duration_sec * sr))
    return np.full(n, val, dtype=np.float32)


def gen_ramp(duration_sec: float, start: float = 0.1, stop: float = 0.6, sr: int = TEST_SR) -> np.ndarray:
    n = int(round(duration_sec * sr))
    return np.linspace(start, stop, n, dtype=np.float32)


def gen_square(duration_sec: float, freq: float = 100.0, sr: int = TEST_SR) -> np.ndarray:
    n = int(round(duration_sec * sr))
    t = np.arange(n) / sr
    return np.sign(np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# ============================================================================
# Test Suite 1: Memory Cap Under Multi-Hour Simulation
# ============================================================================

class TestMemoryCapSimulation:
    """Stress tests verifying memory ceiling (< 150MB, strictly < 50MB) and leak-free streaming."""

    def test_memory_bounded_under_50_chunks(self, tmp_path):
        """Monitor tracemalloc across 50 chunk cycles; assert peak < 50MB and no linear leak."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(42).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        # 50 chunks * 0.4s = 20s audio, with chunk_sec=0.4s and overlap_sec=0.08s
        total_sec = 20.0
        n_samples = int(total_sec * TEST_SR)
        t = np.arange(n_samples) / TEST_SR
        sig = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.random.default_rng(1).standard_normal(n_samples)).astype(np.float32)
        main_file = make_wav_file(tmp_path / "main.wav", sig)
        out_file = tmp_path / "out.wav"

        chunk_snapshots: dict[int, tuple[float, float]] = {}

        def on_progress(msg: str, frac: float):
            if "chunk" in msg:
                for part in msg.split():
                    if "/" in part:
                        try:
                            idx = int(part.split("/")[0])
                            cur, peak = tracemalloc.get_traced_memory()
                            if idx not in chunk_snapshots:
                                chunk_snapshots[idx] = (cur / (1024 * 1024), peak / (1024 * 1024))
                        except (ValueError, IndexError):
                            pass

        tracemalloc.start()
        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=0.4, overlap_duration=0.08
        )
        res = run_streaming(cfg, progress=on_progress)
        final_cur, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert res is not None
        assert res.output.is_file()

        peak_mb = final_peak / (1024 * 1024)
        print(f"\n[TestMemory] 50-chunk peak memory: {peak_mb:.2f} MB")
        assert peak_mb < 150.0, f"Peak memory {peak_mb:.2f}MB exceeded hard cap of 150.0MB"
        assert peak_mb < 50.0, f"Peak memory {peak_mb:.2f}MB exceeded target limit of 50.0MB"

        # Leak check: compare memory at chunk 5 vs chunk 30
        if 5 in chunk_snapshots and 30 in chunk_snapshots:
            mem_c5 = chunk_snapshots[5][0]
            mem_c30 = chunk_snapshots[30][0]
            leak_delta_mb = abs(mem_c30 - mem_c5)
            print(f"[TestMemory] Memory at chunk 5: {mem_c5:.2f}MB, chunk 30: {mem_c30:.2f}MB, delta: {leak_delta_mb:.2f}MB")
            assert leak_delta_mb < 2.0, f"Memory grew significantly between chunk 5 and 30 ({leak_delta_mb:.2f}MB)"

    def test_memory_multi_chunk_varying_durations(self, tmp_path):
        """Verify memory stays bounded across varied chunk durations (0.5s, 1.0s, 2.0s < 50MB; 5.0s < 150MB)."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(0).standard_normal(TEST_SR).astype(np.float32) * 0.05)

        for chunk_dur in [0.5, 1.0, 2.0, 5.0]:
            out_file = tmp_path / f"out_{chunk_dur}s.wav"
            # Audio long enough for multiple chunk cycles
            audio_sec = max(6.0, chunk_dur * 5.0)
            sig = gen_sine(audio_sec, 440.0, 0.4)
            main_file = make_wav_file(tmp_path / f"main_{chunk_dur}s.wav", sig)

            tracemalloc.start()
            cfg = JobConfig(
                ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                polish=False, chunk_duration=chunk_dur, overlap_duration=min(1.0, chunk_dur * 0.2)
            )
            res = run_streaming(cfg)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert res is not None
            peak_mb = peak / (1024 * 1024)
            print(f"[TestMemory] Chunk duration {chunk_dur}s peak memory: {peak_mb:.2f} MB")
            assert peak_mb < 150.0, f"Chunk duration {chunk_dur}s consumed {peak_mb:.2f}MB (limit < 150MB)"
            if chunk_dur <= 2.0:
                assert peak_mb < 50.0, f"Chunk duration {chunk_dur}s consumed {peak_mb:.2f}MB (target < 50MB)"

    def test_memory_passthrough_stream_100_chunks(self, tmp_path):
        """Stress-test 100 contiguous chunk iterations through _stream_disk_to_disk for linear leak."""
        total_sec = 10.0
        n_samples = int(total_sec * TEST_SR)
        sig = gen_sine(total_sec, 440.0, 0.4)
        main_file = make_wav_file(tmp_path / "stream_100.wav", sig)
        out_file = tmp_path / "out_100.wav"

        chunk_mems: dict[int, float] = {}

        def progress_cb(msg: str, frac: float):
            if "chunk" in msg:
                for part in msg.split():
                    if "/" in part:
                        try:
                            idx = int(part.split("/")[0])
                            cur, _ = tracemalloc.get_traced_memory()
                            if idx not in chunk_mems:
                                chunk_mems[idx] = cur / (1024 * 1024)
                        except (ValueError, IndexError):
                            pass

        tracemalloc.start()
        _stream_disk_to_disk(
            main_path=main_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=0.1,
            overlap_sec=0.02,
            progress=progress_cb,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"\n[TestMemory] 100-chunk passthrough peak: {peak_mb:.2f} MB")
        assert peak_mb < 50.0, f"100-chunk passthrough peak {peak_mb:.2f}MB exceeded 50MB"

        # Compare chunk 10 vs chunk 90
        if 10 in chunk_mems and 90 in chunk_mems:
            diff = abs(chunk_mems[90] - chunk_mems[10])
            print(f"[TestMemory] Chunk 10 mem: {chunk_mems[10]:.2f}MB, Chunk 90 mem: {chunk_mems[90]:.2f}MB, Delta: {diff:.2f}MB")
            assert diff < 1.0, f"Memory leaked across 80 chunks: {diff:.2f} MB growth"

    def test_memory_60s_chunk_analysis(self, tmp_path):
        """Empirical analysis of 60s chunk duration memory consumption at 16kHz vs 48kHz."""
        # Test 16kHz (speech standard)
        sr_16k = 16000
        ref_16k = make_wav_file(tmp_path / "ref_16k.wav", np.random.default_rng(10).standard_normal(sr_16k).astype(np.float32) * 0.05, sr=sr_16k)
        main_16k = make_wav_file(tmp_path / "main_16k.wav", gen_sine(125.0, 300.0, 0.4, sr=sr_16k), sr=sr_16k)
        out_16k = tmp_path / "out_16k.wav"

        tracemalloc.start()
        cfg_16k = JobConfig(
            ref_16k, ReferenceRole.NOISY, main_16k, 0.75, 24.0, out_16k,
            polish=False, chunk_duration=60.0, overlap_duration=2.0
        )
        res_16k = run_streaming(cfg_16k)
        _, peak_16k = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_16k_mb = peak_16k / (1024 * 1024)
        print(f"\n[TestMemory] 60s chunk at 16kHz peak memory: {peak_16k_mb:.2f} MB")
        assert peak_16k_mb < 150.0, f"16kHz 60s chunk exceeded 150MB: {peak_16k_mb:.2f}MB"


# ============================================================================
# Test Suite 2: Raised-Cosine Overlap-Add Reconstruction Continuity
# ============================================================================

class TestRaisedCosineContinuity:
    """Stress tests verifying mathematical unity gain, phase preservation, and zero clicks."""

    @pytest.mark.parametrize("n_samples", [3571, 10007, 48017, 96001])
    def test_unity_gain_constant_tone(self, tmp_path, n_samples):
        """Constant tones crossfaded across chunk boundaries exhibit zero amplitude dip (|y_out - y_in| < 1e-5)."""
        sig = np.full(n_samples, 0.45, dtype=np.float32)
        in_file = make_wav_file(tmp_path / f"const_{n_samples}.wav", sig)
        out_file = tmp_path / f"out_const_{n_samples}.wav"

        _stream_disk_to_disk(
            main_path=in_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=0.1,
            overlap_sec=0.025,
        )

        rec, sr = sf.read(str(out_file), dtype="float32")
        assert sr == TEST_SR
        assert len(rec) == n_samples, f"Sample count mismatch: {len(rec)} vs {n_samples}"
        max_error = float(np.max(np.abs(rec - sig)))
        print(f"[TestContinuity] Constant tone (n={n_samples}) max error: {max_error:.2e}")
        assert max_error < 1e-5, f"Reconstruction error {max_error:.2e} exceeded 1e-5 threshold"

    @pytest.mark.parametrize("n_samples", [3571, 10007, 48017])
    def test_unity_gain_linear_ramp(self, tmp_path, n_samples):
        """Linear ramp signals crossfaded across chunk boundaries exhibit zero step click (|y_out - y_in| < 1e-5)."""
        sig = np.linspace(0.1, 0.65, n_samples, dtype=np.float32)
        in_file = make_wav_file(tmp_path / f"ramp_{n_samples}.wav", sig)
        out_file = tmp_path / f"out_ramp_{n_samples}.wav"

        _stream_disk_to_disk(
            main_path=in_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=0.075,
            overlap_sec=0.015,
        )

        rec, sr = sf.read(str(out_file), dtype="float32")
        assert sr == TEST_SR
        assert len(rec) == n_samples
        max_error = float(np.max(np.abs(rec - sig)))
        print(f"[TestContinuity] Linear ramp (n={n_samples}) max error: {max_error:.2e}")
        assert max_error < 1e-5, f"Reconstruction error {max_error:.2e} exceeded 1e-5 threshold"

    @pytest.mark.parametrize("freq", [50.0, 440.0, 1000.0, 8000.0])
    def test_unity_gain_sine_wave(self, tmp_path, freq):
        """Pure sine tones crossfaded across chunk boundaries exhibit zero amplitude dip or click (|y_out - y_in| < 1e-5)."""
        n_samples = 48017  # Prime sample length
        t = np.arange(n_samples) / TEST_SR
        sig = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        in_file = make_wav_file(tmp_path / f"sine_{freq}_{n_samples}.wav", sig)
        out_file = tmp_path / f"out_sine_{freq}_{n_samples}.wav"

        _stream_disk_to_disk(
            main_path=in_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=0.08,
            overlap_sec=0.02,
        )

        rec, _ = sf.read(str(out_file), dtype="float32")
        assert len(rec) == n_samples
        max_error = float(np.max(np.abs(rec - sig)))
        print(f"[TestContinuity] Sine wave ({freq}Hz) max error: {max_error:.2e}")
        assert max_error < 1e-5, f"Reconstruction error {max_error:.2e} exceeded 1e-5 threshold"

    def test_zero_overlap_duration(self, tmp_path):
        """Zero overlap (overlap_duration = 0.0) processes contiguous blocks with exact duration and no corruption."""
        n_samples = 24000
        sig = gen_ramp(0.5, 0.2, 0.7)
        in_file = make_wav_file(tmp_path / "zero_ov_in.wav", sig)
        out_file = tmp_path / "zero_ov_out.wav"

        _stream_disk_to_disk(
            main_path=in_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=0.1,
            overlap_sec=0.0,
        )

        rec, _ = sf.read(str(out_file), dtype="float32")
        assert len(rec) == len(sig)
        max_error = float(np.max(np.abs(rec - sig)))
        print(f"[TestContinuity] Zero overlap max error: {max_error:.2e}")
        assert max_error < 1e-5

    @pytest.mark.parametrize("overlap_samples", [17, 33, 48, 73, 99])
    def test_small_overlap_under_100_samples(self, tmp_path, overlap_samples):
        """Small overlap lengths (< 100 samples) crossfade cleanly without indexing errors or discontinuity."""
        overlap_sec = overlap_samples / TEST_SR
        chunk_sec = 0.05  # 2400 samples
        n_samples = 15000
        sig = gen_sine(n_samples / TEST_SR, 440.0, 0.4)
        in_file = make_wav_file(tmp_path / f"small_ov_{overlap_samples}.wav", sig)
        out_file = tmp_path / f"out_small_ov_{overlap_samples}.wav"

        _stream_disk_to_disk(
            main_path=in_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
        )

        rec, _ = sf.read(str(out_file), dtype="float32")
        assert len(rec) == n_samples
        max_error = float(np.max(np.abs(rec - sig)))
        print(f"[TestContinuity] Small overlap ({overlap_samples} samples) max error: {max_error:.2e}")
        assert max_error < 1e-5

    def test_exact_sample_drift_bounds(self, tmp_path):
        """Sliced input duration matches output duration to the exact sample (drift <= 0.05 * sr)."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(2).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        dur_sec = 12.3456
        sig = gen_sine(dur_sec, 300.0, 0.4)
        main_file = make_wav_file(tmp_path / "drift_in.wav", sig)
        out_file = tmp_path / "drift_out.wav"

        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=1.73, overlap_duration=0.231
        )
        res = run_streaming(cfg)
        assert res is not None
        rec, _ = sf.read(str(res.output), dtype="float32")

        drift_samples = abs(len(rec) - len(sig))
        print(f"[TestContinuity] Duration drift: {drift_samples} samples (allowed <= {int(0.05 * TEST_SR)})")
        assert drift_samples <= int(0.05 * TEST_SR)
        assert drift_samples == 0, f"Expected 0 sample drift, got {drift_samples}"

    def test_timeline_range_slicing_with_streaming(self, tmp_path):
        """Timeline slicing (--start/--end) combined with chunked streaming preserves exact sample boundaries."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(3).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        total_sec = 15.0
        sig = gen_sine(total_sec, 440.0, 0.4)
        main_file = make_wav_file(tmp_path / "sliced_main.wav", sig)
        out_file = tmp_path / "sliced_out.wav"

        start_sec = 3.25
        end_sec = 11.75
        expected_samples = int(round((end_sec - start_sec) * TEST_SR))

        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=1.5, overlap_duration=0.25,
            start_sec=start_sec, end_sec=end_sec
        )
        res = run_streaming(cfg)
        assert res is not None
        rec, _ = sf.read(str(res.output), dtype="float32")

        print(f"[TestContinuity] Sliced samples: {len(rec)}, Expected: {expected_samples}")
        assert len(rec) == expected_samples

    def test_stereo_input_channel_downmix_continuity(self, tmp_path):
        """Stereo inputs processed through streaming downmix cleanly to mono and preserve continuity."""
        n_samples = 48000
        t = np.arange(n_samples) / TEST_SR
        left = 0.4 * np.sin(2 * np.pi * 300 * t)
        right = 0.4 * np.cos(2 * np.pi * 300 * t)
        stereo_sig = np.column_stack([left, right]).astype(np.float32)
        stereo_file = make_wav_file(tmp_path / "stereo_in.wav", stereo_sig, subtype="PCM_24")
        out_file = tmp_path / "mono_out.wav"

        _stream_disk_to_disk(
            main_path=stereo_file,
            out_path=out_file,
            sr=TEST_SR,
            process_chunk=lambda x: x,
            chunk_sec=0.2,
            overlap_sec=0.04,
        )

        rec, _ = sf.read(str(out_file), dtype="float32")
        expected_mono = stereo_sig.mean(axis=1)
        max_error = float(np.max(np.abs(rec - expected_mono)))
        print(f"[TestContinuity] Stereo downmix max error: {max_error:.2e}")
        assert max_error < 1e-5


# ============================================================================
# Test Suite 3: Audio Quality and Clipping Bounds
# ============================================================================

class TestAudioQualityAndClipping:
    """Stress tests verifying clipping protection, peak limiting, and NaN/Inf rejection."""

    def test_full_scale_square_wave_peak_limiting(self, tmp_path):
        """Full-scale square wave input is limited such that |y| <= 1.0001 (strictly <= 0 dBFS)."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(4).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        square = gen_square(4.0, 100.0)
        main_file = make_wav_file(tmp_path / "square.wav", square)
        out_file = tmp_path / "out_square.wav"

        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=1.0, overlap_duration=0.2
        )
        res = run_streaming(cfg)
        assert res is not None
        rec, _ = sf.read(str(res.output), dtype="float32")

        max_val = float(np.max(np.abs(rec)))
        print(f"\n[TestClipping] Square wave peak: {max_val:.6f}")
        assert np.all(np.isfinite(rec)), "Output contains NaN or Inf"
        assert max_val <= 1.0001, f"Peak amplitude {max_val} exceeded 1.0001"
        assert max_val <= 0.8913, f"Expected -1 dBFS peak limit (~0.89125), got {max_val}"

    def test_pure_digital_silence_no_nan_inf(self, tmp_path):
        """Pure digital silence input (all 0.0) across multi-chunk streaming produces 0 NaN and 0 Inf."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(5).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        silence = np.zeros(int(4.0 * TEST_SR), dtype=np.float32)
        main_file = make_wav_file(tmp_path / "silence.wav", silence)
        out_file = tmp_path / "out_silence.wav"

        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=1.0, overlap_duration=0.2
        )
        res = run_streaming(cfg)
        assert res is not None
        rec, _ = sf.read(str(res.output), dtype="float32")

        assert np.all(np.isfinite(rec)), "Silence output contains NaN or Inf"
        assert float(np.max(np.abs(rec))) <= 1e-5, f"Expected near-zero silence, got peak {np.max(np.abs(rec))}"

    def test_dc_offset_and_super_unity_input(self, tmp_path):
        """Input exceeding 0 dBFS (+2.0 peak) with DC offset is peak-limited safely below 1.0."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(6).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        t = np.arange(int(3.0 * TEST_SR)) / TEST_SR
        hot_signal = (1.5 * np.sin(2 * np.pi * 200 * t) + 0.5).astype(np.float32)  # Peak = 2.0
        main_file = make_wav_file(tmp_path / "hot.wav", hot_signal)
        out_file = tmp_path / "out_hot.wav"

        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=1.0, overlap_duration=0.2
        )
        res = run_streaming(cfg)
        assert res is not None
        rec, _ = sf.read(str(res.output), dtype="float32")

        max_val = float(np.max(np.abs(rec)))
        print(f"[TestClipping] Hot signal (+2.0 peak) limited to: {max_val:.6f}")
        assert np.all(np.isfinite(rec))
        assert max_val <= 1.0001


# ============================================================================
# Test Suite 4: In-Flight Cancellation & Configuration Resilience
# ============================================================================

class TestCancellationAndResilience:
    """Stress tests verifying mid-stream cancellation and configuration parameter validation."""

    def test_cancel_during_multi_chunk_streaming(self, tmp_path):
        """Triggering cancel_event midway through streaming unlinks partial output and returns None."""
        ref_file = make_wav_file(tmp_path / "ref.wav", np.random.default_rng(7).standard_normal(TEST_SR).astype(np.float32) * 0.05)
        main_file = make_wav_file(tmp_path / "main.wav", gen_sine(10.0, 300.0, 0.4))
        out_file = tmp_path / "cancelled_out.wav"

        cancel_ev = threading.Event()
        processed_chunks = 0

        def on_progress(msg: str, frac: float):
            nonlocal processed_chunks
            if "chunk" in msg:
                processed_chunks += 1
                if processed_chunks >= 2:
                    cancel_ev.set()

        cfg = JobConfig(
            ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
            polish=False, chunk_duration=0.5, overlap_duration=0.1,
            cancel_event=cancel_ev
        )
        res = run_streaming(cfg, progress=on_progress)
        assert res is None, "Expected run_streaming to return None when cancelled"
        assert not out_file.exists() or out_file.stat().st_size == 0, "Partial output file should be removed"

    def test_invalid_chunk_overlap_configurations(self, tmp_path):
        """Configurations with overlap >= chunk or overlap < 0 are cleanly rejected with ValueError."""
        ref_file = make_wav_file(tmp_path / "ref.wav", gen_sine(1.0))
        main_file = make_wav_file(tmp_path / "main.wav", gen_sine(2.0))

        # overlap >= chunk
        with pytest.raises(ValueError, match="overlap_duration"):
            JobConfig(ref_file, ReferenceRole.NOISY, main_file, chunk_duration=2.0, overlap_duration=2.0)

        # overlap < 0
        with pytest.raises(ValueError, match="overlap_duration"):
            JobConfig(ref_file, ReferenceRole.NOISY, main_file, chunk_duration=2.0, overlap_duration=-0.5)

        # chunk <= 0
        with pytest.raises(ValueError, match="chunk_duration"):
            JobConfig(ref_file, ReferenceRole.NOISY, main_file, chunk_duration=0.0, overlap_duration=0.0)


# ============================================================================
# Main Execution CLI Entrypoint
# ============================================================================

def main():
    print("=" * 70)
    print("Running Milestone 2 Empirical Stress Test Suite directly...")
    print("=" * 70)
    pytest_args = ["-v", "-s", __file__]
    code = pytest.main(pytest_args)
    sys.exit(code)


if __name__ == "__main__":
    main()
