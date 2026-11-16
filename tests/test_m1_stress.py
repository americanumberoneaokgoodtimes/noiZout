"""Milestone 1 (M1) Empirical Adversarial Stress Test Suite for noiZout.

Tests:
1. Thread safety and race conditions under rapid concurrent toggle_ab() calls.
2. Strict monotonic progression of sample_index across hot-swaps.
3. Meter levels dBFS bounds and accuracy (square, sine, silence, overs, corrupt inputs).
4. Headless mode fallback under simulated PortAudio failure scenarios.
5. Boundary edge cases (zero length, single sample, high mismatch, rapid play/stop churn).
"""

import math
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from zout.audio.io import compute_waveform_envelope, load_audio, parse_time_spec
from zout.audio.player import AudioPlayer, HeadlessRunner, _calc_dbfs, _calc_meters
from zout.config import JobConfig


@pytest.fixture
def audio_pair(tmp_path):
    """Generate two 2-second stereo wav files for A/B testing."""
    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    # Track A: 440 Hz tone
    track_a = np.column_stack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t))).astype(np.float32)
    # Track B: 880 Hz tone
    track_b = np.column_stack((np.sin(2 * np.pi * 880 * t), np.sin(2 * np.pi * 880 * t))).astype(np.float32)

    path_a = tmp_path / "track_a.wav"
    path_b = tmp_path / "track_b.wav"
    sf.write(str(path_a), track_a, sr)
    sf.write(str(path_b), track_b, sr)
    return path_a, path_b, sr


# ============================================================================
# 1. Thread Safety: Rapid Concurrent toggle_ab()
# ============================================================================

def test_stress_concurrent_toggle_ab(audio_pair):
    """Stress test: 10 worker threads each calling toggle_ab() 50 times concurrently.

    Simultaneously, a polling thread queries sample_index, meter_levels, and status_text.
    Verifies no deadlocks, exceptions, or race conditions.
    """
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=True)
    assert player.play_pair(path_a, path_b, start_with="B") is True

    num_threads = 10
    toggles_per_thread = 50
    exceptions = []
    stop_poll = threading.Event()
    polled_indices = []

    def poll_worker():
        while not stop_poll.is_set():
            try:
                idx = player.sample_index
                trk = player.get_current_track()
                meters = player.get_meter_levels()
                status = player.status_text()
                polled_indices.append(idx)
                assert trk in ("A", "B")
                assert isinstance(meters, tuple)
                assert isinstance(status, str)
            except Exception as e:
                exceptions.append(e)
            time.sleep(0.001)

    def toggle_worker():
        for _ in range(toggles_per_thread):
            try:
                res = player.toggle_ab()
                assert res in ("A", "B")
            except Exception as e:
                exceptions.append(e)

    poll_t = threading.Thread(target=poll_worker)
    poll_t.start()

    workers = [threading.Thread(target=toggle_worker) for _ in range(num_threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=5.0)
        assert not w.is_alive(), "Worker thread timed out (deadlock detected)"

    stop_poll.set()
    poll_t.join(timeout=2.0)
    assert not poll_t.is_alive(), "Poll thread timed out"

    player.stop()

    assert len(exceptions) == 0, f"Exceptions encountered during concurrent toggle: {exceptions}"
    assert len(polled_indices) > 0, "Polling thread did not record any samples"


# ============================================================================
# 2. Strict Monotonicity: sample_index NEVER resets on toggle_ab()
# ============================================================================

def test_stress_sample_index_strictly_monotonic_across_toggles(audio_pair):
    """Verify sample_index strictly increases and NEVER resets or stutters backwards on toggle_ab()."""
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=True)
    assert player.play_pair(path_a, path_b, start_with="A") is True

    observed_indices = []
    toggle_events = []
    start_time = time.time()

    # Run for 0.5 seconds while toggling repeatedly
    toggle_count = 0
    while time.time() - start_time < 0.5:
        idx = player.sample_index
        observed_indices.append(idx)

        # Toggle roughly every 20ms
        if len(observed_indices) % 5 == 0:
            trk = player.toggle_ab()
            toggle_events.append((len(observed_indices), idx, trk))
            toggle_count += 1
        time.sleep(0.004)

    player.stop()

    # Windows timer resolution is ~15ms, so sleep(0.004) sleeps ~15ms actual.
    # In 0.5s we get ~33 iterations, with toggle every 5th = ~6 toggles minimum.
    assert toggle_count >= 4, f"Expected at least 4 toggles, got {toggle_count}"
    assert len(observed_indices) >= 10, f"Expected >= 10 index samples, got {len(observed_indices)}"

    # Verification: monotonicity
    for i in range(1, len(observed_indices)):
        prev = observed_indices[i - 1]
        curr = observed_indices[i]
        assert curr >= prev, (
            f"Sample index went backwards! Step {i}: prev={prev}, curr={curr}. "
            f"Toggles: {toggle_events}"
        )

    # Verification: sample_index did not reset to 0 mid-playback
    assert observed_indices[-1] > observed_indices[0], (
        f"Playback did not advance: start={observed_indices[0]}, end={observed_indices[-1]}"
    )


def test_sample_index_monotonicity_manual_callback_pumping(audio_pair):
    """Deterministic test: directly pump audio callback blocks and toggle between pumps."""
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=True)
    player.play_pair(path_a, path_b, start_with="B")

    # Stop headless runner to take manual control of callback
    if player._headless_runner:
        player._headless_runner.stop()

    blocksize = 512
    outdata = np.zeros((blocksize, 2), dtype=np.float32)
    history = []

    for step in range(30):
        # Pump one block
        player._audio_callback(outdata, blocksize, None, None)
        idx_after_pump = player.sample_index
        history.append(("pump", idx_after_pump))

        # Toggle track
        trk = player.toggle_ab()
        idx_after_toggle = player.sample_index
        history.append(("toggle", idx_after_toggle))

        # Index must remain identical immediately across toggle
        assert idx_after_toggle == idx_after_pump, (
            f"Step {step}: sample_index changed upon toggle! "
            f"before={idx_after_pump}, after={idx_after_toggle}"
        )

    # Check all indices in history are monotonically non-decreasing
    indices = [h[1] for h in history]
    for i in range(1, len(indices)):
        assert indices[i] >= indices[i - 1], f"Monotonicity failed at step {i}: {indices[i-1]} -> {indices[i]}"

    player.stop()


# ============================================================================
# 3. Meter Levels: dBFS Bounds, Mathematical Accuracy & Non-Finite Rejection
# ============================================================================

def test_meter_full_scale_square_wave():
    """Square wave at +/- 1.0 has RMS = 1.0 (0.0 dBFS) and Peak = 1.0 (0.0 dBFS)."""
    block = np.ones((1024, 2), dtype=np.float32)
    rms_l, rms_r, peak_l, peak_r = _calc_meters(block)
    assert abs(rms_l - 0.0) < 1e-4, f"Square wave RMS L was {rms_l}, expected 0.0 dBFS"
    assert abs(rms_r - 0.0) < 1e-4, f"Square wave RMS R was {rms_r}, expected 0.0 dBFS"
    assert abs(peak_l - 0.0) < 1e-4, f"Square wave Peak L was {peak_l}, expected 0.0 dBFS"
    assert abs(peak_r - 0.0) < 1e-4, f"Square wave Peak R was {peak_r}, expected 0.0 dBFS"


def test_meter_full_scale_sine_wave():
    """Full-scale sine wave at amplitude 1.0 has Peak = 0.0 dBFS and RMS ~ -3.0103 dBFS."""
    t = np.linspace(0, 1, 44100, endpoint=False)
    sine = np.sin(2 * np.pi * 1000 * t[:1024]).astype(np.float32)
    block = np.column_stack((sine, sine))
    rms_l, rms_r, peak_l, peak_r = _calc_meters(block)

    expected_rms = 20.0 * np.log10(1.0 / np.sqrt(2))  # ~ -3.0103 dBFS
    assert abs(rms_l - expected_rms) < 0.1, f"Sine wave RMS was {rms_l}, expected ~ {expected_rms} dBFS"
    assert abs(peak_l - 0.0) < 0.05, f"Sine wave Peak was {peak_l}, expected 0.0 dBFS"


def test_meter_digital_silence():
    """Digital silence must return strictly -80.0 dBFS (the configured floor)."""
    block = np.zeros((1024, 2), dtype=np.float32)
    rms_l, rms_r, peak_l, peak_r = _calc_meters(block)
    assert rms_l == -80.0
    assert rms_r == -80.0
    assert peak_l == -80.0
    assert peak_r == -80.0


def test_meter_sub_noise_floor_signals():
    """Signals below 1e-4 linear (-80 dBFS) must clamp cleanly to -80.0 dBFS."""
    tiny_block = np.full((1024, 2), 1e-6, dtype=np.float32)
    rms_l, rms_r, peak_l, peak_r = _calc_meters(tiny_block)
    assert rms_l == -80.0
    assert peak_l == -80.0


def test_meter_empty_block():
    """Empty block must return -80.0 dBFS without crashing or throwing ZeroDivisionError."""
    empty_block = np.zeros((0, 2), dtype=np.float32)
    levels = _calc_meters(empty_block)
    assert levels == (-80.0, -80.0, -80.0, -80.0)


def test_meter_rejection_of_nan_and_inf():
    """Adversarial stress: audio blocks containing NaN and Inf must NOT output Inf or NaN.

    Expected: meter levels must remain finite and bounded within valid dBFS range.
    """
    nan_block = np.full((1024, 2), np.nan, dtype=np.float32)
    levels_nan = _calc_meters(nan_block)
    for val in levels_nan:
        assert not math.isnan(val), f"Meter level produced NaN: {levels_nan}"
        assert not math.isinf(val), f"Meter level produced Inf: {levels_nan}"

    inf_block = np.full((1024, 2), np.inf, dtype=np.float32)
    levels_inf = _calc_meters(inf_block)
    for val in levels_inf:
        assert not math.isnan(val), f"Meter level produced NaN on Inf block: {levels_inf}"
        assert not math.isinf(val), f"Meter level produced Inf on Inf block: {levels_inf}"


# ============================================================================
# 4. Headless Fallback Under Simulated PortAudio Failure
# ============================================================================

def test_headless_fallback_on_output_stream_portaudio_error(audio_pair):
    """Simulate PortAudioError when creating sounddevice.OutputStream.

    Verify player automatically falls back to HeadlessRunner and continues working.
    """
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=False)

    with patch("sounddevice.OutputStream", side_effect=RuntimeError("PortAudio device error: -9996")):
        ok = player.play_pair(path_a, path_b, start_with="B")
        assert ok is True
        assert player._is_headless is True
        assert player._headless_runner is not None

        time.sleep(0.05)
        assert player.is_playing() is True
        assert player.sample_index > 0

        # Toggle during headless playback
        res = player.toggle_ab()
        assert res == "A"
        assert player.get_current_track() == "A"

        player.stop()
        assert player.is_playing() is False
        assert player._headless_runner is None


def test_headless_fallback_on_sounddevice_import_failure(audio_pair):
    """Simulate missing or unimportable sounddevice module."""
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=False)

    with patch.dict("sys.modules", {"sounddevice": None}):
        ok = player.play_pair(path_a, path_b, start_with="A")
        assert ok is True
        assert player._is_headless is True

        time.sleep(0.05)
        assert player.is_playing() is True

        player.stop()
        assert player.is_playing() is False


def test_headless_runner_thread_cleanup(audio_pair):
    """Verify HeadlessRunner thread is cleanly joined and terminated when stop() is called."""
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=True)
    player.play_pair(path_a, path_b)

    runner = player._headless_runner
    assert runner is not None
    assert runner._thread is not None
    assert runner._thread.is_alive()

    player.stop()
    assert not runner._thread.is_alive(), "HeadlessRunner worker thread leaked after stop()"


# ============================================================================
# 5. Boundary & Corner Cases: Edge Audio Inputs
# ============================================================================

def test_play_pair_with_zero_length_audio(tmp_path):
    """Zero-length audio file must be rejected gracefully without crash."""
    empty_file = tmp_path / "empty.wav"
    sf.write(str(empty_file), np.zeros((0, 2), dtype=np.float32), 44100)

    player = AudioPlayer(force_headless=True)
    ok = player.play_pair(empty_file, empty_file)
    assert ok is False
    assert player.is_playing() is False


def test_play_pair_single_sample_audio(tmp_path):
    """Single-sample audio file should not cause division by zero or index crash."""
    single_file = tmp_path / "single.wav"
    sf.write(str(single_file), np.array([[0.5, 0.5]], dtype=np.float32), 44100)

    player = AudioPlayer(force_headless=True)
    ok = player.play_pair(single_file, single_file)
    assert ok is True
    time.sleep(0.05)
    player.stop()


def test_rapid_play_stop_churn(audio_pair):
    """Rapid alternating calls to play_pair and stop in rapid succession."""
    path_a, path_b, sr = audio_pair
    player = AudioPlayer(force_headless=True)

    for _ in range(20):
        assert player.play_pair(path_a, path_b) is True
        player.toggle_ab()
        player.stop()
        assert player.is_playing() is False


def test_toggle_ab_when_not_playing():
    """Calling toggle_ab on an unstarted player toggles default track without crash."""
    player = AudioPlayer(force_headless=True)
    assert player.is_playing() is False
    assert player.get_current_track() == "B"
    trk1 = player.toggle_ab()
    assert trk1 == "A"
    assert player.get_current_track() == "A"
    trk2 = player.toggle_ab()
    assert trk2 == "B"
    assert player.get_current_track() == "B"


def test_envelope_with_extreme_values():
    """Waveform envelope calculation with constant DC, Nyquist oscillation, and large arrays."""
    # DC offset
    dc = np.full(1000, 0.5, dtype=np.float32)
    env_dc = compute_waveform_envelope(dc, width=8)
    assert len(env_dc) == 8

    # Nyquist oscillation (+1, -1, +1, -1)
    nyquist = np.tile([1.0, -1.0], 500).astype(np.float32)
    env_nyq = compute_waveform_envelope(nyquist, width=8)
    assert len(env_nyq) == 8
    assert env_nyq == "█" * 8


def test_envelope_cache_address_reuse_collision():
    """Adversarial stress: verify memory allocator address reuse does not cause cache collision.

    If array A is deallocated, array B allocated at the same address must NOT inherit array A's envelope.
    """
    import zout.audio.io as aio
    if hasattr(aio, "_ARRAY_ENVELOPE_CACHE"):
        aio._ARRAY_ENVELOPE_CACHE.clear()

    # Step 1: Allocate a loud signal and compute envelope
    loud = np.ones(1000, dtype=np.float32)
    loud_addr = id(loud)
    env_loud = compute_waveform_envelope(loud, width=8)
    assert env_loud == "█" * 8

    # Step 2: Delete loud array so CPython's pymalloc reassigns the address
    del loud

    # Step 3: Allocate a silent signal of the same shape
    silent = np.zeros(1000, dtype=np.float32)
    silent_addr = id(silent)
    env_silent = compute_waveform_envelope(silent, width=8)

    # If pymalloc reused the address, verify it does NOT produce '████████'
    if loud_addr == silent_addr:
        assert env_silent == " " * 8, (
            f"Cache address reuse collision detected! Address {loud_addr} was reused, "
            f"causing silent array to receive loud envelope '{env_silent}' instead of '        '"
        )
    else:
        assert env_silent == " " * 8



if __name__ == "__main__":
    pytest.main(["-v", __file__])
