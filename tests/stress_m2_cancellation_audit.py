"""Empirical Adversarial Stress Test Suite for Milestone 2.

Focus areas:
1. In-Flight Cancellation Latency (< 50ms pre-flight, < 200ms mid-flight).
2. Windows File Lock Discipline (zero WinError 32 leaks, immediate unlinking and re-creations).
3. Zero temporary artifacts (*.tmp) remaining after cancellation.
4. Forensic Audit Log Sidecar True-Peak (ITU-R BS.1770 4x oversampled ISP detection).
5. Digital silence and non-finite signal clamping (-120.0 dBFS, zero NaN/Inf in JSON).
6. VAD selectivity on pure Gaussian noise, acoustic hum, and speech-pause-speech sequences.
7. Full RFC 8259 schema validation with dual aliases and speech summary statistics.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from zout.audio.audit import (
    compute_rms,
    compute_speech_summary,
    compute_true_peak,
    extract_vad_intervals,
    generate_audit_manifest,
)
from zout.config import JobConfig, ReferenceRole
from zout.pipeline import run_job, run_streaming

TEST_SR = 16000


def synth_tone(freq: float, duration_sec: float, sr: int = TEST_SR, amp: float = 0.5) -> np.ndarray:
    """Generate pure sine tone."""
    t = np.arange(int(duration_sec * sr)) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def synth_hum(duration_sec: float, sr: int = TEST_SR, seed: int = 1) -> np.ndarray:
    """Generate 60Hz + 180Hz powerline hum with Gaussian background noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sr)) / sr
    return (
        0.08 * np.sin(2.0 * np.pi * 60.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.02 * rng.standard_normal(len(t))
    ).astype(np.float32)


def synth_speechlike(duration_sec: float, sr: int = TEST_SR, seed: int = 42, f0: float = 140.0) -> np.ndarray:
    """Generate multi-harmonic voiced speech-like signal with syllable rate amplitude modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sr)) / sr
    env = (np.sin(2.0 * np.pi * 3.0 * t) > -0.2).astype(np.float32)
    env = np.convolve(env, np.ones(200) / 200.0, mode="same")
    sig = sum(np.sin(2.0 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 15))
    return (0.25 * sig * env).astype(np.float32)


# ============================================================================
# 1. In-Flight Cancellation & Windows File Lock Discipline
# ============================================================================


def test_preflight_cancellation_latency_under_50ms(tmp_path: Path):
    """Adversarial Challenge: Immediate pre-flight cancellation must return None in < 50ms.

    Tests 50 consecutive runs measuring latency distribution across run_job and run_streaming.
    """
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "main.wav"
    out_file = tmp_path / "out.wav"

    sf.write(ref_file, synth_hum(2.0), TEST_SR)
    sf.write(main_file, synth_tone(440.0, 4.0), TEST_SR)

    latencies_ms: list[float] = []

    # 1. Test via cfg.cancel_event
    for _ in range(50):
        ev = threading.Event()
        ev.set()
        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, cancel_event=ev)
        t0 = time.perf_counter()
        res = run_job(cfg)
        lat = (time.perf_counter() - t0) * 1000.0
        assert res is None, "Pre-flight cancelled job must return None"
        latencies_ms.append(lat)

    p99 = np.percentile(latencies_ms, 99)
    max_lat = max(latencies_ms)
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    assert max_lat < 50.0, f"Max pre-flight cancellation latency {max_lat:.3f}ms exceeded 50ms limit"
    assert p99 < 10.0, f"P99 latency {p99:.3f}ms exceeded 10ms"

    # 2. Test via cancel_token parameter in signature
    token = threading.Event()
    token.set()
    cfg_tok = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file, polish=False)
    t0 = time.perf_counter()
    res_tok = run_job(cfg_tok, cancel_token=token)
    tok_lat = (time.perf_counter() - t0) * 1000.0
    assert res_tok is None
    assert tok_lat < 50.0

    # 3. Test run_streaming directly
    t0 = time.perf_counter()
    res_stream = run_streaming(cfg, cancel_token=token)
    stream_lat = (time.perf_counter() - t0) * 1000.0
    assert res_stream is None
    assert stream_lat < 50.0

    # Ensure output file never created or remains unlinked
    assert not out_file.exists()


def test_midflight_streaming_cancellation_and_windows_file_locks(tmp_path: Path):
    """Adversarial Challenge: Mid-flight cancellation during chunked streaming must cleanly abort,
    immediately release Windows file locks without PermissionError (WinError 32), and unlink out_path.
    """
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "main.wav"
    out_file = tmp_path / "stream_out.wav"

    sf.write(ref_file, synth_hum(2.0), TEST_SR)
    # 8-second file streamed in 1.0s chunks with 0.2s overlap
    sf.write(main_file, synth_speechlike(8.0) + synth_hum(8.0), TEST_SR)

    # Pre-populate directory with a temporary file to stress-test _cleanup_partial
    dummy_tmp = tmp_path / "stale_buffer.tmp"
    dummy_tmp.write_bytes(b"stale_partial_buffer_data")

    cancel_ev = threading.Event()
    progress_calls: list[tuple[str, float, float]] = []

    def on_progress(msg: str, frac: float):
        now = time.perf_counter()
        progress_calls.append((msg, frac, now))
        # Trigger cancellation on 2nd chunk
        if "chunk 2" in msg and not cancel_ev.is_set():
            cancel_ev.set()

    cfg = JobConfig(
        ref_file,
        ReferenceRole.NOISY,
        main_file,
        0.75,
        24.0,
        out_file,
        chunk_duration=1.0,
        overlap_duration=0.2,
        polish=False,
        cancel_event=cancel_ev,
    )

    t0 = time.perf_counter()
    res = run_job(cfg, progress=on_progress)
    t_end = time.perf_counter()

    assert res is None, "Mid-flight cancelled streaming job must return None"

    # Crucial Windows check 1: out_file must be completely unlinked
    assert not out_file.exists(), f"Output file {out_file} was not unlinked on cancellation"

    # Crucial Windows check 2: immediate file lock verification
    # If the SoundFile stream was not closed in finally:, this open will raise:
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process
    try:
        with open(out_file, "wb") as f:
            f.write(b"windows_lock_discipline_verified")
        out_file.unlink()
    except PermissionError as e:
        pytest.fail(f"Windows file lock leak detected! WinError 32 on out_file: {e}")

    # Crucial Windows check 3: verify zero temporary files remain in output directory
    remaining_tmps = list(tmp_path.glob("*.tmp"))
    assert len(remaining_tmps) == 0, f"Temporary files leaked in output folder: {remaining_tmps}"


def test_midflight_monolithic_nmf_cancellation_latency_under_200ms(tmp_path: Path):
    """Adversarial Challenge: Mid-flight cancellation during monolithic NMF loop must cleanly abort
    in < 200ms once cancel_event is set.
    """
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "monolithic_main.wav"
    out_file = tmp_path / "monolithic_out.wav"

    sf.write(ref_file, synth_hum(2.0), TEST_SR)
    sf.write(main_file, synth_speechlike(4.0) + synth_hum(4.0), TEST_SR)

    cancel_ev = threading.Event()
    cancel_time = [0.0]

    def on_progress(msg: str, frac: float):
        # Cancel right as NMF separation starts
        if "Separating noise bases" in msg and not cancel_ev.is_set():
            cancel_time[0] = time.perf_counter()
            cancel_ev.set()

    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                    polish=False, cancel_event=cancel_ev)

    res = run_job(cfg, progress=on_progress)
    t_finish = time.perf_counter()

    assert res is None, "Monolithic job must cleanly return None on cancellation"
    assert cancel_time[0] > 0.0, "Progress hook was never reached"

    abort_latency_ms = (t_finish - cancel_time[0]) * 1000.0
    assert abort_latency_ms < 200.0, (
        f"Monolithic NMF cancellation latency {abort_latency_ms:.2f}ms exceeded 200ms cap"
    )

    # Windows file lock release check
    assert not out_file.exists()
    try:
        out_file.write_bytes(b"monolithic_lock_check")
        out_file.unlink()
    except PermissionError as e:
        pytest.fail(f"Windows file lock leak on monolithic out_file: {e}")


def test_rapid_concurrent_cancellation_churn_no_lock_leaks(tmp_path: Path):
    """Adversarial Challenge: 10 rapid successive jobs with asynchronous thread cancellation
    must complete with zero WinError 32 leaks and zero handle corruption.
    """
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "churn_main.wav"
    sf.write(ref_file, synth_hum(1.0), TEST_SR)
    sf.write(main_file, synth_tone(300.0, 3.0), TEST_SR)

    for i in range(10):
        out_file = tmp_path / f"churn_{i}.wav"
        cancel_ev = threading.Event()

        # Asynchronously cancel after small delay
        def async_canceller():
            time.sleep(0.02 * (i % 3 + 1))
            cancel_ev.set()

        thr = threading.Thread(target=async_canceller)
        thr.start()

        cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        chunk_duration=1.0, overlap_duration=0.2, polish=False,
                        cancel_event=cancel_ev)
        res = run_job(cfg)
        thr.join()

        # If cancelled, output file must be deleted and unlocked
        if res is None:
            assert not out_file.exists()
            with open(out_file, "wb") as f:
                f.write(b"churn_lock_test")
            out_file.unlink()


# ============================================================================
# 2. Forensic Audit Log Sidecar True-Peak & VAD Edge Cases
# ============================================================================


def test_itu_bs1770_true_peak_intersample_peak_fixture():
    """Adversarial Challenge: ITU-R BS.1770 4x oversampled true-peak on an inter-sample peak fixture.

    A continuous sine wave at fs/4 with discrete sample phase offset pi/4 has:
      x[n] = sin(pi/2 * n + pi/4)
    Discrete sample values:
      x[0] = sin(pi/4) = 0.7071 (-3.01 dBFS)
      x[1] = sin(3pi/4) = 0.7071 (-3.01 dBFS)
      x[2] = sin(5pi/4) = -0.7071 (-3.01 dBFS)
      x[3] = sin(7pi/4) = -0.7071 (-3.01 dBFS)
    Sample peak is strictly 0.7071 (-3.01 dBFS).
    The 4x oversampled true-peak algorithm reconstructs intermediate points at 0, pi/2, pi, 3pi/2...
    detecting the true continuous peak of ~1.000 (~0.0 dBFS).
    """
    sr = 48000
    n = np.arange(sr * 2)
    x = np.sin(0.5 * np.pi * n + np.pi / 4.0).astype(np.float32)

    # Discrete peak
    discrete_peak = float(np.max(np.abs(x)))
    discrete_peak_db = 20.0 * math.log10(discrete_peak)
    assert abs(discrete_peak - (math.sqrt(2.0) / 2.0)) < 1e-4
    assert abs(discrete_peak_db - (-3.0103)) < 0.05

    # 4x oversampled true-peak via compute_true_peak
    tp_db = compute_true_peak(x, sr=sr, oversample=4)

    # True-peak must reconstruct the analog peak within 0.15 dB of 0.0 dBFS
    assert -0.20 <= tp_db <= 0.20, (
        f"True-peak failed to detect inter-sample peak! Expected ~0.0 dBFS, got {tp_db:.2f} dBFS"
    )
    # The true-peak reading must be ~3 dB higher than the discrete sample peak
    peak_diff = tp_db - discrete_peak_db
    assert peak_diff >= 2.8, (
        f"True-peak delta {peak_diff:.2f} dB is less than expected ~3 dB ISP gain"
    )


def test_digital_silence_and_nonfinite_signal_audit_safety(tmp_path: Path):
    """Adversarial Challenge: True-peak, RMS, and JSON serialization on pure digital silence
    and non-finite signals (NaN, Inf) must return finite values (-120.0 dBFS) and zero
    NaN or Infinity strings in the serialized JSON.
    """
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "silence_main.wav"
    out_file = tmp_path / "silence_out.wav"

    sf.write(ref_file, synth_hum(1.0), TEST_SR)
    # Pure digital silence
    sf.write(main_file, np.zeros(TEST_SR * 2, dtype=np.float32), TEST_SR)

    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                    polish=False, audit=True)
    res = run_job(cfg)
    assert res is not None

    audit_path = cfg.default_audit_path()
    assert audit_path.is_file(), f"Audit sidecar not found at {audit_path}"

    raw_json = audit_path.read_text(encoding="utf-8")

    # 1. Zero NaN / Infinity strings in the serialized RFC 8259 JSON
    assert "NaN" not in raw_json, "Unsanitized 'NaN' token found in JSON sidecar"
    assert "Infinity" not in raw_json, "Unsanitized 'Infinity' token found in JSON sidecar"
    assert "-Infinity" not in raw_json, "Unsanitized '-Infinity' token found in JSON sidecar"

    data = json.loads(raw_json)

    # 2. Clamped values must be finite and <= -120.0 dBFS
    assert data["true_peak"]["output"] == -120.0
    assert data["true_peak_dbfs"]["output"] == -120.0
    assert data["rms"]["output"] <= -120.0 and math.isfinite(data["rms"]["output"])
    assert data["rms_dbfs"]["output"] <= -120.0 and math.isfinite(data["rms_dbfs"]["output"])

    # 3. Speech summary on pure silence
    assert data["speech_summary"]["speech_duration_sec"] == 0.0
    assert data["speech_summary"]["speech_presence_ratio"] == 0.0
    assert data["vad_speech_timestamps"] == []

    # 4. Direct audit module test with NaN / Inf inputs
    nan_audio = np.array([np.nan, np.inf, -np.inf, 0.0, np.nan], dtype=np.float32)
    tp_nan = compute_true_peak(nan_audio, sr=TEST_SR)
    rms_nan = compute_rms(nan_audio, sr=TEST_SR)
    assert tp_nan == -120.0
    assert rms_nan == -120.0


def test_vad_selectivity_noise_and_hum_rejection():
    """Adversarial Challenge: VAD must strictly reject pure Gaussian white noise and acoustic hum,
    yielding vad_speech_timestamps == [].
    """
    # 1. Pure Gaussian white noise (3 seconds at RMS ~ -14 dBFS)
    rng = np.random.default_rng(1337)
    white_noise = rng.normal(0.0, 0.20, TEST_SR * 3).astype(np.float32)
    vads_noise = extract_vad_intervals(white_noise, sr=TEST_SR)
    assert vads_noise == [], f"VAD falsely triggered on white noise: {vads_noise}"

    # 2. Pure 60Hz and 180Hz powerline hum
    t = np.arange(TEST_SR * 3) / TEST_SR
    hum_signal = (0.35 * np.sin(2.0 * np.pi * 60.0 * t) + 0.25 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)
    vads_hum = extract_vad_intervals(hum_signal, sr=TEST_SR)
    assert vads_hum == [], f"VAD falsely triggered on powerline hum: {vads_hum}"

    # 3. Pure 1000 Hz sine calibration tone
    tone = synth_tone(1000.0, 3.0, sr=TEST_SR, amp=0.8)
    vads_tone = extract_vad_intervals(tone, sr=TEST_SR)
    assert vads_tone == [], f"VAD falsely triggered on pure tone: {vads_tone}"


def test_vad_speech_pause_speech_intervals():
    """Adversarial Challenge: VAD on speech + pause + speech must identify continuous intervals
    with correct timestamps, positive durations, and gap bridging.
    """
    # Test on repo fixture testdata/speech_with_pauses.wav if available
    fixture_path = Path("testdata/speech_with_pauses.wav")
    if fixture_path.is_file():
        vads = extract_vad_intervals(fixture_path)
        assert len(vads) >= 2, f"Expected multiple speech segments in {fixture_path}, found {len(vads)}"

        prev_end = -1.0
        for interval in vads:
            st = interval["start"]
            en = interval["end"]
            dur = interval["duration"]

            assert st >= 0.0
            assert en > st
            assert dur > 0.0
            assert abs(round(en - st, 3) - dur) <= 0.002
            assert st >= prev_end, f"Intervals overlapped or not monotonically ordered: {st} < {prev_end}"
            prev_end = en

    # Synthetic speech + pause + speech test:
    # 0.5s - 2.5s speech, 2.5s - 4.5s pause (2.0s silence), 4.5s - 6.5s speech
    total_sec = 7.0
    synth_audio = np.zeros(int(total_sec * TEST_SR), dtype=np.float32)
    synth_audio[int(0.5 * TEST_SR) : int(2.5 * TEST_SR)] = synth_speechlike(2.0, seed=42)
    synth_audio[int(4.5 * TEST_SR) : int(6.5 * TEST_SR)] = synth_speechlike(2.0, seed=100)

    synth_vads = extract_vad_intervals(synth_audio, sr=TEST_SR)
    assert len(synth_vads) >= 2, f"Failed to detect two distinct speech bursts: {synth_vads}"

    # First interval must be within first speech burst [0.5, 2.5]
    assert 0.4 <= synth_vads[0]["start"] <= 1.0
    assert synth_vads[0]["end"] <= 2.6

    # Last interval must be within second speech burst [4.5, 6.5]
    last_int = synth_vads[-1]
    assert 4.4 <= last_int["start"] <= 6.3
    assert last_int["end"] <= 6.6

    # Verify no speech was detected during pause [2.6, 4.4]
    for interval in synth_vads:
        assert not (interval["start"] > 2.6 and interval["end"] < 4.4), (
            f"VAD falsely triggered during silence pause: {interval}"
        )


def test_full_rfc8259_audit_schema_and_dual_aliases(tmp_path: Path):
    """Adversarial Challenge: Comprehensive RFC 8259 JSON validation ensuring dual aliases
    (noise_reduction_db/suppression_db, true_peak/true_peak_dbfs, rms/rms_dbfs),
    speech_summary, and complete parameters dictionary.
    """
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "full_main.wav"
    out_file = tmp_path / "full_out.wav"

    sf.write(ref_file, synth_hum(2.0), TEST_SR)
    sf.write(main_file, synth_speechlike(4.0) + synth_hum(4.0), TEST_SR)

    cfg = JobConfig(
        ref_file,
        ReferenceRole.NOISY,
        main_file,
        0.80,
        30.0,
        out_file,
        polish=False,
        audit=True,
        chunk_duration=60.0,
        overlap_duration=2.0,
    )

    res = run_job(cfg)
    assert res is not None
    assert res.audit_path is not None
    assert res.audit_path.is_file()

    with open(res.audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Dual Alias: noise reduction
    assert "noise_reduction_db" in data
    assert "suppression_db" in data
    assert data["noise_reduction_db"] == data["suppression_db"]
    assert isinstance(data["noise_reduction_db"], (int, float))

    # 2. Dual Alias: true_peak
    assert "true_peak" in data
    assert "true_peak_dbfs" in data
    assert data["true_peak"] == data["true_peak_dbfs"]
    assert "input" in data["true_peak"] and "output" in data["true_peak"]
    assert math.isfinite(data["true_peak"]["input"])
    assert math.isfinite(data["true_peak"]["output"])

    # 3. Dual Alias: rms
    assert "rms" in data
    assert "rms_dbfs" in data
    assert data["rms"] == data["rms_dbfs"]
    assert "input" in data["rms"] and "output" in data["rms"]
    assert math.isfinite(data["rms"]["input"])
    assert math.isfinite(data["rms"]["output"])

    # 4. Speech summary statistics
    assert "speech_summary" in data
    summ = data["speech_summary"]
    for key in ["speech_duration_sec", "non_speech_duration_sec", "total_duration_sec", "speech_presence_ratio"]:
        assert key in summ
        assert isinstance(summ[key], (int, float))
        assert summ[key] >= 0.0

    assert 0.0 <= summ["speech_presence_ratio"] <= 1.0
    # Sum of speech + non-speech must equal total duration (within rounding)
    calc_sum = round(summ["speech_duration_sec"] + summ["non_speech_duration_sec"], 2)
    assert abs(calc_sum - round(summ["total_duration_sec"], 2)) <= 0.05

    # 5. Parameters dictionary completeness
    assert "parameters" in data
    params = data["parameters"]
    required_params = [
        "role", "strength", "max_atten_db", "polish", "start_sec", "end_sec",
        "chunk_duration", "overlap_duration", "reference", "main", "output"
    ]
    for param in required_params:
        assert param in params, f"Missing parameter {param} in audit manifest"

    assert params["role"] in ["NOISY", "noisy"]
    assert params["strength"] == 0.80
    assert params["max_atten_db"] == 30.0
    assert params["polish"] is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
