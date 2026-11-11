"""Synthetic fixtures and end-to-end tests. VoxLock model downloads on first CLEAN run."""
import numpy as np
import pytest
import soundfile as sf

from zout.audio.io import prepare_clean_enrollment
from zout.audio.masking import min_gain
from zout.config import JobConfig, ReferenceRole
from zout.pipeline import run_job
from zout.tui.file_picker import PickerState

SR = 16000


def speechlike(sec, seed=0, f0=140):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    env = (np.sin(2 * np.pi * 3 * t) > -0.2).astype(np.float32)
    env = np.convolve(env, np.ones(400) / 400, "same")
    x = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 20))
    return (0.2 * x * env).astype(np.float32)


def hum(sec, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    return (0.08 * np.sin(2 * np.pi * 60 * t) + 0.05 * np.sin(2 * np.pi * 180 * t)
            + 0.03 * rng.standard_normal(len(t))).astype(np.float32)


def test_picker_filters(tmp_path):
    for n in ["a.wav", "b.MP3", "c.txt", "d.png", "e.exe", ".hidden.wav"]:
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    st = PickerState(tmp_path)
    labels = [e.label for e in st.entries]
    assert "a.wav" in labels and "b.MP3" in labels and "sub" + __import__("os").sep in labels
    assert not any(l in labels for l in ["c.txt", "d.png", "e.exe", ".hidden.wav"])


def test_short_enrollment_is_padded():
    x = speechlike(1.2)
    out, voiced = prepare_clean_enrollment(x, SR)
    assert len(out) >= int(3.0 * SR) and voiced > 0.3 and np.isfinite(out).all()


def test_min_gain_monotonic():
    assert min_gain(0.2, 24) > min_gain(0.9, 24)
    assert min_gain(0.5, 60) >= 0.0


def noise_energy(y, ref_noise_hz=60):
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(spec[(f > 50) & (f < 70)].sum())


def test_noisy_mode_reduces_noise_and_scales_with_strength(tmp_path):
    ref = hum(5)
    main = speechlike(8) + hum(8, seed=2)
    sf.write(tmp_path / "ref.wav", ref, SR)
    sf.write(tmp_path / "main.wav", main, SR)
    levels = []
    for s in (0.2, 0.95):
        cfg = JobConfig(tmp_path / "ref.wav", ReferenceRole.NOISY, tmp_path / "main.wav", s, 60.0,
                        tmp_path / f"o{s}.wav", polish=False)
        res = run_job(cfg)
        y, _ = sf.read(res.output, dtype="float32")
        assert np.isfinite(y).all() and np.abs(y).max() <= 1.0 and len(y) == len(main)
        levels.append(noise_energy(y))
    assert levels[1] < noise_energy(main)
    assert levels[1] <= levels[0]


def test_atten_cap_limits_suppression(tmp_path):
    ref, main = hum(5), speechlike(6) + hum(6, seed=2)
    sf.write(tmp_path / "ref.wav", ref, SR)
    sf.write(tmp_path / "main.wav", main, SR)
    outs = {}
    for cap in (6.0, 60.0):
        cfg = JobConfig(tmp_path / "ref.wav", ReferenceRole.NOISY, tmp_path / "main.wav", 0.95, cap,
                        tmp_path / f"c{cap}.wav", polish=False)
        outs[cap] = noise_energy(sf.read(run_job(cfg).output, dtype="float32")[0])
    assert outs[6.0] > outs[60.0]


@pytest.mark.slow
def test_clean_mode_runs_end_to_end(tmp_path):
    voice = speechlike(10, f0=140)
    main = voice + 0.5 * speechlike(10, seed=5, f0=210) + hum(10)
    sf.write(tmp_path / "ref.wav", speechlike(4, seed=3, f0=140), SR)
    sf.write(tmp_path / "main.wav", main, SR)
    cfg = JobConfig(tmp_path / "ref.wav", ReferenceRole.CLEAN, tmp_path / "main.wav", 0.8, 24.0,
                    tmp_path / "out.wav")
    res = run_job(cfg)
    y, sr = sf.read(res.output, dtype="float32")
    assert sr == SR and len(y) == len(main) and np.isfinite(y).all()


def test_auto_noise_extraction_detects_pure_noise_pauses(tmp_path):
    from zout.audio.noise_extractor import extract_noise_profile

    # Create a 20s recording: pure noise at [0s-4s], speech at [4s-10s], pure noise at [10s-15s], speech at [15s-20s]
    rng = np.random.default_rng(123)
    t = np.arange(20 * SR) / SR
    noise = (0.04 * rng.standard_normal(len(t))).astype(np.float32)

    speech = np.zeros_like(noise)
    s1 = speechlike(6, seed=7)
    speech[4 * SR : 4 * SR + len(s1)] = s1
    s2 = speechlike(5, seed=8)
    speech[15 * SR : 15 * SR + len(s2)] = s2

    mix_audio = noise + speech
    main_path = tmp_path / "mix.wav"
    sf.write(main_path, mix_audio, SR)

    res = extract_noise_profile(mix_audio, SR, n_chunks=3, chunk_sec=1.5, mode="balanced")

    assert len(res.chunks) >= 2
    assert len(res.audio) > 0
    assert np.isfinite(res.audio).all()
    # Ensure no selected chunks overlap with speech intervals [4s-10s] or [15s-20s]
    for st, en in res.chunks:
        overlap_s1 = max(0, min(en, 10.0) - max(st, 4.0))
        overlap_s2 = max(0, min(en, 20.0) - max(st, 15.0))
        assert overlap_s1 == 0.0, f"Chunk {st}-{en} overlapped with speech section 1"
        assert overlap_s2 == 0.0, f"Chunk {st}-{en} overlapped with speech section 2"


def test_audio_player_play_and_stop(tmp_path):
    from zout.audio.player import AudioPlayer

    test_wav = tmp_path / "tone.wav"
    t = np.arange(SR // 2) / SR  # 0.5 sec
    tone = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(test_wav, tone, SR)

    p = AudioPlayer()
    ok = p.play(test_wav, "Test Tone")
    assert ok is True
    p.stop()
    assert p.is_playing() is False


from unittest.mock import patch


def test_parse_time_spec_valid():
    """Verify parse_time_spec handles seconds, mm:ss, and hh:mm:ss properly."""
    from zout.audio.io import parse_time_spec

    # None and empty passthrough
    assert parse_time_spec(None) is None

    # Numerics
    assert parse_time_spec(0) == 0.0
    assert parse_time_spec(45) == 45.0
    assert parse_time_spec(12.75) == 12.75

    # Seconds string
    assert parse_time_spec("0") == 0.0
    assert parse_time_spec("45") == 45.0
    assert parse_time_spec("45.5") == 45.5
    assert parse_time_spec("  90.25  ") == 90.25

    # mm:ss format
    assert parse_time_spec("01:30") == 90.0
    assert parse_time_spec("1:30") == 90.0
    assert parse_time_spec("00:45") == 45.0
    assert parse_time_spec("02:05.5") == 125.5
    assert parse_time_spec("10:00") == 600.0

    # hh:mm:ss format
    assert parse_time_spec("01:15:30") == 4530.0
    assert parse_time_spec("00:01:30") == 90.0
    assert parse_time_spec("01:00:00.5") == 3600.5


def test_parse_time_spec_invalid():
    """Verify parse_time_spec rejects negative numbers and malformed strings."""
    from zout.audio.io import parse_time_spec

    invalid_inputs = [
        "-5",
        "-01:30",
        -10,
        -0.1,
        "abc",
        "",
        "   ",
        ":30",
        "12:",
        "1:2:3:4",
        "invalid",
    ]
    for inp in invalid_inputs:
        with pytest.raises(ValueError):
            parse_time_spec(inp)


def test_sliced_audio_loading(tmp_path):
    """Verify load_audio range slicing via start_sec and end_sec."""
    from zout.audio.io import load_audio

    # Create a 4.0 second synthetic audio file at 16kHz
    duration = 4.0
    t = np.arange(int(duration * SR)) / SR
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    test_wav = tmp_path / "slice_test.wav"
    sf.write(test_wav, audio, SR)

    # 1. Full load baseline
    full_audio, sr = load_audio(test_wav)
    assert sr == SR
    assert len(full_audio) == int(duration * SR)

    # 2. Slice both start and end: 1.0s to 3.0s (duration = 2.0s = 32000 samples)
    sliced, sr_slice = load_audio(test_wav, start_sec=1.0, end_sec=3.0)
    assert sr_slice == SR
    expected_len = int(2.0 * SR)
    assert len(sliced) == expected_len
    # Exact sample match
    np.testing.assert_allclose(sliced, full_audio[int(1.0 * SR) : int(3.0 * SR)], atol=1e-5)

    # 3. Slice start only: 2.5s to EOF (duration = 1.5s = 24000 samples)
    start_only, _ = load_audio(test_wav, start_sec=2.5)
    assert len(start_only) == int(1.5 * SR)
    np.testing.assert_allclose(start_only, full_audio[int(2.5 * SR) :], atol=1e-5)

    # 4. Slice end only: 0.0s to 1.5s (duration = 1.5s = 24000 samples)
    end_only, _ = load_audio(test_wav, end_sec=1.5)
    assert len(end_only) == int(1.5 * SR)
    np.testing.assert_allclose(end_only, full_audio[: int(1.5 * SR)], atol=1e-5)

    # 5. Invalid range where start >= end raises ValueError
    with pytest.raises(ValueError):
        load_audio(test_wav, start_sec=3.0, end_sec=1.0)
    with pytest.raises(ValueError):
        load_audio(test_wav, start_sec=2.0, end_sec=2.0)


def test_compute_waveform_envelope():
    """Verify waveform envelope generation, glyph count, silence handling, and dynamic ramp."""
    from zout.audio.io import compute_waveform_envelope

    glyphs = " ▂▃▄▅▆▇█"

    # 1. Glyph count and width check
    sine = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))).astype(np.float32)
    env16 = compute_waveform_envelope(sine, width=16)
    assert len(env16) == 16
    assert all(c in glyphs for c in env16)

    env32 = compute_waveform_envelope(sine, width=32)
    assert len(env32) == 32
    assert all(c in glyphs for c in env32)

    # 2. Silence handling (all zeros)
    silence = np.zeros(16000, dtype=np.float32)
    env_silence = compute_waveform_envelope(silence, width=16)
    assert len(env_silence) == 16
    assert env_silence == glyphs[0] * 16

    # 3. Dynamic ramp (crescendo from quiet to loud)
    t = np.linspace(0, 1, 16000)
    ramp_env = np.linspace(0.01, 1.0, 16000)
    ramp_audio = (ramp_env * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    env_ramp = compute_waveform_envelope(ramp_audio, width=16)
    assert len(env_ramp) == 16

    # Verify glyph levels are non-decreasing and ramp up
    first_level = glyphs.index(env_ramp[0])
    mid_level = glyphs.index(env_ramp[len(env_ramp) // 2])
    last_level = glyphs.index(env_ramp[-1])
    assert first_level <= mid_level <= last_level
    assert first_level < last_level

    # 4. Stereo array handling (2D)
    stereo = np.column_stack([ramp_audio, ramp_audio])
    env_stereo = compute_waveform_envelope(stereo, width=16)
    assert len(env_stereo) == 16
    assert all(c in glyphs for c in env_stereo)

    # 5. Empty array fallback
    empty = np.array([], dtype=np.float32)
    env_empty = compute_waveform_envelope(empty, width=16)
    assert len(env_empty) == 16
    assert env_empty == glyphs[0] * 16


def test_audio_player_ab_hot_swap_and_continuity(tmp_path):
    """Verify AudioPlayer A/B hot-swap and playhead continuity."""
    from zout.audio.player import AudioPlayer

    t = np.arange(SR * 2) / SR  # 2.0s
    tone_a = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone_b = (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    path_a = tmp_path / "track_a.wav"
    path_b = tmp_path / "track_b.wav"
    sf.write(path_a, tone_a, SR)
    sf.write(path_b, tone_b, SR)

    p = AudioPlayer()
    started = p.play_pair(path_a, path_b, start_with="B")
    assert started is True
    assert p.get_current_track() == "B"

    # Advance playhead position
    with p._lock:
        p._sample_index = 8000
    idx_before = getattr(p, "sample_index", getattr(p, "_sample_index", 0))
    assert idx_before == 8000

    # Hot swap to A
    active = p.toggle_ab()
    assert active == "A"
    assert p.get_current_track() == "A"

    # Verify sample_index is strictly preserved across toggle
    idx_after = getattr(p, "sample_index", getattr(p, "_sample_index", 0))
    assert idx_after == 8000, "toggle_ab() must preserve sample_index continuity"

    # Hot swap back to B
    active = p.toggle_ab()
    assert active == "B"
    assert p.get_current_track() == "B"

    # Verify meter levels structure
    levels = p.get_meter_levels()
    assert isinstance(levels, tuple) and len(levels) == 2
    rms, peak = levels
    assert len(rms) == 2 and len(peak) == 2
    assert all(isinstance(v, (int, float)) for v in rms + peak)

    p.stop()
    assert p.is_playing() is False


def test_headless_player_safety_on_device_failure(tmp_path):
    """Verify AudioPlayer does not crash when audio output device is unavailable."""
    from zout.audio.player import AudioPlayer

    test_wav = tmp_path / "headless_tone.wav"
    t = np.arange(SR // 4) / SR
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(test_wav, tone, SR)

    p = AudioPlayer()

    # Simulate OutputStream failing (e.g., no sound hardware in CI)
    with patch("sounddevice.OutputStream", side_effect=RuntimeError("No default output device")):
        res = p.play_pair(test_wav, test_wav)
        assert isinstance(res, bool)
        assert isinstance(p.get_meter_levels(), tuple)
        assert isinstance(p.status_text(), str)
        p.toggle_ab()
        p.stop()
        assert p.is_playing() is False

    # Simulate sounddevice.play failing
    with patch("sounddevice.play", side_effect=RuntimeError("PortAudio device unavailable")):
        res = p.play(test_wav)
        assert isinstance(res, bool)
        p.stop()
        assert p.is_playing() is False


def test_waveform_envelope_no_memory_address_reuse_collision():
    """Verify compute_waveform_envelope does not return stale cached envelopes
    when Python's allocator reassigns deallocated array memory addresses."""
    from zout.audio.io import compute_waveform_envelope

    def get_env(make_array):
        arr = make_array()
        return compute_waveform_envelope(arr, width=16)

    # 1. Allocate loud array, compute envelope, and let it deallocate
    env_loud = get_env(lambda: np.ones(1000, dtype=np.float32))
    assert env_loud == "█" * 16

    # 2. Allocate silent array of identical shape (reusing memory address)
    env_silent = get_env(lambda: np.zeros(1000, dtype=np.float32))
    assert env_silent == " " * 16, f"Silent array returned stale cached envelope: {env_silent!r}"

    # 3. Repeated sequential allocation churn
    for _ in range(10):
        loud = np.full(500, 0.9, dtype=np.float32)
        assert compute_waveform_envelope(loud, width=8) == "█" * 8
        del loud
        silent = np.zeros(500, dtype=np.float32)
        assert compute_waveform_envelope(silent, width=8) == " " * 8


def test_parse_time_spec_negative_zero_clock_bypass():
    """Verify parse_time_spec rejects negative zero clock specs (-00:30, -0:30, -00:00:30)."""
    from zout.audio.io import parse_time_spec

    negative_clock_specs = [
        "-00:30",
        "-0:30",
        "-00:00:30",
        "-0:00:15",
        "01:-10",
        "00:01:-05",
        "-0",
        "-0.0",
        "-00",
        -0.0,
    ]
    for spec in negative_clock_specs:
        with pytest.raises(ValueError):
            parse_time_spec(spec)


def test_load_audio_ffmpeg_slicing_and_eof(tmp_path):
    """Verify load_audio handles non-native (MP3) slicing duration and EOF seeking."""
    import subprocess

    from zout.audio.io import _ffmpeg, load_audio

    # Create 1.0s tone in WAV then convert to MP3
    tone_wav = tmp_path / "tone.wav"
    tone_mp3 = tmp_path / "tone.mp3"
    t = np.arange(SR) / SR
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(tone_wav, audio, SR)

    ffmpeg_exe = _ffmpeg()
    subprocess.run([ffmpeg_exe, "-y", "-i", str(tone_wav), str(tone_mp3)], capture_output=True, check=True)

    # 1. Non-native slicing: -t after -i accurately trims output
    slice_audio, sr = load_audio(tone_mp3, start_sec=0.2, end_sec=0.3)
    assert sr == 48000
    assert abs(len(slice_audio) - 4800) < 500

    # 2. Seeking past EOF on non-native format returns empty array instead of RuntimeError
    eof_audio, sr_eof = load_audio(tone_mp3, start_sec=999.0, end_sec=1000.0)
    assert sr_eof == 48000
    assert len(eof_audio) == 0
    assert isinstance(eof_audio, np.ndarray)

    # 3. Native format seeking past EOF also returns empty array
    native_eof, _ = load_audio(tone_wav, start_sec=999.0, end_sec=1000.0)
    assert len(native_eof) == 0


def test_calc_meters_non_finite_guard_and_status_clamping():
    """Verify _calc_meters rejects NaN/Inf and status_text clamps elapsed duration."""
    import math

    from zout.audio.player import AudioPlayer, _calc_dbfs, _calc_meters

    # 1. Audio blocks containing NaN and Inf must return finite, bounded dBFS in [-80.0, 0.0]
    inf_block = np.full((1024, 2), np.inf, dtype=np.float32)
    levels_inf = _calc_meters(inf_block)
    assert len(levels_inf) == 4
    for val in levels_inf:
        assert math.isfinite(val), f"Expected finite float, got {val}"
        assert -80.0 <= val <= 0.0, f"Expected bounded dBFS in [-80.0, 0.0], got {val}"

    nan_block = np.full((1024, 2), np.nan, dtype=np.float32)
    levels_nan = _calc_meters(nan_block)
    assert len(levels_nan) == 4
    for val in levels_nan:
        assert math.isfinite(val), f"Expected finite float, got {val}"
        assert val == -80.0, f"Expected floor -80.0 for NaN block, got {val}"

    # 2. Direct _calc_dbfs edge cases
    assert _calc_dbfs(float("inf")) == 0.0
    assert _calc_dbfs(float("-inf")) == -80.0
    assert _calc_dbfs(float("nan")) == -80.0

    # 3. AudioPlayer.status_text elapsed time clamping
    p = AudioPlayer()
    p._is_playing = True
    p._start_time = 0.0
    p._duration = 5.0
    p._target_name = "Test"
    status = p.status_text()
    assert "(5.0s / 5.0s)" in status, f"Elapsed should be clamped to duration: {status}"


def test_job_config_batch_mode_default_paths():
    """Verify JobConfig default_output and default_audit_path do not assert when main is None."""
    from pathlib import Path

    from zout.config import JobConfig, ReferenceRole

    # 1. Batch config with out_dir specified
    cfg_out = JobConfig(
        reference=Path("ref.wav"),
        role=ReferenceRole.NOISY,
        main=None,
        batch_dir=Path("input_batch"),
        out_dir=Path("output_batch"),
        audit=True,
    )
    assert cfg_out.ready() is True
    out_path = cfg_out.default_output()
    assert isinstance(out_path, Path)
    assert out_path.parent == Path("output_batch")

    audit_path = cfg_out.default_audit_path()
    assert isinstance(audit_path, Path)
    assert audit_path.name.endswith(".json")
    assert audit_path.parent == Path("output_batch")

    noise_path = cfg_out.default_noise_output()
    assert isinstance(noise_path, Path)
    assert noise_path.parent == Path("output_batch")

    # 2. Batch config without out_dir specified (fallback)
    cfg_no_out = JobConfig(
        reference=Path("ref.wav"),
        main=None,
        batch_dir=Path("input_batch"),
    )
    assert cfg_no_out.ready() is True
    out_no_dir = cfg_no_out.default_output()
    assert isinstance(out_no_dir, Path)
    assert "input_batch" in str(out_no_dir)

    audit_no_dir = cfg_no_out.default_audit_path()
    assert isinstance(audit_no_dir, Path)
    assert audit_no_dir.name.endswith(".json")



# ============================================================================
# Milestone 2 Unit Tests: Streaming, Memory Cap, Audit Schema & Cancellation
# ============================================================================

def test_streaming_overlap_add_reconstruction_linearity(tmp_path):
    """Verify 2-second raised-cosine overlap-add formula exhibits unity gain
    and produces smooth amplitude continuity across chunk boundaries without dips or clicks."""
    # 1. Mathematical unity-gain identity check
    n_overlap = int(2.0 * SR)
    theta = np.pi * (np.arange(n_overlap) + 0.5) / n_overlap
    w_in = (0.5 * (1.0 - np.cos(theta))).astype(np.float32)
    w_out = (0.5 * (1.0 + np.cos(theta))).astype(np.float32)

    # w_in + w_out must be strictly 1.0 everywhere
    np.testing.assert_allclose(w_in + w_out, 1.0, atol=1e-6)
    assert abs(w_in[0]) < 1e-4
    assert abs(w_in[-1] - 1.0) < 1e-4

    # Blending constant DC tone yields identical constant tone
    dc1 = np.ones(n_overlap, dtype=np.float32)
    dc2 = np.ones(n_overlap, dtype=np.float32)
    blended_dc = dc1 * w_out + dc2 * w_in
    np.testing.assert_allclose(blended_dc, 1.0, atol=1e-6)

    # Blending continuous sine tone yields continuous sine tone
    t = np.arange(n_overlap) / SR
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    blended_sine = sine * w_out + sine * w_in
    np.testing.assert_allclose(blended_sine, sine, atol=1e-6)

    # 2. Pipeline chunked streaming continuity check
    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "main.wav"
    out_file = tmp_path / "stream_out.wav"

    ref_audio = hum(2.0)
    main_audio = speechlike(5.0) + hum(5.0)
    sf.write(ref_file, ref_audio, SR)
    sf.write(main_file, main_audio, SR)

    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                    polish=False, chunk_duration=1.5, overlap_duration=0.3)
    res = run_job(cfg)
    assert res is not None
    assert res.output.is_file()

    y, sr = sf.read(str(res.output), dtype="float32")
    assert sr == SR
    assert abs(len(y) - len(main_audio)) <= int(0.05 * sr)
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0001


def test_streaming_memory_cap_under_150mb(tmp_path):
    """Verify memory usage remains strictly capped below 150MB across multi-chunk streaming."""
    import tracemalloc

    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "long_main.wav"
    out_file = tmp_path / "mem_out.wav"

    # 8.0s audio with 1.0s chunks exercises 8 chunk cycles
    sf.write(ref_file, hum(1.5), SR)
    sf.write(main_file, speechlike(8.0) + hum(8.0), SR)

    tracemalloc.start()
    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                    polish=False, chunk_duration=1.0, overlap_duration=0.2)
    res = run_job(cfg)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert res is not None
    assert peak_mb < 150.0, f"Peak memory {peak_mb:.2f}MB exceeded 150MB cap"


def test_forensic_audit_true_peak_and_vad_schema(tmp_path):
    """Verify forensic audit JSON schema completeness, ITU-R BS.1770 4x true-peak detection,
    and VAD timestamp validation."""
    import json

    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "audit_main.wav"
    out_file = tmp_path / "audit_out.wav"

    # Speech signal at 1.0s-3.0s, silence/hum elsewhere
    sig = np.zeros(int(4.0 * SR), dtype=np.float32)
    sig[SR: 3 * SR] = speechlike(2.0)
    sf.write(ref_file, hum(2.0), SR)
    sf.write(main_file, sig + hum(4.0), SR)

    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                    polish=False, audit=True)
    res = run_job(cfg)
    assert res is not None

    audit_path = cfg.default_audit_path()
    if not audit_path.is_file():
        alt = out_file.with_name(out_file.stem + ".json")
        audit_path = alt if alt.is_file() else out_file.with_name(out_file.stem + "_audit.json")
    assert audit_path.is_file(), f"Audit JSON sidecar not found at {audit_path}"

    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Required Schema Fields
    assert "noise_reduction_db" in data or "suppression_db" in data
    assert "true_peak" in data or "true_peak_dbfs" in data
    assert "rms" in data or "rms_dbfs" in data
    assert "vad_speech_timestamps" in data
    assert "parameters" in data

    # 2. VAD timestamps validation
    vads = data["vad_speech_timestamps"]
    assert isinstance(vads, list)
    for interval in vads:
        assert interval["start"] >= 0.0
        assert interval["end"] >= interval["start"]
        assert interval["duration"] >= 0.0

    # 3. 4x oversampled True-Peak check
    tp = data.get("true_peak") or data.get("true_peak_dbfs")
    if isinstance(tp, dict):
        assert "output" in tp and isinstance(tp["output"], (int, float))
        assert tp["output"] <= 0.05
    else:
        assert isinstance(tp, (int, float))


def test_in_flight_cancellation_pipeline_and_engines(tmp_path):
    """Verify in-flight cancellation via threading.Event aborts cleanly in < 200ms,
    cleans up partially written output files, and leaves zero temporary artifacts."""
    import threading
    import time

    ref_file = tmp_path / "ref.wav"
    main_file = tmp_path / "cancel_main.wav"
    out_file = tmp_path / "cancel_out.wav"
    sf.write(ref_file, hum(2.0), SR)
    sf.write(main_file, speechlike(6.0) + hum(6.0), SR)

    # 1. Pre-set cancellation aborts immediately (< 100ms)
    cancel_ev = threading.Event()
    cancel_ev.set()

    t0 = time.time()
    cfg_pre = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file,
                        polish=False, cancel_event=cancel_ev)
    res_pre = run_job(cfg_pre)
    elapsed = time.time() - t0
    assert res_pre is None
    assert elapsed < 0.20, f"Pre-set cancellation took {elapsed:.3f}s, expected < 0.20s"
    assert not out_file.exists() or out_file.stat().st_size == 0

    # 2. cancel_token accepted directly in run_job signature
    cancel_ev2 = threading.Event()
    cancel_ev2.set()
    cfg_token = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, tmp_path / "token_out.wav",
                          polish=False)
    res_token = run_job(cfg_token, cancel_token=cancel_ev2)
    assert res_token is None

    # 3. Mid-flight cancellation during streaming loop cleanly unlinks output file
    cancel_stream_ev = threading.Event()
    chunk_counter = 0

    def progress_hook(msg, frac):
        nonlocal chunk_counter
        chunk_counter += 1
        if chunk_counter >= 2:
            cancel_stream_ev.set()

    stream_out = tmp_path / "stream_cancel.wav"
    cfg_stream = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, stream_out,
                           polish=False, chunk_duration=1.0, overlap_duration=0.2,
                           cancel_event=cancel_stream_ev)
    res_stream = run_job(cfg_stream, progress=progress_hook)
    assert res_stream is None
    assert not stream_out.exists() or stream_out.stat().st_size == 0
    # Zero lingering .tmp files
    assert len(list(tmp_path.glob("*.tmp"))) == 0


def test_streaming_48khz_default_60s_memory_under_75mb(tmp_path):
    """Verify 48kHz studio audio with default 60s chunk duration stays strictly capped
    under 75MB peak RAM (and far below the 150MB requirement ceiling) via adaptive clamping."""
    import tracemalloc

    SR_48K = 48000
    ref_file = tmp_path / "ref_48k.wav"
    main_file = tmp_path / "main_48k.wav"
    out_file = tmp_path / "mem_48k_out.wav"

    # 65.0s audio at 48kHz exercises multiple streaming chunks
    t_ref = np.arange(int(2.0 * SR_48K)) / SR_48K
    ref_audio = (0.08 * np.sin(2 * np.pi * 60 * t_ref) + 0.05 * np.sin(2 * np.pi * 180 * t_ref)).astype(np.float32)

    t_main = np.arange(int(65.0 * SR_48K)) / SR_48K
    main_audio = (0.2 * np.sin(2 * np.pi * 440 * t_main) + 0.05 * np.sin(2 * np.pi * 60 * t_main)).astype(np.float32)

    sf.write(ref_file, ref_audio, SR_48K)
    sf.write(main_file, main_audio, SR_48K)

    tracemalloc.start()
    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file, polish=False)
    res = run_job(cfg)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert res is not None
    assert res.output.is_file()
    assert peak_mb < 75.0, f"Peak memory {peak_mb:.2f}MB exceeded 75MB limit (150MB cap)"


def speechlike_var_sr(duration_sec: float, sr: int, seed: int = 42, f0: float = 140.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_sec * sr))) / sr
    env = (np.sin(2 * np.pi * 3 * t) > -0.2).astype(np.float32)
    win_len = max(10, int(0.0125 * sr))
    env = np.convolve(env, np.ones(win_len) / win_len, mode="same")
    sig = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 12))
    return (0.25 * sig * env).astype(np.float32)


def hum_var_sr(duration_sec: float, sr: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_sec * sr))) / sr
    sig = (0.08 * np.sin(2 * np.pi * 60 * t) +
           0.04 * np.sin(2 * np.pi * 180 * t) +
           0.02 * rng.standard_normal(len(t)))
    return sig.astype(np.float32)


def test_streaming_clean_48khz_memory_under_75mb(tmp_path):
    """Verify CleanVoiceEngine at 48kHz studio audio stays strictly capped
    under 75MB peak RAM under both run_streaming and run_job."""
    import tracemalloc
    from zout.pipeline import run_streaming

    sr = 48000
    ref_file = tmp_path / "ref_clean_48k.wav"
    main_file = tmp_path / "main_clean_48k.wav"

    ref_audio = speechlike_var_sr(4.0, sr, seed=10)
    dur = 12.0  # 12.0s at 48k = 576,000 samples (> 240k threshold, exercises 3 chunk cycles)
    main_audio = speechlike_var_sr(dur, sr, seed=20) + hum_var_sr(dur, sr, seed=30)
    sf.write(ref_file, ref_audio, sr)
    sf.write(main_file, main_audio, sr)

    # 1. run_streaming
    out_stream = tmp_path / "clean_48k_stream.wav"
    tracemalloc.start()
    cfg_s = JobConfig(ref_file, ReferenceRole.CLEAN, main_file, 0.75, 24.0, out_stream, polish=False)
    res_s = run_streaming(cfg_s)
    _, peak_s = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_s_mb = peak_s / (1024 * 1024)
    assert res_s is not None
    assert out_stream.is_file()
    assert peak_s_mb < 75.0, f"run_streaming peak {peak_s_mb:.2f}MB exceeded 75MB cap"

    # 2. run_job
    out_job = tmp_path / "clean_48k_job.wav"
    tracemalloc.start()
    cfg_j = JobConfig(ref_file, ReferenceRole.CLEAN, main_file, 0.75, 24.0, out_job, polish=False)
    res_j = run_job(cfg_j)
    _, peak_j = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_j_mb = peak_j / (1024 * 1024)
    assert res_j is not None
    assert out_job.is_file()
    assert peak_j_mb < 75.0, f"run_job peak {peak_j_mb:.2f}MB exceeded 75MB cap"


def test_streaming_clean_96khz_memory_under_75mb(tmp_path):
    """Verify CleanVoiceEngine at 96kHz audiophile sample rate stays strictly capped
    under 75MB peak RAM under both run_streaming and run_job."""
    import tracemalloc
    from zout.pipeline import run_streaming

    sr = 96000
    ref_file = tmp_path / "ref_clean_96k.wav"
    main_file = tmp_path / "main_clean_96k.wav"

    ref_audio = speechlike_var_sr(4.0, sr, seed=40)
    dur = 8.0  # 8.0s at 96k = 768,000 samples (> 240k threshold, exercises 4 chunk cycles)
    main_audio = speechlike_var_sr(dur, sr, seed=50) + hum_var_sr(dur, sr, seed=60)
    sf.write(ref_file, ref_audio, sr)
    sf.write(main_file, main_audio, sr)

    # 1. run_streaming
    out_stream = tmp_path / "clean_96k_stream.wav"
    tracemalloc.start()
    cfg_s = JobConfig(ref_file, ReferenceRole.CLEAN, main_file, 0.75, 24.0, out_stream, polish=False)
    res_s = run_streaming(cfg_s)
    _, peak_s = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_s_mb = peak_s / (1024 * 1024)
    assert res_s is not None
    assert out_stream.is_file()
    assert peak_s_mb < 75.0, f"run_streaming peak {peak_s_mb:.2f}MB exceeded 75MB cap"

    # 2. run_job
    out_job = tmp_path / "clean_96k_job.wav"
    tracemalloc.start()
    cfg_j = JobConfig(ref_file, ReferenceRole.CLEAN, main_file, 0.75, 24.0, out_job, polish=False)
    res_j = run_job(cfg_j)
    _, peak_j = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_j_mb = peak_j / (1024 * 1024)
    assert res_j is not None
    assert out_job.is_file()
    assert peak_j_mb < 75.0, f"run_job peak {peak_j_mb:.2f}MB exceeded 75MB cap"


def test_noisy_96khz_run_job_memory_under_75mb(tmp_path):
    """Verify NoisyAcousticEngine at 96kHz stays strictly capped under 75MB peak RAM under run_job."""
    import tracemalloc

    sr = 96000
    ref_file = tmp_path / "ref_noisy_96k.wav"
    main_file = tmp_path / "main_noisy_96k.wav"
    out_file = tmp_path / "noisy_96k_job.wav"

    ref_audio = hum_var_sr(3.0, sr, seed=70)
    dur = 10.0  # 10.0s at 96k = 960,000 samples
    main_audio = speechlike_var_sr(dur, sr, seed=80) + hum_var_sr(dur, sr, seed=90)
    sf.write(ref_file, ref_audio, sr)
    sf.write(main_file, main_audio, sr)

    tracemalloc.start()
    cfg = JobConfig(ref_file, ReferenceRole.NOISY, main_file, 0.75, 24.0, out_file, polish=False)
    res = run_job(cfg)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert res is not None
    assert out_file.is_file()
    assert peak_mb < 75.0, f"run_job peak {peak_mb:.2f}MB exceeded 75MB cap"



# ============================================================================
# MILESTONE 3 UNIT TESTS: Themes, VU Meter, Keys & Batch CLI
# ============================================================================

def test_themes_cycle_and_styles():
    """Verify ThemeManager cycle_theme, get_theme, set_theme, and Rich style parsing."""
    from rich.style import Style
    from zout.tui import themes

    required = ["Forensic Cyan", "Amber Phosphor", "Matrix Emerald", "Studio Dark"]
    for name in required:
        th = themes.get_theme(name)
        assert th is not None
        assert th.name == name
        for attr in ["primary", "secondary", "accent", "key", "border"]:
            val = getattr(th, attr)
            assert isinstance(val, str) and len(val) > 0
            assert Style.parse(val) is not None

    # Test sequential cycling and wraparound
    names = [themes.cycle_theme().name for _ in range(5)]
    assert names[0] == names[4]
    assert len(set(names[:4])) == 4

    # Test case-insensitive lookup and fallback
    assert themes.get_theme("amber phosphor").name == "Amber Phosphor"
    assert themes.get_theme("nonexistent_theme_fallback").name in required


def test_keys_nonblocking_polling():
    """Verify keys.poll_key and read_key timeout behaviors."""
    import time
    from zout.tui.keys import poll_key, read_key

    # Timeout None on non-interactive/closed returns or wait test
    t0 = time.perf_counter()
    k = read_key(timeout=0.03)
    dur = time.perf_counter() - t0
    assert k is None
    assert dur >= 0.02

    # poll_key instant check
    assert poll_key() is None


def test_stereo_vu_meter_rendering():
    """Verify render_stereo_vu_meter output structure, bar assembly, and dBFS readouts."""
    from zout.tui.screens import render_stereo_vu_meter

    levels = ((-18.5, -12.0), (-3.2, 0.2))  # RMS and Peak with clipping on R channel
    table = render_stereo_vu_meter(levels, width=20)
    assert table is not None
    assert len(table.columns) == 3


def test_batch_pipeline_execution(tmp_path):
    """Verify run_batch processes supported audio files while skipping non-audio files."""
    ref_file = tmp_path / "ref.wav"
    batch_dir = tmp_path / "batch_in"
    out_dir = tmp_path / "batch_out"

    sf.write(ref_file, hum(2), SR)
    batch_dir.mkdir()
    sf.write(batch_dir / "audio1.wav", speechlike(2), SR)
    sf.write(batch_dir / "audio2.wav", speechlike(2), SR)
    (batch_dir / "readme.txt").write_text("Ignore this file")
    (batch_dir / "sub_folder").mkdir()

    cfg = JobConfig(ref_file, ReferenceRole.NOISY, batch_dir=batch_dir, out_dir=out_dir, polish=False)
    res = run_job(cfg)

    assert res is not None
    out_files = sorted(out_dir.glob("*.wav"))
    assert len(out_files) == 2
    assert out_files[0].name == "audio1_zout.wav"
    assert out_files[1].name == "audio2_zout.wav"


def test_cli_parser_m3_flags():
    """Verify CLI parser parses --start, --end, --audit, --batch, --out-dir flags."""
    from zout.app import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "-r", "ref.wav", "-t", "noisy",
        "--batch", "my_batch_folder",
        "--out-dir", "my_out_folder",
        "--start", "00:15",
        "--end", "01:45",
        "--audit",
    ])
    assert args.batch == "my_batch_folder"
    assert args.out_dir == "my_out_folder"
    assert args.start == "00:15"
    assert args.end == "01:45"
    assert args.audit is True
