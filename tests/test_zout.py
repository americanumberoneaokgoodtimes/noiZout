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
