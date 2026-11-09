"""Build pink-noise test mixes from the TTS voices in testdata/ and evaluate zout on them.

Usage: python scripts/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from zout.audio.io import load_audio, resample
from zout.config import JobConfig, ReferenceRole
from zout.pipeline import run_job

D = Path(__file__).resolve().parent.parent / "testdata"
OUT = D / "out"
SR = 22050


def pink(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / np.sqrt(np.mean(y ** 2))).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = noise[: len(speech)]
    return speech + n * (rms(speech) / rms(n)) * 10 ** (-snr_db / 20)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    est -= est.mean(); ref -= ref.mean()
    t = (est @ ref) / (ref @ ref + 1e-12) * ref
    return float(10 * np.log10((t @ t) / (((est - t) ** 2).sum() + 1e-12)))


def load(name: str) -> np.ndarray:
    x, sr = load_audio(D / name)
    return resample(x, sr, SR)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    voice = load("voiceA_main.wav")
    interferer = load("voiceB_interferer.wav")
    clean_ref = load("voiceA_ref_clean.wav")
    L = len(voice)

    sf.write(D / "noise_ref_pink.wav", 0.3 * pink(SR * 6, seed=11) , SR)  # NOISY reference, unrelated segment
    rows = []

    print("== NOISY mode: reference = separate heavy pink noise sample ==")
    for snr in (5, 0, -5, -10):
        m = mix(voice, pink(L, seed=22), snr)
        m = m / max(1.0, np.abs(m).max() / 0.9)
        p = D / f"main_pink_snr{snr:+d}.wav"
        sf.write(p, m, SR)
        base = si_sdr(m, voice)
        for pol in (False, True):
            cfg = JobConfig(D / "noise_ref_pink.wav", ReferenceRole.NOISY, p, 0.8, 40.0,
                            OUT / f"{p.stem}_noisy_pol{int(pol)}.wav", polish=pol)
            r = run_job(cfg)
            y, _ = sf.read(r.output, dtype="float32")
            rows.append((f"NOISY pink SNR {snr:+d} dB s=0.8 DF3={pol}", base, si_sdr(y, voice)))

    print("== CLEAN mode: reference = clean voice A; main = A + voice B + pink ==")
    for snr in (0, -5):
        m = mix(voice + 0.7 * interferer[:L] if len(interferer) >= L else voice + 0.7 * np.pad(interferer, (0, L - len(interferer))),
                pink(L, seed=33), snr)
        m = m / max(1.0, np.abs(m).max() / 0.9)
        p = D / f"main_voices_pink_snr{snr:+d}.wav"
        sf.write(p, m, SR)
        sf.write(D / "voiceA_ref_clean_3s.wav", clean_ref[: SR * 3], SR)
        base = si_sdr(m, voice)
        for pol, ref in ((True, "voiceA_ref_clean.wav"), (True, "voiceA_ref_clean_3s.wav"), (False, "voiceA_ref_clean.wav")):
            cfg = JobConfig(D / ref, ReferenceRole.CLEAN, p, 0.8, 40.0,
                            OUT / f"{p.stem}_clean_{Path(ref).stem}_pol{int(pol)}.wav", polish=pol)
            r = run_job(cfg)
            y, _ = sf.read(r.output, dtype="float32")
            rows.append((f"CLEAN A+B+pink {snr:+d} dB {Path(ref).stem} DF3={pol}", base, si_sdr(y, voice)))

    print(f"\n{'case':58s} {'in SI-SDR':>10s} {'out':>8s} {'gain':>7s}")
    for name, a, b in rows:
        print(f"{name:58s} {a:10.1f} {b:8.1f} {b - a:+7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

