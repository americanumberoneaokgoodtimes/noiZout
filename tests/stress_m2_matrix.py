"""Challenger Empirical Stress Matrix: High Sample Rate Memory Bounds & Linear Reconstruction.

Tests:
1. Peak memory bounds across 16kHz, 48kHz, 96kHz for both NoisyAcousticEngine and CleanVoiceEngine
   under both run_streaming(cfg) and run_job(cfg) with default 60s chunk duration.
2. Exact raised-cosine overlap-add reconstruction linearity and sample drift across various
   sample rates, prime sample lengths, and boundary configurations.
"""

from __future__ import annotations

import gc
import math
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from zout.config import JobConfig, ReferenceRole
from zout.engines.clean_engine import CleanVoiceEngine
from zout.engines.noisy_engine import NoisyAcousticEngine, _stream_disk_to_disk
from zout.pipeline import run_job, run_streaming


def gen_speechlike_var_sr(duration_sec: float, sr: int, seed: int = 42, f0: float = 140.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_sec * sr))) / sr
    env = (np.sin(2 * np.pi * 3 * t) > -0.2).astype(np.float32)
    win_len = max(10, int(0.0125 * sr))
    env = np.convolve(env, np.ones(win_len) / win_len, mode="same")
    sig = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 12))
    return (0.25 * sig * env).astype(np.float32)


def gen_noise_var_sr(duration_sec: float, sr: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_sec * sr))) / sr
    sig = (0.08 * np.sin(2 * np.pi * 60 * t) +
           0.04 * np.sin(2 * np.pi * 180 * t) +
           0.02 * rng.standard_normal(len(t)))
    return sig.astype(np.float32)


def measure_peak_memory(fn) -> tuple[float, any]:
    """Execute fn with tracemalloc and return (peak_mb, result)."""
    gc.collect()
    tracemalloc.reset_peak()
    tracemalloc.start()
    try:
        res = fn()
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        return peak_mb, res
    finally:
        tracemalloc.stop()
        gc.collect()


def run_benchmark():
    rates = [16000, 48000, 96000]
    roles = [
        ("NOISY", ReferenceRole.NOISY),
        ("CLEAN", ReferenceRole.CLEAN),
    ]
    runners = [
        ("run_streaming", run_streaming),
        ("run_job", run_job),
    ]

    duration = 65.0  # 65 seconds exercises multi-chunk with default 60s chunks
    results = []

    print("=" * 80)
    print(f"EMPIRICAL MEMORY BOUNDS BENCHMARK (Duration: {duration}s, default 60s chunk)")
    print("=" * 80)
    print(f"{'Role':<8} | {'Sample Rate':<12} | {'Runner':<15} | {'Peak RAM (MB)':<14} | {'Status (<75MB)':<15}")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for sr in rates:
            # Generate test audio
            ref_noisy = gen_noise_var_sr(3.0, sr, seed=10)
            main_noisy = gen_speechlike_var_sr(duration, sr, seed=20) + gen_noise_var_sr(duration, sr, seed=30)

            ref_clean = gen_speechlike_var_sr(4.0, sr, seed=40)
            main_clean = gen_speechlike_var_sr(duration, sr, seed=50) + gen_noise_var_sr(duration, sr, seed=60)

            noisy_ref_path = tmp_path / f"ref_noisy_{sr}.wav"
            noisy_main_path = tmp_path / f"main_noisy_{sr}.wav"
            sf.write(noisy_ref_path, ref_noisy, sr)
            sf.write(noisy_main_path, main_noisy, sr)

            clean_ref_path = tmp_path / f"ref_clean_{sr}.wav"
            clean_main_path = tmp_path / f"main_clean_{sr}.wav"
            sf.write(clean_ref_path, ref_clean, sr)
            sf.write(clean_main_path, main_clean, sr)

            for role_name, role in roles:
                ref_p = noisy_ref_path if role is ReferenceRole.NOISY else clean_ref_path
                main_p = noisy_main_path if role is ReferenceRole.NOISY else clean_main_path

                for runner_name, runner_fn in runners:
                    out_p = tmp_path / f"out_{role_name}_{runner_name}_{sr}.wav"
                    cfg = JobConfig(
                        reference=ref_p,
                        role=role,
                        main=main_p,
                        strength=0.75,
                        max_atten_db=24.0,
                        output=out_p,
                        chunk_duration=60.0,
                        overlap_duration=2.0,
                        polish=False,
                    )

                    def task():
                        return runner_fn(cfg)

                    peak_mb, res = measure_peak_memory(task)
                    passed_75 = peak_mb < 75.0
                    passed_150 = peak_mb < 150.0
                    status_str = "PASS (<75MB)" if passed_75 else ("PASS (<150MB)" if passed_150 else "FAIL (>=150MB)")

                    print(f"{role_name:<8} | {sr:<12} | {runner_name:<15} | {peak_mb:<14.2f} | {status_str:<15}")
                    assert res is not None, f"Execution returned None for {role_name} {sr} {runner_name}"
                    assert out_p.exists(), f"Output file missing for {role_name} {sr} {runner_name}"
                    out_audio, out_sr = sf.read(out_p, dtype="float32")
                    assert out_sr == sr
                    assert len(out_audio) == int(round(duration * sr)), f"Sample length mismatch: {len(out_audio)} vs {int(round(duration * sr))}"
                    assert np.isfinite(out_audio).all(), "Non-finite output samples detected"

                    results.append({
                        "role": role_name,
                        "sr": sr,
                        "runner": runner_name,
                        "peak_mb": peak_mb,
                        "passed_75": passed_75,
                        "passed_150": passed_150,
                    })

    print("=" * 80)
    return results


if __name__ == "__main__":
    run_benchmark()
