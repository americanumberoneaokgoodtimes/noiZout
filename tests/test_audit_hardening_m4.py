"""Comprehensive Unit & Regression Test Suite for Audio Pipeline & System Hardening (M4).

Covers all 14 audit resolution and hardening items:
1. Overlap-Add input energy accumulation continuity
2. Short audio STFT zero-padding & boundary-safe reconstruction
3. Reference recording minimum length guard
4. Persistent ffmpeg streaming pipe in audio audit (single subprocess)
5. Headless AudioPlayer clean EOF termination (_PlaybackFinished)
6. Crossfade concat safety clamping for short noise chunks
7. Multi-channel ITU-R BS.1770 true-peak & RMS without phase cancellation
8. Partial output file atomic cleanup on unhandled exceptions
9. Empty array -120 dB guard in noise_suppression_db
10. Auto-noise candidate scanning length cap (memory safety)
11. Audit VAD speech intervals non-redundancy (no second pass on speech-free input)
12. ffprobe stream duration fallback when format duration is N/A
13. OS-specific ffmpeg installation hints
14. Air-gapped / offline model pre-cache CLI (--download-models / --preload)
15. Batch errors log sidecar generation
16. Strict zero-textual dependency verification
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from zout.app import _build_parser, _preload_models, main
from zout.audio.audit import (
    _iter_audio_chunks,
    compute_rms,
    compute_true_peak,
    generate_audit_manifest,
)
from zout.audio.io import get_audio_info, load_audio, probe_audio
from zout.audio.masking import _istft, _stft, noise_suppression_db
from zout.audio.noise_extractor import _crossfade_concat
from zout.audio.player import AudioPlayer, HeadlessRunner, _PlaybackFinished
from zout.config import (
    JobConfig,
    ReferenceRole,
    collect_deps_status,
    get_ffmpeg_install_hint,
)
from zout.engines.noisy_engine import _stream_disk_to_disk
from zout.pipeline import run_batch, run_job, run_streaming


# ===========================================================================
# 1. Overlap-Add Input Energy Accumulation
# ===========================================================================

def test_ola_input_energy_accumulation_continuity():
    """Verify in_rms in _stream_disk_to_disk matches exact continuous signal RMS.

    Ensures that N_overlap samples at intermediate chunk boundaries are not omitted.
    """
    with tempfile.TemporaryDirectory() as td:
        in_f = Path(td) / "in.wav"
        out_f = Path(td) / "out.wav"
        sr = 48000
        # 10 seconds of constant amplitude 0.1 (-20 dBFS)
        data = np.ones(sr * 10, dtype=np.float32) * 0.1
        sf.write(in_f, data, sr)

        stats = _stream_disk_to_disk(
            in_f, out_f, sr, lambda x: x, chunk_sec=2.0, overlap_sec=0.5
        )
        # Theoretical RMS: 20 * log10(0.1) = -20.0 dBFS
        expected_rms = 20.0 * math.log10(0.1)
        assert abs(stats["in_rms"] - expected_rms) < 0.05, (
            f"Input RMS distorted: expected {expected_rms:.2f}, got {stats['in_rms']:.2f}"
        )
        assert abs(stats["out_rms"] - expected_rms) < 0.05


# ===========================================================================
# 2. Short Audio STFT Zero-Padding & Boundary Handling
# ===========================================================================

@pytest.mark.parametrize("length", [0, 5, 50, 500, 1024, 1536, 2047, 2048, 4096])
def test_short_audio_stft_zero_padding_and_inversion(length):
    """Verify _stft handles signals shorter than N_FFT (2048) without raising
    ValueError: noverlap must be less than nperseg, and _istft reconstructs cleanly.
    """
    sr = 48000
    x = np.random.randn(length).astype(np.float32) if length > 0 else np.zeros(0, dtype=np.float32)
    Z = _stft(x, sr)
    assert Z is not None
    y = _istft(Z, sr, length)
    assert len(y) == length
    assert y.dtype == np.float32


# ===========================================================================
# 3. Minimum Reference Audio Length Guard
# ===========================================================================

def test_reference_recording_minimum_length_guard():
    """Verify that reference audio shorter than max(2048, sr // 4) raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ref_f = root / "short_ref.wav"
        main_f = root / "main.wav"
        sr = 48000
        # 100 samples is much shorter than 2048 or sr // 4 (12000)
        sf.write(ref_f, np.zeros(100, dtype=np.float32), sr)
        sf.write(main_f, np.zeros(sr * 2, dtype=np.float32), sr)

        cfg = JobConfig(
            reference=ref_f,
            role=ReferenceRole.NOISY,
            main=main_f,
            output=root / "out.wav",
        )
        with pytest.raises(ValueError, match="reference recording is too short"):
            run_job(cfg)

        with pytest.raises(ValueError, match="reference recording is too short"):
            run_streaming(cfg)


# ===========================================================================
# 4. Persistent Single-Process FFmpeg Pipe in Audio Audit
# ===========================================================================

def test_audit_persistent_ffmpeg_pipe_no_process_explosion(monkeypatch):
    """Verify non-native audio streaming in _iter_audio_chunks uses a single
    persistent subprocess rather than spawning a process per chunk.
    """
    with tempfile.TemporaryDirectory() as td:
        fake_mp3 = Path(td) / "test.mp3"
        fake_mp3.write_bytes(b"ID3" + b"\x00" * 1000)

        # Mock get_audio_info to return 30s duration
        from zout.audio.io import AudioInfo
        monkeypatch.setattr("zout.audio.audit.get_audio_info", lambda p: AudioInfo(duration=30.0, samplerate=48000, channels=2, frames=30 * 48000))

        # Track Popen calls
        popen_count = 0
        real_popen = subprocess.Popen

        class MockPipe:
            def __init__(self, *args, **kwargs):
                nonlocal popen_count
                popen_count += 1
                self.stdout = MagicMock()
                # Simulate 3 chunks of 10s audio
                sr = 48000
                chunk_bytes = 10 * sr * 2 * 4
                raw_chunk = np.zeros((10 * sr, 2), dtype=np.float32).tobytes()
                self.stdout.read.side_effect = [raw_chunk, raw_chunk, raw_chunk, b""]

            def kill(self):
                pass

            def wait(self):
                return 0

        monkeypatch.setattr(subprocess, "Popen", MockPipe)

        chunks = list(_iter_audio_chunks(fake_mp3, chunk_sec=10.0, start_sec=0.0, end_sec=30.0))
        assert len(chunks) == 3
        # Must be exactly 1 process spawned for all 3 chunks, NOT 3 processes!
        assert popen_count == 1, f"Expected 1 persistent process, but spawned {popen_count}"


# ===========================================================================
# 5. Headless AudioPlayer Clean EOF Termination
# ===========================================================================

def test_headless_audio_player_clean_eof_termination():
    """Verify HeadlessRunner cleanly terminates when audio callback signals EOF."""
    player = AudioPlayer(force_headless=True)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "tiny.wav"
        # 0.1s of audio @ 44.1kHz = 4410 samples
        sf.write(f, np.zeros(4410, dtype=np.float32), 44100)
        ok = player.play(f)
        assert ok
        assert player.is_playing()

        # Wait up to 1.5s for playback to finish
        t_start = time.time()
        while player.is_playing() and (time.time() - t_start) < 1.5:
            time.sleep(0.05)

        assert not player.is_playing(), "Player failed to terminate at EOF in headless mode"
        assert player._headless_runner is None or not player._headless_runner._thread.is_alive()


def test_audio_player_callback_raises_playback_finished_when_no_sounddevice(monkeypatch):
    """Verify _audio_callback raises _PlaybackFinished if sounddevice is absent."""
    player = AudioPlayer(force_headless=True)
    outdata = np.zeros((1024, 2), dtype=np.float32)

    # Force sounddevice import to fail
    import builtins
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("Mocked missing sounddevice")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Set sample_index beyond total frames to trigger stop_after
    with player._lock:
        player._audio_a = np.zeros((100, 2), dtype=np.float32)
        player._audio_b = np.zeros((100, 2), dtype=np.float32)
        player._sample_index = 200
        player._is_playing = True

    with pytest.raises(_PlaybackFinished):
        player._audio_callback(outdata, 1024, None, None)


# ===========================================================================
# 6. Crossfade Concat Chunk Safety Clamp
# ===========================================================================

def test_crossfade_concat_short_chunks_safety():
    """Verify _crossfade_concat clamps xfade_samples to half the shortest chunk."""
    sr = 48000
    # At 48kHz, default xfade_ms=25 is 1200 samples.
    # Provide chunks shorter than 1200 samples.
    chunks = [
        np.ones(100, dtype=np.float32),
        np.ones(60, dtype=np.float32),
        np.ones(150, dtype=np.float32),
    ]
    # Without fix, this raises ValueError: operands could not be broadcast together with shapes (60,) (1200,)
    res = _crossfade_concat(chunks, sr=sr, xfade_ms=25.0)
    assert len(res) > 0
    assert np.all(np.isfinite(res))

    # Single or tiny chunks
    tiny = [np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32)]
    res_tiny = _crossfade_concat(tiny, sr=sr, xfade_ms=25.0)
    assert len(res_tiny) == 5


# ===========================================================================
# 7. Multi-Channel True-Peak & RMS (ITU-R BS.1770)
# ===========================================================================

def test_multichannel_true_peak_itu_bs1770():
    """Verify true-peak measures maximum channel independently without phase cancellation."""
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    # Hard panned stereo: Left channel full sine, Right channel silence
    panned = np.zeros((sr, 2), dtype=np.float32)
    panned[:, 0] = sine
    tp_panned = compute_true_peak(panned, sr)
    # Peak of pure sine is 0 dBFS (or slightly above with sinc oversampling)
    assert abs(tp_panned - 0.0) < 0.2, f"Expected ~0 dBFS for hard panned sine, got {tp_panned}"

    # Out of phase stereo: Left = sine, Right = -sine
    # If mono downmixed by (L+R)/2, this cancels to silence (-120 dBFS)!
    ooph = np.zeros((sr, 2), dtype=np.float32)
    ooph[:, 0] = sine
    ooph[:, 1] = -sine
    tp_ooph = compute_true_peak(ooph, sr)
    assert abs(tp_ooph - 0.0) < 0.2, f"Out-of-phase true peak cancelled out: {tp_ooph}"

    rms_ooph = compute_rms(ooph, sr)
    # Total power across channels for amplitude 1.0 sine: RMS = -3.01 dBFS
    assert abs(rms_ooph - (-3.01)) < 0.2, f"Out-of-phase RMS cancelled out: {rms_ooph}"


# ===========================================================================
# 8. Unhandled Exception Partial File Cleanup
# ===========================================================================

def test_unhandled_exception_cleans_up_partial_output():
    """Verify any unhandled exception in run_job unlinks partial output files."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ref_f = root / "ref.wav"
        main_f = root / "main.wav"
        out_f = root / "out.wav"
        sr = 48000
        sf.write(ref_f, np.ones(sr * 2, dtype=np.float32) * 0.1, sr)
        sf.write(main_f, np.ones(sr * 2, dtype=np.float32) * 0.1, sr)

        cfg = JobConfig(
            reference=ref_f,
            role=ReferenceRole.NOISY,
            main=main_f,
            output=out_f,
        )

        with patch("zout.engines.noisy_engine.NoisyAcousticEngine.run", side_effect=RuntimeError("Simulated DSP Crash")):
            # Create a dummy partial output to simulate in-flight write
            out_f.write_bytes(b"partial header data")
            assert out_f.exists()

            with pytest.raises(RuntimeError, match="Simulated DSP Crash"):
                run_job(cfg)

            assert not out_f.exists(), "Partial output file was not cleaned up after exception"


# ===========================================================================
# 9. Empty Array -120 dB Guard in noise_suppression_db
# ===========================================================================

def test_noise_suppression_db_empty_arrays():
    """Verify noise_suppression_db returns -120.0 for empty arrays without NaN."""
    empty = np.zeros(0, dtype=np.float32)
    normal = np.ones(1024, dtype=np.float32) * 0.1

    assert noise_suppression_db(empty, empty) == -120.0
    assert noise_suppression_db(empty, normal) == -120.0
    assert noise_suppression_db(normal, empty) == -120.0


# ===========================================================================
# 10. Auto-Noise Candidate Scanning Length Cap
# ===========================================================================

def test_auto_noise_candidate_scan_length_capping(monkeypatch):
    """Verify prompt_auto_noise uses bounded streaming on long 3-hour recordings instead of unbounded load_audio."""
    from rich.console import Console
    from zout.tui.keys import Key
    from zout.tui.screens import prompt_auto_noise
    from zout.audio.io import AudioInfo

    with tempfile.TemporaryDirectory() as td:
        main_f = Path(td) / "3hour_meeting.wav"
        main_f.write_bytes(b"RIFF" + b"\x00" * 1000)

        monkeypatch.setattr(
            "zout.audio.noise_extractor.get_audio_info",
            lambda p: AudioInfo(duration=10800.0, samplerate=48000, channels=2, frames=10800 * 48000),
        )

        load_called = False

        def forbid_full_load(*args, **kwargs):
            nonlocal load_called
            load_called = True
            return np.zeros(48000, dtype=np.float32), 48000

        streamed_chunk_secs: list[float] = []

        def fake_stream(path, chunk_sec=10.0, start_sec=None, end_sec=None, mono=True, cancel_event=None, **_):
            streamed_chunk_secs.append(chunk_sec)
            if cancel_event is not None:
                cancel_event.set()
            raise InterruptedError("Cancelled")

        monkeypatch.setattr("zout.audio.noise_extractor.load_audio", forbid_full_load)
        monkeypatch.setattr("zout.audio.noise_extractor.iter_audio_chunks", fake_stream)
        monkeypatch.setattr("zout.tui.screens.read_key", lambda: Key("enter"))
        monkeypatch.setattr(
            "zout.tui.screens.poll_key",
            lambda timeout=0.05: Key("esc", "\x1b") if streamed_chunk_secs else None,
        )

        cfg = JobConfig(main=main_f)
        res = prompt_auto_noise(Console(record=True), cfg)
        assert res is None
        assert not load_called, "3-hour file must stream in bounded blocks rather than loading fully into RAM"
        assert streamed_chunk_secs == [30.0]


# ===========================================================================
# 11. Audit VAD Redundant Second Pass Elimination
# ===========================================================================

def test_audit_vad_does_not_rerun_on_speech_free_input(monkeypatch):
    """Verify that if in_src was analyzed and had zero speech, audit generator
    does NOT run a redundant second pass over out_src.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out_f = root / "out.wav"
        sf.write(out_f, np.zeros(48000, dtype=np.float32), 48000)

        cfg = JobConfig(
            reference=root / "ref.wav",
            main=root / "main.wav",
            output=out_f,
            audit=True,
        )

        vad_calls = []

        def mock_vad(src, *args, **kwargs):
            vad_calls.append(src)
            return []  # No speech detected in input

        monkeypatch.setattr("zout.audio.audit.extract_vad_intervals", mock_vad)
        monkeypatch.setattr("zout.audio.audit._get_audio_duration", lambda *args, **kwargs: 10.0)

        generate_audit_manifest(
            cfg=cfg,
            output_path=out_f,
            input_audio=np.ones(48000, dtype=np.float32) * 0.01,
            output_audio=out_f,
            sr=48000,
        )

        # Must have only run VAD ONCE on in_src, and NOT a second time on out_src!
        assert len(vad_calls) == 1


# ===========================================================================
# 12. ffprobe Stream Duration Fallback
# ===========================================================================

def test_ffprobe_stream_duration_fallback(monkeypatch):
    """Verify probe_audio falls back to stream duration when format duration is N/A."""
    with tempfile.TemporaryDirectory() as td:
        frag_mp4 = Path(td) / "fragmented.mp4"
        frag_mp4.write_bytes(b"\x00" * 500)

        # Mock ffprobe output with stream duration but format duration=N/A
        mock_output = (
            "sample_rate=44100\n"
            "channels=2\n"
            "duration=72.500000\n"  # from stream=duration
            "duration=N/A\n"        # from format=duration
        )

        class MockSubprocessRun:
            stdout = mock_output
            returncode = 0

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockSubprocessRun())
        monkeypatch.setattr("shutil.which", lambda cmd: "ffprobe")

        info = probe_audio(frag_mp4)
        assert info is not None
        assert info.duration == 72.5
        assert info.channels == 2


# ===========================================================================
# 13. OS-Specific FFmpeg Installation Hints
# ===========================================================================

def test_os_specific_ffmpeg_install_hint(monkeypatch):
    """Verify get_ffmpeg_install_hint returns package manager command matching host OS."""
    monkeypatch.setattr(sys, "platform", "win32")
    hint_win = get_ffmpeg_install_hint()
    assert "winget install Gyan.FFmpeg" in hint_win

    monkeypatch.setattr(sys, "platform", "darwin")
    hint_mac = get_ffmpeg_install_hint()
    assert "brew install ffmpeg" in hint_mac

    monkeypatch.setattr(sys, "platform", "linux")
    hint_linux = get_ffmpeg_install_hint()
    assert "apt install ffmpeg" in hint_linux


def test_collect_deps_status_contains_install_hint_when_ffmpeg_missing(monkeypatch):
    """Verify collect_deps_status includes OS install command if ffmpeg is missing."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    rows = collect_deps_status()
    ffmpeg_row = [r for r in rows if r[0] == "Media format support"][0]
    assert ffmpeg_row[2] == "MISSING"
    assert "install via:" in ffmpeg_row[3]


# ===========================================================================
# 14. Air-Gapped / Offline Model Pre-Cache CLI
# ===========================================================================

def test_download_models_cli_flag_parsing():
    """Verify --download-models and --preload CLI options are recognized."""
    ap = _build_parser()
    args1 = ap.parse_args(["--download-models"])
    assert args1.preload_models is True

    args2 = ap.parse_args(["--preload"])
    assert args2.preload_models is True


def test_preload_models_function():
    """Verify _preload_models executes cleanly."""
    rc = _preload_models()
    assert rc == 0


# ===========================================================================
# 15. Batch Errors Log Sidecar Generation
# ===========================================================================

def test_batch_errors_log_sidecar_generated_on_failure():
    """Verify run_batch creates batch_errors.log in out_dir on corrupted files."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        batch_dir = root / "batch"
        batch_dir.mkdir()
        out_dir = root / "out"
        out_dir.mkdir()
        ref_f = root / "ref.wav"
        sf.write(ref_f, np.ones(48000, dtype=np.float32) * 0.05, 48000)

        # 1 valid file, 1 corrupted file
        sf.write(batch_dir / "valid.wav", np.ones(48000, dtype=np.float32) * 0.1, 48000)
        (batch_dir / "broken.wav").write_bytes(b"NOT_A_VALID_AUDIO_HEADER")

        cfg = JobConfig(
            reference=ref_f,
            role=ReferenceRole.NOISY,
            batch_dir=batch_dir,
            out_dir=out_dir,
        )
        res = run_batch(cfg)
        assert res is not None
        assert res.notes["success"] == 1
        assert len(res.notes["failures"]) == 1

        err_log = out_dir / "batch_errors.log"
        assert err_log.is_file(), "batch_errors.log was not generated"
        content = err_log.read_text(encoding="utf-8")
        assert "broken.wav" in content
        assert "Batch processing errors" in content


# ===========================================================================
# 16. Strict Zero Textual Dependency Enforcement
# ===========================================================================

def test_strict_zero_textual_in_all_new_and_modified_files():
    """Ensure zero occurrences of textual import across entire zout package."""
    import ast
    repo_root = Path(__file__).resolve().parent.parent
    zout_dir = repo_root / "zout"

    for py_file in zout_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "textual" not in alias.name, f"Forbidden textual import in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "textual" not in node.module, f"Forbidden textual import in {py_file}"


# ===========================================================================
# 17. Streaming Non-Native Output Extension Parity & Audit Integrity
# ===========================================================================

def test_streaming_non_native_output_extension_returns_wav_and_warns_and_audits():
    """Verify run_streaming with -o out.mp3 returns the written .wav path, records warning, and audits real audio."""
    import json
    from zout.pipeline import run_streaming

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ref_f = root / "ref.wav"
        main_f = root / "main.wav"
        requested_out = root / "out.mp3"
        sr = 48000
        rng = np.random.default_rng(7)
        sf.write(ref_f, (rng.standard_normal(sr * 2) * 0.05).astype(np.float32), sr)
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        sig = (0.3 * np.sin(2 * np.pi * 440 * t) + rng.standard_normal(sr * 3) * 0.05).astype(np.float32)
        sf.write(main_f, sig, sr)

        cfg = JobConfig(
            reference=ref_f,
            role=ReferenceRole.NOISY,
            main=main_f,
            output=requested_out,
            polish=False,
            audit=True,
        )
        res = run_streaming(cfg)
        assert res is not None
        assert res.output.suffix == ".wav"
        assert res.output.is_file()
        assert not requested_out.exists()
        assert "warning" in res.notes
        assert ".mp3" in res.notes["warning"] and "saved as WAV" in res.notes["warning"]
        assert res.audit_path is not None and res.audit_path.is_file()
        manifest = json.loads(res.audit_path.read_text(encoding="utf-8"))
        assert manifest["rms_dbfs"]["output"] > -60.0
        assert manifest["true_peak_dbfs"]["output"] > -60.0


# ===========================================================================
# 18. Effective Streaming Window & Custom Max Chunk Samples
# ===========================================================================

def test_effective_streaming_window_and_custom_max_chunk_samples():
    """Verify effective_streaming_window reports capping accurately and honors custom max_samples."""
    from zout.engines.noisy_engine import effective_streaming_window

    eff_c, eff_o, capped = effective_streaming_window(60.0, 2.0, 48000, max_samples=240_000)
    assert capped is True
    assert abs(eff_c - 5.0) < 1e-6
    assert abs(eff_o - 1.25) < 1e-6

    eff_c2, eff_o2, capped2 = effective_streaming_window(10.0, 2.0, 48000, max_samples=720_000)
    assert capped2 is False
    assert abs(eff_c2 - 10.0) < 1e-6
    assert abs(eff_o2 - 2.0) < 1e-6

    with pytest.raises(ValueError, match="max_chunk_samples"):
        JobConfig(max_chunk_samples=1024)


# ===========================================================================
# 19. Dashboard Controls Layout & Capped Streaming Window Display
# ===========================================================================

def test_dashboard_controls_no_split_item_and_shows_effective_cap():
    """Verify dashboard at 80 and 110 cols never splits '[B] Batch' across lines and shows effective cap."""
    from io import StringIO
    from rich.console import Console
    from zout.tui import screens

    cfg = JobConfig(chunk_duration=60.0, overlap_duration=2.0, max_chunk_samples=240_000)
    for width in (80, 110):
        buf = StringIO()
        c = Console(file=buf, width=width, force_terminal=False, color_system=None)
        c.print(screens.dashboard(cfg))
        rendered = buf.getvalue()
        assert "effective 5.0s" in rendered
        assert "240k-sample cap" in rendered
        for line in rendered.splitlines():
            # Ensure 'B' key and 'Batch' label are on the same line whenever Batch is rendered
            if "Batch" in line and "Controls" not in line and "Pick a reference" not in line:
                assert " B  Batch" in line or "Batch folder" in line


# ===========================================================================
# 20. File Picker Single Waveform Preview (No Duplicate Table Info Envelope)
# ===========================================================================

def test_file_picker_no_duplicate_waveform_in_info_column():
    """Verify file_picker renders Waveform Dynamics only in the bottom preview bar, not inside the table row."""
    from io import StringIO
    from rich.console import Console
    from zout.tui import file_picker

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wav = root / "sample.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sf.write(wav, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)

        st = file_picker.PickerState(start=root)
        st.refresh()
        # Select sample.wav
        for idx, entry in enumerate(st.entries):
            if entry.path == wav:
                st.index = idx
                break

        buf = StringIO()
        c = Console(file=buf, width=110, height=28, force_terminal=False, color_system=None)
        c.print(file_picker.render(st, "Pick Audio", 110, 28))
        out = buf.getvalue()
        assert "Waveform Dynamics: [" in out
        # Find the row line containing sample.wav and ensure it has only 2 pipe separators (size | dur | sr | ch)
        row_lines = [ln for ln in out.splitlines() if "sample.wav" in ln]
        assert len(row_lines) == 1
        assert row_lines[0].count("|") == 3  # e.g. "31 KB | 00:01 | 16kHz | Ch1"


# ===========================================================================
# 21. Sampled Fast Multi-Hour Auto-Noise Scan & Auto Fallback
# ===========================================================================

def test_sampled_noise_scan_multi_hour_and_auto_fallback(monkeypatch):
    """Verify extract_noise_profile_from_file uses distributed probing on >600s files and falls back when needed."""
    from zout.audio.io import AudioInfo
    import zout.audio.noise_extractor as nx

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "long.wav"
        sr = 16000
        rng = np.random.default_rng(99)
        room = (rng.standard_normal(sr * 12) * 0.02).astype(np.float32)
        sf.write(wav, room, sr)

        # Simulate a 3600s (1-hour) recording
        monkeypatch.setattr(nx, "get_audio_info", lambda p: AudioInfo(3600.0, sr, 1, frames=3600 * sr))
        probed_Windows: list[tuple[float | None, float | None]] = []

        def fake_iter_chunks(path, chunk_sec=10.0, start_sec=None, end_sec=None, mono=True, cancel_event=None):
            probed_Windows.append((start_sec, end_sec))
            yield room, sr

        monkeypatch.setattr(nx, "iter_audio_chunks", fake_iter_chunks)
        monkeypatch.setattr(nx, "_load_vad_model", lambda: (None, None))

        res = nx.extract_noise_profile_from_file(wav, n_chunks=4, chunk_sec=1.5, mode="fast", scan_mode="auto")
        assert len(res.chunks) == 4
        # Should have probed distributed windows (12 windows for n_chunks=4), NOT 120 sequential 30s blocks
        assert 8 <= len(probed_Windows) <= 24
        assert probed_Windows[0][0] == 0.0
        assert probed_Windows[-1][0] > 3000.0


# ===========================================================================
# 22. CLI Parser Flags (--max-chunk-samples & --noise-scan)
# ===========================================================================

def test_cli_max_chunk_samples_and_noise_scan_flags():
    """Verify _build_parser and _validate support --max-chunk-samples and --noise-scan."""
    from zout.app import _build_parser, _validate

    ap = _build_parser()
    args = ap.parse_args(["--auto-noise", "-m", __file__, "--noise-scan", "sampled", "--max-chunk-samples", "480000"])
    assert args.noise_scan == "sampled"
    assert args.max_chunk_samples == 480000


# ===========================================================================
# 23. Gate Verification Regression Fixes
# ===========================================================================

def test_run_batch_propagates_max_chunk_samples(monkeypatch):
    """Verify run_batch forwards cfg.max_chunk_samples to per-item JobConfig."""
    from zout import pipeline

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        batch_dir = root / "batch"
        batch_dir.mkdir()
        out_dir = root / "out"
        ref_f = root / "ref.wav"
        sf.write(ref_f, np.zeros(16000, dtype=np.float32), 16000)
        sf.write(batch_dir / "item.wav", np.zeros(16000, dtype=np.float32), 16000)

        seen_caps: list[int] = []

        def fake_run_job(item_cfg, progress=None, cancel_token=None):
            seen_caps.append(item_cfg.max_chunk_samples)
            return pipeline.Result(item_cfg.output, 0.1, -20.0, -30.0, 10.0)

        monkeypatch.setattr(pipeline, "run_job", fake_run_job)
        cfg = JobConfig(
            reference=ref_f,
            role=ReferenceRole.NOISY,
            batch_dir=batch_dir,
            out_dir=out_dir,
            max_chunk_samples=65536,
        )
        res = pipeline.run_batch(cfg)
        assert res is not None
        assert seen_caps == [65536]


def test_ref_length_guard_uses_ref_sr_and_preserves_existing_output():
    """Verify 0.5s@8kHz ref (4000 samples) works with 48kHz main, and pre-flight error preserves existing output."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ref_8k = root / "ref_8k.wav"
        main_48k = root / "main_48k.wav"
        out_f = root / "existing_out.wav"

        rng = np.random.default_rng(11)
        sf.write(ref_8k, (rng.standard_normal(4000) * 0.05).astype(np.float32), 8000)
        sf.write(main_48k, (rng.standard_normal(48000 * 2) * 0.1).astype(np.float32), 48000)

        cfg = JobConfig(
            reference=ref_8k,
            role=ReferenceRole.NOISY,
            main=main_48k,
            output=out_f,
            polish=False,
        )
        res = run_job(cfg)
        assert res is not None and out_f.is_file()
        orig_bytes = out_f.read_bytes()

        # Now trigger a pre-flight error with a too-short ref: existing out_f must remain intact
        short_ref = root / "too_short.wav"
        sf.write(short_ref, np.zeros(100, dtype=np.float32), 48000)
        bad_cfg = JobConfig(
            reference=short_ref,
            role=ReferenceRole.NOISY,
            main=main_48k,
            output=out_f,
            polish=False,
        )
        with pytest.raises(ValueError, match="reference recording is too short"):
            run_job(bad_cfg)
        assert out_f.is_file() and out_f.read_bytes() == orig_bytes


def test_get_audio_info_corrupt_native_raises_runtime_error():
    """Verify get_audio_info wraps soundfile errors on corrupt native files as RuntimeError."""
    from zout.audio.io import get_audio_info

    with tempfile.TemporaryDirectory() as td:
        corrupt = Path(td) / "corrupt.wav"
        corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt_corrupt_bytes")
        with pytest.raises(RuntimeError, match="Failed to read audio metadata"):
            get_audio_info(corrupt)
