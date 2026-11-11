# noiZout Test Infrastructure Specification (`TEST_INFRA.md`)

## 1. Overview & Objectives

This document specifies the testing architecture, methodology, environment, and verification framework for **noiZout**—a high-performance, reference-conditioned audio noise reduction and voice isolation system.

The testing infrastructure guarantees:
1. **Opaque-Box Requirement Coverage**: Tests exercise high-level application interfaces (`JobConfig`, `run_job`, `AudioPlayer`, CLI entry point `_cli`) and validate observable behaviors without depending on internal implementation details.
2. **Speed & Determinism**: Uses fast, mathematically derived synthetic audio fixtures (sine waves, harmonic speech envelopes, acoustic hum, Gaussian noise) without downloading remote models or accessing hardware sound cards. Full 125-test suite executes in ~2 minutes (KL-NMF is compute-intensive at 48kHz).
3. **Headless & CI Resilience**: Provides a drop-in `MockOutputStream` for `sounddevice` to ensure playback and meter callback logic execute reliably on headless servers and CI nodes lacking physical DAC hardware.
4. **Progressive Testability**: Employs dynamic interface discovery guards (`is_r1_implemented()`, `is_r2_implemented()`, etc.) allowing tests to run cleanly during active milestone development, automatically activating as each milestone lands.

---

## 2. Directory Layout & Artifacts

```
C:\Users\o\Desktop\git\zout\
├── pyproject.toml              # Build config, dependencies, entry points
├── TEST_INFRA.md               # Test infrastructure architecture (this document)
├── TEST_READY.md               # Test execution summary & coverage matrix
├── tests/
│   ├── test_zout.py            # Existing unit & baseline regression tests (8 tests)
│   └── test_e2e.py             # 4-Tier comprehensive E2E test suite (93 tests)
└── zout/                       # Application source tree
    ├── app.py                  # CLI parser & TUI loop
    ├── config.py               # JobConfig & ReferenceRole
    ├── pipeline.py             # Execution orchestration (standard & streaming)
    ├── audio/                  # Audio I/O, player, audit sidecar, masking
    ├── engines/                # KL-NMF & VoxLock/DeepFilter engines
    └── tui/                    # Screens, themes, keys, file picker
```

---

## 3. Environment & Execution Protocol

### 3.1 Python Virtual Environment
- **Python Version**: 3.10+
- **Key Test Libraries**: `pytest`, `soundfile`, `soxr`, `numpy`, `rich`, `silero-vad`, `noisereduce`
- **Optional (required for CLEAN mode tests)**: `voxlock` (no public install path — see project docs)

### 3.2 Primary Test Execution Commands

```bash
# Run unit tests + E2E suite (recommended)
pytest tests/test_zout.py tests/test_e2e.py -v

# Run only the E2E suite
pytest tests/test_e2e.py -v

# Check which runtime dependencies are available
zout --check-deps
```

---

## 4. 4-Tier Test Suite Architecture

The E2E test suite in `tests/test_e2e.py` is organized into four distinct tiers:

### Tier 1: Feature Coverage (40 Tests, >=5 per feature R1–R8)
Exercises the primary happy path and interface contracts for every requirement:
- **R1. Chunked Streaming**: Validates output duration preservation, raised-cosine crossfade continuity ($w_{in} + w_{out} \equiv 1.0$), memory capping (< 150MB), and noisy/clean engine chunking parameters.
- **R2. Instant Seamless A/B Hot-Swap**: Validates dual-track initialization in `AudioPlayer.play_pair`, `toggle_ab()` sample index preservation, rapid cycling, status display, and audio buffer callback accuracy.
- **R3. Live VU Meter & Waveform Envelope**: Validates `compute_waveform_envelope` 8-level Unicode glyph output (` ▂▃▄▅▆▇█`), dynamic amplitude tracking, silence detection, and `get_meter_levels()` dBFS output.
- **R4. In-Flight Cancellation**: Validates `cancel_event` / `cancel_token` aborts, partial file cleanup, concurrent thread interruption, interface signatures, and streaming loop cancellation.
- **R5. Timeline Range Slicing**: Validates `parse_time_spec` formatting (seconds, mm:ss, hh:mm:ss), invalid timestamp rejection, `load_audio` frame-accurate slicing, and CLI `--start` / `--end` flags.
- **R6. Forensic Audit Log**: Validates `.json` sidecar generation, schema completeness (dB reduction, true-peak, RMS, VAD intervals, parameters), 4x oversampled true-peak, and CLI `--audit`/`--json` flags.
- **R7. Terminal Color Themes**: Validates existence of all 4 themes (`Forensic Cyan`, `Amber Phosphor`, `Matrix Emerald`, `Studio Dark`), `cycle_theme()` wrapping, `get_theme()` lookups, palette attributes, and state persistence.
- **R8. Batch Folder Processing**: Validates CLI `--batch` and `--out-dir` arguments, multi-file discovery, skipping non-audio files, default `processed/` output directory, and error resilience.

### Tier 2: Boundary & Corner Cases (40 Tests, >=5 per feature R1–R8)
Exercises edge cases, resource boundaries, and extreme input combinations:
- **R1 Boundaries**: Audio shorter than chunk size, exact chunk boundary multiples, zero overlap duration, pure digital silence streaming, and clipped square wave inputs.
- **R2 Boundaries**: Non-existent audio paths, `toggle_ab()` when idle, mismatched track lengths, immediate `stop()` after `play_pair()`, and sub-second audio clips.
- **R3 Boundaries**: Zero-length audio envelope, extreme widths (1 and 128), constant non-zero DC offsets, -120 dBFS attenuation floor, and independent stereo channel levels.
- **R4 Boundaries**: Pre-set cancel tokens (< 100ms exit), multiple idempotent cancel signals, cancel when idle, temporary buffer cleanup, and input file preservation.
- **R5 Boundaries**: `start == end` (0-duration), `start > end`, `start > duration`, sub-second millisecond timestamps, and slice starting at zero.
- **R6 Boundaries**: Audit on pure digital silence (no NaN/inf), pure Gaussian noise (empty VAD timestamps), custom `audit_path` overrides, uninterrupted speech, and RFC 8259 syntax validation.
- **R7 Boundaries**: Invalid theme fallback to default, case-insensitive lookups, `None` argument handling, 100x rapid theme cycles, and Rich `Style.parse` compatibility.
- **R8 Boundaries**: Empty batch directory (0 files processed), non-existent batch folder rejection, single file passed as batch folder, automatic creation of missing output directories, and mixed file format discovery (.wav, .flac, .ogg).

### Tier 3: Cross-Feature Combinations (8 Tests)
Tests pairwise interactions across feature boundaries:
1. **Range Slicing + Chunked Streaming**: Slicing a target range from a multi-minute recording, processed through chunked streaming.
2. **Range Slicing + Audit Sidecar**: Range-sliced processing documenting sliced parameters in the audit manifest.
3. **Batch Processing + Audit Sidecar**: Batch folder execution generating individual audit JSON manifests for every file in `--out-dir`.
4. **A/B Hot-Swap + VU Meters**: Audio playback toggling between A and B while VU meters actively measure each track's level.
5. **In-Flight Cancellation + Chunked Streaming**: Cancellation triggered while the pipeline is midway through chunk processing.
6. **Batch Processing + Range Slicing**: Uniform time-bracket slicing applied across all audio files in a batch directory.
7. **Auto-Noise Extraction + Audit Sidecar**: Noise profile extraction (`--auto-noise`) combined with audit manifest generation.
8. **Clean Voice Mode + Audit Sidecar**: Clean voice isolation mode generating a forensic audit log with speaker profile metadata.

### Tier 4: Real-World Scenarios (5 Tests)
Simulates end-to-end production scenarios:
1. **Multi-Minute Interview Cleanup**: Long speech recording with continuous HVAC hum cleaned using NOISY profile with peak limiting.
2. **Noisy Lecture Isolation**: Lecture hall recording with competing murmur and background HVAC cleaned using CLEAN speaker enrollment.
3. **Studio Dark Batch Processing**: Multi-track mastering session processing vocal, guitar, and bass stems into a target output directory.
4. **Forensic Audit Examination**: Wiretap audio enhanced with timeline slicing and verified against a forensic audit manifest.
5. **Seamless A/B Evaluation**: Quality assurance evaluation with uninterrupted playback, playhead sample-continuity verification, and stereo VU metering.

---

## 5. Synthetic Fixtures & Mathematical Verification

### 5.1 Deterministic Signal Generators
All tests in `tests/test_e2e.py` use local synthetic generators:
- `gen_sine(duration, freq, sr, amp)`: Generates pure sinusoidal test tones.
- `gen_speechlike(duration, sr, seed, f0)`: Generates harmonic signals with formant modulation mimicking voiced human speech.
- `gen_hum(duration, sr, seed)`: Generates 60Hz and 180Hz harmonic hum with Gaussian noise mimicking electrical/HVAC interference.
- `gen_white_noise(duration, sr, seed, amp)`: Generates random Gaussian noise for SNR testing.

### 5.2 Raised-Cosine Crossfade Verification
For R1 chunk streaming overlap-add:
$$\theta_n = \frac{\pi (n + 0.5)}{N_{overlap}}, \quad n \in [0, N_{overlap}-1]$$
$$w_{in}[n] = 0.5 \times (1 - \cos(\theta_n)), \quad w_{out}[n] = 0.5 \times (1 + \cos(\theta_n))$$
The test asserts that $w_{in}[n] + w_{out}[n] \equiv 1.0 \pm 10^{-6}$ and edge derivatives equal zero.

### 5.3 ITU-R BS.1770 True-Peak Verification
For R6 forensic audit logging, true peak levels are verified using 4x oversampling via `soxr.resample(x, sr, sr * 4, quality="VHQ")`:
$$TruePeak_{dBFS} = 20 \log_{10}\left(\max |x_{4x}| + 10^{-9}\right)$$

---

## 6. Headless Audio Simulation (`MockOutputStream`)

In headless environments without physical sound output devices, `MockOutputStream` intercepts `sounddevice.OutputStream`. It captures the audio callback function, allows synchronous pumping of audio frames via `mock_stream.pump(n_frames)`, and computes RMS and Peak levels in the callback thread without requiring PortAudio hardware.
