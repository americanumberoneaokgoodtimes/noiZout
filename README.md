# noiZout

> **Reference-Conditioned Audio Differentiation, Speech De-Noising, and Voice Restoration**

noiZout (`zout`) is a terminal-based audio differentiation engine built with Python and **Rich** (strictly zero Textual dependency). Given a reference audio sample, it conditions its machine-learning and DSP pipeline to either eliminate matching acoustic noise patterns or isolate and restore a specific speaker's vocal harmonics.

---

## Features

- **NOISY Reference Mode**: Learn room noise, HVAC hum, mic gain hiss, or non-stationary noise patterns from a reference sample and subtract them from the main recording using supervised KL-NMF and adaptive spectral gating.
- **CLEAN Reference Mode**: Conditioned on even a short (1-3 second) clean utterance, isolats and restores the target speaker's voice using VoxLock Target-Speaker Extraction (TSE), retaining natural vocal harmonics while rejecting background noise and competing voices.
- **Automatic Room Noise Extraction (`[A]`)**: Automatically scans recordings for candidate speech-free room noise sections with lowest SNR. Rigorously rejects voice fragments using Silero VAD v5 ($\le 0.06$ probability threshold) and crossfades sections into a pure reference profile.
- **In-Terminal Audio Playback**: Listen to audio directly from the terminal without leaving the app:
  - `[P]` — Play noise reference / processed result
  - `[p]` — Play original main recording
  - `[Space]` — Stop playback
- **Interactive Terminal UI**: Menu-driven interface featuring a built-in audio-only file picker with drive switching (`D`) and `/` search, real-time visual parameter steppers, and live progress bars.
- **Scriptable Headless CLI**: Full command-line interface for headless execution and automated batch pipelines.
- **CPU Optimized**: Runs on ONNX Runtime INT8 and native C DLL runtime (`libDF`). No CUDA GPU or PyTorch install required.

---

## Installation

### Requirements
- Python 3.10+
- FFmpeg (required for MP3/M4A/AAC input)

```bash
# Clone the repository
git clone https://github.com/americanumberoneaokgoodtimes/noiZout.git
cd noiZout

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install core dependencies
pip install -e .
```

### Runtime Feature Dependencies

Three optional packages unlock additional processing modes. Run `zout --check-deps` at any time
to see which are available.

| Feature | Package | Install |
|---------|---------|---------|
| NOISY mode (spectral denoising) | `noisereduce` | `pip install noisereduce` |
| CLEAN mode (speaker extraction) | `voxlock` | see notes below |
| DeepFilterNet3 polish stage | `deepfilternet-slim` | see notes below |

**`noisereduce`** is available on PyPI and is the most commonly used mode.

**`voxlock`** is a proprietary Target-Speaker Extraction runtime. Contact the project maintainer
for distribution. Without it, all CLEAN mode operations raise an `ImportError` with a clear message.

**`deepfilternet-slim`** is an optional C runtime for the DeepFilterNet3 speech-enhancement polish
pass. If absent, the polish stage is silently skipped — NOISY and CLEAN processing still run
normally, only the final polish step is omitted.

### FFmpeg

FFmpeg is required for reading compressed audio formats (MP3, M4A, AAC, OGG Vorbis, etc.).
WAV and FLAC work without FFmpeg.

- **Windows**: `winget install ffmpeg` or download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `apt install ffmpeg` / `dnf install ffmpeg`

Verify installation: `zout --check-deps`

---

## Usage

### 1. Interactive Terminal UI
Launch the menu-driven dashboard by running `zout` without arguments:

```bash
zout
```

#### Hotkey Navigation
| Key | Action |
| :--- | :--- |
| `[1]` | Select Reference Audio File (File Picker) |
| `[M]` | Toggle Reference Role (`NOISY` ↔ `CLEAN`) |
| `[2]` | Select Main Audio Recording (File Picker) |
| `[B]` | Select Batch Directory (process a whole folder) |
| `[N]` | Auto-extract room noise profile from main recording |
| `[T]` | Set timeline range (`--start` / `--end`) |
| `[3]` / `[4]` | Adjust Strength & Max Attenuation Cap |
| `[F]` | Toggle DeepFilterNet3 Polish (`ON` / `OFF`) |
| `[A]` | Toggle Forensic Audit sidecar (`.json` manifest) |
| `[O]` | Set custom output path |
| `[P]` | Play reference noise / Processed output |
| `[p]` | Play main subject recording |
| `[Space]` | Stop playback |
| `[Tab]` | A/B hot-swap during result playback |
| `[C]` | Cycle color theme |
| `[R]` | Run processing pipeline |
| `[Q]` | Quit |

---

### 2. Command-Line Interface (CLI)

#### Noise Differentiation (Remove Matching Noise)
```bash
# Remove noise using a separate noise reference sample
zout -r noise_profile.wav -m interview.mp3 -t noisy -s 0.8 -a 24 -o cleaned.wav

# Automatically scan main audio for room noise and denoise in one shot
zout -m recording.wav --auto-noise --noise-chunks 6 --noise-len 2.0 -s 0.85
```

#### Voice Restoration (Target-Speaker Extraction)
```bash
# Isolate voice matching the clean reference sample
zout -r clean_voice_sample.wav -m noisy_meeting.wav -t clean -s 0.85 -a 30 -o isolated_voice.wav
```

#### Timeline Range Slicing
```bash
# Process only the segment from 0:30 to 2:00
zout -r noise.wav -m speech.wav -t noisy --start 00:30 --end 02:00

# Accepts seconds (30.5), mm:ss (00:30), or hh:mm:ss (01:30:00)
```

#### Forensic Audit Sidecar
```bash
# Generate a JSON audit manifest alongside the output
zout -r noise.wav -m recording.wav -t noisy --audit -o cleaned.wav
# Writes cleaned_audit.json with true-peak, RMS, VAD timestamps, parameters
```

#### Batch Processing
```bash
# Process every audio file in a directory
zout -r noise_profile.wav --batch ./raw_recordings/ --out-dir ./processed/ --role noisy

# Output files are named <original_name>_zout.wav
```

#### Dependency Check
```bash
# See which optional features are available
zout --check-deps
```

#### CLI Flags

| Flag | Description |
|------|-------------|
| `-r, --ref FILE` | Reference audio (noise profile or clean voice) |
| `-m, --main FILE` | Main recording to process |
| `-t, --role [noisy\|clean]` | `noisy` = remove it; `clean` = keep it |
| `-s, --strength 0-1` | Processing strength (default 0.75) |
| `-a, --atten-db DB` | Max attenuation cap in dB (default 24; ≥60 = uncapped) |
| `--polish [auto\|on\|off]` | DeepFilterNet3 speech polish (requires deepfilternet-slim) |
| `-o, --out FILE` | Output path (default: `<main>_zout.wav`) |
| `-q, --quiet` | No progress bar; prints only the output path |
| `--start TIME` | Start offset: seconds, `mm:ss`, or `hh:mm:ss` |
| `--end TIME` | End offset: seconds, `mm:ss`, or `hh:mm:ss` |
| `--audit` | Generate forensic JSON audit manifest sidecar |
| `-b, --batch DIR` | Process entire directory of audio files |
| `--out-dir DIR` | Output directory for batch results (default: `<batch>/processed/`) |
| `--auto-noise` | Scan `--main` for speech-free noise sections and use as reference |
| `--noise-chunks N` | Number of noise sections to extract (default 6) |
| `--noise-len SEC` | Duration of each noise section (default 2.0s) |
| `--noise-mode` | `balanced` / `deep` / `fast` extraction strategy |
| `--save-noise FILE` | Save extracted noise profile to disk |
| `--check-deps` | Report which runtime features are available and exit |

---

## CLEAN Mode — Speaker Extraction Guide

CLEAN mode uses VoxLock Target-Speaker Extraction (TSE) to isolate one specific voice from a
multi-speaker recording. It requires the `voxlock` package.

### How It Works

1. **Enrollment**: A short reference clip of the target speaker's voice is processed to build
   a speaker embedding (voice fingerprint).
2. **Extraction**: The main recording is scanned and only audio that matches the speaker
   fingerprint is preserved. All other voices, background noise, and room reflections are
   suppressed.
3. **Fullband Restoration**: VoxLock operates in the 0–8 kHz band. Frequencies above 8 kHz
   are retained from the original and gated by the speech-presence ratio in the extracted band,
   preserving natural air and sibilance.

### Reference Clip Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Duration | 1 second | 3–10 seconds |
| Voiced speech | 0.5 seconds | ≥ 2 seconds |
| Environment | Any | Quiet room, same mic as main recording |
| Format | Any supported audio | WAV/FLAC 16kHz–48kHz |

**Tips for best results:**
- The reference clip should contain only the target speaker — no background music, no other voices.
- Longer references with varied pitch and rhythm improve speaker fingerprint accuracy.
- If `zout` warns `"only X.XXs of voice found in reference"`, the clip has too much silence or
  noise. Trim to the voiced section.
- The same microphone and room as the main recording generally gives better results, but is
  not required.

### Example

```bash
# Conference call: isolate Alice's voice (you have a clean clip of Alice speaking)
zout -r alice_reference.wav -m full_meeting.wav -t clean -s 0.80 -a 30 -o alice_only.wav

# Short enrollment clip is fine — 3 seconds of clear speech is enough
zout -r alice_3sec.wav -m two_hour_recording.wav -t clean -s 0.75 -o alice_isolated.wav
```

### Strength and Attenuation Settings

| Setting | Recommended range | Effect |
|---------|------------------|--------|
| `--strength` | 0.65–0.85 | Higher = more aggressive speaker separation; may cut quiet speech |
| `--atten-db` | 24–40 dB | How much to attenuate rejected content; 24 dB is a natural-sounding floor |

---

## Testing

```bash
# Run the full test suite (requires noisereduce)
pytest tests/test_zout.py tests/test_e2e.py -v

# Check which features are available before running
zout --check-deps
```

Current test status: **120/125 passing** (5 blocked by voxlock dependency).

---

```text
                  +-------------------------------+
                  |       Input Main Audio        |
                  +---------------+---------------+
                                  |
               +------------------+------------------+
               |                                     |
    [ NOISY Reference Mode ]              [ CLEAN Reference Mode ]
               |                                     |
      Train KL-NMF Noise Bases             Silero VAD Enrollment Tiling
               |                                     |
    Spectral Gating (noisereduce)         Broadband Pre-Filter (DFNet3)
               |                                     |
       Supervised NMF Split                  VoxLock TSE Inference
               |                                     |
     DeepFilterNet3 Polish                 Harmonic High-Band (>8kHz)
               |                                     |
               +------------------+------------------+
                                  |
                     Adaptive Leak Mask Blending
                                  |
                       EBU R128 Peak Limiting
                                  |
                        Clean Output Audio
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
