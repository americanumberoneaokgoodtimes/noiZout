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
- FFmpeg (recommended for reading compressed audio formats like `.mp3`, `.m4a`, etc.)

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

# Install dependencies
pip install -e .
```

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
| `[T]` | Toggle Reference Role (`NOISY` / `CLEAN`) |
| `[2]` | Select Main Audio Recording (File Picker) |
| `[A]` | Auto-extract room noise profile from main recording |
| `[3]` / `[4]` | Adjust Strength & Max Attenuation Cap |
| `[F]` | Toggle DeepFilterNet3 Polish (`ON` / `OFF`) |
| `[O]` | Set custom output path |
| `[P]` | Play reference noise / Processed output |
| `[p]` | Play main subject recording |
| `[Space]` | Stop playback |
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

#### CLI Flags
- `-r, --ref FILE`: Reference audio sample (noise profile or clean voice).
- `-m, --main FILE`: Main recording to differentiate.
- `-t, --role [noisy|clean]`: Classify reference as `noisy` (remove it) or `clean` (keep it).
- `-s, --strength FLOAT`: Inference strength / adaptive threshold (0.0 to 1.0, default 0.75).
- `-a, --atten-db DB`: Max attenuation cap in dB (6 to 60 dB, $\ge 60$ uncapped, default 24).
- `--auto-noise`: Automatically scan main recording for speech-free room noise chunks.
- `--noise-chunks N`: Number of noise sections to find (default 6).
- `--noise-len SEC`: Duration of each section in seconds (default 2.0).
- `--noise-mode [balanced|deep|fast]`: Extraction sensitivity strategy.
- `--save-noise FILE`: Save extracted noise profile to disk.
- `-o, --out FILE`: Output audio file path (defaults to `<main>_zout.wav`).
- `-q, --quiet`: Quiet execution (suppress progress bar; prints only output path).

---

## Architecture

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

## Testing

Run the test suite with pytest:

```bash
pytest -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
