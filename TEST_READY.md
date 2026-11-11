# Test Readiness Report (`TEST_READY.md`)

## Executive Summary

The comprehensive, requirement-driven, opaque-box End-to-End (E2E) test suite for **noiZout** is
implemented and verified. The test suite is located in `tests/test_e2e.py` alongside the existing
unit regression test suite `tests/test_zout.py`.

All 93 E2E tests across 4 tiers are fully specified with authoritative expected outputs, zero facade
tests, deterministic synthetic fixtures, and progressive testability markers.

---

## Current Test Status

```
tests/test_zout.py  :  29 passed,  3 failed  (voxlock blocked)
tests/test_e2e.py   :  91 passed,  2 failed  (voxlock blocked)
─────────────────────────────────────────────────────────────
Total               : 120 passed,  5 failed  (5/125 = 4% dep-blocked)
```

**All 5 remaining failures require `voxlock` (CLEAN mode / speaker extraction), which has no
public install path. Install voxlock to clear them.**

---

## Test Execution Command

Run the complete test suite:

```bash
pytest tests/test_zout.py tests/test_e2e.py -v
```

Execution characteristics:
- **Total Test Cases**: 125 (32 unit + 93 E2E)
- **Execution Time**: ~2 minutes (noisereduce KL-NMF is compute-intensive)
- **Network Access**: 0 downloads required (100% offline synthetic fixtures)
- **Audio Hardware**: Headless-safe (`MockOutputStream` when audio device absent)

To run only the E2E suite:
```bash
pytest tests/test_e2e.py -v
```

---

## Runtime Dependency Status

| Package           | Feature            | Status   | Install                                |
|-------------------|--------------------|----------|----------------------------------------|
| `noisereduce`     | NOISY mode         | ✅ OK    | `pip install noisereduce`              |
| `silero_vad`      | Forensic audit VAD | ✅ OK    | `pip install silero-vad`               |
| `sounddevice`     | Audio playback     | ✅ OK    | `pip install sounddevice`              |
| `ffmpeg`+`ffprobe`| MP3/M4A/AAC        | ✅ OK    | system package                         |
| `voxlock`         | CLEAN mode         | ❌ MISSING | see project docs (no public path)    |
| `deepfilternet_slim` | Polish stage    | ❌ MISSING | see project docs (no public path)    |

Run `zout --check-deps` to get a live report at any time.

---

## Known Limitations

**`voxlock` (CLEAN mode)** — 5 tests that exercise speaker extraction via VoxLock are blocked
until `voxlock` is installed. These tests are:
- `tests/test_zout.py::test_clean_mode_runs_end_to_end`
- `tests/test_zout.py::test_streaming_clean_48khz_memory_under_75mb`
- `tests/test_zout.py::test_streaming_clean_96khz_memory_under_75mb`
- `tests/test_e2e.py::TestTier3CrossFeatureCombinations::test_r3_combo_08_clean_mode_and_audit_sidecar`
- `tests/test_e2e.py::TestTier4RealWorldScenarios::test_r4_scenario_02_noisy_lecture_isolation`

**`deepfilternet_slim`** — The polish stage is silently skipped (no test failures) because
`DeepFilter` import errors are caught and ignored. DeepFilterNet3 output is not tested.

---

## 4-Tier Test Suite Breakdown

| Tier | Category | Count | Status |
|------|----------|-------|--------|
| **Tier 1** | Feature Coverage (R1–R8) | 40 | 40/40 passing |
| **Tier 2** | Boundary & Corner Cases (R1–R8) | 40 | 40/40 passing |
| **Tier 3** | Cross-Feature Combinations | 8 | 7/8 passing (1 voxlock) |
| **Tier 4** | Real-World Scenarios | 5 | 4/5 passing (1 voxlock) |
| **Total** | **All 4 Tiers** | **93** | **91/93 passing** |

---

## Requirement Coverage Matrix (R1 – R9)

| Req | Description | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|-----|-------------|--------|--------|--------|--------|
| R1 | Chunked Streaming (<150MB, OLA) | 5 tests ✅ | 5 tests ✅ | combo_01, combo_05 ✅ | scenario_01 ✅ |
| R2 | A/B Hot-Swap ([Tab]) | 5 tests ✅ | 5 tests ✅ | combo_04 ✅ | scenario_05 ✅ |
| R3 | VU Meter & Waveform Envelope | 5 tests ✅ | 5 tests ✅ | combo_04 ✅ | scenario_05 ✅ |
| R4 | In-Flight Cancellation ([Esc]) | 5 tests ✅ | 5 tests ✅ | combo_05 ✅ | — |
| R5 | Timeline Range Slicing | 5 tests ✅ | 5 tests ✅ | combo_01,02,06 ✅ | scenario_04 ✅ |
| R6 | Forensic Audit Log (.json) | 5 tests ✅ | 5 tests ✅ | combo_02,03,07 ✅; combo_08 ❌ voxlock | scenario_04 ✅ |
| R7 | Terminal Color Themes | 5 tests ✅ | 5 tests ✅ | — | scenario_03 ✅ |
| R8 | Batch Folder Processing | 5 tests ✅ | 5 tests ✅ | combo_03,06 ✅ | scenario_03 ✅ |
| R9 | Test Integrity | Evaluated across entire suite |
