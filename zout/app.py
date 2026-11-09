from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .config import SUPPORTED_AUDIO_EXTENSIONS, JobConfig, ReferenceRole


def _cli(args) -> int:
    from rich.progress import Progress

    from .audio.io import load_audio, save_audio
    from .audio.noise_extractor import extract_noise_profile
    from .pipeline import run_job

    console = Console()

    ref_path = Path(args.ref) if args.ref else None
    role = ReferenceRole(args.role.upper()) if args.role else ReferenceRole.NOISY
    main_path = Path(args.main)

    if args.auto_noise:
        if not args.quiet:
            console.print(f"[cyan]Auto-extracting noise profile from:[/] {main_path.name}")
        main_audio, sr = load_audio(main_path, mono=True)
        noise_res = extract_noise_profile(
            main_audio,
            sr=sr,
            n_chunks=args.noise_chunks,
            chunk_sec=args.noise_len,
            mode=args.noise_mode,
        )
        save_path = Path(args.save_noise) if args.save_noise else main_path.with_name(main_path.stem + "_noise_profile.wav")
        save_audio(save_path, noise_res.audio, sr)
        ref_path = save_path
        role = ReferenceRole.NOISY
        if not args.quiet:
            console.print(f"[green]Extracted {len(noise_res.chunks)} noise sections ({len(noise_res.audio) / sr:.1f}s)[/] -> {save_path}")
            if noise_res.warnings:
                for w in noise_res.warnings:
                    console.print(f"[yellow]Notice:[/] {w}")

    cfg = JobConfig(ref_path, role, main_path, args.strength,
                    args.atten_db, Path(args.out) if args.out else None,
                    None if args.polish == "auto" else args.polish == "on")
    if args.quiet:
        res = run_job(cfg)
        print(res.output)
        return 0
    with Progress(console=console) as pg:
        task = pg.add_task("zout", total=100)
        res = run_job(cfg, lambda m, f: pg.update(task, completed=f * 100, description=m))
    console.print(f"[green]Wrote[/] {res.output}  ({res.seconds:.1f}s, noise floor -{res.suppression_db:.1f} dB)")
    if res.notes.get("warning"):
        console.print(f"[yellow]warning:[/] {res.notes['warning']}")
    return 0


def _interactive() -> int:
    from .audio.player import player
    from .tui import file_picker, screens

    console = Console()
    cfg = JobConfig()
    msg = ""
    last_dir: Path | None = None
    try:
        while True:
            console.clear()
            console.print(screens.dashboard(cfg, msg))
            msg = ""
            k = screens.read_key()
            if k.name != "char":
                continue
            c = k.char
            if c.lower() == "q":
                return 0
            elif c == "1":
                player.stop()
                p = file_picker.pick_file(console, "Select REFERENCE audio file", last_dir)
                if p:
                    last_dir = p.parent
                    cfg.reference = p
                    screens.prompt_role(console, cfg)
            elif c.lower() == "t":
                cfg.role = ReferenceRole.CLEAN if cfg.role is ReferenceRole.NOISY else ReferenceRole.NOISY
            elif c == "2":
                player.stop()
                p = file_picker.pick_file(console, "Select MAIN recording", last_dir)
                if p:
                    last_dir = p.parent
                    cfg.main = p
                    cfg.output = None
                    screens.prompt_params(console, cfg)
            elif c.lower() == "a":
                player.stop()
                if not cfg.main or not cfg.main.is_file():
                    msg = "Pick main recording [2] first before auto-extracting noise."
                else:
                    out = screens.prompt_auto_noise(console, cfg)
                    if out:
                        msg = f"Auto-extracted noise profile loaded from {out.name}"
            elif c in "34":
                screens.prompt_params(console, cfg)
            elif c.lower() == "f":
                cfg.polish = not cfg.use_polish()
            elif c == "P":  # Shift+P / Capital P: Play noise reference
                if cfg.reference and cfg.reference.is_file():
                    if player.is_playing() and player._target_name == "Noise Reference":
                        player.stop()
                        msg = "Stopped noise playback."
                    else:
                        player.play(cfg.reference, "Noise Reference")
                        msg = f"▶ Playing noise reference: {cfg.reference.name}"
                else:
                    msg = "No reference noise audio available yet. Pick one [1] or auto-extract [A]."
            elif c == "p":  # Lowercase p: Play main subject audio
                if cfg.main and cfg.main.is_file():
                    if player.is_playing() and player._target_name == "Main Recording":
                        player.stop()
                        msg = "Stopped main recording playback."
                    else:
                        player.play(cfg.main, "Main Recording")
                        msg = f"▶ Playing main recording: {cfg.main.name}"
                else:
                    msg = "No main recording selected yet. Pick one [2]."
            elif c in (" ", "s", "S"):
                if player.is_playing():
                    player.stop()
                    msg = "Playback stopped."
            elif c.lower() == "o" and cfg.main:
                player.stop()
                console.print("\nOutput path (blank = default): ", end="")
                s = input().strip().strip('"')
                cfg.output = Path(s) if s else None
            elif c.lower() == "r":
                player.stop()
                if not cfg.ready():
                    msg = "Select both a reference (or Auto-extract A) and a main recording first."
                    continue
                console.clear()
                res = screens.run_with_progress(console, cfg)
                if isinstance(res, screens.Result):
                    while True:
                        console.clear()
                        console.print(screens.result_card(res, player.status_text()))
                        rk = screens.read_key()
                        if rk.name == "char" and rk.char.upper() == "P":
                            if player.is_playing() and player._target_name == "Processed Output":
                                player.stop()
                            else:
                                player.play(res.output, "Processed Output")
                        elif rk.name == "char" and rk.char.lower() == "p":
                            if player.is_playing() and player._target_name == "Main Recording":
                                player.stop()
                            else:
                                player.play(cfg.main, "Main Recording")
                        elif rk.name == "char" and rk.char in (" ", "s", "S"):
                            player.stop()
                        elif rk.name in ("enter", "esc"):
                            player.stop()
                            break
                else:
                    console.clear()
                    console.print(screens.result_card(res))
                    while screens.read_key().name != "enter":
                        pass
    finally:
        player.stop()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="zout",
        description="Reference-conditioned audio denoise / voice restoration. "
                    "With no arguments, starts the interactive menu.",
        epilog="examples:\n"
               "  zout -r room_noise.wav -m interview.mp3 -t noisy -s 0.8\n"
               "  zout -m long_interview.wav --auto-noise -s 0.85\n"
               "  zout -r my_voice.wav -m meeting.wav -t clean --atten-db 40 -o out.wav",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--ref", metavar="FILE", help="reference sample (training file)")
    ap.add_argument("-m", "--main", metavar="FILE", help="main recording to process")
    ap.add_argument("-t", "--role", "--type", choices=["noisy", "clean"],
                    help="is the reference a NOISY sample (remove it) or CLEAN sample (keep it)")
    ap.add_argument("-s", "--strength", type=float, default=0.75, metavar="0-1",
                    help="adaptive threshold / power (default 0.75)")
    ap.add_argument("-a", "--atten-db", type=float, default=24.0, metavar="DB",
                    help="max attenuation cap in dB, 6-60 (>=60 = uncapped, default 24)")
    ap.add_argument("--polish", choices=["auto", "on", "off"], default="auto",
                    help="DeepFilterNet3 speech polish (auto: on, off)")
    ap.add_argument("-o", "--out", metavar="FILE", help="output file (default: <main>_zout.wav)")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress bar; print only the output path")
    ap.add_argument("-V", "--version", action="version", version=f"zout {__version__}")

    # Auto-noise extraction options
    ag = ap.add_argument_group("auto-noise extraction options")
    ag.add_argument("--auto-noise", action="store_true",
                    help="automatically scan --main recording for speech-free noise chunks and use as reference")
    ag.add_argument("--noise-chunks", type=int, default=6, metavar="N",
                    help="number of noise chunks to extract (default 6)")
    ag.add_argument("--noise-len", type=float, default=2.0, metavar="SEC",
                    help="duration of each noise chunk in seconds (default 2.0)")
    ag.add_argument("--noise-mode", choices=["balanced", "deep", "fast"], default="balanced",
                    help="noise extraction strategy (default: balanced)")
    ag.add_argument("--save-noise", metavar="FILE",
                    help="destination path for auto-extracted noise profile WAV")
    return ap


def _validate(ap: argparse.ArgumentParser, args) -> None:
    if args.auto_noise and not args.main:
        ap.error("--auto-noise requires --main <FILE>")

    for flag in ("ref", "main"):
        p = getattr(args, flag)
        if p and not Path(p).is_file():
            ap.error(f"--{flag}: file not found: {p}")
        if p and Path(p).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            ap.error(f"--{flag}: unsupported audio type '{Path(p).suffix}' "
                     f"(supported: {' '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))})")
    if not 0.0 <= args.strength <= 1.0:
        ap.error("--strength must be between 0 and 1")
    if not 6.0 <= args.atten_db:
        ap.error("--atten-db must be >= 6")


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    if args.auto_noise and args.main:
        # Auto-noise specifies noise extraction from main
        _validate(ap, args)
        return _cli(args)

    given = [args.ref, args.main, args.role]
    if any(given) and not all(given):
        ap.error("--ref, --main and --role must be given together (or use --auto-noise with --main)")
    _validate(ap, args)
    try:
        if all(given):
            return _cli(args)
        if not sys.stdin.isatty():
            ap.error("non-interactive use requires --ref, --main and --role (or --auto-noise with --main)")
        return _interactive()
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"zout: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
