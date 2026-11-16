from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .config import SUPPORTED_AUDIO_EXTENSIONS, JobConfig, ReferenceRole, collect_deps_status


def _cli(args) -> int:
    from rich.progress import Progress

    from .audio.io import load_audio, parse_time_spec, save_audio
    from .audio.noise_extractor import extract_noise_profile
    from .pipeline import run_job

    console = Console()

    ref_path = Path(args.ref) if args.ref else None
    role = ReferenceRole(args.role.upper()) if args.role else ReferenceRole.NOISY
    main_path = Path(args.main) if args.main else None
    batch_dir = Path(args.batch) if getattr(args, "batch", None) else None
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None

    if args.auto_noise and main_path:
        size_mb = main_path.stat().st_size / 1_048_576
        if not args.quiet:
            with Progress(console=console) as _pg:
                _task = _pg.add_task(f"Loading audio ({size_mb:.0f} MB)...", total=100)
                main_audio, sr = load_audio(main_path, mono=True)
                dur_s = len(main_audio) / sr
                _pg.update(_task, completed=8,
                           description=f"Scanning {dur_s / 60:.1f} min for speech-free noise...")
                noise_res = extract_noise_profile(
                    main_audio,
                    sr=sr,
                    n_chunks=args.noise_chunks,
                    chunk_sec=args.noise_len,
                    mode=args.noise_mode,
                    progress=lambda msg, f: _pg.update(_task, completed=8 + f * 92, description=msg),
                )
        else:
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

    start_sec = parse_time_spec(args.start) if getattr(args, "start", None) else None
    end_sec = parse_time_spec(args.end) if getattr(args, "end", None) else None
    audit = getattr(args, "audit", False)

    cfg = JobConfig(
        reference=ref_path,
        role=role,
        main=main_path,
        strength=args.strength,
        max_atten_db=args.atten_db,
        output=Path(args.out) if args.out else None,
        polish=None if args.polish == "auto" else args.polish == "on",
        start_sec=start_sec,
        end_sec=end_sec,
        chunk_duration=getattr(args, "chunk_duration", 60.0),
        overlap_duration=getattr(args, "overlap_duration", 2.0),
        audit=audit,
        batch_dir=batch_dir,
        out_dir=out_dir,
    )

    if args.quiet:
        res = run_job(cfg)
        if res is not None:
            print(res.output)
            return 0
        return 1

    with Progress(console=console) as pg:
        task = pg.add_task("zout", total=100)
        res = run_job(cfg, lambda m, f: pg.update(task, completed=f * 100, description=m))

    if res is None:
        console.print("[bold yellow]Job cancelled.[/bold yellow]")
        return 1

    if getattr(res, "notes", {}).get("batch"):
        success = res.notes.get("success", 0)
        total = res.notes.get("total", 0)
        failures = res.notes.get("failures", [])
        console.print(f"[green]Batch processing complete:[/] {success}/{total} files processed into {res.output}")
        for f in failures:
            console.print(f"[yellow]  failed:[/] {f}")
    else:
        console.print(f"[green]Wrote[/] {res.output}  ({res.seconds:.1f}s, noise floor -{res.suppression_db:.1f} dB)")
    if res.notes.get("warning"):
        console.print(f"[yellow]warning:[/] {res.notes['warning']}")
    if res.audit_path and res.audit_path.is_file():
        console.print(f"[cyan]Forensic audit manifest saved:[/] {res.audit_path}")
    return 0


def _interactive() -> int:
    import shutil

    from rich.live import Live

    from .audio.player import player
    from .tui import file_picker, screens, themes

    console = Console()
    cfg = JobConfig()
    # Preflight: warn if ffmpeg is missing (needed for MP3/M4A/AAC)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        msg = "ffmpeg not found — MP3/M4A/AAC files will fail to load. Run 'zout --check-deps' for details."
    else:
        msg = ""
    last_dir: Path | None = None
    try:
        while True:
            cur_th = themes.get_theme()
            console.clear()
            if player.is_playing():
                k = None
                with Live(screens.dashboard(cfg, msg, theme=cur_th),
                          console=console, refresh_per_second=12, transient=True) as live:
                    msg = ""
                    while player.is_playing():
                        live.update(screens.dashboard(cfg, msg, theme=themes.get_theme()))
                        k = screens.poll_key(timeout=0.08)
                        if k is not None:
                            break
                if k is None:
                    console.clear()
                    console.print(screens.dashboard(cfg, msg, theme=themes.get_theme()))
                    k = screens.read_key()
            else:
                console.print(screens.dashboard(cfg, msg, theme=cur_th))
                msg = ""
                k = screens.read_key()

            if k is None:
                continue
            if k.name == "space":
                if player.is_playing():
                    player.stop()
                    msg = "Playback stopped."
                continue
            if k.name != "char":
                continue
            c = k.char
            cur_th = themes.get_theme()
            if c.lower() == "q":
                return 0
            elif c == "1":
                player.stop()
                p = file_picker.pick_file(console, "Select REFERENCE audio file", last_dir, theme=cur_th)
                if p:
                    last_dir = p.parent
                    cfg.reference = p
                    screens.prompt_role(console, cfg, theme=cur_th)
            elif c.lower() == "m":
                cfg.role = ReferenceRole.CLEAN if cfg.role is ReferenceRole.NOISY else ReferenceRole.NOISY
            elif c == "2":
                player.stop()
                p = file_picker.pick_file(console, "Select MAIN recording", last_dir, theme=cur_th)
                if p:
                    last_dir = p.parent
                    cfg.main = p
                    cfg.batch_dir = None
                    cfg.output = None
                    screens.prompt_params(console, cfg, theme=cur_th)
            elif c.lower() == "t":
                player.stop()
                screens.prompt_timeline_range(console, cfg, theme=cur_th)
            elif c.lower() == "a":
                cfg.audit = not cfg.audit
                msg = f"Forensic audit log {'ENABLED (.json manifest)' if cfg.audit else 'DISABLED'}"
            elif c.lower() == "b":
                player.stop()
                d = file_picker.pick_directory(console, "Select BATCH directory of audio files", last_dir, theme=cur_th)
                if d:
                    last_dir = d
                    cfg.batch_dir = d
                    cfg.main = None
                    cfg.output = None
                    msg = f"Batch directory selected: {d.name}"
            elif c.lower() == "c":
                new_theme = themes.cycle_theme()
                msg = f"Theme switched to: {new_theme.name}"
            elif c.lower() == "n":
                player.stop()
                if not cfg.main or not cfg.main.is_file():
                    msg = "Pick main recording [2] first before auto-extracting noise."
                else:
                    out = screens.prompt_auto_noise(console, cfg, theme=cur_th)
                    if out:
                        msg = f"Auto-extracted noise profile loaded from {out.name}"
            elif c in "34":
                screens.prompt_params(console, cfg, theme=cur_th)
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
                    msg = "No reference noise audio available yet. Pick one [1] or auto-extract [N]."
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
            elif c.lower() == "o":
                player.stop()
                screens.prompt_output_path(console, cfg, theme=cur_th)
            elif c.lower() == "d":
                player.stop()
                screens.prompt_deps_check(console, theme=cur_th)
            elif c.lower() == "r":
                player.stop()
                if not cfg.ready():
                    msg = "Select both a reference (or Auto-extract N) and a main recording (2) or batch directory (B) first."
                    continue
                console.clear()
                res = screens.run_with_progress(console, cfg, theme=cur_th)
                if res is None:
                    # Clean return upon [Esc] cancellation without crashing
                    msg = "[bold yellow]Job cancelled by user.[/bold yellow]"
                    continue
                elif isinstance(res, Exception):
                    console.clear()
                    console.print(screens.result_card(res, theme=cur_th))
                    while True:
                        rk = screens.read_key()
                        if rk and rk.name in ("enter", "esc"):
                            break
                    continue
                else:
                    # Auto-start A/B hot-swap playback with processed track B (single-file jobs only)
                    has_pair = bool(
                        not res.notes.get("batch")
                        and cfg.main and cfg.main.is_file()
                        and res.output and res.output.is_file()
                    )
                    if has_pair:
                        player.play_pair(cfg.main, res.output, start_with="B")

                    with Live(screens.result_card(res, player.status_text(), theme=cur_th),
                              console=console, refresh_per_second=15) as live:
                        while True:
                            active_th = themes.get_theme()
                            live.update(screens.result_card(res, player.status_text(), theme=active_th))
                            rk = screens.poll_key(timeout=0.08)
                            if rk is None:
                                continue
                            if rk.name == "tab" and has_pair and player.is_playing():
                                player.toggle_ab()
                            elif rk.name == "space" or (rk.name == "char" and rk.char == " "):
                                if player.is_playing():
                                    player.stop()
                                elif has_pair:
                                    player.play_pair(cfg.main, res.output, start_with="B")
                            elif rk.name == "char" and rk.char == "P":
                                if has_pair:
                                    if player.is_playing():
                                        if player.get_current_track() == "B":
                                            player.stop()
                                        else:
                                            player.toggle_ab()
                                    else:
                                        player.play_pair(cfg.main, res.output, start_with="B")
                                elif res.output and res.output.is_file():
                                    if player.is_playing():
                                        player.stop()
                                    else:
                                        player.play(res.output, "Processed Output")
                            elif rk.name == "char" and rk.char == "p":
                                if has_pair:
                                    if player.is_playing():
                                        if player.get_current_track() == "A":
                                            player.stop()
                                        else:
                                            player.toggle_ab()
                                    else:
                                        player.play_pair(cfg.main, res.output, start_with="A")
                                elif cfg.main and cfg.main.is_file():
                                    if player.is_playing():
                                        player.stop()
                                    else:
                                        player.play(cfg.main, "Main Recording")
                            elif rk.name == "char" and rk.char.lower() == "c":
                                themes.cycle_theme()
                            elif rk.name in ("enter", "esc"):
                                player.stop()
                                break
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
               "  zout -r noise.wav -m speech.wav -t noisy --start 00:30 --end 02:00 --audit\n"
               "  zout -r noise.wav --batch ./raw_recordings/ --out-dir ./processed/\n"
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
    ap.add_argument("--chunk-duration", type=float, default=60.0, metavar="SEC",
                    help="streaming chunk duration in seconds (default 60.0)")
    ap.add_argument("--overlap-duration", type=float, default=2.0, metavar="SEC",
                    help="streaming raised-cosine overlap duration in seconds (default 2.0)")
    ap.add_argument("-o", "--out", metavar="FILE", help="output file (default: <main>_zout.wav)")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress bar; print only the output path")
    ap.add_argument("-V", "--version", action="version", version=f"zout {__version__}")
    ap.add_argument("--check-deps", action="store_true",
                    help="report which optional runtime features are available and exit")

    # Timeline range slicing options (R5)
    ts_group = ap.add_argument_group("timeline range slicing options")
    ts_group.add_argument("--start", metavar="TIME", dest="start",
                          help="start offset in seconds or mm:ss / hh:mm:ss")
    ts_group.add_argument("--end", metavar="TIME", dest="end",
                          help="end offset in seconds or mm:ss / hh:mm:ss")

    # Forensic audit sidecar (R6)
    ap.add_argument("--audit", "--json", action="store_true", dest="audit",
                    help="generate forensic JSON audit manifest sidecar alongside output audio")

    # Batch folder processing options (R8)
    batch_group = ap.add_argument_group("batch folder processing options")
    batch_group.add_argument("-b", "--batch", metavar="DIR", dest="batch",
                             help="process an entire directory of audio files against reference profile")
    batch_group.add_argument("--out-dir", metavar="DIR", dest="out_dir",
                             help="destination directory for processed files (default: <batch>/processed/)")

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
    from .audio.io import parse_time_spec

    if args.auto_noise and not args.main:
        ap.error("--auto-noise requires --main <FILE>")

    if getattr(args, "batch", None):
        if args.main:
            ap.error("--batch and --main are mutually exclusive")
        bp = Path(args.batch)
        if not bp.exists():
            ap.error(f"--batch: directory not found: {args.batch}")
        if not bp.is_dir():
            ap.error(f"--batch: must be a directory: {args.batch}")
        if not args.ref:
            ap.error("--batch requires --ref <FILE> and --role <noisy|clean>")

    for flag in ("ref", "main"):
        p = getattr(args, flag, None)
        if p and not Path(p).is_file():
            ap.error(f"--{flag}: file not found: {p}")
        if p and Path(p).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            ap.error(f"--{flag}: unsupported audio type '{Path(p).suffix}' "
                     f"(supported: {' '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))})")

    if getattr(args, "start", None) is not None:
        try:
            s = parse_time_spec(args.start)
            if s is not None and s < 0:
                ap.error("--start cannot be negative")
        except Exception as ex:
            ap.error(f"invalid --start value '{args.start}': {ex}")

    if getattr(args, "end", None) is not None:
        try:
            e = parse_time_spec(args.end)
            if e is not None and e <= 0:
                ap.error("--end must be positive")
        except Exception as ex:
            ap.error(f"invalid --end value '{args.end}': {ex}")

    if getattr(args, "start", None) is not None and getattr(args, "end", None) is not None:
        try:
            s = parse_time_spec(args.start)
            e = parse_time_spec(args.end)
            if s is not None and e is not None and s >= e:
                ap.error(f"--start ({s}s) must be strictly less than --end ({e}s)")
        except Exception:
            pass

    if not 0.0 <= args.strength <= 1.0:
        ap.error("--strength must be between 0 and 1")
    if not 6.0 <= args.atten_db:
        ap.error("--atten-db must be >= 6")
    if getattr(args, "chunk_duration", 60.0) <= 0:
        ap.error("--chunk-duration must be positive")
    if getattr(args, "overlap_duration", 2.0) < 0:
        ap.error("--overlap-duration cannot be negative")
    if getattr(args, "overlap_duration", 2.0) >= getattr(args, "chunk_duration", 60.0):
        ap.error("--overlap-duration must be strictly less than --chunk-duration")


def _check_deps() -> int:
    """Print a feature-availability table and exit."""
    console = Console()
    console.print("[bold]noiZout runtime dependency check[/]")
    last_section = None
    for section, label, status, detail in collect_deps_status():
        if section != last_section:
            console.print(f"\n[bold cyan]{section}[/]")
            last_section = section
        color = "green" if status == "OK" else ("yellow" if status == "SLIM" else "red")
        tag = f"[{color}]{status:^9}[/]"
        hint = f"  [dim]{detail}[/]" if detail else ""
        console.print(f"  {tag}  {label}{hint}")
    console.print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    if getattr(args, "check_deps", False):
        return _check_deps()

    if getattr(args, "batch", None):
        if not args.ref or not args.role:
            ap.error("--batch requires --ref and --role")
        _validate(ap, args)
        return _cli(args)

    if args.auto_noise and args.main:
        _validate(ap, args)
        return _cli(args)

    given = [args.ref, args.main, args.role]
    if any(given) and not all(given):
        ap.error("--ref, --main and --role must be given together (or use --auto-noise with --main, or --batch with --ref and --role)")
    _validate(ap, args)
    try:
        if all(given):
            return _cli(args)
        if not sys.stdin.isatty():
            ap.error("non-interactive use requires --ref, --main and --role (or --auto-noise with --main, or --batch with --ref and --role)")
        return _interactive()
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"zout: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
