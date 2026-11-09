from __future__ import annotations

from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..audio.io import probe_audio, save_audio
from ..audio.noise_extractor import extract_noise_profile
from ..audio.player import player
from ..config import JobConfig, ReferenceRole
from ..pipeline import Result, run_job
from .keys import read_key

KEY = "bold black on yellow"


def key(k: str) -> Text:
    return Text(f" {k} ", style=KEY)


def bar(value: float, lo: float, hi: float, width: int = 22, color: str = "magenta") -> Text:
    f = (value - lo) / (hi - lo)
    n = int(round(f * width))
    return Text.assemble(("█" * n, color), ("░" * (width - n), "grey37"))


def strength_label(s: float) -> str:
    return "Gentle" if s < 0.45 else "Balanced" if s < 0.65 else "Strong" if s < 0.85 else "Maximum"


def atten_label(a: float) -> str:
    return "Uncapped" if a >= 60 else f"{a:.0f} dB"


def badge(role: ReferenceRole) -> Text:
    return (Text(" NOISY ", style="bold white on red") if role is ReferenceRole.NOISY
            else Text(" CLEAN ", style="bold black on green"))


def _file_line(p: Path | None) -> Text:
    if not p:
        return Text("<not selected>", style="dim italic")
    ai = probe_audio(p)
    t = Text(str(p), style="bright_white")
    if ai:
        t.append(f"  ({ai.short()})", style="dim cyan")
    return t


def dashboard(cfg: JobConfig, msg: str = "") -> Group:
    head = Panel(Align.center(Text.assemble(("ZOUT", "bold bright_magenta"), "  ::  ",
                                            ("Audio Differential & Voice Restoration Studio", "bold bright_cyan"))),
                 border_style="bright_blue", box=box.ROUNDED)
    t = Table.grid(padding=(0, 2))
    t.add_column(); t.add_column(style="bold cyan"); t.add_column()
    role_desc = ("Eliminate matching noise from main recording" if cfg.role is ReferenceRole.NOISY
                 else "Keep & restore voice matching this sample")
    t.add_row(key("1"), "Reference sample", _file_line(cfg.reference))
    t.add_row(key("T"), "Reference type", Text.assemble(badge(cfg.role), f"  {role_desc}", style="dim"))
    t.add_row(key("2"), "Main recording", _file_line(cfg.main))

    # Auto-extract noise feature row
    if cfg.main:
        auto_text = Text.assemble(("Scan main recording for speech-free room noise", "green"),
                                  ("  [Press A to run]", "dim cyan"))
    else:
        auto_text = Text("(Pick main recording [2] first to enable auto-noise extraction)", style="dim italic")
    t.add_row(key("A"), "Auto-extract noise", auto_text)

    t.add_row(key("3"), "Strength", Text.assemble(bar(cfg.strength, 0, 1), f"  {cfg.strength:.2f} ({strength_label(cfg.strength)})"))
    t.add_row(key("4"), "Max attenuation", Text.assemble(bar(min(cfg.max_atten_db, 60), 6, 60, color="cyan"),
                                                         f"  {atten_label(cfg.max_atten_db)}"))
    t.add_row(key("F"), "DeepFilterNet3 polish", Text("ON" if cfg.use_polish() else "OFF",
                                                      style="bold green" if cfg.use_polish() else "bold red"))
    out = cfg.output or (cfg.default_output() if cfg.main else None)
    t.add_row(key("O"), "Output file", Text(str(out) if out else "<auto>", style="dim"))

    # Playback status display
    play_status = player.status_text()
    if play_status:
        t.add_row(Text("▶", style="bold green"), "Audio playback", Text(play_status, style="bold bright_green"))

    cfgp = Panel(t, title="[bold]Pipeline Configuration[/]", border_style="cyan", box=box.ROUNDED)

    ctl1 = Text.assemble(
        key("1"), " Ref  ",
        key("T"), " Type  ",
        key("2"), " Main  ",
        key("A"), " Auto-Noise  ",
        key("3/4"), " Params  ",
        key("F"), " Polish  ",
        key("O"), " Out",
    )
    ctl2 = Text.assemble(
        key("P"), " Play Noise  ",
        key("p"), " Play Main  ",
        key("Space"), " Stop Playback  ",
        key("R"), " RUN  ",
        key("Q"), " Quit",
    )
    ready = cfg.ready()
    status_line = (
        Text(msg, style="bold bright_yellow") if msg
        else Text("Ready — press R to process." if ready else "Pick a reference (1) or Auto-extract noise (A), and a main recording (2).",
                  style="green" if ready else "dim")
    )
    foot = Panel(Group(ctl1, ctl2, Text(""), status_line),
                 title="[bold]Controls[/]", border_style="yellow", box=box.ROUNDED)
    return Group(head, cfgp, foot)


def _make_role_line(r: ReferenceRole, k: str, title: str, desc: str, sel: ReferenceRole) -> Text:
    on = sel is r
    return Text.assemble(
        ("> " if on else "  ", "bold yellow"),
        f"[{k}] ",
        (title, "bold " + ("red" if r is ReferenceRole.NOISY else "green")),
        f"\n      {desc}\n",
        style="reverse" if on else "",
    )


def prompt_role(console: Console, cfg: JobConfig) -> bool:
    sel = cfg.role
    while True:
        body = Group(
            Text(f"Reference: {cfg.reference.name if cfg.reference else ''}\n", style="bright_white"),
            Text("How should the ML system learn from this file?\n"),
            _make_role_line(ReferenceRole.NOISY, "N", "NOISY SAMPLE", "Remove audio matching this file from the main recording.", sel),
            _make_role_line(ReferenceRole.CLEAN, "C", "CLEAN SAMPLE", "Keep & restore voice matching these harmonics/phonemics.", sel),
            Text.assemble(key("↑/↓"), " or ", key("N"), key("C"), " choose  ", key("Enter"), " confirm  ", key("Esc"), " back"))
        console.clear()
        console.print(Panel(body, title="[bold magenta]Step 2 :: Classify Reference[/]", border_style="bright_cyan"))
        k = read_key()
        if k.name in ("up", "down", "tab"):
            sel = ReferenceRole.CLEAN if sel is ReferenceRole.NOISY else ReferenceRole.NOISY
        elif k.name == "char" and k.char.lower() == "n":
            sel = ReferenceRole.NOISY
        elif k.name == "char" and k.char.lower() == "c":
            sel = ReferenceRole.CLEAN
        elif k.name == "enter":
            cfg.role = sel
            return True
        elif k.name == "esc":
            return False


def _make_param_row(i: int, current_field: int, title: str, widget: Text, hints: str) -> Text:
    on = current_field == i
    return Text.assemble(
        ("> " if on else "  ", "bold yellow"),
        (title, "bold"),
        "  ",
        widget,
        "\n",
        (f"      {hints}\n", "dim"),
    )


def prompt_params(console: Console, cfg: JobConfig) -> None:
    field = 0
    while True:
        body = Group(
            _make_param_row(0, field, "1. Strength (adaptive threshold / power)", Text.assemble(bar(cfg.strength, 0, 1), f"  {cfg.strength:.2f} {strength_label(cfg.strength)}"),
                "Low 0.1-0.4 gentle | 0.5-0.8 recommended | 0.85-1.0 maximum separation"),
            _make_param_row(1, field, "2. Max attenuation cap (ambience guard)", Text.assemble(bar(min(cfg.max_atten_db, 60), 6, 60, color="cyan"), f"  {atten_label(cfg.max_atten_db)}"),
                "Caps maximum cut so quiet syllables never drop to dead silence. [L]12 [B]24 [H]40 [U]uncapped"),
            Text.assemble(key("Tab"), " switch  ", key("←/→"), " adjust  ", key("Enter"), " done"))
        console.clear()
        console.print(Panel(body, title="[bold magenta]Step 5 :: Inference Threshold & Guard[/]", border_style="bright_cyan"))
        k = read_key()
        if k.name == "tab" or k.name in ("up", "down"):
            field = 1 - field
        elif k.name in ("left", "right"):
            d = 1 if k.name == "right" else -1
            if field == 0:
                cfg.strength = round(min(1.0, max(0.0, cfg.strength + 0.05 * d)), 2)
            else:
                cur = 60.0 if cfg.max_atten_db >= 60 else cfg.max_atten_db
                cfg.max_atten_db = float(min(60, max(6, cur + 3 * d)))
        elif k.name == "char" and field == 1:
            preset = {"l": 12.0, "b": 24.0, "h": 40.0, "u": 60.0}.get(k.char.lower())
            if preset:
                cfg.max_atten_db = preset
        elif k.name == "enter":
            return


def run_with_progress(console: Console, cfg: JobConfig) -> Result | Exception:
    bar_ = ProgressBar(total=100, width=50, complete_style="magenta", finished_style="green")
    state = {"msg": "Starting", "f": 0.0}

    def view():
        bar_.completed = state["f"] * 100
        return Panel(Group(Text.assemble(("Mode: ", "bold"), badge(cfg.role), f"  strength={cfg.strength:.2f}  cap={atten_label(cfg.max_atten_db)}"),
                           Text(""), bar_, Text(f"{state['f'] * 100:5.1f}%  {state['msg']}", style="cyan")),
                     title="[bold magenta]Processing[/]", border_style="bright_cyan")

    with Live(view(), console=console, refresh_per_second=8) as live:
        def cb(msg: str, f: float):
            state["msg"], state["f"] = msg, f
            live.update(view())
        try:
            return run_job(cfg, cb)
        except Exception as e:  # shown to the user, not crashed
            return e


def result_card(res: Result | Exception, play_status: str = "") -> Panel:
    if isinstance(res, Exception):
        return Panel(Group(Text(f"{type(res).__name__}: {res}", style="bold red"),
                           Text.assemble(key("Enter"), " back")),
                     title="[bold red]Failed[/]", border_style="red")
    lines = [Text.assemble(("Output: ", "bold"), (str(res.output), "bright_green")),
             Text(f"Level: {res.in_rms:.1f} dBFS -> {res.out_rms:.1f} dBFS | Noise floor: -{res.suppression_db:.1f} dB | {res.seconds:.1f}s")]
    if res.notes.get("warning"):
        lines.append(Text(res.notes["warning"], style="yellow"))
    if res.notes.get("polish_error"):
        lines.append(Text("Polish skipped: " + res.notes["polish_error"], style="yellow"))
    if play_status:
        lines.append(Text(""))
        lines.append(Text(play_status, style="bold bright_green"))
    lines.append(Text(""))
    lines.append(Text.assemble(
        key("P"), " Play Processed  ",
        key("p"), " Play Original  ",
        key("Space"), " Stop  ",
        key("Enter"), " Back to menu"
    ))
    return Panel(Group(*lines), title="[bold green]Processing Complete[/]", border_style="green")


def prompt_auto_noise(console: Console, cfg: JobConfig) -> Path | None:
    """Interactive modal to configure and run automatic noise floor extraction."""
    from ..audio.io import load_audio

    if not cfg.main or not cfg.main.is_file():
        return None

    n_chunks = 6
    chunk_sec = 2.0
    mode_idx = 0
    modes = [("balanced", "Balanced (VAD + Flatness) [Recommended]"),
             ("deep", "Deep Hunt (Strict - Catches Low-Energy Whispers)"),
             ("fast", "Fast (Quick Energy Scan + VAD)")]

    field = 0  # 0: n_chunks, 1: chunk_sec, 2: mode

    while True:
        body = Group(
            Text.assemble(("Source Audio: ", "bold"), (cfg.main.name, "cyan"), "\n"),
            _make_param_row(0, field, "Number of noise sections to find",
                Text.assemble(bar(n_chunks, 1, 15, color="green"), f"  {n_chunks} chunks"),
                "Recommended: 4 to 8 sections. [3] [6] [9] presets"),
            _make_param_row(1, field, "Duration of each section (seconds)",
                Text.assemble(bar(chunk_sec, 0.5, 6.0, color="cyan"), f"  {chunk_sec:.1f} s"),
                "Recommended: 1.5 to 3.0 seconds. [1] [2] [3] [5] presets"),
            _make_param_row(2, field, "Extraction Strategy",
                Text(f"● {modes[mode_idx][1]}", style="bold magenta"),
                "[B] Balanced  [D] Deep Hunt (ultra-strict VAD)  [F] Fast scan"),
            Text.assemble(
                key("Tab"), " switch field  ",
                key("←/→"), " adjust  ",
                key("Enter"), " START EXTRACTION  ",
                key("Esc"), " cancel"
            )
        )
        console.clear()
        console.print(Panel(body, title="[bold magenta]Auto-Extract Room Noise Profile[/]", border_style="bright_cyan"))

        k = read_key()
        if k.name in ("tab", "up", "down"):
            d = 1 if (k.name == "tab" or k.name == "down") else -1
            field = (field + d) % 3
        elif k.name in ("left", "right"):
            d = 1 if k.name == "right" else -1
            if field == 0:
                n_chunks = max(1, min(20, n_chunks + d))
            elif field == 1:
                chunk_sec = round(max(0.5, min(10.0, chunk_sec + 0.5 * d)), 1)
            elif field == 2:
                mode_idx = (mode_idx + d) % len(modes)
        elif k.name == "char":
            c = k.char.lower()
            if field == 0 and c in "123456789":
                n_chunks = int(c)
            elif field == 1 and c in "12345":
                chunk_sec = float(c)
            elif c == "b":
                mode_idx = 0
            elif c == "d":
                mode_idx = 1
            elif c == "f":
                mode_idx = 2
        elif k.name == "esc":
            return None
        elif k.name == "enter":
            break

    # Run extraction with live progress
    bar_ = ProgressBar(total=100, width=50, complete_style="magenta", finished_style="green")
    state = {"msg": "Loading audio file...", "f": 0.05}

    def view():
        bar_.completed = state["f"] * 100
        return Panel(
            Group(
                Text.assemble(
                    ("Target: ", "bold"), (f"{n_chunks} chunks x {chunk_sec:.1f}s", "cyan"),
                    ("  Strategy: ", "bold"), (modes[mode_idx][0].upper(), "magenta"),
                ),
                Text(""),
                bar_,
                Text(f"{state['f'] * 100:5.1f}%  {state['msg']}", style="cyan"),
            ),
            title="[bold magenta]Extracting Speech-Free Room Noise Profile[/]",
            border_style="bright_cyan",
        )

    result = None
    with Live(view(), console=console, refresh_per_second=8) as live:
        def cb(msg: str, f: float):
            state["msg"], state["f"] = msg, f
            live.update(view())

        try:
            cb("Loading main audio...", 0.05)
            main_audio, sr = load_audio(cfg.main, mono=True)
            cb("Analyzing acoustic timeline...", 0.15)
            res = extract_noise_profile(
                main_audio,
                sr=sr,
                n_chunks=n_chunks,
                chunk_sec=chunk_sec,
                mode=modes[mode_idx][0],
                progress=cb,
            )
            # Save extracted noise WAV
            out_noise_path = cfg.default_noise_output()
            save_audio(out_noise_path, res.audio, sr)
            res.output_path = out_noise_path
            result = res
        except Exception as e:
            result = e

    # Display result card
    console.clear()
    if isinstance(result, Exception):
        console.print(
            Panel(
                Group(
                    Text(f"Extraction failed: {result}", style="bold red"),
                    Text.assemble(key("Enter"), " back to menu"),
                ),
                title="[bold red]Noise Extraction Error[/]",
                border_style="red",
            )
        )
        while read_key().name != "enter":
            pass
        return None

    # Success: automatically configure job reference
    cfg.reference = result.output_path
    cfg.role = ReferenceRole.NOISY

    # Present success modal with playback option
    while True:
        play_st = player.status_text()
        card_lines = [
            Text.assemble(("Noise Profile Saved: ", "bold"), (str(result.output_path), "bright_green")),
            Text(f"Extracted {len(result.chunks)} noise sections ({len(result.audio) / result.sr:.1f}s total)"),
            Text(f"Noise floor: {result.rms_db:.1f} dBFS"),
            Text(""),
            Text("Detected Speech-Free Timeline Sections:", style="bold cyan"),
        ]
        for idx, (st_sec, en_sec) in enumerate(result.chunks[:6], 1):
            st_m, st_s = divmod(int(st_sec), 60)
            en_m, en_s = divmod(int(en_sec), 60)
            card_lines.append(Text(f"  Chunk #{idx}:  {st_m:02d}:{st_s:02d} - {en_m:02d}:{en_s:02d}  ({en_sec - st_sec:.1f}s)", style="dim white"))
        if len(result.chunks) > 6:
            card_lines.append(Text(f"  ... and {len(result.chunks) - 6} more sections", style="dim"))

        if result.warnings:
            card_lines.append(Text(""))
            for w in result.warnings:
                card_lines.append(Text(f"Notice: {w}", style="yellow"))

        if play_st:
            card_lines.append(Text(""))
            card_lines.append(Text(play_st, style="bold bright_green"))

        card_lines.append(Text(""))
        card_lines.append(
            Text.assemble(
                key("P"), " Play extracted noise  ",
                key("Space"), " Stop playback  ",
                key("Enter"), " Accept & return to menu"
            )
        )

        console.clear()
        console.print(
            Panel(
                Group(*card_lines),
                title="[bold green]Room Noise Profile Successfully Extracted[/]",
                border_style="green",
            )
        )

        k = read_key()
        if k.name == "char" and k.char.upper() == "P":
            if player.is_playing():
                player.stop()
            else:
                player.play(result.output_path, "Extracted Noise Profile")
        elif k.name == "char" and (k.char == " " or k.char.lower() == "s"):
            player.stop()
        elif k.name == "enter":
            player.stop()
            return result.output_path
