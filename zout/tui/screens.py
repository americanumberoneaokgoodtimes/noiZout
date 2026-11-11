from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..audio.io import compute_waveform_envelope, parse_time_spec, probe_audio, save_audio
from ..audio.noise_extractor import extract_noise_profile
from ..audio.player import player
from ..config import JobConfig, ReferenceRole
from ..pipeline import Result, run_job
from . import themes
from .keys import poll_key, read_key


def key(k: str, theme: themes.Theme | None = None) -> Text:
    th = theme or themes.get_theme()
    return Text(f" {k} ", style=th.key)


def bar(value: float, lo: float, hi: float, width: int = 22, color: str | None = None, theme: themes.Theme | None = None) -> Text:
    th = theme or themes.get_theme()
    c = color or th.bar_fill
    f = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.0
    n = int(round(f * width))
    return Text.assemble(("█" * n, c), ("░" * (width - n), th.bar_empty))


def strength_label(s: float) -> str:
    return "Gentle" if s < 0.45 else "Balanced" if s < 0.65 else "Strong" if s < 0.85 else "Maximum"


def atten_label(a: float) -> str:
    return "Uncapped" if a >= 60 else f"{a:.0f} dB"


def badge(role: ReferenceRole) -> Text:
    return (Text(" NOISY ", style="bold white on red") if role is ReferenceRole.NOISY
            else Text(" CLEAN ", style="bold black on green"))


def _file_line(p: Path | None, theme: themes.Theme | None = None) -> Text:
    th = theme or themes.get_theme()
    if not p:
        return Text("<not selected>", style="dim italic")
    ai = probe_audio(p)
    t = Text(str(p), style="bright_white")
    if ai:
        t.append(f"  ({ai.short()})", style=f"dim {th.primary}")
    return t


def render_stereo_vu_meter(
    player_or_levels: Any,
    width: int = 24,
    theme: themes.Theme | None = None,
    db_floor: float = -60.0,
) -> Table:
    """Render dual-channel ASCII/Unicode stereo VU meter (RMS filled bar + Peak transient marker)."""
    th = theme or themes.get_theme()
    if hasattr(player_or_levels, "get_meter_levels"):
        levels = player_or_levels.get_meter_levels()
    else:
        levels = player_or_levels
    (rms_l, rms_r), (peak_l, peak_r) = levels

    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style=f"bold {th.primary}")
    t.add_column()
    t.add_column(style="dim")

    for ch_name, rms_db, peak_db in [("L", rms_l, peak_l), ("R", rms_r, peak_r)]:
        f_rms = min(1.0, max(0.0, (rms_db - db_floor) / (0.0 - db_floor)))
        n_rms = int(round(f_rms * width))

        f_peak = min(1.0, max(0.0, (peak_db - db_floor) / (0.0 - db_floor)))
        pos_peak = min(width - 1, max(0, int(round(f_peak * (width - 1)))))

        parts = []
        for i in range(width):
            db_i = db_floor + (i / width) * (0.0 - db_floor)
            if db_i < -18.0:
                color = th.meter_low or "bright_cyan"
            elif db_i < -6.0:
                color = th.meter_mid or "bright_yellow"
            else:
                color = th.meter_high or "bright_red"

            if i == pos_peak and peak_db > db_floor + 1.0:
                peak_style = th.meter_peak if peak_db >= -0.5 else f"bold {color}"
                parts.append(("▌", peak_style))
            elif i < n_rms:
                parts.append(("█", color))
            else:
                parts.append(("░", th.bar_empty))

        bar_text = Text.assemble("[", *parts, "]")
        rms_txt = f"{rms_db:5.1f} dB" if rms_db > db_floor else "  -∞ dB"
        peak_txt = f"{peak_db:5.1f} dB" if peak_db > db_floor else "  -∞ dB"
        clip_tag = " [CLIP]" if peak_db >= -0.5 else ""
        desc = f"RMS: {rms_txt}  Peak: {peak_txt}{clip_tag}"
        t.add_row(f"{ch_name}:", bar_text, Text(desc, style=th.meter_peak if clip_tag else "dim"))

    return t


def render_playback_card(player_inst: Any, title: str = "Audio Playback", theme: themes.Theme | None = None) -> Panel:
    """Render real-time playback panel with stereo VU meter, track badge, and playhead."""
    th = theme or themes.get_theme()
    trk = player_inst.get_current_track()
    desc = "Original Raw" if trk == "A" else "Processed Output"
    badge_style = "bold white on red" if trk == "A" else "bold black on green"
    track_badge = Text(f" TRACK {trk}: {desc} ", style=badge_style)

    dur = getattr(player_inst, "duration", 0.0)
    sample_idx = getattr(player_inst, "sample_index", 0)
    sr = getattr(player_inst, "_sr", 44100)
    elapsed = min(dur, max(0.0, sample_idx / sr)) if dur > 0 else 0.0

    m_el, s_el = divmod(int(elapsed), 60)
    m_dur, s_dur = divmod(int(dur), 60)
    time_str = f"{m_el:02d}:{s_el:02d} / {m_dur:02d}:{s_dur:02d}"

    pbar = ProgressBar(total=dur if dur > 0 else 1.0, completed=elapsed, width=32,
                       complete_style=th.primary, finished_style=th.success)

    vu_table = render_stereo_vu_meter(player_inst, width=24, theme=th)

    header = Text.assemble(track_badge, "  ", (f"Elapsed: {time_str}", f"bold {th.primary}"))
    controls = Text.assemble(
        key("Tab", th), " Hot-Swap A/B  ",
        key("Space", th), " Stop Playback  ",
        key("C", th), " Theme  ",
        key("Enter", th), " Back"
    )

    return Panel(
        Group(header, Text(""), pbar, Text(""), vu_table, Text(""), controls),
        title=f"[bold {th.accent}]{title}[/]",
        border_style=th.border,
        box=box.ROUNDED,
    )


def dashboard(cfg: JobConfig, msg: str = "", theme: themes.Theme | None = None) -> Group:
    th = theme or themes.get_theme()
    head = Panel(
        Align.center(
            Text.assemble(
                ("ZOUT", th.title),
                "  ::  ",
                ("Audio Differential & Voice Restoration Studio", f"bold {th.primary}"),
                (f"  [{th.name}]", "dim")
            )
        ),
        border_style=th.secondary,
        box=box.ROUNDED,
    )
    t = Table.grid(padding=(0, 2))
    t.add_column()
    t.add_column(style=f"bold {th.primary}")
    t.add_column()

    role_desc = ("Eliminate matching noise from main recording" if cfg.role is ReferenceRole.NOISY
                 else "Keep & restore voice matching this sample")
    t.add_row(key("1", th), "Reference sample", _file_line(cfg.reference, th))
    t.add_row(key("M", th), "Reference type", Text.assemble(badge(cfg.role), f"  {role_desc}", style="dim"))
    t.add_row(key("2", th), "Main recording", _file_line(cfg.main, th))

    # Timeline range slicing row (R5)
    if cfg.start_sec is not None or cfg.end_sec is not None:
        def _fmt(s: float | None) -> str:
            if s is None:
                return "end"
            m, sec = divmod(int(s), 60)
            return f"{m:02d}:{sec:02d}"
        s_txt = _fmt(cfg.start_sec or 0.0)
        e_txt = _fmt(cfg.end_sec)
        range_txt = Text.assemble((f"Slice [{s_txt} -> {e_txt}]", f"bold {th.accent}"), ("  [Press T to edit]", "dim"))
    else:
        range_txt = Text.assemble(("Full recording", "dim"), ("  [Press T to slice range]", "dim"))
    t.add_row(key("T", th), "Timeline range", range_txt)

    # Auto-extract noise feature row
    if cfg.main:
        auto_text = Text.assemble(("Scan main recording for speech-free room noise", f"bold {th.success}"),
                                  ("  [Press N to run]", f"dim {th.primary}"))
    else:
        auto_text = Text("(Pick main recording [2] first to enable auto-noise extraction)", style="dim italic")
    t.add_row(key("N", th), "Auto-extract noise", auto_text)

    # Forensic audit log row (R6)
    audit_style = f"bold {th.success}" if cfg.audit else "dim"
    audit_label = "ENABLED (.json sidecar generated)" if cfg.audit else "DISABLED (press A to toggle)"
    t.add_row(key("A", th), "Forensic audit log", Text(audit_label, style=audit_style))

    t.add_row(key("3", th), "Strength", Text.assemble(bar(cfg.strength, 0, 1, theme=th), f"  {cfg.strength:.2f} ({strength_label(cfg.strength)})"))
    t.add_row(key("4", th), "Max attenuation", Text.assemble(bar(min(cfg.max_atten_db, 60), 6, 60, color=th.secondary, theme=th),
                                                             f"  {atten_label(cfg.max_atten_db)}"))
    t.add_row(key("F", th), "DeepFilterNet3 polish", Text("ON" if cfg.use_polish() else "OFF",
                                                          style=f"bold {th.success}" if cfg.use_polish() else "bold red"))
    out = cfg.output or (cfg.default_output() if cfg.main else None)
    t.add_row(key("O", th), "Output file", Text(str(out) if out else "<auto>", style="dim"))

    # Batch directory display (R8)
    if cfg.batch_dir:
        t.add_row(key("B", th), "Batch folder", Text.assemble((str(cfg.batch_dir), f"bold {th.accent}"), (" (Batch mode active)", "dim")))

    # Playback status display
    play_status = player.status_text()
    if play_status:
        vu_grid = render_stereo_vu_meter(player, width=18, theme=th)
        t.add_row(Text("▶", style=f"bold {th.success}"), "Audio playback", Group(Text(play_status, style=f"bold {th.primary}"), vu_grid))

    cfgp = Panel(t, title="[bold]Pipeline Configuration[/]", border_style=th.border, box=box.ROUNDED)

    ctl1 = Text.assemble(
        key("1", th), " Ref  ",
        key("M", th), " Mode  ",
        key("2", th), " Main  ",
        key("T", th), " Range  ",
        key("N", th), " Noise  ",
        key("A", th), " Audit  ",
        key("3/4", th), " Params  ",
        key("F", th), " Polish  ",
        key("C", th), " Theme  ",
        key("B", th), " Batch  ",
        key("O", th), " Out",
    )
    ctl2 = Text.assemble(
        key("P", th), " Play Ref  ",
        key("p", th), " Play Main  ",
        key("Space", th), " Stop Playback  ",
        key("R", th), " RUN  ",
        key("Q", th), " Quit",
    )
    ready = cfg.ready()
    status_line = (
        Text(msg, style="bold bright_yellow") if msg
        else Text("Ready — press R to process." if ready else "Pick a reference (1) or Auto-extract noise (N), and a main recording (2) or batch (B).",
                  style=f"bold {th.success}" if ready else "dim")
    )
    foot = Panel(Group(ctl1, ctl2, Text(""), status_line),
                 title="[bold]Controls[/]", border_style=th.border, box=box.ROUNDED)
    return Group(head, cfgp, foot)


def _make_role_line(r: ReferenceRole, k: str, title: str, desc: str, sel: ReferenceRole, theme: themes.Theme | None = None) -> Text:
    th = theme or themes.get_theme()
    on = sel is r
    color = "red" if r is ReferenceRole.NOISY else "green"
    title_style = f"bold underline {color}" if on else f"bold {color}"
    return Text.assemble(
        ("> " if on else "  ", "bold yellow"),
        f"[{k}] ",
        (title, title_style),
        (" ◀ SELECTED" if on else "", "bold yellow"),
        f"\n      {desc}\n",
        style="reverse" if on else "",
    )


def prompt_role(console: Console, cfg: JobConfig, theme: themes.Theme | None = None) -> bool:
    th = theme or themes.get_theme()
    sel = cfg.role
    while True:
        body = Group(
            Text(f"Reference: {cfg.reference.name if cfg.reference else ''}\n", style="bright_white"),
            Text("How should the ML system learn from this file?\n"),
            _make_role_line(ReferenceRole.NOISY, "N", "NOISY SAMPLE", "Remove audio matching this file from the main recording.", sel, th),
            _make_role_line(ReferenceRole.CLEAN, "C", "CLEAN SAMPLE", "Keep & restore voice matching these harmonics/phonemics.", sel, th),
            Text.assemble(key("↑/↓", th), " or ", key("N", th), key("C", th), " choose  ", key("Enter", th), " confirm  ", key("Esc", th), " back"))
        console.clear()
        console.print(Panel(body, title=f"[bold {th.accent}]Step 2 :: Classify Reference[/]", border_style=th.border))
        k = read_key()
        if k is None:
            continue
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


def prompt_params(console: Console, cfg: JobConfig, theme: themes.Theme | None = None) -> None:
    th = theme or themes.get_theme()
    field = 0
    while True:
        body = Group(
            _make_param_row(0, field, "1. Strength (adaptive threshold / power)", Text.assemble(bar(cfg.strength, 0, 1, theme=th), f"  {cfg.strength:.2f} {strength_label(cfg.strength)}"),
                "Low 0.1-0.4 gentle | 0.5-0.8 recommended | 0.85-1.0 maximum separation"),
            _make_param_row(1, field, "2. Max attenuation cap (ambience guard)", Text.assemble(bar(min(cfg.max_atten_db, 60), 6, 60, color=th.secondary, theme=th), f"  {atten_label(cfg.max_atten_db)}"),
                "Caps maximum cut so quiet syllables never drop to dead silence. [L]12 [B]24 [H]40 [U]uncapped"),
            Text.assemble(key("Tab", th), " switch  ", key("←/→", th), " adjust  ", key("Enter", th), " done"))
        console.clear()
        console.print(Panel(body, title=f"[bold {th.accent}]Step 5 :: Inference Threshold & Guard[/]", border_style=th.border))
        k = read_key()
        if k is None:
            continue
        if k.name in ("tab", "up", "down"):
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


def prompt_timeline_range(console: Console, cfg: JobConfig, theme: themes.Theme | None = None) -> tuple[float | None, float | None]:
    """Interactive modal to bracket start and end timestamps with visual timeline bar (R5)."""
    th = theme or themes.get_theme()
    total_dur: float | None = None
    if cfg.main and cfg.main.is_file():
        ai = probe_audio(cfg.main)
        if ai:
            total_dur = ai.duration

    def _fmt(s: float | None) -> str:
        if s is None:
            return ""
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"

    start_str = _fmt(cfg.start_sec) if cfg.start_sec is not None else ""
    end_str = _fmt(cfg.end_sec) if cfg.end_sec is not None else ""
    field = 0  # 0: start, 1: end
    err_msg = ""

    while True:
        s_val, e_val = None, None
        err_msg = ""
        try:
            if start_str.strip():
                s_val = parse_time_spec(start_str)
        except Exception as ex:
            err_msg = f"Invalid start time: {ex}"

        try:
            if end_str.strip():
                e_val = parse_time_spec(end_str)
        except Exception as ex:
            err_msg = f"Invalid end time: {ex}"

        if not err_msg and s_val is not None and e_val is not None:
            if s_val >= e_val:
                err_msg = f"Start time ({s_val:.1f}s) must be strictly before end time ({e_val:.1f}s)"

        if not err_msg and total_dur is not None:
            if s_val is not None and s_val > total_dur:
                err_msg = f"Start time exceeds file duration ({_fmt(total_dur)})"

        bar_w = 32
        if total_dur and total_dur > 0:
            s_s = s_val if s_val is not None else 0.0
            e_s = min(total_dur, e_val) if e_val is not None else total_dur
            p1 = int(round(min(1.0, max(0.0, s_s / total_dur)) * bar_w))
            p2 = int(round(min(1.0, max(0.0, e_s / total_dur)) * bar_w))
            p2 = max(p1, p2)

            bar_chars = [
                ("░" * p1, th.bar_empty),
                ("[", "bold yellow"),
                ("█" * (p2 - p1), f"bold {th.primary}"),
                ("]", "bold yellow"),
                ("░" * (bar_w - p2), th.bar_empty),
            ]
            bracket_bar = Text.assemble("00:00 ", *bar_chars, f" {_fmt(total_dur)}")
            slice_dur = max(0.0, e_s - s_s)
            dur_info = f"Selected Duration: {_fmt(slice_dur)} / {_fmt(total_dur)} ({(slice_dur / total_dur) * 100:.1f}%)"
        else:
            bracket_bar = Text("[-- No duration probe available --]", style="dim")
            dur_info = "Targeting custom range"

        start_disp = Text.assemble(
            ("> " if field == 0 else "  ", "bold yellow"),
            ("1. Start Timestamp:  ", "bold"),
            (start_str or "<00:00 - beginning>", f"bold {th.primary}" if field == 0 else "white"),
        )
        end_disp = Text.assemble(
            ("> " if field == 1 else "  ", "bold yellow"),
            ("2. End Timestamp:    ", "bold"),
            (end_str or f"<{_fmt(total_dur) if total_dur else 'end of file'}>", f"bold {th.primary}" if field == 1 else "white"),
        )

        body_items = [
            Text(f"Target Recording: {cfg.main.name if cfg.main else 'None'}\n", style="bright_white"),
            start_disp,
            Text("      Type seconds (e.g. 45), mm:ss (01:30), or use ←/→ to nudge ±1s\n", style="dim"),
            end_disp,
            Text("      Type seconds (e.g. 120), mm:ss (02:00), or use ←/→ to nudge ±1s\n", style="dim"),
            bracket_bar,
            Text(dur_info, style=f"bold {th.primary}"),
            Text(""),
        ]
        if err_msg:
            body_items.append(Text(f"✗ {err_msg}", style="bold red"))
        else:
            body_items.append(Text("✓ Valid timeline bracket", style=f"bold {th.success}"))

        body_items.append(Text(""))
        body_items.append(Text.assemble(
            key("Tab/↑/↓", th), " Switch field  ",
            key("←/→", th), " Nudge ±1s  ",
            key("R", th), " Reset full  ",
            key("Enter", th), " Confirm  ",
            key("Esc", th), " Cancel"
        ))

        console.clear()
        console.print(Panel(Group(*body_items), title=f"[bold {th.accent}]Timeline Range Slicing (R5)[/]",
                            border_style=th.border, box=box.ROUNDED))

        k = read_key()
        if k is None:
            continue
        if k.name in ("tab", "up", "down"):
            field = 1 - field
        elif k.name == "backspace":
            if field == 0:
                start_str = start_str[:-1]
            else:
                end_str = end_str[:-1]
        elif k.name in ("left", "right"):
            delta = 1.0 if k.name == "right" else -1.0
            if field == 0:
                cur = s_val if s_val is not None else 0.0
                start_str = _fmt(max(0.0, cur + delta))
            else:
                cur = e_val if e_val is not None else (total_dur or 60.0)
                end_str = _fmt(max(0.0, cur + delta))
        elif k.name == "char":
            c = k.char
            if c in "0123456789:.":
                if field == 0:
                    start_str += c
                else:
                    end_str += c
            elif c.lower() == "r":
                start_str = ""
                end_str = ""
        elif k.name == "enter":
            if not err_msg:
                cfg.start_sec = s_val
                cfg.end_sec = e_val
                return (cfg.start_sec, cfg.end_sec)
        elif k.name == "esc":
            return (cfg.start_sec, cfg.end_sec)


def run_with_progress(console: Console, cfg: JobConfig, theme: themes.Theme | None = None) -> Result | Exception | None:
    """Execute job with live progress bar and non-blocking in-flight cancellation via [Esc] (R4)."""
    import time as _time
    th = theme or themes.get_theme()
    cancel_ev = threading.Event()
    cfg.cancel_event = cancel_ev

    bar_ = ProgressBar(total=100, width=50, complete_style=th.primary, finished_style=th.success)
    state = {"msg": "Starting...", "f": 0.0, "cancelled": False}
    t0 = _time.time()

    def view():
        bar_.completed = state["f"] * 100
        elapsed = _time.time() - t0
        f = state["f"]
        if f > 0.02:
            eta = (elapsed / f) * (1.0 - f)
            time_str = f"{elapsed:.0f}s elapsed  —  ETA ~{eta:.0f}s"
        else:
            time_str = f"{elapsed:.0f}s elapsed"
        hint = (
            Text("Cancellation requested... aborting inference", style="bold yellow")
            if state["cancelled"]
            else Text.assemble(key("Esc", th), " Cancel processing", style="dim")
        )
        return Panel(
            Group(
                Text.assemble(("Mode: ", "bold"), badge(cfg.role), f"  strength={cfg.strength:.2f}  cap={atten_label(cfg.max_atten_db)}"),
                Text(""),
                bar_,
                Text(f"{state['f'] * 100:5.1f}%  {state['msg']}", style=f"bold {th.primary}"),
                Text(time_str, style="dim"),
                Text(""),
                hint,
            ),
            title=f"[bold {th.accent}]Processing[/]",
            border_style=th.border,
            box=box.ROUNDED,
        )

    res_box: dict[str, Any] = {"result": None, "error": None, "done": False}

    def worker():
        try:
            res_box["result"] = run_job(cfg, progress=cb, cancel_token=cancel_ev)
        except Exception as e:
            res_box["error"] = e
        finally:
            res_box["done"] = True

    with Live(view(), console=console, refresh_per_second=10) as live:
        def cb(msg: str, f: float):
            state["msg"], state["f"] = msg, f
            live.update(view())

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        while not res_box["done"]:
            k = poll_key(timeout=0.05)
            if k is not None and (k.name == "esc" or (k.name == "char" and k.char == "\x1b")):
                state["cancelled"] = True
                cancel_ev.set()
                live.update(view())
                break

        if cancel_ev.is_set():
            worker_thread.join(timeout=3.0)
            return None

        worker_thread.join(timeout=1.0)
        if res_box["error"] is not None:
            return res_box["error"]
        return res_box["result"]


def result_card(res: Result | Exception, play_status: str = "", theme: themes.Theme | None = None) -> Panel:
    th = theme or themes.get_theme()
    if isinstance(res, Exception):
        return Panel(Group(Text(f"{type(res).__name__}: {res}", style="bold red"),
                           Text.assemble(key("Enter", th), " back")),
                     title="[bold red]Failed[/]", border_style="red", box=box.ROUNDED)

    lines = [Text.assemble(("Output: ", "bold"), (str(res.output), f"bold {th.success}")),
             Text(f"Level: {res.in_rms:.1f} dBFS -> {res.out_rms:.1f} dBFS | Noise floor: -{res.suppression_db:.1f} dB | {res.seconds:.1f}s")]
    if res.notes.get("warning"):
        lines.append(Text(res.notes["warning"], style="yellow"))
    if res.notes.get("polish_error"):
        lines.append(Text("Polish skipped: " + res.notes["polish_error"], style="yellow"))
    if res.notes.get("batch"):
        success = res.notes.get("success", 0)
        total = res.notes.get("total", 0)
        failures: list = res.notes.get("failures", [])
        lines.append(Text(f"Batch: {success}/{total} files processed  (avg {res.in_rms:.1f} → {res.out_rms:.1f} dBFS)", style=f"bold {th.primary}"))
        if failures:
            lines.append(Text(f"  {len(failures)} file(s) failed:", style="bold yellow"))
            for f in failures[:5]:
                lines.append(Text(f"    ✗ {f}", style="yellow"))
            if len(failures) > 5:
                lines.append(Text(f"    … and {len(failures) - 5} more", style="dim yellow"))
    if play_status:
        lines.append(Text(""))
        lines.append(Text(play_status, style=f"bold {th.primary}"))
        if player.is_playing():
            lines.append(render_stereo_vu_meter(player, width=22, theme=th))

    lines.append(Text(""))
    lines.append(Text.assemble(
        key("Tab", th), " Hot-Swap A/B  ",
        key("Space", th), " Toggle Play  ",
        key("C", th), " Theme  ",
        key("Enter", th), " Back to menu"
    ))
    return Panel(Group(*lines), title=f"[bold {th.accent}]Processing Complete[/]", border_style=th.border, box=box.ROUNDED)


def prompt_auto_noise(console: Console, cfg: JobConfig, theme: themes.Theme | None = None) -> Path | None:
    """Interactive modal to configure and run automatic noise floor extraction."""
    from ..audio.io import load_audio

    th = theme or themes.get_theme()
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
            Text.assemble(("Source Audio: ", "bold"), (cfg.main.name, f"bold {th.primary}"), "\n"),
            _make_param_row(0, field, "Number of noise sections to find",
                Text.assemble(bar(n_chunks, 1, 15, color=th.success, theme=th), f"  {n_chunks} chunks"),
                "Recommended: 4 to 8 sections. [3] [6] [9] presets"),
            _make_param_row(1, field, "Duration of each section (seconds)",
                Text.assemble(bar(chunk_sec, 0.5, 6.0, color=th.secondary, theme=th), f"  {chunk_sec:.1f} s"),
                "Recommended: 1.5 to 3.0 seconds. [1] [2] [3] [5] presets"),
            _make_param_row(2, field, "Extraction Strategy",
                Text(f"● {modes[mode_idx][1]}", style=f"bold {th.accent}"),
                "[B] Balanced  [D] Deep Hunt (ultra-strict VAD)  [F] Fast scan"),
            Text.assemble(
                key("Tab", th), " switch field  ",
                key("←/→", th), " adjust  ",
                key("Enter", th), " START EXTRACTION  ",
                key("Esc", th), " cancel"
            )
        )
        console.clear()
        console.print(Panel(body, title=f"[bold {th.accent}]Auto-Extract Room Noise Profile[/]", border_style=th.border, box=box.ROUNDED))

        k = read_key()
        if k is None:
            continue
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
    bar_ = ProgressBar(total=100, width=50, complete_style=th.primary, finished_style=th.success)
    state = {"msg": "Loading audio file...", "f": 0.05}

    def view():
        bar_.completed = state["f"] * 100
        return Panel(
            Group(
                Text.assemble(
                    ("Target: ", "bold"), (f"{n_chunks} chunks x {chunk_sec:.1f}s", f"bold {th.primary}"),
                    ("  Strategy: ", "bold"), (modes[mode_idx][0].upper(), f"bold {th.accent}"),
                ),
                Text(""),
                bar_,
                Text(f"{state['f'] * 100:5.1f}%  {state['msg']}", style=f"bold {th.primary}"),
            ),
            title=f"[bold {th.accent}]Extracting Speech-Free Room Noise Profile[/]",
            border_style=th.border,
            box=box.ROUNDED,
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
            out_noise_path = cfg.default_noise_output()
            save_audio(out_noise_path, res.audio, sr)
            res.output_path = out_noise_path
            result = res
        except Exception as e:
            result = e

    console.clear()
    if isinstance(result, Exception):
        console.print(
            Panel(
                Group(
                    Text(f"Extraction failed: {result}", style="bold red"),
                    Text.assemble(key("Enter", th), " back to menu"),
                ),
                title="[bold red]Noise Extraction Error[/]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        while True:
            k = read_key()
            if k and k.name == "enter":
                break
        return None

    cfg.reference = result.output_path
    cfg.role = ReferenceRole.NOISY

    while True:
        play_st = player.status_text()
        env_display = getattr(result, "waveform_envelope", "")
        card_lines = [
            Text.assemble(("Noise Profile Saved: ", "bold"), (str(result.output_path), f"bold {th.success}")),
            Text(f"Extracted {len(result.chunks)} noise sections ({len(result.audio) / result.sr:.1f}s total)"),
            Text(f"Noise floor: {result.rms_db:.1f} dBFS | Waveform: [{env_display}]"),
            Text(""),
            Text("Detected Speech-Free Timeline Sections:", style=f"bold {th.primary}"),
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
            card_lines.append(Text(play_st, style=f"bold {th.primary}"))
            if player.is_playing():
                card_lines.append(render_stereo_vu_meter(player, width=22, theme=th))

        card_lines.append(Text(""))
        card_lines.append(
            Text.assemble(
                key("P", th), " Play noise  ",
                key("Space", th), " Stop playback  ",
                key("C", th), " Theme  ",
                key("Enter", th), " Accept & return"
            )
        )

        console.clear()
        console.print(
            Panel(
                Group(*card_lines),
                title=f"[bold {th.accent}]Room Noise Profile Successfully Extracted[/]",
                border_style=th.border,
                box=box.ROUNDED,
            )
        )

        k = read_key()
        if k is None:
            continue
        if k.name == "char" and k.char.upper() == "P":
            if player.is_playing():
                player.stop()
            else:
                player.play(result.output_path, "Extracted Noise Profile")
        elif k.name in ("space", "char") and (k.name == "space" or k.char in (" ", "s", "S")):
            player.stop()
        elif k.name == "char" and k.char.lower() == "c":
            themes.cycle_theme()
            th = themes.get_theme()
        elif k.name in ("enter", "esc"):
            player.stop()
            return result.output_path
