from __future__ import annotations

import os
import string
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..audio.io import probe_audio
from ..config import SUPPORTED_AUDIO_EXTENSIONS
from .keys import read_key


@dataclass
class Entry:
    path: Path
    is_dir: bool
    label: str


class PickerState:
    """Filesystem navigation logic: lists only directories and supported audio files."""

    def __init__(self, start: Path | None = None):
        self.cwd = self._valid(start)
        self.filter = ""
        self.index = 0
        self.at_drives = False
        self.refresh()

    @staticmethod
    def _valid(p: Path | None) -> Path:
        p = Path(p) if p else Path.cwd()
        while not p.is_dir() and p != p.parent:
            p = p.parent
        return p if p.is_dir() else Path.home()

    def refresh(self) -> None:
        self.entries: list[Entry] = []
        if self.at_drives:
            for d in string.ascii_uppercase:
                root = Path(f"{d}:\\")
                if root.exists():
                    self.entries.append(Entry(root, True, f"{d}:\\"))
        else:
            if self.cwd.parent != self.cwd:
                self.entries.append(Entry(self.cwd.parent, True, ".. (Parent Directory)"))
            dirs, files = [], []
            try:
                with os.scandir(self.cwd) as it:
                    for e in it:
                        if e.name.startswith("."):
                            continue
                        try:
                            if e.is_dir():
                                dirs.append(Entry(Path(e.path), True, e.name + os.sep))
                            elif Path(e.name).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                                files.append(Entry(Path(e.path), False, e.name))
                        except OSError:
                            continue
            except (PermissionError, OSError):
                pass
            key = lambda e: e.label.lower()
            f = self.filter.lower()
            if f:
                dirs = [d for d in dirs if f in d.label.lower()]
                files = [x for x in files if f in x.label.lower()]
            self.entries += sorted(dirs, key=key) + sorted(files, key=key)
        self.index = min(self.index, max(0, len(self.entries) - 1))

    def move(self, delta: int) -> None:
        if self.entries:
            self.index = max(0, min(len(self.entries) - 1, self.index + delta))

    def enter(self) -> Path | None:
        """Open dir or return the selected audio file."""
        if not self.entries:
            return None
        e = self.entries[self.index]
        if e.is_dir:
            self.at_drives = False
            self.cwd, self.filter, self.index = e.path, "", 0
            self.refresh()
            return None
        return e.path

    def up(self) -> None:
        if self.cwd.parent != self.cwd:
            old = self.cwd
            self.cwd, self.filter = self.cwd.parent, ""
            self.at_drives = False
            self.refresh()
            for i, e in enumerate(self.entries):
                if e.path == old:
                    self.index = i

    def home(self) -> None:
        self.cwd, self.filter, self.index, self.at_drives = Path.home(), "", 0, False
        self.refresh()

    def drives(self) -> None:
        if os.name == "nt":
            self.at_drives, self.index = True, 0
            self.refresh()


def render(state: PickerState, title: str, typing_filter: bool, height: int) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold bright_cyan", pad_edge=False)
    t.add_column("Type", width=7)
    t.add_column("Name", ratio=3, overflow="ellipsis", no_wrap=True)
    t.add_column("Info", ratio=2, no_wrap=True)
    rows = max(5, height - 10)
    start = max(0, min(state.index - rows // 2, len(state.entries) - rows))
    for i in range(start, min(len(state.entries), start + rows)):
        e = state.entries[i]
        sel = i == state.index
        if e.is_dir:
            typ, style, info = "[DIR]", "bold bright_blue", ""
        else:
            ext = e.path.suffix.lower().lstrip(".").upper()
            typ, style = f"[{ext}]", "bright_green"
            ai = probe_audio(e.path) if sel else None
            try:
                size = e.path.stat().st_size / 1024
                info = f"{size / 1024:.1f} MB" if size > 1024 else f"{size:.0f} KB"
            except OSError:
                info = ""
            if ai:
                info += f" | {ai.short()}"
        mark = Text("> " if sel else "  ", style="bold yellow")
        name = Text.assemble(mark, (e.label, style))
        row_style = "reverse" if sel else ""
        t.add_row(typ, name, info, style=row_style)
    if not state.entries:
        t.add_row("", Text("(no folders or audio files here)", style="dim italic"), "")
    path_line = Text.assemble(("Path: ", "bold"), (str("Drives" if state.at_drives else state.cwd), "cyan"),
                              ("   Filter: ", "bold"), (state.filter or "*", "magenta" if typing_filter else "dim"))
    keys = Text("[↑/↓] Move  [Enter] Open/Select  [Bksp/←] Up  [/] Filter  [~] Home  [D] Drives  [Esc] Cancel",
                style="bold yellow")
    note = Text("Only folders and supported audio files are shown.", style="dim")
    return Panel(Group(path_line, t, note, keys), title=f"[bold magenta]{title}[/]",
                 border_style="bright_cyan", box=box.ROUNDED)


def pick_file(console: Console, title: str, start: Path | None = None) -> Path | None:
    state = PickerState(start)
    typing = False
    with Live(render(state, title, typing, console.height), console=console, screen=True,
              auto_refresh=False) as live:
        while True:
            live.update(render(state, title, typing, console.height), refresh=True)
            k = read_key()
            if typing:
                if k.name == "enter" or k.name == "esc":
                    typing = False
                elif k.name == "backspace":
                    state.filter = state.filter[:-1]
                    state.refresh()
                elif k.name == "char":
                    state.filter += k.char
                    state.refresh()
                elif k.name in ("up", "down"):
                    state.move(-1 if k.name == "up" else 1)
                continue
            if k.name == "esc":
                return None
            elif k.name == "up":
                state.move(-1)
            elif k.name == "down":
                state.move(1)
            elif k.name == "pgup":
                state.move(-10)
            elif k.name == "pgdn":
                state.move(10)
            elif k.name == "home":
                state.index = 0
            elif k.name == "end":
                state.move(len(state.entries))
            elif k.name in ("backspace", "left"):
                state.up()
            elif k.name in ("enter", "right"):
                chosen = state.enter()
                if chosen:
                    return chosen
            elif k.name == "char":
                c = k.char.lower()
                if c == "/":
                    typing, state.filter = True, ""
                    state.refresh()
                elif c == "~":
                    state.home()
                elif c == "d":
                    state.drives()
                elif c == "q":
                    return None
