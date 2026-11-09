from __future__ import annotations

import os
import sys
from dataclasses import dataclass

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


@dataclass(frozen=True)
class Key:
    name: str   # "up","down","left","right","enter","esc","backspace","tab","home","end","pgup","pgdn","char"
    char: str = ""


_WIN_SPECIAL = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
                "I": "pgup", "Q": "pgdn"}


def read_key() -> Key:
    """Block until a single keypress. Cross-platform, no third-party deps."""
    if os.name == "nt":
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return Key(_WIN_SPECIAL.get(msvcrt.getwch(), "unknown"))
        return _simple(ch)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            import select
            if select.select([sys.stdin], [], [], 0.05)[0]:
                seq = sys.stdin.read(2)
                m = {"[A": "up", "[B": "down", "[C": "right", "[D": "left",
                     "[H": "home", "[F": "end", "[5": "pgup", "[6": "pgdn"}
                return Key(m.get(seq, "unknown"))
            return Key("esc")
        return _simple(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _simple(ch: str) -> Key:
    if ch in ("\r", "\n"):
        return Key("enter")
    if ch == "\x1b":
        return Key("esc")
    if ch in ("\x08", "\x7f"):
        return Key("backspace")
    if ch == "\t":
        return Key("tab")
    if ch == "\x03":
        raise KeyboardInterrupt
    return Key("char", ch)
