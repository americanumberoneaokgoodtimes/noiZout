from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


@dataclass(frozen=True)
class Key:
    name: str  # "up","down","left","right","enter","esc","backspace","tab","space","home","end","pgup","pgdn","char"
    char: str = ""


_WIN_SPECIAL = {
    "H": "up",
    "P": "down",
    "K": "left",
    "M": "right",
    "G": "home",
    "O": "end",
    "I": "pgup",
    "Q": "pgdn",
}


def read_key(timeout: float | None = None) -> Key | None:
    """Read a single keypress. Cross-platform, no third-party deps.
    If timeout is None: blocks until a key is pressed.
    If timeout is float: waits up to timeout seconds. If no key is pressed, returns None.
    """
    if os.name == "nt":
        try:
            if timeout is not None:
                start = time.perf_counter()
                while not msvcrt.kbhit():
                    if time.perf_counter() - start >= timeout:
                        return None
                    time.sleep(0.005)
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                sp = msvcrt.getwch()
                return Key(_WIN_SPECIAL.get(sp, "unknown"), "")
            return _simple(ch)
        except Exception:
            return None
    else:
        if not sys.stdin.isatty():
            return None
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if timeout is not None:
                import select
                r, _, _ = select.select([sys.stdin], [], [], timeout)
                if not r:
                    return None
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                import select
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    seq = sys.stdin.read(2)
                    m = {
                        "[A": "up", "[B": "down", "[C": "right", "[D": "left",
                        "[H": "home", "[F": "end", "[5": "pgup", "[6": "pgdn",
                    }
                    return Key(m.get(seq, "unknown"), "")
                return Key("esc", "\x1b")
            return _simple(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def poll_key(timeout: float | None = 0.0) -> Key | None:
    """Non-blocking check for a pending keypress. Returns Key or None immediately."""
    return read_key(timeout=timeout if timeout is not None else 0.0)


def _simple(ch: str) -> Key:
    if ch in ("\r", "\n"):
        return Key("enter", "\r")
    if ch == "\x1b":
        return Key("esc", "\x1b")
    if ch in ("\x08", "\x7f"):
        return Key("backspace", "\x08")
    if ch == "\t":
        return Key("tab", "\t")
    if ch == " ":
        return Key("space", " ")
    if ch == "\x03":
        raise KeyboardInterrupt
    return Key("char", ch)
