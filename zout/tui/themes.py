"""Terminal color themes and styling manager for noiZout TUI.

Provides 4 distinct color themes:
- Forensic Cyan (default): High-tech digital forensic analysis palette
- Amber Phosphor: Vintage monochrome CRT phosphor monitor aesthetic
- Matrix Emerald: Classic cyberpunk green terminal aesthetic
- Studio Dark: Sleek modern professional DAW dark mode
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Theme:
    name: str
    primary: str
    secondary: str
    accent: str
    key: str
    border: str
    title: str = ""
    muted: str = "dim"
    status: str = "bold yellow"
    success: str = "bold green"
    warning: str = "bold yellow"
    error: str = "bold red"
    bar_fill: str = ""
    bar_empty: str = "grey37"
    meter_low: str = ""
    meter_mid: str = ""
    meter_high: str = "bright_red"
    meter_peak: str = "bright_white on red"

    def __post_init__(self) -> None:
        if not self.title:
            object.__setattr__(self, "title", f"bold {self.accent}")
        if not self.bar_fill:
            object.__setattr__(self, "bar_fill", self.accent)
        if not self.meter_low:
            object.__setattr__(self, "meter_low", self.primary)
        if not self.meter_mid:
            object.__setattr__(self, "meter_mid", self.secondary)


# The 4 Mandated Color Themes
FORENSIC_CYAN = Theme(
    name="Forensic Cyan",
    primary="bright_cyan",
    secondary="cyan",
    accent="magenta",
    key="bold black on bright_cyan",
    border="cyan",
    title="bold bright_magenta",
    bar_fill="bright_cyan",
    bar_empty="grey37",
    meter_low="bright_cyan",
    meter_mid="cyan",
    meter_high="bright_magenta",
    meter_peak="bright_white on red",
)

AMBER_PHOSPHOR = Theme(
    name="Amber Phosphor",
    primary="bright_yellow",
    secondary="dark_orange",
    accent="gold1",
    key="bold black on bright_yellow",
    border="dark_orange",
    title="bold bright_yellow",
    bar_fill="bright_yellow",
    bar_empty="grey27",
    meter_low="yellow",
    meter_mid="dark_orange",
    meter_high="bright_red",
    meter_peak="bright_white on red",
)

MATRIX_EMERALD = Theme(
    name="Matrix Emerald",
    primary="bright_green",
    secondary="green",
    accent="spring_green1",
    key="bold black on bright_green",
    border="green",
    title="bold bright_green",
    bar_fill="bright_green",
    bar_empty="grey23",
    meter_low="bright_green",
    meter_mid="green",
    meter_high="spring_green1",
    meter_peak="bright_white on red",
)

STUDIO_DARK = Theme(
    name="Studio Dark",
    primary="bright_blue",
    secondary="grey70",
    accent="bright_white",
    key="bold white on grey23",
    border="bright_blue",
    title="bold bright_white",
    bar_fill="bright_blue",
    bar_empty="grey30",
    meter_low="bright_blue",
    meter_mid="bright_cyan",
    meter_high="bright_magenta",
    meter_peak="bright_white on red",
)

THEMES: List[Theme] = [
    FORENSIC_CYAN,
    AMBER_PHOSPHOR,
    MATRIX_EMERALD,
    STUDIO_DARK,
]


class ThemeManager:
    """Thread-safe state manager for TUI terminal color themes."""

    def __init__(self, default_theme: str = "Forensic Cyan") -> None:
        self._themes: Dict[str, Theme] = {t.name: t for t in THEMES}
        self._order: List[str] = [t.name for t in THEMES]
        self._lookup: Dict[str, Theme] = {t.name.lower(): t for t in THEMES}
        self._current_idx: int = 0
        self._default: str = default_theme
        self._lock = threading.Lock()

    @property
    def current_theme(self) -> Theme:
        with self._lock:
            return self._themes[self._order[self._current_idx]]

    def cycle_theme(self) -> Theme:
        """Advance sequentially to the next theme in the cycle and wrap around."""
        with self._lock:
            self._current_idx = (self._current_idx + 1) % len(self._order)
            return self._themes[self._order[self._current_idx]]

    def get_theme(self, name: Optional[str] = None) -> Theme:
        """Get a theme by name (case-insensitive). If None, return active theme.
        If unknown name, fallback gracefully to default theme.
        """
        if name is None:
            return self.current_theme
        normalized = name.strip().lower()
        return self._lookup.get(normalized, self._themes[self._default])

    def set_theme(self, name: str) -> Theme:
        """Set the active theme by name."""
        with self._lock:
            theme = self.get_theme(name)
            self._current_idx = self._order.index(theme.name)
            return theme

    def reset(self) -> None:
        """Reset to initial default theme."""
        with self._lock:
            self._current_idx = 0


# Module-level singleton and public API
_theme_manager = ThemeManager()

cycle_theme = _theme_manager.cycle_theme
get_theme = _theme_manager.get_theme
set_theme = _theme_manager.set_theme
reset = _theme_manager.reset
