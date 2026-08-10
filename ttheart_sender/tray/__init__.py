"""System tray front-end.

Wraps the same :class:`~ttheart_sender.app.Application` the CLI drives, so the
tray is purely a way to pick a mode and start/stop it -- no automation logic
lives here.
"""

from __future__ import annotations

from .modes import DEFAULT_MODE, MODES, Mode, get_mode
from .service import AutomationService, RunState

__all__ = [
    "AutomationService",
    "DEFAULT_MODE",
    "MODES",
    "Mode",
    "RunState",
    "get_mode",
    "run_tray",
]


def run_tray(app, *, mode: str = DEFAULT_MODE, autostart: bool = False) -> int:
    """Show the tray icon and pump messages until the user quits."""
    from .app import TrayApp

    return TrayApp(app, mode=mode, autostart=autostart).run()
