"""Wires the tray icon to the automation service.

This module owns the menu layout and nothing else: the icon knows how to draw a
menu, the service knows how to run a flow, and this decides what the menu says.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import win32api
import win32event
import winerror

from ..version import __version__
from .icon import MenuItem, TrayIcon, separator
from .modes import DEFAULT_MODE, MODES
from .service import AutomationService, RunState

log = logging.getLogger(__name__)

APP_TITLE = "ttheart-sender"
#: Shown in the menu and the tooltip, so a screenshot of the tray is enough to
#: tell which build someone is running.
APP_VERSION = f"{APP_TITLE} v{__version__}"
#: One tray per session -- two of them would fight over the same emulator.
MUTEX_NAME = "ttheart-sender-tray-singleton"

_ASSETS = Path(__file__).resolve().parent / "assets"
ICON_IDLE = _ASSETS / "tray-idle.ico"
ICON_RUNNING = _ASSETS / "tray-running.ico"


class TrayApp:
    """The tray front-end: a mode picker, Start/Stop, and a way out."""

    def __init__(
        self,
        app,
        *,
        mode: str = DEFAULT_MODE,
        play: bool = False,
        autostart: bool = False,
    ) -> None:
        self._app = app
        self._autostart = autostart
        self._service = AutomationService(
            app,
            mode=mode,
            play=play,
            on_change=self._on_change,
            on_notify=self._on_notify,
        )
        self._icon = TrayIcon(
            title=APP_TITLE,
            menu=self._build_menu,
            tooltip=self._tooltip,
            icon_path=self._icon_path,
            on_double_click=self._service.toggle,
        )

    # -- lifecycle -------------------------------------------------------
    def run(self) -> int:
        handle = _claim_singleton()
        if handle is None:
            log.error("ttheart-sender is already running (check the system tray).")
            return 1
        try:
            log.info(
                "%s started -- right-click the icon to pick a mode. Stop key: %s",
                APP_VERSION,
                str(self._app.config.runner.stop_key).upper(),
            )
            if self._autostart:
                self._service.start()
            return self._icon.run()
        finally:
            self._service.shutdown()
            win32api.CloseHandle(handle)

    # -- menu ------------------------------------------------------------
    def _build_menu(self) -> List[MenuItem]:
        state = self._service.state
        idle = state is RunState.IDLE
        current = self._service.mode

        modes = [
            MenuItem(
                label=mode.label,
                action=lambda key=mode.key: self._service.set_mode(key),
                radio=True,
                checked=mode is current,
            )
            for mode in MODES
        ]

        stop_key = str(self._app.config.runner.stop_key or "").upper()
        stop_label = f"Stop ({stop_key})" if stop_key else "Stop"

        return [
            MenuItem(label=APP_VERSION, enabled=False),
            MenuItem(label=self._service.status_text(), enabled=False),
            separator(),
            MenuItem(label="Mode", items=modes),
            # A tick, not a mode: the send-hearts cycle either rolls the dice on
            # playing a round or it never does. Takes effect on the next Start.
            MenuItem(
                label="Play rounds",
                action=self._service.toggle_play,
                checked=self._service.play,
            ),
            separator(),
            MenuItem(
                label=f"Start {current.label}",
                action=self._service.start,
                enabled=idle,
                default=True,
            ),
            MenuItem(
                label=stop_label,
                action=self._service.stop,
                enabled=state is RunState.RUNNING,
            ),
            separator(),
            MenuItem(label="Open logs folder", action=self._open_logs),
            separator(),
            MenuItem(label="Exit", action=self._exit),
        ]

    # -- presentation ----------------------------------------------------
    def _tooltip(self) -> str:
        return f"{APP_VERSION} - {self._service.status_text()}"

    def _icon_path(self) -> Optional[Path]:
        return ICON_RUNNING if self._service.busy else ICON_IDLE

    # -- actions ---------------------------------------------------------
    def _open_logs(self) -> None:
        log_dir = self._app.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(log_dir))  # noqa: S606 - opening a folder in Explorer

    def _exit(self) -> None:
        # Ask the run to stop, then let the message loop unwind; run() joins the
        # worker afterwards so this handler never blocks the GUI thread.
        self._service.stop()
        self._icon.quit()

    # -- service callbacks (worker thread) -------------------------------
    def _on_change(self) -> None:
        self._icon.post(self._icon.refresh)

    def _on_notify(self, title: str, message: str, is_error: bool) -> None:
        self._icon.post(lambda: self._icon.notify(title, message, error=is_error))


def _claim_singleton() -> Optional[int]:
    """Return a mutex handle, or ``None`` if another tray already holds it."""
    handle = win32event.CreateMutex(None, False, MUTEX_NAME)
    if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
        if handle:
            win32api.CloseHandle(handle)
        return None
    return handle
