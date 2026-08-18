"""Wires the tray icon and the control panel to the automation service.

The tray icon is now only a launcher and a status light: left-clicking it opens
the panel, which owns every control the old right-click menu used to. That is
why Exit and Open logs live on the panel -- with no context menu, they are the
only way out of the app.

The panel holds no state. It reads from :meth:`TrayApp._panel_state` and writes
back through the callbacks below, so the saved settings, the service and the
controls cannot drift apart.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import win32api
import win32event
import winerror

from ..version import __version__
from .icon import TrayIcon
from .modes import DEFAULT_MODE, MODES
from .panel import ControlPanel
from .service import AutomationService, RunState
from .settings import PanelSettings, settings_path

log = logging.getLogger(__name__)

APP_TITLE = "ttheart-sender"
#: Shown on the panel and in the tooltip, so a screenshot is enough to tell
#: which build someone is running.
APP_VERSION = f"{APP_TITLE} v{__version__}"
#: One tray per session -- two of them would fight over the same emulator.
MUTEX_NAME = "ttheart-sender-tray-singleton"
#: The flow behind the panel's "Buy tsum" button.
PURCHASE_FLOW = "purchase_box"
PURCHASE_LABEL = "Buy tsum"

_ASSETS = Path(__file__).resolve().parent / "assets"
ICON_IDLE = _ASSETS / "tray-idle.ico"
ICON_RUNNING = _ASSETS / "tray-running.ico"


class TrayApp:
    """Tray icon + control panel + the service that runs the flows."""

    def __init__(
        self,
        app,
        *,
        mode: Optional[str] = None,
        play: Optional[bool] = None,
        autostart: bool = False,
    ) -> None:
        self._app = app
        self._autostart = autostart
        self._settings_path = settings_path(app.config)
        self._settings = PanelSettings.load(self._settings_path)
        # Command-line flags win over the saved file for this session only.
        if mode is not None and mode != DEFAULT_MODE:
            self._settings.mode = mode
        if play is not None and play:
            self._settings.auto_play = True

        self._service = AutomationService(
            app,
            mode=self._settings.mode,
            play=self._settings.auto_play,
            on_change=self._on_change,
            on_notify=self._on_notify,
        )
        self._panel = ControlPanel(
            title=APP_TITLE,
            version_text=APP_VERSION,
            modes=MODES,
            get_state=self._panel_state,
            on_mode=self._set_mode,
            on_toggle=self._set_toggle,
            on_purchase=self._set_purchase,
            on_run=self._service.toggle,
            on_buy=self._buy_tsum,
            on_logs=self._open_logs,
            on_exit=self._exit,
        )
        self._icon = TrayIcon(
            title=APP_TITLE,
            tooltip=self._tooltip,
            icon_path=self._icon_path,
            # No menu factory: right-clicking the icon does nothing, by design.
            on_left_click=self._panel.toggle,
        )

    # -- lifecycle -------------------------------------------------------
    def run(self) -> int:
        handle = _claim_singleton()
        if handle is None:
            log.error("ttheart-sender is already running (check the system tray).")
            return 1
        try:
            log.info(
                "%s started -- left-click the tray icon for the panel. Stop key: %s",
                APP_VERSION,
                str(self._app.config.runner.stop_key).upper(),
            )
            if self._autostart:
                self._service.start()
            return self._icon.run()
        finally:
            self._service.shutdown()
            self._panel.destroy()
            win32api.CloseHandle(handle)

    # -- panel state -----------------------------------------------------
    def _panel_state(self) -> Dict[str, Any]:
        state = self._service.state
        return {
            "mode": self._service.mode.key,
            "always_on_top": self._settings.always_on_top,
            "auto_play": self._service.play,
            "purchase": dict(self._settings.purchase),
            "running": state is RunState.RUNNING,
            "stopping": state is RunState.STOPPING,
            "status": self._service.status_text(),
        }

    def _save(self) -> None:
        self._settings.save(self._settings_path)

    # -- panel callbacks -------------------------------------------------
    def _set_mode(self, key: str) -> None:
        self._service.set_mode(key)
        self._settings.mode = self._service.mode.key
        self._save()

    def _set_toggle(self, name: str, value: bool) -> None:
        if name == "auto_play":
            self._service.set_play(value)
            self._settings.auto_play = self._service.play
        elif name == "always_on_top":
            self._settings.always_on_top = bool(value)
        else:
            log.debug("Unknown panel toggle %r", name)
            return
        self._save()

    def _set_purchase(self, key: str, value: bool) -> None:
        if key not in self._settings.purchase:
            log.debug("Unknown purchase toggle %r", key)
            return
        self._settings.purchase[key] = bool(value)
        self._save()

    def _buy_tsum(self) -> None:
        """Run purchase_box with the panel's tick boxes as flow variables.

        In-process rather than shelling out to ``python main.py run
        purchase_box``: the .exe has no interpreter beside it, and a second
        process would drive the same emulator as the tray without either
        knowing about the other.
        """
        self._service.start_job(
            PURCHASE_LABEL,
            PURCHASE_FLOW,
            variables=dict(self._settings.purchase),
        )

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
        self._panel.hide()
        self._icon.quit()

    # -- service callbacks (worker thread) -------------------------------
    def _on_change(self) -> None:
        self._icon.post(self._refresh)

    def _refresh(self) -> None:
        self._icon.refresh()
        self._panel.refresh()

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
