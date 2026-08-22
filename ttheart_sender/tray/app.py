"""Wires the tray icon and the control panel to the automation service.

The tray icon is a launcher and a status light: left-clicking it opens the
panel, which owns every control the old right-click menu used to. Right-clicking
offers Exit and nothing else -- a way out that does not depend on finding the
panel first.

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
from .icon import MenuItem, TrayIcon
from .modes import DEFAULT_MODE, MODES
from .panel import ControlPanel
from .service import AutomationService, RunState
from ..housekeeping import clear_dataset, clear_logs
from .settings import PanelSettings, settings_path
from .updater import UpdateService

log = logging.getLogger(__name__)

APP_TITLE = "ttheart-sender"
#: The panel's caption and the tray tooltip, so a screenshot of either is
#: enough to tell which build someone is running.
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
        self._seed_collection(self._settings_path.exists())

        self._service = AutomationService(
            app,
            mode=self._settings.mode,
            play=self._settings.auto_play,
            return_heart=self._settings.return_heart,
            return_heart_minutes=self._settings.return_heart_minutes,
            claim_pattern=self._settings.claim_pattern,
            restart_when_stuck=self._settings.restart_when_stuck,
            on_change=self._on_change,
            on_notify=self._on_notify,
        )
        self._updater = UpdateService(
            app.config.update,
            auto=self._settings.auto_update,
            on_change=self._on_change,
            on_notify=self._on_notify,
            # Restarting mid-flow would abandon the run half way through, so
            # an automatic install waits for the service to go idle.
            can_apply=lambda: not self._service.busy,
            on_restart=self._restart_for_update,
        )
        self._panel = ControlPanel(
            # The caption carries the version, so a screenshot identifies the
            # build without spending a row of the panel on it.
            title=APP_VERSION,
            modes=MODES,
            get_state=self._panel_state,
            on_mode=self._set_mode,
            on_toggle=self._set_toggle,
            on_return_minute=self._set_return_minute,
            on_claim=self._set_claim_pattern,
            on_purchase=self._set_purchase,
            on_run=self._service.toggle,
            on_buy=self._buy_tsum,
            on_update=self._updater.activate,
            on_logs=self._open_logs,
            on_clear_logs=self._clear_logs,
            on_clear_data=self._clear_dataset,
            on_exit=self._exit,
        )
        self._icon = TrayIcon(
            title=APP_TITLE,
            tooltip=self._tooltip,
            icon_path=self._icon_path,
            menu=self._menu,
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
            self._updater.start()
            if self._autostart:
                self._service.start()
            # Open the panel as soon as the message loop is up: a first-time
            # user has no reason to guess that the tray icon is clickable.
            # Queued rather than shown here because the panel must be created
            # on the thread that will pump it.
            self._icon.post(self._panel.show)
            return self._icon.run()
        finally:
            self._updater.shutdown()
            self._service.shutdown()
            self._panel.destroy()
            win32api.CloseHandle(handle)

    # -- panel state -----------------------------------------------------
    def _seed_collection(self, has_saved_settings: bool) -> None:
        """Agree on one answer for "is collection on" at startup.

        Two switches name the same thing: `dataset.enabled` in config.yaml and
        the panel's tick box. A first run has nothing saved, so config.yaml
        decides and the box shows what the file already said. After that the
        box is the switch -- someone who unticks it means it, and a config
        file that still says `true` must not turn it back on at the next
        launch.
        """
        if has_saved_settings:
            self._app.config.dataset.enabled = self._settings.collect_data
        else:
            self._settings.collect_data = bool(self._app.config.dataset.enabled)

    def _panel_state(self) -> Dict[str, Any]:
        state = self._service.state
        return {
            "mode": self._service.mode.key,
            "always_on_top": self._settings.always_on_top,
            "collect_data": self._settings.collect_data,
            "auto_play": self._service.play,
            "return_heart": self._service.return_heart,
            "return_heart_minutes": self._service.return_heart_minutes,
            "claim_pattern": self._service.claim_pattern,
            "restart_when_stuck": self._service.restart_when_stuck,
            "purchase": dict(self._settings.purchase),
            "auto_update": self._settings.auto_update,
            "update_status": self._updater.status_text(),
            "update_button": self._updater.button_label(),
            "update_ready": not self._updater.busy,
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
        elif name == "return_heart":
            self._service.set_return_heart(value)
            self._settings.return_heart = self._service.return_heart
        elif name == "restart_when_stuck":
            self._service.set_restart_when_stuck(value)
            self._settings.restart_when_stuck = self._service.restart_when_stuck
        elif name == "auto_update":
            self._updater.set_auto(value)
            self._settings.auto_update = self._updater.auto
        elif name == "always_on_top":
            self._settings.always_on_top = bool(value)
        elif name == "collect_data":
            # Applied to the live config rather than only saved: the flow
            # action reads `ctx.config.dataset`, so this takes effect on the
            # next round without a restart.
            self._settings.collect_data = bool(value)
            self._app.config.dataset.enabled = self._settings.collect_data
            log.info("Data collection %s (%s)",
                     "on" if self._settings.collect_data else "off",
                     self._app.config.dataset_dir)
        else:
            log.debug("Unknown panel toggle %r", name)
            return
        self._save()

    def _set_return_minute(self, index: int, minute: int) -> None:
        """Store one Return Heart mark. Only the next run picks it up.

        The panel hands over one box at a time, so the other mark is read back
        off the service rather than the box beside it -- that box may be
        half-typed, and it is the stored value that the flow will be given.
        """
        minutes = list(self._service.return_heart_minutes)
        if not 0 <= index < len(minutes):
            log.debug("Unknown Return Heart mark %r", index)
            return
        minutes[index] = minute
        if not self._service.set_return_heart_minutes(minutes):
            return
        self._settings.return_heart_minutes = self._service.return_heart_minutes
        self._save()

    def _set_claim_pattern(self, key: str) -> None:
        """Store which way the mailbox is emptied. The next run picks it up."""
        if not self._service.set_claim_pattern(key):
            return
        self._settings.claim_pattern = self._service.claim_pattern
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

    def _restart_for_update(self) -> None:
        """Close down so the new build, already in place, can take over.

        Called from the update worker: the swap has happened and a helper
        script is waiting for this process to let go of the old image. Exiting
        has to happen on the GUI thread, so it goes through the icon's queue
        like every other cross-thread call.
        """
        self._icon.post(self._exit)

    # -- presentation ----------------------------------------------------
    def _menu(self):
        """The right-click menu: one way out, wherever the panel happens to be."""
        return [MenuItem("Exit", self._exit)]

    def _tooltip(self) -> str:
        return f"{APP_VERSION} - {self._service.status_text()}"

    def _icon_path(self) -> Optional[Path]:
        return ICON_RUNNING if self._service.busy else ICON_IDLE

    # -- actions ---------------------------------------------------------
    def _open_logs(self) -> None:
        log_dir = self._app.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(log_dir))  # noqa: S606 - opening a folder in Explorer

    def _clear_logs(self) -> None:
        """Empty the log directory. The panel has already asked."""
        self._report_clear("Clear logs", clear_logs(self._app.config))

    def _clear_dataset(self) -> None:
        """Empty the dataset directory. The panel has already asked."""
        self._report_clear("Clear data collection", clear_dataset(self._app.config))

    def _report_clear(self, title: str, result) -> None:
        """Say what was removed, through the tray balloon the rest of the app uses.

        A file left behind is reported as an error rather than swallowed: it
        means something still has it open, and the user asked for it to be gone.
        """
        self._on_notify(title, result.describe(), not result.ok)
        self._on_change()

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
