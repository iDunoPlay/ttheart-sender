"""Watches for a newer release and, when asked to, installs it.

Same shape as :mod:`.service`, and for the same reason: the GUI thread owns the
message loop, so an HTTPS call -- which can sit there for a whole timeout --
has to happen somewhere else. One worker thread does the checking, downloading
and swapping; the panel only ever reads a snapshot of the state and pushes
buttons through the methods below.

What the tick box means: the *check* always happens, so the panel can say a
newer build exists, and "Auto Update" decides whether finding one is enough to
act on. Even then an update never lands mid-run -- restarting would abandon a
flow half way through -- so an automatic install waits for the service to go
idle.
"""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
from enum import Enum
from typing import Callable, Optional

from ..update import (
    DEFAULT_REPO,
    Release,
    UpdateError,
    Version,
    can_self_update,
    clean_leftovers,
    current_exe,
    download,
    install,
    latest_release,
    staged_path,
    why_not_self_update,
)
from ..version import __version__

log = logging.getLogger(__name__)

#: How long after startup the first check runs. Long enough that launching the
#: app never waits on the network, short enough that the panel someone opens
#: straight away is answered while they are still looking at it.
FIRST_CHECK_DELAY = 20.0
#: How often to look again while an install is waiting for the run to finish.
#: Minutes rather than hours: the point is to catch the moment it goes idle.
PENDING_POLL = 60.0
#: Floor and ceiling on the configured interval, so a hand-edited config can
#: neither hammer the API nor switch the feature off by accident.
MIN_INTERVAL_HOURS = 0.25
MAX_INTERVAL_HOURS = 24 * 7
#: The status line is one row of a 300px panel, and the text that lands in it
#: includes whatever :mod:`urllib` had to say about a failure.
STATUS_LIMIT = 46


class UpdateState(Enum):
    IDLE = "idle"
    CHECKING = "checking"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


class UpdateService:
    """Release checks, downloads and the restart, off the GUI thread."""

    def __init__(
        self,
        config,
        *,
        auto: bool = False,
        on_change: Optional[Callable[[], None]] = None,
        on_notify: Optional[Callable[[str, str, bool], None]] = None,
        #: Whether an install may restart the app right now -- the tray says
        #: no while a flow is running.
        can_apply: Optional[Callable[[], bool]] = None,
        #: Called once the new build is in place; the tray exits, and the
        #: helper script starts the replacement.
        on_restart: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._auto = bool(auto)
        self._on_change = on_change or (lambda: None)
        self._on_notify = on_notify or (lambda title, message, is_error: None)
        self._can_apply = can_apply or (lambda: True)
        self._on_restart = on_restart or (lambda: None)

        self._lock = threading.RLock()
        self._state = UpdateState.IDLE
        self._latest: Optional[Release] = None
        self._error = ""
        self._percent = 0
        self._checked_at: Optional[float] = None
        #: Set by the panel; picked up by the worker at its next turn.
        self._command: Optional[str] = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- state -----------------------------------------------------------
    @property
    def state(self) -> UpdateState:
        with self._lock:
            return self._state

    @property
    def latest(self) -> Optional[Release]:
        with self._lock:
            return self._latest

    @property
    def auto(self) -> bool:
        with self._lock:
            return self._auto

    @property
    def busy(self) -> bool:
        return self.state in (UpdateState.CHECKING, UpdateState.DOWNLOADING, UpdateState.READY)

    @property
    def update_available(self) -> bool:
        release = self.latest
        return bool(release and release.is_newer_than())

    def status_text(self) -> str:
        """The panel's one-line summary of where the update stands."""
        with self._lock:
            state, release, error, percent = (
                self._state, self._latest, self._error, self._percent
            )
            checked = self._checked_at
        return _fit(self._status_text(state, release, error, percent, checked))

    @staticmethod
    def _status_text(state, release, error, percent, checked) -> str:
        if state is UpdateState.CHECKING:
            return "Checking for updates..."
        if state is UpdateState.DOWNLOADING:
            name = release.tag if release else "update"
            return f"Downloading {name} - {percent}%"
        if state is UpdateState.READY:
            return f"Restarting into {release.tag}..." if release else "Restarting..."
        if state is UpdateState.FAILED:
            return error or "Update failed"
        if state is UpdateState.AVAILABLE and release is not None:
            reason = "" if can_self_update() else f" ({why_not_self_update()})"
            return f"{release.tag} available{reason}"
        if checked is None:
            return f"v{__version__}"
        return f"v{__version__} - up to date"

    def button_label(self) -> str:
        """What the button beside the tick box currently offers."""
        state = self.state
        if state is UpdateState.CHECKING:
            return "Checking"
        if state is UpdateState.DOWNLOADING:
            return "Working"
        if state is UpdateState.READY:
            return "Restarting"
        if state is UpdateState.AVAILABLE:
            return "Update" if can_self_update() else "Open"
        if state is UpdateState.FAILED:
            return "Retry"
        return "Check"

    def _set(self, state: UpdateState, *, error: str = "") -> None:
        with self._lock:
            unchanged = self._state is state and self._error == error
            self._state = state
            self._error = error
            if state is not UpdateState.DOWNLOADING:
                self._percent = 0
        if not unchanged:
            self._on_change()

    # -- commands (GUI thread) -------------------------------------------
    def start(self) -> None:
        """Begin checking in the background. Safe to call once."""
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._loop, name="ttheart-update", daemon=True
            )
            self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop checking. A download in flight notices at its next chunk."""
        self._stopping.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def set_auto(self, enabled: bool) -> bool:
        """Turn automatic installing on or off."""
        enabled = bool(enabled)
        with self._lock:
            if self._auto is enabled:
                return False
            self._auto = enabled
        log.info("Auto Update %s", "on" if enabled else "off")
        # Ticking the box with an update already found should act on it now
        # rather than at the next scheduled check.
        if enabled:
            self._wake.set()
        self._on_change()
        return True

    def activate(self) -> None:
        """The button beside the tick box: check, install, or open the page."""
        state = self.state
        release = self.latest
        if state in (UpdateState.CHECKING, UpdateState.DOWNLOADING, UpdateState.READY):
            return
        if state is UpdateState.AVAILABLE and release is not None:
            if not can_self_update():
                self._open_release_page(release)
                return
            self._request("install")
            return
        self._request("check")

    def _open_release_page(self, release: Release) -> None:
        """Hand a source checkout or folder install over to the browser."""
        url = release.url or "https://github.com/"
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - no browser is not a crash
            log.warning("Could not open %s", url, exc_info=True)

    def _request(self, command: str) -> None:
        with self._lock:
            self._command = command
        self._wake.set()

    # -- worker ----------------------------------------------------------
    def _loop(self) -> None:
        exe = current_exe()
        if exe is not None:
            # A build that started from a swap leaves the one it replaced
            # behind if the helper script lost the race; tidy it here.
            clean_leftovers(exe.parent)
        if self._sleep(FIRST_CHECK_DELAY):
            return
        while not self._stopping.is_set():
            with self._lock:
                command, self._command = self._command, None
            try:
                if command == "install":
                    self._install()
                elif command == "check" or self._check_due():
                    self._check()
                self._maybe_auto_install()
            except Exception:  # noqa: BLE001 - the worker must outlive a bad turn
                log.exception("Update worker failed")
                self._set(UpdateState.FAILED, error="Update check failed")
            if self._sleep(self._next_delay()):
                return

    def _sleep(self, seconds: float) -> bool:
        """Wait, unless a command or shutdown arrives. True means stop."""
        if self._stopping.is_set():
            return True
        self._wake.wait(max(0.0, seconds))
        self._wake.clear()
        return self._stopping.is_set()

    def _next_delay(self) -> float:
        """How long to doze for -- shorter while an install is pending."""
        if self.state is UpdateState.AVAILABLE and self.auto:
            return PENDING_POLL
        return self._interval_seconds()

    def _interval_seconds(self) -> float:
        hours = float(getattr(self._config, "check_interval_hours", 6.0) or 6.0)
        return max(MIN_INTERVAL_HOURS, min(MAX_INTERVAL_HOURS, hours)) * 3600.0

    def _check_due(self) -> bool:
        if not getattr(self._config, "enabled", True):
            return False
        with self._lock:
            checked = self._checked_at
        return checked is None or (time.monotonic() - checked) >= self._interval_seconds()

    def _check(self) -> None:
        """Ask GitHub what the newest release is and record the answer."""
        if not getattr(self._config, "enabled", True):
            log.debug("Update checks are disabled in config.yaml")
            return
        self._set(UpdateState.CHECKING)
        try:
            release = latest_release(
                getattr(self._config, "repo", None) or DEFAULT_REPO,
                include_prereleases=bool(getattr(self._config, "include_prereleases", False)),
                timeout=float(getattr(self._config, "timeout", 10.0)),
            )
        except UpdateError as exc:
            log.info("Update check failed: %s", exc)
            self._set(UpdateState.FAILED, error=str(exc))
            return
        with self._lock:
            self._latest = release
            self._checked_at = time.monotonic()
        if release is None:
            log.info("No published release to compare against")
            self._set(UpdateState.IDLE)
            return
        if not release.is_newer_than():
            log.info("Up to date (running %s, latest %s)", __version__, release.tag)
            self._set(UpdateState.IDLE)
            return
        log.info("Update available: %s (running %s)", release.tag, __version__)
        self._set(UpdateState.AVAILABLE)
        self._notify("Update available", f"{release.tag} is ready to install", False)

    def _maybe_auto_install(self) -> None:
        """Install by ourselves, if that is what the tick box asked for."""
        if not self.auto or self.state is not UpdateState.AVAILABLE:
            return
        if not can_self_update():
            return
        if not self._apply_allowed():
            log.debug("Holding %s back until the run finishes", self.latest and self.latest.tag)
            return
        self._install()

    def _apply_allowed(self) -> bool:
        try:
            return bool(self._can_apply())
        except Exception:  # noqa: BLE001 - never let a callback strand an update
            log.debug("can_apply callback failed", exc_info=True)
            return False

    def _install(self) -> None:
        """Download the new build, put it in place and ask for a restart."""
        release = self.latest
        if release is None or not release.is_newer_than():
            self._check()
            return
        if not release.has_asset:
            self._set(UpdateState.FAILED, error=f"{release.tag} has no .exe to download")
            return
        exe = current_exe()
        if exe is None or not can_self_update(exe):
            self._set(UpdateState.FAILED, error=f"Update by hand ({why_not_self_update(exe)})")
            return
        if not self._apply_allowed():
            self._set(UpdateState.AVAILABLE)
            self._notify("Update waiting", "Stop the run to install it", False)
            return

        staged = staged_path(exe)
        self._set(UpdateState.DOWNLOADING)
        try:
            download(
                str(release.asset_url),
                staged,
                expected_size=release.asset_size,
                timeout=float(getattr(self._config, "timeout", 10.0)) * 3,
                on_progress=self._progress,
                should_stop=self._stopping.is_set,
            )
            log.info("Installing %s over %s", release.tag, exe.name)
            install(staged, exe)
        except UpdateError as exc:
            log.error("Update to %s failed: %s", release.tag, exc)
            self._set(UpdateState.FAILED, error=str(exc))
            self._notify("Update failed", str(exc), True)
            return
        self._set(UpdateState.READY)
        self._notify("Updating", f"Restarting into {release.tag}", False)
        try:
            self._on_restart()
        except Exception:  # noqa: BLE001 - the swap already happened
            log.exception("Restart handler failed after installing %s", release.tag)

    def _progress(self, written: int, total: int) -> None:
        """Report whole percent steps only -- each one repaints the panel."""
        percent = int(written * 100 / total) if total else 0
        with self._lock:
            if percent == self._percent:
                return
            self._percent = percent
        self._on_change()

    def _notify(self, title: str, message: str, is_error: bool) -> None:
        try:
            self._on_notify(title, message, is_error)
        except Exception:  # noqa: BLE001 - notification failure is cosmetic
            log.debug("Update notification failed", exc_info=True)


def _fit(text: str) -> str:
    """Cut a status down to what the label can actually show."""
    if len(text) <= STATUS_LIMIT:
        return text
    return text[: STATUS_LIMIT - 3].rstrip() + "..."


__all__ = ["UpdateService", "UpdateState", "Version"]
