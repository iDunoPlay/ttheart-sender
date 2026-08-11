"""Runs a mode on a background thread so the tray stays responsive.

The GUI thread must never block: it owns the window message loop, and a frozen
message loop means a tray icon that cannot even be used to press Stop. So every
run happens on a worker thread, and the only things crossing the thread
boundary are an immutable state snapshot and two callbacks.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..exceptions import TTHeartError
from .modes import DEFAULT_MODE, MODES, Mode, get_mode

log = logging.getLogger(__name__)

#: How long :meth:`AutomationService.shutdown` waits for a run to notice the
#: stop request. Runs check the stop flag between steps and during waits, so
#: this only ever expires if a single action is wedged.
SHUTDOWN_TIMEOUT = 8.0

#: Flow variable holding the odds (in percent) that a cycle plays a round.
#: launch.yaml / resume.yaml declare their own value; the tray overrides it with
#: :data:`PLAY_CHANCE_OFF` unless the user ticks "Play rounds".
PLAY_CHANCE_VAR = "play_chance_percent"
#: What that variable becomes while the tray toggle is off -- never play.
PLAY_CHANCE_OFF = 0


class RunState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"


class AutomationService:
    """Start/stop one flow at a time, off the GUI thread."""

    def __init__(
        self,
        app,
        *,
        mode: str = DEFAULT_MODE,
        play: bool = False,
        on_change: Optional[Callable[[], None]] = None,
        on_notify: Optional[Callable[[str, str, bool], None]] = None,
    ) -> None:
        self._app = app
        self._mode: Mode = get_mode(mode) or MODES[0]
        self._play = bool(play)
        self._state = RunState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._on_change = on_change or (lambda: None)
        self._on_notify = on_notify or (lambda title, message, is_error: None)

    # -- state -----------------------------------------------------------
    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    @property
    def play(self) -> bool:
        """Whether a run is allowed to break off and play a round."""
        with self._lock:
            return self._play

    @property
    def state(self) -> RunState:
        with self._lock:
            return self._state

    @property
    def busy(self) -> bool:
        return self.state is not RunState.IDLE

    def status_text(self) -> str:
        state = self.state
        mode = self.mode
        if state is RunState.RUNNING:
            return f"Running: {mode.label}"
        if state is RunState.STOPPING:
            return f"Stopping: {mode.label}"
        return f"Idle ({mode.label})"

    def _set_state(self, state: RunState) -> None:
        with self._lock:
            if self._state is state:
                return
            self._state = state
        self._on_change()

    # -- commands --------------------------------------------------------
    def set_mode(self, key: str) -> bool:
        """Pick the mode the next Start will run. Never disturbs a live run."""
        mode = get_mode(key)
        if mode is None:
            log.warning("Unknown tray mode %r", key)
            return False
        with self._lock:
            if self._mode is mode:
                return False
            self._mode = mode
        log.info("Mode set to %s (%s)", mode.label, mode.command)
        self._on_change()
        return True

    def set_play(self, enabled: bool) -> bool:
        """Turn the play-a-round dice roll on or off for the next Start.

        Like :meth:`set_mode` this only decides what the *next* run is handed:
        a live run keeps the variables it was started with.
        """
        enabled = bool(enabled)
        with self._lock:
            if self._play is enabled:
                return False
            self._play = enabled
        log.info("Play rounds %s (%s)", "on" if enabled else "off", self._describe_play())
        self._on_change()
        return True

    def toggle_play(self) -> bool:
        return self.set_play(not self.play)

    def _describe_play(self) -> str:
        return "flow default" if self.play else f"{PLAY_CHANCE_VAR}={PLAY_CHANCE_OFF}"

    def _variables(self) -> Optional[Dict[str, Any]]:
        """Overrides for the next run; ``None`` leaves the flow's own vars be."""
        return None if self.play else {PLAY_CHANCE_VAR: PLAY_CHANCE_OFF}

    def start(self) -> bool:
        """Run the selected mode. Does nothing if a run is already going."""
        with self._lock:
            if self._state is not RunState.IDLE:
                log.info("Ignoring Start: a run is already %s", self._state.value)
                return False
            mode = self._mode
            self._state = RunState.RUNNING
            self._thread = threading.Thread(
                target=self._run,
                # Snapshot the overrides here so a mid-run toggle cannot change
                # what this run was started with.
                args=(mode, self._variables()),
                name=f"ttheart-{mode.key}",
                daemon=True,
            )
            self._thread.start()
        self._on_change()
        return True

    def stop(self) -> bool:
        """Ask the running flow to stop at its next checkpoint."""
        with self._lock:
            if self._state is not RunState.RUNNING:
                return False
        # Same switch the F12 hotkey flips, so both paths unwind identically.
        self._app.stop.request_stop()
        self._set_state(RunState.STOPPING)
        log.info("Stop requested from the tray")
        return True

    def toggle(self) -> None:
        if self.busy:
            self.stop()
        else:
            self.start()

    def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """Stop any run and wait (briefly) for the worker to unwind."""
        self.stop()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                log.warning("Automation thread did not stop within %.0fs", timeout)

    # -- worker ----------------------------------------------------------
    def _run(self, mode: Mode, variables: Optional[Dict[str, Any]] = None) -> None:
        log.info("=== tray: %s (%s) ===", mode.label, mode.command)
        try:
            self._app.stop.reset()
            # Re-detect and re-park every run: LDPlayer may have been closed,
            # restarted or moved since the last one, which would leave the
            # cached window handle pointing at nothing.
            self._app.startup(require_window=True, prepare=True)
            report = self._app.run_flow(
                mode.flow,
                variables=variables,
                loops=mode.loops,
                loop_delay=mode.loop_delay,
            )
        except TTHeartError as exc:
            log.error("%s failed: %s", mode.label, exc)
            self._notify(f"{mode.label} failed", str(exc), True)
        except Exception as exc:  # noqa: BLE001 - a crash must not kill the tray
            log.exception("Unexpected error while running %s", mode.label)
            self._notify(f"{mode.label} crashed", f"{type(exc).__name__}: {exc}", True)
        else:
            self._report(mode, report)
        finally:
            with self._lock:
                self._state = RunState.IDLE
                self._thread = None
            self._on_change()

    def _report(self, mode: Mode, report) -> None:
        if report.stopped_early:
            log.info("%s stopped: %s", mode.label, report.summary())
            self._notify(f"{mode.label} stopped", report.summary(), False)
        elif report.success:
            self._notify(f"{mode.label} finished", report.summary(), False)
        else:
            self._notify(f"{mode.label} failed", report.summary(), True)

    def _notify(self, title: str, message: str, is_error: bool) -> None:
        try:
            self._on_notify(title, message, is_error)
        except Exception:  # noqa: BLE001 - notification failure is cosmetic
            log.debug("Tray notification failed", exc_info=True)
