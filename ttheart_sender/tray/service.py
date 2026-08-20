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
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..exceptions import TTHeartError
from .modes import DEFAULT_MODE, MODES, Mode, get_mode
from .settings import (
    CLAIM_PATTERN_DEFAULT,
    RETURN_HEART_MINUTES_DEFAULT,
    TSUM_MODE_DEFAULT,
    bowl_reject_value,
    claim_all_flag,
    clamp_minutes,
    normalize_claim_pattern,
    normalize_tsum_mode,
)

log = logging.getLogger(__name__)

#: How long :meth:`AutomationService.shutdown` waits for a run to notice the
#: stop request. Runs check the stop flag between steps and during waits, so
#: this only ever expires if a single action is wedged.
SHUTDOWN_TIMEOUT = 8.0

#: Flow variable holding the odds (in percent) that a cycle plays a round.
#: launch.yaml / resume.yaml declare their own value; the tray always overrides
#: it. The panel offers a tick box rather than odds, so from here it is only
#: ever all or nothing -- a console run keeps the finer grain via `--var`.
PLAY_CHANCE_VAR = "play_chance_percent"
#: What that variable becomes while the tray toggle is off -- never play.
PLAY_CHANCE_OFF = 0
#: ...and while it is on -- play every cycle.
PLAY_CHANCE_ON = 100

#: Flow variables behind the panel's "Return Heart" section: whether hearts go
#: out on the clock at all, and the minutes of the hour they go out on.
RETURN_HEART_VAR = "return_heart_timed"
RETURN_HEART_MINUTES_VAR = "return_heart_minutes"

#: Flow variable behind the panel's "Claim pattern" radio. The panel picks a
#: pattern by name; claim_mailbox.yaml only wants to know which of its two
#: branches to take, so the name is boiled down to this flag on the way out.
CLAIM_ALL_VAR = "claim_all"

#: Flow variables behind the panel's "Play Settings" section. play.yaml
#: forwards both into the `play_tsum` step's `options:`, so they reach the
#: board reader without the panel having to know that step exists.
TSUM_MODE_VAR = "tsum_mode"
BOWL_REJECT_VAR = "bowl_reject"


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
        return_heart: bool = False,
        return_heart_minutes: Optional[Sequence[int]] = None,
        claim_pattern: str = CLAIM_PATTERN_DEFAULT,
        tsum_mode: str = TSUM_MODE_DEFAULT,
        bowl_reject: bool = True,
        on_change: Optional[Callable[[], None]] = None,
        on_notify: Optional[Callable[[str, str, bool], None]] = None,
    ) -> None:
        self._app = app
        self._mode: Mode = get_mode(mode) or MODES[0]
        self._play = bool(play)
        self._return_heart = bool(return_heart)
        self._return_heart_minutes = clamp_minutes(
            RETURN_HEART_MINUTES_DEFAULT if return_heart_minutes is None else return_heart_minutes
        )
        self._claim_pattern = normalize_claim_pattern(claim_pattern)
        self._tsum_mode = normalize_tsum_mode(tsum_mode)
        self._bowl_reject = bool(bowl_reject)
        self._state = RunState.IDLE
        #: What the live run is called -- the mode's label, or "Buy tsum" for
        #: a one-off job, so the panel can say what it is waiting on.
        self._job_label: Optional[str] = None
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
    def return_heart(self) -> bool:
        """Whether hearts go out on the clock rather than every cycle."""
        with self._lock:
            return self._return_heart

    @property
    def return_heart_minutes(self) -> List[int]:
        """The minutes of the hour hearts go out on, while that is on."""
        with self._lock:
            return list(self._return_heart_minutes)

    @property
    def claim_pattern(self) -> str:
        """Which pattern the mailbox pass uses -- see ``CLAIM_PATTERNS``."""
        with self._lock:
            return self._claim_pattern

    @property
    def tsum_mode(self) -> str:
        """How a played round decides two tsums are the same character."""
        with self._lock:
            return self._tsum_mode

    @property
    def bowl_reject(self) -> bool:
        """Whether finds that are really the bowl showing get thrown away."""
        with self._lock:
            return self._bowl_reject

    @property
    def state(self) -> RunState:
        with self._lock:
            return self._state

    @property
    def busy(self) -> bool:
        return self.state is not RunState.IDLE

    @property
    def job_label(self) -> Optional[str]:
        with self._lock:
            return self._job_label

    def status_text(self) -> str:
        state = self.state
        label = self.job_label or self.mode.label
        if state is RunState.RUNNING:
            return f"Running: {label}"
        if state is RunState.STOPPING:
            return f"Stopping: {label}"
        return f"Idle ({self.mode.label})"

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

    def set_return_heart(self, enabled: bool) -> bool:
        """Turn timed heart-sending on or off for the next Start."""
        enabled = bool(enabled)
        with self._lock:
            if self._return_heart is enabled:
                return False
            self._return_heart = enabled
        log.info(
            "Return Heart %s (%s)",
            "on" if enabled else "off",
            self._describe_return_heart(),
        )
        self._on_change()
        return True

    def set_return_heart_minutes(self, minutes: Sequence[int]) -> bool:
        """Set the marks the next Start hands the flow. Clamped to 0-59."""
        marks = clamp_minutes(minutes)
        with self._lock:
            if self._return_heart_minutes == marks:
                return False
            self._return_heart_minutes = marks
        log.info("Return Heart marks set to %s", self._describe_return_heart())
        self._on_change()
        return True

    def set_claim_pattern(self, pattern: str) -> bool:
        """Pick the mailbox claim pattern the next Start will hand the flow.

        Like :meth:`set_mode`, a live run keeps what it started with: the flow
        reads ``claim_all`` once per mailbox pass, and swapping it underneath
        a pass in progress would leave half the mailbox claimed each way.
        """
        chosen = normalize_claim_pattern(pattern, self.claim_pattern)
        with self._lock:
            if self._claim_pattern == chosen:
                return False
            self._claim_pattern = chosen
        log.info("Claim pattern set to %s (%s=%s)",
                 chosen, CLAIM_ALL_VAR, claim_all_flag(chosen))
        self._on_change()
        return True

    def set_tsum_mode(self, mode: str) -> bool:
        """Pick the board reading the next Start will hand the flow.

        Like every other setting here, a live run keeps what it started with:
        the reading decides what counts as a chain, and swapping it mid-round
        would make the round's own log unreadable.
        """
        chosen = normalize_tsum_mode(mode, self.tsum_mode)
        with self._lock:
            if self._tsum_mode == chosen:
                return False
            self._tsum_mode = chosen
        log.info("Tsum mode set to %s (%s=%s)", chosen, TSUM_MODE_VAR, chosen)
        self._on_change()
        return True

    def set_bowl_reject(self, enabled: bool) -> bool:
        """Turn the "ignore the bowl" filter on or off for the next run."""
        enabled = bool(enabled)
        with self._lock:
            if self._bowl_reject is enabled:
                return False
            self._bowl_reject = enabled
        log.info("Bowl reject %s (%s=%s)", "on" if enabled else "off",
                 BOWL_REJECT_VAR, bowl_reject_value(enabled))
        self._on_change()
        return True

    def _describe_play(self) -> str:
        return f"{PLAY_CHANCE_VAR}={self._chance()}"

    def _describe_return_heart(self) -> str:
        if not self.return_heart:
            return "every cycle"
        return ", ".join(f":{minute:02d}" for minute in self.return_heart_minutes)

    def _chance(self) -> int:
        """All or nothing: the panel's tick box is the only switch there is."""
        return PLAY_CHANCE_ON if self.play else PLAY_CHANCE_OFF

    def _variables(self) -> Optional[Dict[str, Any]]:
        """Overrides for the next run. The panel always has the last word."""
        return {
            PLAY_CHANCE_VAR: self._chance(),
            RETURN_HEART_VAR: self.return_heart,
            RETURN_HEART_MINUTES_VAR: self.return_heart_minutes,
            CLAIM_ALL_VAR: claim_all_flag(self.claim_pattern),
            TSUM_MODE_VAR: self.tsum_mode,
            BOWL_REJECT_VAR: bowl_reject_value(self.bowl_reject),
        }

    def start(self) -> bool:
        """Run the selected mode. Does nothing if a run is already going."""
        mode = self.mode
        # Snapshot the overrides here so a mid-run toggle cannot change what
        # this run was started with.
        return self.start_job(
            mode.label,
            mode.flow,
            variables=self._variables(),
            loops=mode.loops,
            loop_delay=mode.loop_delay,
            name=mode.key,
        )

    def start_job(
        self,
        label: str,
        flow: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
        loops: int = 1,
        loop_delay: float = 0.0,
        name: Optional[str] = None,
    ) -> bool:
        """Run any flow through the same one-at-a-time worker as the modes.

        The panel's "Buy tsum" is a flow like any other -- it just is not a
        mode, because picking it must not change what the Run button will do
        next. Routing it through here keeps the invariant that matters: one
        run at a time, on a worker thread, stoppable by the same switch.
        """
        with self._lock:
            if self._state is not RunState.IDLE:
                log.info("Ignoring %s: a run is already %s", label, self._state.value)
                return False
            self._state = RunState.RUNNING
            self._job_label = label
            self._thread = threading.Thread(
                target=self._run,
                args=(label, flow, variables, loops, loop_delay),
                name=f"ttheart-{name or flow}",
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
    def _run(
        self,
        label: str,
        flow: str,
        variables: Optional[Dict[str, Any]],
        loops: int,
        loop_delay: float,
    ) -> None:
        log.info("=== tray: %s (run %s) ===", label, flow)
        try:
            self._app.stop.reset()
            # Re-detect and re-park every run: LDPlayer may have been closed,
            # restarted or moved since the last one, which would leave the
            # cached window handle pointing at nothing.
            self._app.startup(require_window=True, prepare=True)
            report = self._app.run_flow(
                flow,
                variables=variables,
                loops=loops,
                loop_delay=loop_delay,
            )
        except TTHeartError as exc:
            log.error("%s failed: %s", label, exc)
            self._notify(f"{label} failed", str(exc), True)
        except Exception as exc:  # noqa: BLE001 - a crash must not kill the tray
            log.exception("Unexpected error while running %s", label)
            self._notify(f"{label} crashed", f"{type(exc).__name__}: {exc}", True)
        else:
            self._report(label, report)
        finally:
            with self._lock:
                self._state = RunState.IDLE
                self._job_label = None
                self._thread = None
            self._on_change()

    def _report(self, label: str, report) -> None:
        if report.stopped_early:
            log.info("%s stopped: %s", label, report.summary())
            self._notify(f"{label} stopped", report.summary(), False)
        elif report.success:
            self._notify(f"{label} finished", report.summary(), False)
        else:
            self._notify(f"{label} failed", report.summary(), True)

    def _notify(self, title: str, message: str, is_error: bool) -> None:
        try:
            self._on_notify(title, message, is_error)
        except Exception:  # noqa: BLE001 - notification failure is cosmetic
            log.debug("Tray notification failed", exc_info=True)
