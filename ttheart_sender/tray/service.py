"""Runs a mode on a background thread so the tray stays responsive.

The GUI thread must never block: it owns the window message loop, and a frozen
message loop means a tray icon that cannot even be used to press Stop. So every
run happens on a worker thread, and the only things crossing the thread
boundary are an immutable state snapshot and two callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

from ..exceptions import TTHeartError
from .modes import DEFAULT_MODE, MODES, Mode, get_mode
from .settings import (
    CLAIM_PATTERN_DEFAULT,
    RETURN_HEART_MINUTES_DEFAULT,
    claim_all_flag,
    clamp_minutes,
    normalize_claim_pattern,
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

#: Flow variable behind the panel's "Restart when stucked" tick box. resume.yaml
#: and play.yaml declare their own value; the tray always overrides it.
STUCK_CHECK_VAR = "stuck_check"

#: `fit_effort` is deliberately NOT a variable here. It was the panel's
#: "Steady colour fit" box while it was an experiment; the thirteenth round
#: settled it and `flows/*.yaml` declare `fit_effort: 3` outright.
#:
#: The tray must not send it, and that is a correctness point rather than
#: tidiness: "the panel always has the last word" in `_variables()` is exactly
#: what would defeat the documented revert. Edit `fit_effort: 1` into a flow's
#: `vars:` while the tray still passed the variable, and every run started
#: from the panel would overwrite it -- the revert would appear to do nothing,
#: silently, and only when driven from the tray. A settled setting gets one
#: home.

#: `verify_clears` is settled the same way and is NOT here either. It is ON
#: in flows/play.yaml, permanently, because it is the ruler every other rule
#: is priced with -- it re-reads the board after each drag using the frame
#: `--verify` already grabbed, so it costs a few disk means and no capture.
#: The panel box that used to send it defaulted to OFF, so any run started
#: from the tray without ticking it silently turned the measurement off and
#: the round came back unmeasurable. One home: the flow.

#: `verify_extend` is NOT sent by the tray and no longer has a box. It was one,
#: and the box had been ticked for 248 consecutive rounds -- the whole corpus.
#: When the experiments became a radio group, choosing any other one silently
#: unticked it, and the 18 rounds that followed measured two changes at once.
#: It now lives in `flows/*.yaml` at `true`, which is what the baseline is.
#: Kept as a name because the flows still have to declare and forward it.
VERIFY_EXTEND_VAR = "verify_extend"


class Experiment(NamedTuple):
    """One unproven play rule, and the two values its tick box chooses between.

    **What belongs here is exactly what a played round has not yet settled.**
    A rule the rounds have decided gets one home in `flows/*.yaml` and no box
    -- `fit_effort` and `verify_clears` are both here in spirit and neither is
    in this table, for the reason written above them: the panel has the last
    word in `_variables()`, so a settled setting that is ALSO sent from the
    tray cannot be reverted by editing the flow. The revert would appear to do
    nothing, silently, and only when driven from the tray.

    The value is not always a boolean, because a flow variable is not always a
    switch. `step_px` picks between two distances and `kinds` between two group
    counts; the box still reads as on/off because that is the only question
    being asked of it -- run the round the new way, or the way every number in
    docs/DATASET-FINDINGS.md was taken.
    """

    key: str        #: the PanelSettings field and the panel-state key
    var: str        #: the flow variable it writes
    on: Any         #: value sent when the box is ticked
    off: Any        #: value sent when it is not -- the flow's own default
    label: str      #: what the box says
    note: str       #: what a round should show if it is working


#: What "no experiment" is called. Not a sixth Experiment row, because it has
#: no variable of its own: it is the absence of all of them, and every switch
#: sends its `off` value under it.
NO_EXPERIMENT = ""

#: Every switch under "Experiments" in the panel, in the order they appear.
#:
#: **They are RADIO buttons, and that is a correctness point.** Each one
#: changes a different part of the pipeline, so a round played with two of them
#: armed cannot say which was responsible for whatever it shows -- the house
#: rule at the top of flows/play.yaml, made structural instead of advisory. A
#: tick box asks the player to remember; a radio makes arming two impossible.
#:
#: Three rules deliberately NOT here:
#:
#: * The **board filter** (`reject_model`). It had a row, it was played, and
#:   it lost: 143 rounds on one build, alternating, 72 ON against 71 OFF.
#:   Cleared 256.9 against 275.4 (p=0.036), FEVER 41.2% against 47.1%
#:   (p=0.030), and rounds that never reached FEVER at all went from 1.4% to
#:   12.5% (p=0.009). It did exactly what it was built for -- dead drags fell
#:   48% -- and lost anyway. A row for it would invite a round spent
#:   re-finding that, which is the same reason the character model has none.
#:   The code, the model and the corpus all stay; only the invitation goes.
#:
#: * The **character model**. It is measured, and the measurement is that it
#:   loses: -10.9% tsums cleared per drag, because naming the fifth of a board
#:   it can read splits characters whose buried members keep a colour id. See
#:   docs/IDENTITY.md. A row for it would invite a round spent re-finding that.
#: * **`verify_extend`**. It was a row here, and that was a mistake with a
#:   cost. It had been ticked for 248 consecutive rounds -- the whole corpus --
#:   so choosing any OTHER experiment silently switched it off too, and the
#:   next 18 rounds measured two changes at once. It now lives in
#:   `flows/*.yaml` at `true`, which is what the baseline actually is, and
#:   reverts by editing that line the way `fit_effort` does.
#:
#: The general rule this pair illustrates: a row here must be OFF in the
#: corpus the experiment will be compared against. Anything already on is part
#: of the baseline, and putting it in a radio group turns every other row into
#: a two-variable experiment.
#: WHY A ROW SAYS "(no evidence)" AND IS STILL HERE.
#:
#: There are three states a rule can be in, and only two of them used to have
#: a home. A rule that has been PLAYED AND LOST leaves the panel entirely --
#: the character model and the board filter both did, because a row for one
#: invites a round spent re-finding a settled answer. A rule that is UNPROVEN
#: gets a row, which is what this table is for.
#:
#: The third state had no name until 2026-09-05: **the rule was never played,
#: and the reason for proposing it has since been withdrawn.** All three rows
#: below are in it. Deleting them would claim they had lost, which is not
#: true and no round has said so. Leaving them unmarked would invite five
#: hours of play for a question nobody currently has. So they are marked, and
#: the label is where a person actually looks.
EXPERIMENTS: tuple = (
    Experiment(
        "chain_ranker", "chain_model", "models/chain.onnx", "",
        "Pick the chain the game will accept",
        "THE ONLY ROW WITH EVIDENCE BEHIND IT. Offline, on rounds it never "
        "trained on: 0.876 AUC per member against `adjacency`'s 0.500, a "
        "calibrated expected total (2.274 predicted, 2.310 actual), and "
        "+0.109 accepted members a press that survives scoring by a second "
        "independently trained model. Never played. Watch accepted members "
        "per press, then `cleared`; +4.7% needs ~136 rounds an arm."),
    Experiment(
        "settle_board", "settle_board", True, False,
        "Wait for the board, not the screen (no evidence)",
        "PREMISE WITHDRAWN. It was proposed because score was said to track "
        "chains played at r=+0.91 -- five rounds. Over the 140 scored rounds "
        "of the clean single-build baseline that is -0.17, so more chains per "
        "round is not the objective. The wait is still the biggest slice of a "
        "round, so this may yet pay for some other reason; it has no case "
        "today."),
    Experiment(
        "fast_stroke", "step_px", 12, 8,
        "Faster stroke (no evidence)",
        "PREMISE WITHDRAWN, the same one as `settle_board`: it is the second "
        "throughput lever, and throughput is not what the 140-round baseline "
        "says the score is made of."),
    Experiment(
        "four_groups", "kinds", 4, 0,
        "Force 4 colour groups (no evidence)",
        "SCORED ON A DISQUALIFIED METRIC. The +3-4% that motivated it was "
        "simulated tsums-cleared-per-drag, which the twenty-sixth round "
        "disqualified when an ORACLE grouping scored BELOW the shipped rule "
        "on it. Never played."),
)


#: The two variables that turn the armed experiment into an A/B.
#:
#: NOT an `EXPERIMENTS` row, and that is the point. The rows are a radio group
#: because two rules armed at once cannot be told apart -- but this is not a
#: rule, it is a way of RUNNING one, and it has to compose with whichever row
#: is armed rather than replace it. A row for it would make "A/B the board
#: filter" unselectable, because selecting it would unselect the board filter.
#:
#: Both values are derived from the armed row, so the pair cannot drift: `ab`
#: takes that row's `var` and `ab_off` takes its `off`. That matters for
#: `step_px`, whose off-value is 8 rather than an empty one -- an A/B that
#: guessed the off-value would alternate 12 against 0 and measure nothing
#: anybody chose.
AB_VAR = "ab"
AB_OFF_VAR = "ab_off"


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
        restart_when_stuck: bool = False,
        experiment: Optional[str] = None,
        ab_experiment: bool = False,
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
        self._restart_when_stuck = bool(restart_when_stuck)
        # One dict rather than a field per switch: adding an experiment should
        # be a row in EXPERIMENTS, not five edits across three files, and a
        # switch that exists in the table but nowhere else is the failure this
        # avoids.
        # One selection, not a set. See EXPERIMENTS.
        # A SET, because the panel is tick boxes again. See `experiments`.
        want = experiment if experiment is not None else ()
        if isinstance(want, str):
            want = (want,) if want else ()
        valid = {e.key for e in EXPERIMENTS}
        self._experiments = {k for k in want if k in valid}
        self._ab_experiment = bool(ab_experiment)
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
    def restart_when_stuck(self) -> bool:
        """Whether a run restarts the emulator when it decides it has wedged."""
        with self._lock:
            return self._restart_when_stuck


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

    @property
    def ab_experiment(self) -> bool:
        """Whether the armed experiment alternates round by round."""
        with self._lock:
            return self._ab_experiment

    def set_ab_experiment(self, enabled: bool) -> bool:
        """Alternate the armed experiment ON/OFF between rounds, or do not.

        Off is the baseline: the armed row is on for every round, which is what
        every experiment before this one was played as. On makes the run its
        own control -- `scripts/ab_eval.py` reads the arms back out.
        """
        enabled = bool(enabled)
        with self._lock:
            if self._ab_experiment is enabled:
                return False
            self._ab_experiment = enabled
        log.info("A/B the armed experiment %s", "on" if enabled else "off")
        self._on_change()
        return True

    def set_restart_when_stuck(self, enabled: bool) -> bool:
        """Arm or disarm the stuck watchdog for the next Start.

        Like every other switch here this only decides what the *next* run is
        handed -- a live run keeps the variables it started with, so disarming
        it cannot call off a restart already under way.
        """
        enabled = bool(enabled)
        with self._lock:
            if self._restart_when_stuck is enabled:
                return False
            self._restart_when_stuck = enabled
        log.info("Restart when stucked %s (%s=%s)",
                 "on" if enabled else "off", STUCK_CHECK_VAR, enabled)
        self._on_change()
        return True

    @property
    def experiments(self) -> frozenset:
        """Every armed experiment's key. Empty is the baseline."""
        with self._lock:
            return frozenset(self._experiments)

    @property
    def experiment_states(self) -> Dict[str, bool]:
        """Every switch and whether it is ticked, for the panel's rows.

        Named apart from `experiments` on purpose: it used to BE `experiments`,
        and when the set-valued property arrived below it the later definition
        won silently. `e.key in armed` then tested a dict's KEYS, every row
        read as armed, and a baseline run sent every experiment's ON value.
        Caught by the suite; the lesson is that two properties differing only
        in what they return is a shadowing waiting to happen.
        """
        armed = self.experiments
        return {e.key: (e.key in armed) for e in EXPERIMENTS}

    @property
    def experiment(self) -> str:
        """The armed experiment when there is EXACTLY one, else "".

        Not a convenience. Several things downstream -- the A/B pair most of
        all -- are only answerable when one rule is under test, and returning
        the "first" of two would answer them wrongly rather than not at all.
        """
        with self._lock:
            return (next(iter(self._experiments))
                    if len(self._experiments) == 1 else NO_EXPERIMENT)

    def set_experiment(self, key: str, enabled: bool = True) -> bool:
        """Tick or untick one experiment. Nothing ticked is the baseline.

        These were tick boxes, then a radio group, and are tick boxes again at
        the player's request. The radio existed for a real reason and the
        reason has not gone away: **a round played with two rules armed cannot
        say which was responsible for what it shows.** The thirty-fifth round
        lost 18 rounds to exactly that, twice over.

        So the constraint is kept and moved from the control to the record.
        Ticking a second row is allowed and is said out loud -- in the log, in
        the panel's own line, and through a notification -- rather than being
        made unreachable. Nothing downstream guesses: `experiment` answers only
        when exactly one is armed, so the A/B pair goes empty rather than
        picking one of two.

        Returns False when nothing changed, so a repaint driven by the panel
        does not loop back through `_on_change`. A live run keeps what it
        started with either way.
        """
        spec = next((e for e in EXPERIMENTS if e.key == key), None)
        if spec is None:
            log.warning("unknown experiment %r -- ignored", key)
            return False
        with self._lock:
            armed = set(self._experiments)
            if (key in armed) == bool(enabled):
                return False
            armed.add(key) if enabled else armed.discard(key)
            self._experiments = armed
        if not armed:
            log.info("Experiments: none -- baseline round")
        else:
            log.info("Experiment %s: %s (%s=%s). Armed: %s",
                     "on" if enabled else "off", spec.label, spec.var,
                     spec.on if enabled else spec.off,
                     ", ".join(sorted(armed)))
        if len(armed) > 1:
            log.warning(
                "%d experiments armed at once (%s). A round cannot say which "
                "of them was responsible for what it shows -- this is the "
                "shape that cost the thirty-fifth round 18 rounds.",
                len(armed), ", ".join(sorted(armed)))
            self._on_notify(
                "Two experiments armed",
                f"{len(armed)} rules are on at once. A round played this way "
                f"cannot attribute its result to either of them.", False)
        self._on_change()
        return True

    def clear_experiments(self) -> bool:
        """Back to the baseline -- the round every recorded number is against."""
        with self._lock:
            if not self._experiments:
                return False
            self._experiments = set()
        log.info("Experiments: none -- baseline round")
        self._on_change()
        return True


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
        armed = self.experiments
        return {
            PLAY_CHANCE_VAR: self._chance(),
            RETURN_HEART_VAR: self.return_heart,
            RETURN_HEART_MINUTES_VAR: self.return_heart_minutes,
            CLAIM_ALL_VAR: claim_all_flag(self.claim_pattern),
            STUCK_CHECK_VAR: self.restart_when_stuck,
            # Every experiment sends a value either way, never only when armed.
            # Sending nothing when off would leave the flow's own default in
            # place, which happens to be the same value today -- and would stop
            # being so the moment a default changed, turning an unticked box
            # into a silent opt-in.
            **{e.var: (e.on if e.key in armed else e.off)
               for e in EXPERIMENTS},
            # Derived from the armed row, never typed. Empty when nothing is
            # armed or the box is off, which is the baseline: `_ab_arm`
            # refuses an empty name and the round plays exactly as before.
            **self._ab_variables(armed),
        }

    def _ab_variables(self, armed) -> Dict[str, Any]:
        """`ab` and `ab_off` for the one armed row, or the pair that means "no".

        Deliberately silent when two rows are armed. Alternating one of them
        while the other stays on for every round produces a corpus that reads
        like a clean A/B and is not one, and that is worse than no A/B at all.
        """
        if not self.ab_experiment:
            return {AB_VAR: "", AB_OFF_VAR: ""}
        if len(armed) > 1:
            log.warning("A/B is on with %d experiments armed -- alternating "
                        "nothing, because a corpus that looks like a clean A/B "
                        "and is not one is worse than none", len(armed))
            return {AB_VAR: "", AB_OFF_VAR: ""}
        for spec in EXPERIMENTS:
            if spec.key in armed:
                return {AB_VAR: spec.var, AB_OFF_VAR: str(spec.off)}
        # Ticked with nothing armed. There is no experiment to alternate, and
        # saying so beats alternating the baseline against itself.
        log.info("A/B is on but no experiment is armed -- nothing to alternate")
        return {AB_VAR: "", AB_OFF_VAR: ""}

    def start(self) -> bool:
        """Run the selected mode. Does nothing if a run is already going.

        **Never refuses a press.** A cooldown used to sit here, meant to stop
        the run being restarted seconds after the cursor safety had ended it.
        It was wrong on its own evidence: those restarts were already known to
        be explicit `start()` calls, and refusing an explicit press is exactly
        what must not happen. It made the first click after every cursor-stop
        do nothing, so Run appeared to need pressing twice.
        """
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
