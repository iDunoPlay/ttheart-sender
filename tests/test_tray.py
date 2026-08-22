"""Tray tests: mode selection, run lifecycle, and the menu it produces.

No real screen, no real emulator and no message loop -- the application is a
stub that records what the service asked it to do.
"""

from __future__ import annotations

import json
import threading
from typing import Optional
from unittest import mock

import pytest

from ttheart_sender.automation.flow import load_flow_by_name
from ttheart_sender.automation.runner import RunReport
from ttheart_sender.config import Config
from ttheart_sender.control.hotkey import StopKeyWatcher
from ttheart_sender.exceptions import WindowNotFoundError
from ttheart_sender.tray.modes import MODES, get_mode
from ttheart_sender.tray.service import (
    CLAIM_ALL_VAR,
    PLAY_CHANCE_OFF,
    PLAY_CHANCE_ON,
    PLAY_CHANCE_VAR,
    RETURN_HEART_MINUTES_VAR,
    RETURN_HEART_VAR,
    STUCK_CHECK_VAR,
    AutomationService,
    RunState,
)
from ttheart_sender.tray.settings import (
    CLAIM_PATTERN_DEFAULT,
    CLAIM_PATTERNS,
    RETURN_HEART_MINUTES_DEFAULT,
    claim_all_flag,
    clamp_minute,
    clamp_minutes,
    normalize_claim_pattern,
)

DEFAULT_MARKS = list(RETURN_HEART_MINUTES_DEFAULT)


def overrides(chance=PLAY_CHANCE_OFF, timed=False, marks=None, claim_all=False,
              stuck_check=False):
    """What a run started from the panel should be handed."""
    return {
        PLAY_CHANCE_VAR: chance,
        RETURN_HEART_VAR: timed,
        RETURN_HEART_MINUTES_VAR: DEFAULT_MARKS if marks is None else list(marks),
        CLAIM_ALL_VAR: claim_all,
        STUCK_CHECK_VAR: stuck_check,
    }


class FakeApp:
    """Stands in for :class:`~ttheart_sender.app.Application`."""

    def __init__(self, *, block: bool = False, error: Optional[Exception] = None) -> None:
        self.config = Config()
        self.stop = StopKeyWatcher(None)  # no real hotkey polling in tests
        self.calls = []
        self.variables = []
        self.startups = 0
        self._error = error
        # When blocking, the "flow" waits until the test releases it, which is
        # how we observe the RUNNING state from the outside.
        self._block = block
        self.entered = threading.Event()
        self.release = threading.Event()

    def startup(self, *, require_window=True, prepare=None):
        self.startups += 1
        if self._error is not None:
            raise self._error
        return None

    def run_flow(self, name, *, variables=None, loops=1, loop_delay=0.0, **kwargs):
        self.calls.append({"flow": name, "loops": loops, "loop_delay": loop_delay})
        self.variables.append(variables)
        if self._block:
            self.entered.set()
            self.release.wait(5)
        return RunReport(flow=name, success=True, steps_run=1)


def wait_for(predicate, timeout=5.0):
    deadline = threading.Event()
    threading.Timer(timeout, deadline.set).start()
    while not predicate():
        if deadline.is_set():
            raise AssertionError("timed out waiting for condition")
    return True


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def test_default_mode_is_resume():
    service = AutomationService(FakeApp())
    assert service.mode.key == "resume"
    assert service.mode.command == "python main.py run resume"


def test_every_mode_maps_to_a_real_flow():
    flows_dir = Config().flows_dir
    for mode in MODES:
        assert get_mode(mode.key) is mode
        # The flows the tray offers must actually ship with the app.
        assert (flows_dir / f"{mode.flow}.yaml").exists(), mode.flow


def test_set_mode_notifies_and_ignores_unknown():
    changes = []
    service = AutomationService(FakeApp(), on_change=lambda: changes.append(1))

    assert service.set_mode("play") is True
    assert service.mode.key == "play"
    assert len(changes) == 1

    assert service.set_mode("play") is False, "re-selecting should be a no-op"
    assert service.set_mode("nonsense") is False
    assert len(changes) == 1


def test_play_mode_repeats_until_stopped():
    play = get_mode("play")
    assert play.loops == 0, "play.yaml is one round; the tray must loop it"


# --------------------------------------------------------------------------
# Play toggle
# --------------------------------------------------------------------------
def test_play_is_off_by_default_and_zeroes_the_chance_variable():
    app = FakeApp()
    service = AutomationService(app)

    assert service.play is False
    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [overrides(PLAY_CHANCE_OFF)]


def test_play_on_is_all_or_nothing():
    """The panel offers a tick box, so the flow only ever sees 100 or 0."""
    app = FakeApp()
    changes = []
    service = AutomationService(app, on_change=lambda: changes.append(1))

    assert service.set_play(True) is True
    assert service.play is True
    assert len(changes) == 1
    assert service.set_play(True) is False, "re-selecting should be a no-op"

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [overrides(PLAY_CHANCE_ON)]


def test_toggling_play_mid_run_does_not_change_the_live_run():
    app = FakeApp(block=True)
    service = AutomationService(app)

    service.start()
    app.entered.wait(5)
    service.toggle_play()
    assert service.play is True

    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [overrides(PLAY_CHANCE_OFF)], (
        "the run keeps the variables it started with"
    )

    # ...but the next one picks the new setting up.
    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables[-1] == overrides(PLAY_CHANCE_ON)


# --------------------------------------------------------------------------
# Return Heart
# --------------------------------------------------------------------------
def test_return_heart_is_off_by_default_so_every_cycle_sends():
    app = FakeApp()
    service = AutomationService(app)

    assert service.return_heart is False
    assert service.return_heart_minutes == DEFAULT_MARKS

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [overrides(timed=False)]


def test_restart_when_stucked_is_off_until_the_panel_ticks_it():
    """The watchdog is opt-in: wrong, it costs a needless emulator restart."""
    app = FakeApp()
    changes = []
    service = AutomationService(app, on_change=lambda: changes.append(1))

    assert service.restart_when_stuck is False
    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [overrides(stuck_check=False)]

    # Counted from here: start() fires on_change on each state transition too,
    # so only the delta across the setter says anything about the setter.
    before = len(changes)
    assert service.set_restart_when_stuck(True) is True
    assert service.set_restart_when_stuck(True) is False, "re-ticking should be a no-op"
    assert len(changes) - before == 1

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables[-1] == overrides(stuck_check=True)


def test_a_live_run_keeps_the_watchdog_setting_it_started_with():
    """Ticking mid-run must not arm a restart the run was not started with."""
    app = FakeApp(block=True)
    service = AutomationService(app, restart_when_stuck=False)

    service.start()
    app.entered.wait(5)
    service.set_restart_when_stuck(True)
    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)

    assert app.variables == [overrides(stuck_check=False)]


def test_the_marks_reach_the_flow_and_are_clamped():
    app = FakeApp()
    changes = []
    service = AutomationService(app, on_change=lambda: changes.append(1))

    assert service.set_return_heart(True) is True
    assert service.set_return_heart(True) is False, "re-ticking should be a no-op"
    assert service.set_return_heart_minutes([5, 35]) is True
    assert service.set_return_heart_minutes([5, 35]) is False, "re-setting is a no-op"

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [overrides(timed=True, marks=[5, 35])]

    # A minute of the hour is what it is; the spinner cannot ask for more.
    service.set_return_heart_minutes([99, -4])
    assert service.return_heart_minutes == [59, 0]


def test_the_marks_survive_return_heart_being_unticked():
    """Untick, retick: the schedule is still the one that was typed."""
    service = AutomationService(FakeApp(), return_heart=True)
    service.set_return_heart_minutes([5, 35])
    service.set_return_heart(False)

    assert service.return_heart_minutes == [5, 35]
    service.set_return_heart(True)
    assert service.return_heart_minutes == [5, 35]


def test_the_marks_a_caller_hands_over_cannot_be_mutated_from_outside():
    """The service owns its schedule -- a stray edit must not reach the flow."""
    marks = [5, 35]
    service = AutomationService(FakeApp(), return_heart_minutes=marks)

    marks[0] = 42
    assert service.return_heart_minutes == [5, 35]

    service.return_heart_minutes.append(99)
    assert service.return_heart_minutes == [5, 35]


def test_unusable_marks_fall_back_instead_of_raising():
    """The edit box and a hand-edited settings file both feed this."""
    assert clamp_minute("12") == 12
    assert clamp_minute("", 15) == 15
    assert clamp_minute(None, 50) == 50
    assert clamp_minute("nonsense", 15) == 15
    assert clamp_minute(7.6) == 8
    assert clamp_minute(99) == 59

    # A stored list of the wrong length or shape still yields two usable marks.
    assert clamp_minutes([5]) == [5, RETURN_HEART_MINUTES_DEFAULT[1]]
    assert clamp_minutes([1, 2, 3]) == [1, 2]
    assert clamp_minutes("nonsense") == DEFAULT_MARKS
    assert clamp_minutes(None) == DEFAULT_MARKS


def find_steps(steps, action):
    for step in steps:
        if step.action == action:
            yield step
        for children in step.children.values():
            yield from find_steps(children, action)


def test_the_chance_step_reads_the_variable_the_tray_overrides():
    """The override is only meaningful if the flow reads it."""
    flow = load_flow_by_name(Config().flows_dir, "resume")

    assert PLAY_CHANCE_VAR in flow.vars, "resume.yaml must ship a default"
    rolls = [step for step in find_steps(flow.steps, "chance")]
    assert rolls, "resume.yaml no longer rolls for a play round"
    assert all(step.params["percent"] == f"${{{PLAY_CHANCE_VAR}}}" for step in rolls)


def test_a_played_round_replaces_the_idle_wait():
    """A cycle either plays or waits -- never both, and never neither."""
    flow = load_flow_by_name(Config().flows_dir, "resume")
    rolls = [step for step in find_steps(flow.steps, "chance")]

    for roll in rolls:
        played = list(find_steps(roll.children.get("then", []), "run_flow"))
        # Only the first call is pinned: the branch may go on to act on what
        # the round reported (unlocking the level cap, say), and those are
        # consequences of having played, not a second thing the cycle chose
        # to do instead of waiting.
        assert [s.params.get("flow") for s in played][:1] == ["play"]
        # Without the else branch an unticked Auto Play would spin the cycle
        # with no pause at all.
        assert list(find_steps(roll.children.get("else", []), "wait")), (
            "the wait has to survive as the branch taken when the roll misses"
        )


def test_launch_forwards_every_override_to_resume():
    """run_flow re-applies the called flow's vars, so they have to be passed on."""
    flow = load_flow_by_name(Config().flows_dir, "launch")
    calls = [s for s in find_steps(flow.steps, "run_flow") if s.params.get("flow") == "resume"]

    assert calls, "launch.yaml no longer hands off to resume"
    for name in (PLAY_CHANCE_VAR, RETURN_HEART_VAR, RETURN_HEART_MINUTES_VAR):
        assert name in flow.vars, "launch.yaml needs its own standalone default"
        for call in calls:
            forwarded = call.params.get("vars", {})
            assert forwarded.get(name) == f"${{{name}}}", (
                "without this, resume.yaml's own default overwrites the tray's choice"
            )


def test_resume_sends_hearts_on_the_clock_only_when_asked_to():
    """Both halves of the gate: the schedule, and the flag it raises."""
    flow = load_flow_by_name(Config().flows_dir, "resume")

    for name in (RETURN_HEART_VAR, RETURN_HEART_MINUTES_VAR):
        assert name in flow.vars, "resume.yaml must ship a standalone default"

    gates = list(find_steps(flow.steps, "time_gate"))
    assert len(gates) == 1, "one schedule, so one gate"
    gate = gates[0]
    assert gate.params["minutes"] == f"${{{RETURN_HEART_MINUTES_VAR}}}"
    flag = gate.params["save_as"]

    # The gate has to be the timed branch of a switch, or an unticked
    # Return Heart would stop hearts going out altogether.
    switch = next(
        step for step in find_steps(flow.steps, "if")
        if step.params.get("value") == f"${{{RETURN_HEART_VAR}}}"
    )
    assert gate in switch.children["then"]
    assert [s.params for s in find_steps(switch.children["else"], "set")] == [{flag: True}]

    # ...and the flag has to be what actually decides whether hearts go out,
    # then be put back down so the next mark is the next send.
    branch = next(
        step for step in find_steps(flow.steps, "if")
        if step.params.get("value") == f"${{{flag}}}"
    )
    sends = list(find_steps(branch.children["then"], "run_flow"))
    assert [s.params.get("flow") for s in sends] == ["send_heart"]
    assert {flag: False} in [s.params for s in find_steps(branch.children["then"], "set")]


def test_claim_pattern_defaults_to_single_and_reaches_the_flow():
    """Unchanged panel = the item-by-item pass, the way it always ran."""
    service = AutomationService(FakeApp())

    assert service.claim_pattern == CLAIM_PATTERN_DEFAULT == "single"
    assert service._variables()[CLAIM_ALL_VAR] is False

    assert service.set_claim_pattern("all") is True
    assert service._variables()[CLAIM_ALL_VAR] is True
    # Setting the same one twice is not a change, so nothing is announced.
    assert service.set_claim_pattern("all") is False


def test_an_unknown_claim_pattern_leaves_the_chosen_one_alone():
    service = AutomationService(FakeApp())
    service.set_claim_pattern("all")

    assert service.set_claim_pattern("sideways") is False
    assert service.claim_pattern == "all"
    assert normalize_claim_pattern(None) == CLAIM_PATTERN_DEFAULT
    assert normalize_claim_pattern(" ALL ") == "all"
    assert [claim_all_flag(key) for key, _label, _flag in CLAIM_PATTERNS] == [False, True]


def test_the_two_claim_patterns_are_the_two_branches_of_one_switch():
    """Exactly one runs: claim-all and item-by-item must never both fire."""
    flow = load_flow_by_name(Config().flows_dir, "claim_mailbox")

    assert CLAIM_ALL_VAR in flow.vars, "claim_mailbox.yaml must ship a default"
    assert flow.vars[CLAIM_ALL_VAR] is False

    switch = next(
        step for step in find_steps(flow.steps, "if")
        if step.params.get("value") == f"${{{CLAIM_ALL_VAR}}}"
    )
    # Claim all: the game's own dialog, behind the badge that says there is
    # something to claim.
    claim_all = list(find_steps(switch.children["then"], "find_click"))
    assert "claim_all_button" in [s.params.get("template") for s in claim_all]
    # Single claim: one Check per item, until the mailbox stops offering them.
    # find_steps walks outermost-first, so the pass itself is loops[0]; the
    # loops nested in it re-tap a swallowed Check and see the dialog off, and
    # both of those wait on the dialog rather than on Check.
    loops = list(find_steps(switch.children["else"], "repeat"))
    assert loops[0].params.get("while_found") == "check_button"
    assert [
        s.params.get("until_found") or s.params.get("while_found") for s in loops[1:]
    ] == ["ok_button", "ok_button"]

    # ...and neither pattern may leak into the other branch.
    assert not list(find_steps(switch.children["then"], "repeat"))
    assert "claim_all_button" not in [
        s.params.get("template") for s in find_steps(switch.children["else"], "find_click")
    ]


def test_the_claim_pattern_is_handed_down_the_whole_chain():
    """launch -> resume -> claim_mailbox: every hop re-applies its own vars."""
    config = Config()
    for name, called in (("launch", "resume"), ("resume", "claim_mailbox")):
        flow = load_flow_by_name(config.flows_dir, name)
        assert CLAIM_ALL_VAR in flow.vars, f"{name}.yaml needs a standalone default"
        calls = [s for s in find_steps(flow.steps, "run_flow") if s.params.get("flow") == called]
        assert calls, f"{name}.yaml no longer calls {called}"
        for call in calls:
            assert call.params.get("vars", {}).get(CLAIM_ALL_VAR) == f"${{{CLAIM_ALL_VAR}}}", (
                f"without this, {called}.yaml's own default overwrites the panel's choice"
            )


# --------------------------------------------------------------------------
# Run lifecycle
# --------------------------------------------------------------------------
def test_start_runs_the_selected_mode_and_returns_to_idle():
    app = FakeApp()
    service = AutomationService(app, mode="resume")

    assert service.start() is True
    wait_for(lambda: service.state is RunState.IDLE)

    assert app.startups == 1, "the emulator must be re-detected for every run"
    assert app.calls == [{"flow": "resume", "loops": 1, "loop_delay": 0.0}]


def test_second_start_is_ignored_while_running():
    app = FakeApp(block=True)
    service = AutomationService(app)

    service.start()
    app.entered.wait(5)
    assert service.state is RunState.RUNNING
    assert service.busy is True
    assert service.status_text() == "Running: Resume"

    assert service.start() is False, "must not run two flows at once"

    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)
    assert len(app.calls) == 1


def test_stop_flips_the_same_switch_as_the_hotkey():
    app = FakeApp(block=True)
    service = AutomationService(app)
    service.start()
    app.entered.wait(5)

    assert service.stop() is True
    assert app.stop.triggered() is True, "the flow's stop watcher must be set"
    assert service.state is RunState.STOPPING
    assert service.status_text() == "Stopping: Resume"

    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)


def test_stop_while_idle_does_nothing():
    service = AutomationService(FakeApp())
    assert service.stop() is False
    assert service.state is RunState.IDLE


def test_missing_emulator_is_reported_not_raised():
    app = FakeApp(error=WindowNotFoundError("No LDPlayer window"))
    notes = []
    service = AutomationService(app, on_notify=lambda t, m, e: notes.append((t, m, e)))

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)

    assert app.calls == [], "the flow must not run without a window"
    assert notes and notes[0][2] is True
    assert "No LDPlayer window" in notes[0][1]


def test_unexpected_error_does_not_wedge_the_service():
    app = FakeApp(error=RuntimeError("boom"))
    service = AutomationService(app)

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)

    # Still usable afterwards.
    app._error = None
    assert service.start() is True
    wait_for(lambda: service.state is RunState.IDLE)
    assert len(app.calls) == 1


def test_toggle_starts_then_stops():
    app = FakeApp(block=True)
    service = AutomationService(app)

    service.toggle()
    app.entered.wait(5)
    assert service.state is RunState.RUNNING

    service.toggle()
    assert service.state is RunState.STOPPING
    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)


# --------------------------------------------------------------------------
# Menu
# --------------------------------------------------------------------------
@pytest.fixture()
def tray(tmp_path):
    from ttheart_sender.tray.app import TrayApp

    app = FakeApp()
    # Keep the saved-settings file out of the repo: TrayApp writes one on
    # every toggle, and output_root defaults to the working directory.
    app.config.output_root = tmp_path
    return TrayApp(app)


def labels(items):
    return [item.label for item in items if item.label]


def test_panel_offers_every_mode_including_the_beta_label(tray):
    state = tray._panel_state()
    assert state["mode"] == "resume"
    assert [mode.label for mode in MODES] == ["Resume", "Launch", "Play (beta)"]


def test_panel_title_carries_the_version(tray):
    """A screenshot of the panel should be enough to identify the build."""
    from ttheart_sender import __version__

    assert tray._panel._title == f"ttheart-sender v{__version__}"


def test_right_click_offers_a_way_out(tray):
    """Left-click opens the panel; right-click is the escape hatch."""
    assert tray._icon._on_left_click is not None
    assert tray._icon._on_double_click is None
    assert labels(tray._menu()) == ["Exit"]


def test_panel_toggles_reach_the_service_and_the_saved_file(tray):
    assert tray._panel_state()["auto_play"] is False
    assert tray._panel_state()["return_heart"] is False
    assert tray._panel_state()["return_heart_minutes"] == DEFAULT_MARKS

    tray._set_toggle("auto_play", True)
    assert tray._service.play is True
    assert tray._panel_state()["auto_play"] is True

    tray._set_toggle("return_heart", True)
    # One box at a time, which is all the panel ever reports.
    tray._set_return_minute(1, 35)
    assert tray._panel_state()["return_heart_minutes"] == [DEFAULT_MARKS[0], 35]

    tray._set_toggle("always_on_top", False)
    tray._set_mode("play")
    assert tray._panel_state()["mode"] == "play"

    # Everything above must survive a restart.
    from ttheart_sender.tray.settings import PanelSettings

    reloaded = PanelSettings.load(tray._settings_path)
    assert reloaded.auto_play is True
    assert reloaded.return_heart is True
    assert reloaded.return_heart_minutes == [DEFAULT_MARKS[0], 35]
    assert reloaded.always_on_top is False
    assert reloaded.mode == "play"


def test_the_claim_radio_reaches_the_service_and_the_saved_file(tray):
    from ttheart_sender.tray.settings import PanelSettings

    assert tray._panel_state()["claim_pattern"] == "single"

    tray._set_claim_pattern("all")
    assert tray._service.claim_pattern == "all"
    assert tray._panel_state()["claim_pattern"] == "all"
    assert PanelSettings.load(tray._settings_path).claim_pattern == "all"

    # A hand-mangled file lights the default radio rather than none of them.
    assert PanelSettings.from_dict({"claim_pattern": "everything"}).claim_pattern == "single"


def test_auto_update_is_on_by_default_and_survives_being_unticked(tray):
    """The check always runs; the tick box only decides whether it acts."""
    from ttheart_sender.tray.settings import PanelSettings

    state = tray._panel_state()
    assert state["auto_update"] is True
    assert state["update_button"] == "Check"
    assert state["update_ready"] is True

    tray._set_toggle("auto_update", False)
    assert tray._updater.auto is False
    assert tray._panel_state()["auto_update"] is False
    assert PanelSettings.load(tray._settings_path).auto_update is False


def test_the_update_button_is_wired_to_the_updater(tray):
    assert tray._panel._on_update == tray._updater.activate
    # Nothing found yet, so pressing it asks GitHub rather than installing.
    tray._panel._on_update()
    assert tray._updater._command == "check"


def test_an_update_never_restarts_a_running_flow(tray):
    tray._app._block = True
    tray._service.start()
    tray._app.entered.wait(5)
    assert tray._updater._apply_allowed() is False

    tray._app.release.set()
    wait_for(lambda: tray._service.state is RunState.IDLE)
    assert tray._updater._apply_allowed() is True


def test_an_unknown_mark_is_ignored_rather_than_stored(tray):
    """The panel has a fixed number of boxes; a stray index must not add one."""
    tray._set_return_minute(7, 42)
    assert tray._panel_state()["return_heart_minutes"] == DEFAULT_MARKS


def test_purchase_ticks_are_saved_and_passed_to_the_flow(tray):
    defaults = tray._panel_state()["purchase"]
    assert defaults == {
        "premium_box_plus": False,
        "premium_box": True,
        "pick_up_capsule": True,
        "happiness_box": True,
    }

    tray._set_purchase("premium_box_plus", True)
    tray._set_purchase("happiness_box", False)
    tray._buy_tsum()
    wait_for(lambda: tray._service.state is RunState.IDLE)

    assert tray._app.calls[-1]["flow"] == "purchase_box"
    assert tray._app.variables[-1] == {
        "premium_box_plus": True,
        "premium_box": True,
        "pick_up_capsule": True,
        "happiness_box": False,
    }


def test_run_and_buy_availability_swaps_with_state(tray):
    idle = tray._panel_state()
    assert idle["running"] is False and idle["status"] == "Idle (Resume)"

    tray._app._block = True
    tray._service.start()
    tray._app.entered.wait(5)

    running = tray._panel_state()
    assert running["running"] is True
    assert running["status"] == "Running: Resume"

    tray._app.release.set()
    wait_for(lambda: tray._service.state is RunState.IDLE)


def test_buy_tsum_is_named_in_the_status_while_it_runs(tray):
    """The Run button's status has to say which job is holding the service."""
    tray._app._block = True
    tray._buy_tsum()
    tray._app.entered.wait(5)

    assert tray._panel_state()["status"] == "Running: Buy tsum"
    # A mode run must not be startable underneath it.
    assert tray._service.start() is False

    tray._app.release.set()
    wait_for(lambda: tray._service.state is RunState.IDLE)
    assert tray._panel_state()["status"] == "Idle (Resume)"


def test_tooltip_reports_the_version_and_the_state(tray):
    from ttheart_sender import __version__

    assert tray._tooltip() == f"ttheart-sender v{__version__} - Idle (Resume)"


def test_one_version_number_for_the_whole_project():
    """version.py is the source; pyproject must not carry a second copy."""
    from pathlib import Path

    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; we support 3.9+

    from ttheart_sender import __version__
    from ttheart_sender.version import __version__ as file_version

    assert __version__ == file_version

    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "version" not in project, "hardcoded version would drift from version.py"
    assert "version" in project.get("dynamic", [])


def test_icon_changes_while_running(tray):
    assert tray._icon_path().name == "tray-idle.ico"
    tray._app._block = True
    tray._service.start()
    tray._app.entered.wait(5)
    assert tray._icon_path().name == "tray-running.ico"
    tray._app.release.set()
    wait_for(lambda: tray._service.state is RunState.IDLE)


def test_shipped_icons_exist():
    from ttheart_sender.tray.app import ICON_IDLE, ICON_RUNNING

    assert ICON_IDLE.exists() and ICON_RUNNING.exists()


# --------------------------------------------------------------------------
# Teardown safety
# --------------------------------------------------------------------------
def test_icon_operations_are_no_ops_without_a_live_window(tray):
    """TrackPopupMenu pumps messages, so the window can die mid-menu.

    Everything that touches the handle has to tolerate that instead of raising
    inside a window procedure.
    """
    icon = tray._icon
    assert icon._alive() is False, "no window has been created yet"

    icon._show_menu()
    icon.refresh()
    icon.notify("title", "message")
    icon.quit()

    icon._hwnd = 0xDEAD  # a handle whose window has been destroyed
    assert icon._alive() is False
    icon._show_menu()
    icon.refresh()


def test_tooltip_is_clipped_to_what_the_shell_accepts():
    from ttheart_sender.tray.icon import _TOOLTIP_LIMIT, _clip

    assert _clip("  spaced   out  ", 99) == "spaced out"
    assert len(_clip("x" * 500, _TOOLTIP_LIMIT)) == _TOOLTIP_LIMIT


# -- data collection -------------------------------------------------------
def _tray(tmp_path, app=None, *, saved=None):
    """A TrayApp with its settings file inside tmp_path, and no windows."""
    from ttheart_sender.tray import app as tray_app

    app = app or FakeApp()
    app.config.output_root = tmp_path
    if saved is not None:
        (tmp_path / "ttheart-settings.json").write_text(json.dumps(saved), encoding="utf-8")
    # The panel and the icon both create real windows; neither is under test.
    with mock.patch.object(tray_app, "ControlPanel"), mock.patch.object(tray_app, "TrayIcon"):
        return tray_app.TrayApp(app)


def test_the_panel_starts_from_config_when_nothing_is_saved(tmp_path):
    """First run: config.yaml is the only opinion, so it wins."""
    app = FakeApp()
    app.config.dataset.enabled = True

    tray = _tray(tmp_path, app)

    assert tray._panel_state()["collect_data"] is True


def test_an_unticked_box_survives_a_config_that_still_says_true(tmp_path):
    """The box is the switch once it exists.

    Otherwise unticking it would last until the next launch and then quietly
    turn itself back on, which is the kind of thing you only notice when the
    disk fills.
    """
    app = FakeApp()
    app.config.dataset.enabled = True

    tray = _tray(tmp_path, app, saved={"collect_data": False})

    assert tray._panel_state()["collect_data"] is False
    assert app.config.dataset.enabled is False, "the live config follows the box"


def test_ticking_the_box_takes_effect_without_a_restart(tmp_path):
    # The play action reads ctx.config.dataset, and the tray hands the same
    # Config object to every run, so the next round picks this up.
    app = FakeApp()
    tray = _tray(tmp_path, app)

    tray._set_toggle("collect_data", True)

    assert app.config.dataset.enabled is True
    saved = json.loads((tmp_path / "ttheart-settings.json").read_text())
    assert saved["collect_data"] is True


def test_ticking_the_box_creates_no_folder(tmp_path):
    """Nothing is written until a round actually samples a drag."""
    tray = _tray(tmp_path)

    tray._set_toggle("collect_data", True)

    assert not (tmp_path / "dataset").exists()
