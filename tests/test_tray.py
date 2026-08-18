"""Tray tests: mode selection, run lifecycle, and the menu it produces.

No real screen, no real emulator and no message loop -- the application is a
stub that records what the service asked it to do.
"""

from __future__ import annotations

import threading
from typing import Optional

import pytest

from ttheart_sender.automation.flow import load_flow_by_name
from ttheart_sender.automation.runner import RunReport
from ttheart_sender.config import Config
from ttheart_sender.control.hotkey import StopKeyWatcher
from ttheart_sender.exceptions import WindowNotFoundError
from ttheart_sender.tray.modes import MODES, get_mode
from ttheart_sender.tray.service import (
    PLAY_CHANCE_OFF,
    PLAY_CHANCE_VAR,
    AutomationService,
    RunState,
)


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
    assert app.variables == [{PLAY_CHANCE_VAR: PLAY_CHANCE_OFF}]


def test_play_on_leaves_the_flow_alone():
    app = FakeApp()
    changes = []
    service = AutomationService(app, on_change=lambda: changes.append(1))

    assert service.set_play(True) is True
    assert service.play is True
    assert len(changes) == 1
    assert service.set_play(True) is False, "re-selecting should be a no-op"

    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [None], "the flow's own play_chance_percent must win"


def test_toggling_play_mid_run_does_not_change_the_live_run():
    app = FakeApp(block=True)
    service = AutomationService(app)

    service.start()
    app.entered.wait(5)
    service.toggle_play()
    assert service.play is True

    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables == [{PLAY_CHANCE_VAR: PLAY_CHANCE_OFF}], (
        "the run keeps the variables it started with"
    )

    # ...but the next one picks the new setting up.
    service.start()
    wait_for(lambda: service.state is RunState.IDLE)
    assert app.variables[-1] is None


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


def test_launch_forwards_the_override_to_resume():
    """run_flow re-applies the called flow's vars, so it has to be passed on."""
    flow = load_flow_by_name(Config().flows_dir, "launch")
    calls = [s for s in find_steps(flow.steps, "run_flow") if s.params.get("flow") == "resume"]

    assert calls, "launch.yaml no longer hands off to resume"
    assert PLAY_CHANCE_VAR in flow.vars, "launch.yaml needs its own standalone default"
    for call in calls:
        forwarded = call.params.get("vars", {})
        assert forwarded.get(PLAY_CHANCE_VAR) == f"${{{PLAY_CHANCE_VAR}}}", (
            "without this, resume.yaml's own default overwrites the tray's choice"
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


def test_panel_shows_the_version(tray):
    """A screenshot of the panel should be enough to identify the build."""
    from ttheart_sender import __version__

    assert tray._panel._version_text == f"ttheart-sender v{__version__}"


def test_right_click_menu_is_gone(tray):
    """Left-click opens the panel; the context menu was retired with it."""
    assert tray._icon._menu_factory is None
    assert tray._icon._on_left_click is not None
    assert tray._icon._on_double_click is None


def test_panel_toggles_reach_the_service_and_the_saved_file(tray):
    assert tray._panel_state()["auto_play"] is False

    tray._set_toggle("auto_play", True)
    assert tray._service.play is True
    assert tray._panel_state()["auto_play"] is True

    tray._set_toggle("always_on_top", False)
    tray._set_mode("play")
    assert tray._panel_state()["mode"] == "play"

    # Everything above must survive a restart.
    from ttheart_sender.tray.settings import PanelSettings

    reloaded = PanelSettings.load(tray._settings_path)
    assert reloaded.auto_play is True
    assert reloaded.always_on_top is False
    assert reloaded.mode == "play"


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
