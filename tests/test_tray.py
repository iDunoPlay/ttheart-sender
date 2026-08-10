"""Tray tests: mode selection, run lifecycle, and the menu it produces.

No real screen, no real emulator and no message loop -- the application is a
stub that records what the service asked it to do.
"""

from __future__ import annotations

import threading
from typing import Optional

import pytest

from ttheart_sender.automation.runner import RunReport
from ttheart_sender.config import Config
from ttheart_sender.control.hotkey import StopKeyWatcher
from ttheart_sender.exceptions import WindowNotFoundError
from ttheart_sender.tray.modes import MODES, get_mode
from ttheart_sender.tray.service import AutomationService, RunState


class FakeApp:
    """Stands in for :class:`~ttheart_sender.app.Application`."""

    def __init__(self, *, block: bool = False, error: Optional[Exception] = None) -> None:
        self.config = Config()
        self.stop = StopKeyWatcher(None)  # no real hotkey polling in tests
        self.calls = []
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

    def run_flow(self, name, *, loops=1, loop_delay=0.0, **kwargs):
        self.calls.append({"flow": name, "loops": loops, "loop_delay": loop_delay})
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
def tray():
    from ttheart_sender.tray.app import TrayApp

    return TrayApp(FakeApp())


def labels(items):
    return [item.label for item in items if item.label]


def test_menu_has_a_mode_group_a_start_and_a_stop(tray):
    rows = tray._build_menu()
    submenus = [item for item in rows if item.items]
    assert len(submenus) == 1
    assert labels(submenus[0].items) == ["Resume", "Start", "Play"]
    assert all(item.radio for item in submenus[0].items)

    assert "Start Resume" in labels(rows)
    assert "Stop (F12)" in labels(rows)
    assert "Exit" in labels(rows)


def test_menu_checks_the_selected_mode(tray):
    tray._service.set_mode("play")
    modes = [item for item in tray._build_menu() if item.items][0].items
    assert [item.checked for item in modes] == [False, False, True]


def test_start_and_stop_swap_availability_with_state(tray):
    idle = {item.label: item for item in tray._build_menu() if item.label}
    assert idle["Start Resume"].enabled is True
    assert idle["Stop (F12)"].enabled is False
    assert idle["Idle (Resume)"].enabled is False, "status row is a label, not a button"

    tray._app._block = True
    tray._service.start()
    tray._app.entered.wait(5)

    running = {item.label: item for item in tray._build_menu() if item.label}
    assert running["Start Resume"].enabled is False
    assert running["Stop (F12)"].enabled is True

    tray._app.release.set()
    wait_for(lambda: tray._service.state is RunState.IDLE)


def test_tooltip_reports_the_state(tray):
    assert tray._tooltip() == "ttheart-sender - Idle (Resume)"


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
