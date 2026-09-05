"""Pressing Run always starts. No exceptions, no cooldowns, no "press again".

A cooldown was added here to stop a run restarting seconds after the cursor
safety had ended it. It was wrong on its own evidence -- those restarts were
already known to be explicit `start()` calls -- and refusing an explicit press
is exactly what must not happen. The first click after every cursor-stop did
nothing, so Run appeared to need pressing twice.

The rule this leaves: the service may decline to start only because a run is
ALREADY GOING. Nothing else.
"""

from __future__ import annotations

from pathlib import Path

from ttheart_sender.tray.service import AutomationService, RunState

from test_tray import FakeApp, wait_for


class Report:
    stopped_early = True
    success = False

    def summary(self):
        return ("flow 'resume_send_heart': STOPPED | 4 step(s) | Stop "
                "requested (cursor: cursor left the emulator: (855, 1014))")


def test_the_first_press_after_a_cursor_stop_starts():
    app = FakeApp()
    service = AutomationService(app)
    service._report("Resume", Report())

    assert service.start() is True, "Run must never need a second press"
    wait_for(lambda: service.state is RunState.IDLE)
    assert len(app.variables) == 1


def test_every_press_starts_a_run():
    app = FakeApp()
    service = AutomationService(app)
    for n in range(1, 4):
        service._report("Resume", Report())
        assert service.start() is True
        wait_for(lambda: service.state is RunState.IDLE)
        assert len(app.variables) == n


def test_start_only_declines_while_a_run_is_going():
    app = FakeApp(block=True)
    service = AutomationService(app)
    assert service.start() is True
    app.entered.wait(5)
    assert service.start() is False, "one run at a time"
    app.release.set()
    wait_for(lambda: service.state is RunState.IDLE)
    assert service.start() is True


def test_no_cooldown_survives_in_the_service():
    src = Path(AutomationService.__module__.replace(".", "/") + ".py")
    text = src.read_text(encoding="utf-8")
    for gone in ("CURSOR_COOLDOWN", "cursor_hold", "_cursor_until"):
        assert gone not in text, f"{gone} refuses a press the user meant"
