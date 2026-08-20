"""The clock gate behind the panel's "Return Heart" section.

`time_gate` is the only action that reads the wall clock, so every test here
runs against a fake one: the real time of day must never decide whether the
suite passes.
"""

from __future__ import annotations

from datetime import datetime

import pytest
import yaml

from ttheart_sender.automation import actions
from ttheart_sender.automation.context import RunContext
from ttheart_sender.automation.flow import parse_flow, parse_step
from ttheart_sender.automation.params import Params
from ttheart_sender.automation.registry import ActionResult, action
from ttheart_sender.automation.runner import FlowRunner
from ttheart_sender.config import Config
from ttheart_sender.control.hotkey import StopKeyWatcher
from ttheart_sender.control.keyboard import NullKeyboard
from ttheart_sender.control.mouse import NullMouse
from ttheart_sender.exceptions import ActionError

FLAG = "send_heart_check"
MARKS = [15, 50]


class _Clock:
    """Stands in for :mod:`datetime`'s module object; only ``now`` is used."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def at(self, hour: int, minute: int) -> None:
        self.moment = self.moment.replace(hour=hour, minute=minute)

    def now(self) -> datetime:
        return self.moment


@pytest.fixture()
def clock(monkeypatch):
    fake = _Clock(datetime(2026, 8, 19, 21, 20))
    monkeypatch.setattr(actions, "datetime", fake)
    return fake


@pytest.fixture()
def ctx() -> RunContext:
    return RunContext(
        config=Config(),
        capture=None,
        matcher=None,
        templates=None,
        mouse=NullMouse(),
        keyboard=NullKeyboard(),
        stop=StopKeyWatcher(None),
    )


def check(ctx: RunContext, minutes=MARKS, flag: str = FLAG) -> bool:
    """Run one `time_gate` step and report whether the flag came up."""
    step = parse_step(
        {"time_gate": {"minutes": minutes, "save_as": flag}},
        source="test.yaml",
        path="step[0]",
    )
    params = Params(step, ctx.variables)
    result = actions.act_time_gate(ctx, params)
    params.ensure_consumed()
    assert result.success, "the gate itself never fails -- only the branch can"
    assert ctx.variables[flag] is result.value
    return bool(result.value)


# --------------------------------------------------------------------------
# Firing
# --------------------------------------------------------------------------
def test_a_mark_already_past_does_not_fire_on_the_first_check(clock, ctx):
    """Starting at 9:20pm waits for 9:50 rather than firing for 9:15.

    Otherwise stopping and starting the bot would be a way to send hearts as
    often as you like, which is exactly what the schedule is there to stop.
    """
    assert check(ctx) is False


def test_the_flag_comes_up_once_when_a_mark_goes_by(clock, ctx):
    assert check(ctx) is False  # armed at :20

    clock.at(21, 49)
    assert check(ctx) is False

    clock.at(21, 50)
    assert check(ctx) is True

    # Every later cycle within the same mark is a miss -- one send, not one
    # per cycle for the rest of the minute.
    clock.at(21, 50)
    assert check(ctx) is False
    clock.at(21, 58)
    assert check(ctx) is False


def test_both_marks_fire_every_hour(clock, ctx):
    check(ctx)  # armed at 21:20
    fired = []
    for hour, minute in [(21, 50), (22, 10), (22, 15), (22, 40), (22, 50), (23, 15)]:
        clock.at(hour, minute)
        if check(ctx):
            fired.append(f"{hour}:{minute:02d}")
    assert fired == ["21:50", "22:15", "22:50", "23:15"]


def test_a_long_cycle_still_fires_once_after_stepping_over_a_mark(clock, ctx):
    """A played round can outlast a mark; the send happens on the far side.

    Only once, though: two marks went by here and the flag still comes up a
    single time, because there is only one heart-sending pass to do.
    """
    clock.at(21, 49)
    assert check(ctx) is False  # armed just before :50

    clock.at(22, 20)  # :50 and :15 both went by while a round was playing
    assert check(ctx) is True
    assert check(ctx) is False


def test_a_mark_before_the_first_one_of_the_hour_looks_back_an_hour(clock, ctx):
    """At :05 the last mark was :50 of the hour before, not :50 of this one."""
    clock.at(21, 5)
    assert check(ctx) is False  # armed against 20:50

    clock.at(21, 14)
    assert check(ctx) is False
    clock.at(21, 15)
    assert check(ctx) is True


def test_two_gates_keep_separate_histories(clock, ctx):
    """Each flag remembers its own last mark, so gates do not shadow each other."""
    assert check(ctx, [15, 50], "hearts") is False
    assert check(ctx, [30], "boxes") is False

    clock.at(21, 30)
    assert check(ctx, [15, 50], "hearts") is False
    assert check(ctx, [30], "boxes") is True


# --------------------------------------------------------------------------
# Reading the marks
# --------------------------------------------------------------------------
def test_marks_can_be_written_as_a_string(clock, ctx):
    """`--var return_heart_minutes=15,50` can only ever hand over a string."""
    assert check(ctx, "15, 50") is False
    clock.at(21, 50)
    assert check(ctx, "15, 50") is True


def test_a_single_mark_needs_no_list(clock, ctx):
    assert check(ctx, 50) is False
    clock.at(21, 50)
    assert check(ctx, 50) is True


def test_an_empty_schedule_never_fires(clock, ctx):
    """An unset panel field is "never", not a crash."""
    assert check(ctx, []) is False
    clock.at(21, 50)
    assert check(ctx, []) is False


def test_duplicate_marks_fire_once(clock, ctx):
    assert check(ctx, [50, 50]) is False
    clock.at(21, 50)
    assert check(ctx, [50, 50]) is True
    assert check(ctx, [50, 50]) is False


@pytest.mark.parametrize("minutes", [[60], [-1], ["quarter past"]])
def test_impossible_marks_are_rejected(clock, ctx, minutes):
    with pytest.raises(ActionError):
        check(ctx, minutes)


# --------------------------------------------------------------------------
# The gate as resume.yaml wires it
# --------------------------------------------------------------------------
SENDS = []


@action("_test_heart_sent", summary="test helper")
def _heart_sent(ctx, params):
    """Counts a send from *outside* the flow, where run_flow cannot undo it."""
    params.ensure_consumed()
    SENDS.append(ctx.variables.get("loop"))
    return ActionResult.ok()


CYCLE = """
name: cycle
vars:
  return_heart_timed: true
  return_heart_minutes: [15, 50]
  send_heart_check: false
steps:
  - if:
      value: ${return_heart_timed}
      then:
      - time_gate: {minutes: "${return_heart_minutes}", save_as: send_heart_check}
      else:
      - set: {send_heart_check: true}
  - if:
      value: ${send_heart_check}
      then:
      - run_flow: send_heart
      - set: {send_heart_check: false}
"""


@pytest.fixture()
def cycle(clock, ctx, tmp_path):
    """One turn of resume.yaml's heart-sending decision, wired up for real.

    A stub send_heart.yaml stands in for the real one so the run_flow call --
    and the variable isolation that comes with it -- is the genuine article.
    """
    SENDS.clear()
    (tmp_path / "send_heart.yaml").write_text(
        "name: send_heart\nsteps:\n  - _test_heart_sent\n", encoding="utf-8"
    )
    ctx.config.runner.flows_dir = str(tmp_path)
    ctx.config.runner.step_delay = 0.0
    flow = parse_flow(yaml.safe_load(CYCLE), source="cycle.yaml")
    runner = FlowRunner(ctx)

    def turn(hour: int, minute: int, **overrides) -> int:
        clock.at(hour, minute)
        # Overrides go through `variables` rather than straight into the
        # context, because that is where the tray puts them -- and because a
        # run re-applies the flow's own `vars:` over anything already there.
        assert runner.run(flow, variables=overrides or None).success
        return len(SENDS)

    return turn


def test_hearts_go_out_at_the_marks_and_nowhere_else(cycle):
    assert cycle(21, 20) == 0, "the mark at :15 was already past when we started"
    assert cycle(21, 40) == 0
    assert cycle(21, 50) == 1
    assert cycle(21, 52) == 1, "one send per mark, not one per cycle"
    assert cycle(22, 15) == 2


def test_the_flag_is_back_down_after_a_send(cycle, ctx):
    """run_flow restores the caller's variables, so the `set` after it lands.

    If it did not, send_heart_check would stay up and every later cycle would
    send again -- the schedule would only ever delay the first pass.
    """
    cycle(21, 20)
    cycle(21, 50)
    assert ctx.variables["send_heart_check"] is False


def test_an_unticked_return_heart_sends_every_cycle(cycle):
    assert cycle(21, 20, return_heart_timed=False) == 1
    assert cycle(21, 21, return_heart_timed=False) == 2
