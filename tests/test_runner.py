"""Engine-level tests: params, control flow, retries, optional steps.

These run without a real screen: the context is built with stub capture /
matcher objects and only actions that never touch pixels are exercised.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from ttheart_sender.automation.context import RunContext
from ttheart_sender.automation.flow import parse_flow, parse_step
from ttheart_sender.automation.params import Params, interpolate
from ttheart_sender.automation.registry import ActionResult, action, get_action
from ttheart_sender.automation.runner import FlowRunner
from ttheart_sender.config import Config
from ttheart_sender.control.hotkey import StopKeyWatcher
from ttheart_sender.control.keyboard import NullKeyboard
from ttheart_sender.control.mouse import NullMouse
from ttheart_sender.exceptions import ActionError


class _NoScreen:
    """Fails loudly if a test accidentally reaches for real pixels."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected screen access: {name}")


def make_context(**overrides) -> RunContext:
    config = Config()
    config.runner.step_delay = 0.0
    config.runner.stop_key = None
    kwargs = dict(
        config=config,
        capture=_NoScreen(),
        matcher=_NoScreen(),
        templates=_NoScreen(),
        mouse=NullMouse(),
        keyboard=NullKeyboard(),
        stop=StopKeyWatcher(None),
    )
    # Overrides replace the stubs rather than colliding with them, so a test
    # can hand in just the one seam it cares about.
    kwargs.update(overrides)
    return RunContext(**kwargs)


def run(text: str, **kwargs):
    flow = parse_flow(yaml.safe_load(text), source="test.yaml")
    ctx = make_context()
    runner = FlowRunner(ctx)
    report = runner.run(flow, **kwargs)
    return report, ctx


# --------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------
def test_interpolation_preserves_type_for_whole_string_references():
    assert interpolate("${n}", {"n": 7}) == 7
    assert interpolate("tap ${n} times", {"n": 7}) == "tap 7 times"
    assert interpolate("${missing}", {}) == "${missing}"
    assert interpolate({"a": ["${n}"]}, {"n": 2}) == {"a": [2]}


def test_typed_getters_and_defaults():
    step = parse_step({"click": {"at": [10, 20], "clicks": "3"}}, source="t", path="steps[0]")
    params = Params(step, {}, {})
    assert params.point("at").as_tuple() == (10, 20)
    assert params.integer("clicks") == 3
    assert params.string("button", "left") == "left"
    assert params.number("interval", 0.08) == 0.08


def test_flow_defaults_feed_step_params():
    step = parse_step({"find": {"template": "x"}}, source="t", path="steps[0]")
    params = Params(step, {}, {"confidence": 0.7})
    assert params.optional_number("confidence") == 0.7


def test_unknown_parameter_is_reported():
    step = parse_step({"wait": {"secnods": 1}}, source="t", path="steps[0]")
    params = Params(step, {}, {})
    params.duration("seconds", 1.0)
    with pytest.raises(ActionError) as exc:
        params.ensure_consumed()
    assert "secnods" in str(exc.value)


def test_duration_accepts_a_range():
    step = parse_step({"wait": {"seconds": {"min": 1.0, "max": 1.0}}}, source="t", path="steps[0]")
    assert Params(step, {}, {}).duration("seconds", 0) == 1.0


def test_bad_point_value_raises_a_helpful_error():
    step = parse_step({"click": {"at": "middle"}}, source="t", path="steps[0]")
    with pytest.raises(ActionError) as exc:
        Params(step, {}, {}).point("at")
    assert "steps[0]" in str(exc.value)


# --------------------------------------------------------------------------
# Control flow
# --------------------------------------------------------------------------
def test_set_and_variable_interpolation_across_steps(caplog):
    caplog.set_level(logging.INFO)
    report, ctx = run(
        """
        name: t
        vars: {greeting: hi}
        steps:
          - set: {name: world}
          - log: "${greeting} ${name}"
        """
    )
    assert report.success
    assert ctx.variables["name"] == "world"
    assert "hi world" in caplog.text


def test_add_counts_flags_and_starts_from_zero():
    report, ctx = run(
        """
        name: t
        vars: {a: true, b: false, c: true}
        steps:
          - add: {enabled: "${a}"}
          - add: {enabled: "${b}"}
          - add: {enabled: "${c}"}
          - add: {enabled: 1}
        """
    )
    assert report.success
    # No initialiser needed, booleans count as 1/0, and 3 stays an int.
    assert ctx.variables["enabled"] == 3
    assert isinstance(ctx.variables["enabled"], int)


def test_add_accepts_string_flags_from_the_command_line():
    report, ctx = run(
        """
        name: t
        vars: {a: true}
        steps:
          - add: {enabled: "${a}"}
        """,
        variables={"a": "false"},
    )
    assert report.success
    assert ctx.variables["enabled"] == 0


def test_add_subtracts_and_keeps_fractions():
    report, ctx = run(
        """
        name: t
        steps:
          - set: {n: 10}
          - add: {n: -2.5}
        """
    )
    assert report.success
    assert ctx.variables["n"] == 7.5


def test_repeat_times_from_a_counted_variable():
    report, ctx = run(
        """
        name: t
        vars: {a: true, b: false, c: true}
        steps:
          - add: {expected: "${a}"}
          - add: {expected: "${b}"}
          - add: {expected: "${c}"}
          - repeat:
              times: ${expected}
              counter: pass
              steps:
                - add: {ran: 1}
                - set: {last: "${pass_1based}"}
        """
    )
    assert report.success
    assert ctx.variables["ran"] == 2
    assert ctx.variables["last"] == 2


class _FrameCapture:
    """Capture stub that plays back a scripted list of frames."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.grabs = 0

    def virtual_screen_rect(self):
        from ttheart_sender.geometry import Rect

        return Rect(0, 0, 4, 4)

    def grab(self, rect):
        frame = self.frames[min(self.grabs, len(self.frames) - 1)]
        self.grabs += 1
        return frame


def _frame(value):
    import numpy as np

    return np.full((4, 4, 3), value, dtype=np.uint8)


def run_with_capture(text: str, capture, **kwargs):
    """Run a flow against scripted frames, with matching stubbed out.

    Only ``find`` is replaced: these tests are about when the loop decides to
    stop, not about template matching, and the real find would drag the whole
    matcher/template stack in behind it.
    """
    import yaml

    from ttheart_sender.automation.flow import parse_flow

    flow = parse_flow(yaml.safe_load(text), source="test.yaml")
    ctx = make_context(capture=capture)
    ctx.find = lambda name, **kw: object() if name == "always_there" else None
    runner = FlowRunner(ctx)
    return runner.run(flow, **kwargs), ctx


SCROLL_LOOP = """
name: t
steps:
  - repeat:
      until_found: never_appears
      max_iterations: 20
      stop_when_still: true
      steps:
        - add: {scrolls: 1}
"""


def test_repeat_stops_once_the_screen_stops_changing():
    # The first scroll moves the list, the second does nothing (clamped).
    # Two scrolls is the floor: a scroll can only be known to be useless
    # after it has been made. Without the check this would burn all 20.
    capture = _FrameCapture([_frame(1), _frame(2), _frame(2), _frame(2)])
    _, ctx = run_with_capture(SCROLL_LOOP, capture)
    assert ctx.variables["scrolls"] == 2


def test_repeat_keeps_going_while_the_screen_moves():
    capture = _FrameCapture([_frame(i) for i in range(1, 21)])
    _, ctx = run_with_capture(SCROLL_LOOP, capture)
    assert ctx.variables["scrolls"] == 20, "only the exhausted budget should end it"


def test_still_tolerance_absorbs_small_differences():
    text = SCROLL_LOOP.replace(
        "stop_when_still: true",
        "stop_when_still: true\n      still_tolerance: 3",
    )
    capture = _FrameCapture([_frame(10), _frame(12), _frame(13)])
    _, ctx = run_with_capture(text, capture)
    # 12 -> 10 is a mean difference of 2, inside the tolerance: treated as
    # the same picture rather than as movement.
    assert ctx.variables["scrolls"] == 1


def test_stillness_does_not_override_a_met_condition():
    """A frozen screen showing the target is a success, not a giving-up."""
    text = SCROLL_LOOP.replace("never_appears", "always_there")
    capture = _FrameCapture([_frame(7)])
    report, ctx = run_with_capture(text, capture)
    assert report.success
    assert "scrolls" not in ctx.variables


def test_stillness_can_drive_the_loop_on_its_own():
    """No times/until_found: "scroll until the screen stops moving"."""
    text = """
    name: t
    steps:
      - repeat:
          stop_when_still: true
          max_iterations: 10
          steps:
            - add: {scrolls: 1}
    """
    capture = _FrameCapture([_frame(1), _frame(2), _frame(3), _frame(3)])
    report, ctx = run_with_capture(text, capture)
    assert report.success
    assert ctx.variables["scrolls"] == 3


def test_repeat_zero_times_is_not_a_failure():
    report, ctx = run(
        """
        name: t
        vars: {a: false}
        steps:
          - add: {expected: "${a}"}
          - repeat:
              times: ${expected}
              steps:
                - add: {ran: 1}
        """
    )
    assert report.success
    assert "ran" not in ctx.variables


def test_if_branches_on_a_variable():
    report, ctx = run(
        """
        name: t
        vars: {flag: true, other: false}
        steps:
          - if:
              value: ${flag}
              then:
                - set: {first: then}
              else:
                - set: {first: else}
          - if:
              value: ${other}
              then:
                - set: {second: then}
              else:
                - set: {second: else}
        """
    )
    assert report.success
    assert ctx.variables["first"] == "then"
    assert ctx.variables["second"] == "else"


def test_if_accepts_a_var_overridden_from_the_command_line():
    # --var only ever produces strings, so 'false' has to read as false.
    report, ctx = run(
        """
        name: t
        vars: {flag: true}
        steps:
          - if:
              value: ${flag}
              then:
                - set: {branch: then}
              else:
                - set: {branch: else}
        """,
        variables={"flag": "false"},
    )
    assert report.success
    assert ctx.variables["branch"] == "else"


def test_if_on_an_undefined_variable_is_an_error():
    report, _ = run(
        """
        name: t
        steps:
          - if:
              value: ${nope}
              then:
                - set: {branch: then}
        """
    )
    assert not report.success


def test_chance_takes_the_then_branch_when_the_roll_lands():
    report, ctx = run(
        """
        name: t
        steps:
          - chance:
              percent: 100
              save_as: rolled
              then:
                - set: {branch: then}
              else:
                - set: {branch: else}
        """
    )
    assert report.success
    assert ctx.variables["branch"] == "then"
    assert ctx.variables["rolled"] is True


def test_chance_takes_the_else_branch_when_the_roll_misses():
    report, ctx = run(
        """
        name: t
        steps:
          - chance:
              p: 0
              save_as: rolled
              then:
                - set: {branch: then}
              else:
                - set: {branch: else}
        """
    )
    assert report.success
    assert ctx.variables["branch"] == "else"
    assert ctx.variables["rolled"] is False


def test_chance_without_a_matching_branch_is_not_a_failure():
    report, _ = run(
        """
        name: t
        steps:
          - chance:
              p: 0
              then:
                - set: {branch: then}
        """
    )
    assert report.success


def test_repeat_runs_n_times_and_exposes_the_counter():
    report, ctx = run(
        """
        name: t
        vars: {n: 4}
        steps:
          - repeat:
              times: ${n}
              counter: i
              steps:
                - set: {seen: "${i}"}
        """
    )
    assert report.success
    assert ctx.variables["seen"] == 3  # last iteration, 0-based


def test_repeat_respects_max_iterations():
    report, ctx = run(
        """
        name: t
        steps:
          - repeat:
              forever: true
              max_iterations: 5
              steps:
                - set: {i: "${index}"}
        """
    )
    assert report.success
    assert ctx.variables["index"] == 4


def test_repeat_duration_stops_after_the_time_budget_elapses():
    report, ctx = run(
        """
        name: t
        steps:
          - repeat:
              duration: 0.2
              delay: 0.05
              steps:
                - set: {i: "${index}"}
        """
    )
    assert report.success
    # Should run several quick iterations, then stop on its own -- not hit
    # max_iterations, and not loop forever.
    assert 1 <= ctx.variables["i"] < 100


def test_repeat_rejects_two_modes_at_once():
    report, _ = run(
        """
        name: t
        steps:
          - repeat:
              times: 2
              forever: true
              steps:
                - wait: 0
        """
    )
    assert not report.success


def test_repeat_rejects_duration_with_times():
    report, _ = run(
        """
        name: t
        steps:
          - repeat:
              times: 2
              duration: 5
              steps:
                - wait: 0
        """
    )
    assert not report.success


def test_repeat_until_found_stays_lenient_by_default_when_never_found():
    flow = parse_flow(
        yaml.safe_load(
            """
            name: t
            steps:
              - repeat:
                  until_found: something
                  max_iterations: 3
                  steps:
                    - wait: 0
            """
        ),
        source="test.yaml",
    )
    ctx = make_context()
    ctx.find = lambda *a, **k: None  # simulates a popup covering the screen
    report = FlowRunner(ctx).run(flow)
    assert report.success


def test_repeat_until_found_require_found_fails_when_never_found():
    flow = parse_flow(
        yaml.safe_load(
            """
            name: t
            steps:
              - repeat:
                  until_found: something
                  max_iterations: 3
                  require_found: true
                  steps:
                    - wait: 0
            """
        ),
        source="test.yaml",
    )
    ctx = make_context()
    ctx.find = lambda *a, **k: None
    report = FlowRunner(ctx).run(flow)
    assert not report.success


def test_repeat_until_found_require_found_succeeds_once_found():
    flow = parse_flow(
        yaml.safe_load(
            """
            name: t
            steps:
              - repeat:
                  until_found: something
                  max_iterations: 5
                  require_found: true
                  steps:
                    - wait: 0
            """
        ),
        source="test.yaml",
    )
    ctx = make_context()
    calls = {"n": 0}

    def fake_find(*_args, **_kwargs):
        calls["n"] += 1
        return object() if calls["n"] >= 2 else None

    ctx.find = fake_find
    report = FlowRunner(ctx).run(flow)
    assert report.success


def test_repeat_delay_param_does_not_error_when_condition_met_immediately():
    # Regression: 'delay' is re-rolled fresh inside the loop body (so a
    # {min, max} range varies per iteration) rather than read once up front.
    # If until_found is already satisfied on the very first check, the loop
    # body -- and therefore the delay read -- never runs at all; 'delay'
    # must still count as a recognised parameter, not an unknown one.
    flow = parse_flow(
        yaml.safe_load(
            """
            name: t
            steps:
              - repeat:
                  until_found: something
                  delay: {min: 0.1, max: 0.2}
                  steps:
                    - wait: 0
            """
        ),
        source="test.yaml",
    )
    ctx = make_context()
    ctx.find = lambda *a, **k: object()  # already visible -> zero iterations
    report = FlowRunner(ctx).run(flow)
    assert report.success


def test_while_found_delay_param_does_not_error_when_never_found():
    flow = parse_flow(
        yaml.safe_load(
            """
            name: t
            steps:
              - while_found:
                  template: something
                  delay: {min: 0.1, max: 0.2}
                  steps:
                    - wait: 0
            """
        ),
        source="test.yaml",
    )
    ctx = make_context()
    ctx.find = lambda *a, **k: None  # never visible -> zero iterations
    report = FlowRunner(ctx).run(flow)
    assert report.success


def test_click_all_delay_param_does_not_error_when_no_matches():
    flow = parse_flow(
        yaml.safe_load(
            """
            name: t
            steps:
              - click_all:
                  template: something
                  delay: {min: 0.1, max: 0.2}
                optional: true
            """
        ),
        source="test.yaml",
    )
    ctx = make_context()
    ctx.find_all = lambda *a, **k: []  # no matches -> loop body never runs
    report = FlowRunner(ctx).run(flow)
    assert report.success


def test_stop_action_ends_the_flow_early(caplog):
    caplog.set_level(logging.INFO)
    report, ctx = run(
        """
        name: t
        steps:
          - set: {before: 1}
          - stop: enough
          - set: {after: 1}
        """
    )
    assert report.success
    assert "before" in ctx.variables and "after" not in ctx.variables
    assert "enough" in report.message


def test_stop_can_mark_the_run_as_failed():
    report, _ = run(
        """
        name: t
        steps:
          - stop: {message: broke, success: false}
        """
    )
    assert not report.success


def test_loops_repeat_the_whole_flow():
    report, ctx = run("name: t\nsteps:\n  - set: {x: '${loop}'}\n", loops=3)
    assert report.success
    assert ctx.variables["x"] == 2


def test_cli_variables_override_flow_vars():
    _, ctx = run("name: t\nvars: {n: 1}\nsteps:\n  - log: x\n", variables={"n": 9})
    assert ctx.variables["n"] == 9


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------
@action("_test_flaky", summary="test helper")
def _flaky(ctx, params):
    attempts = ctx.variables.get("attempts", 0) + 1
    ctx.set_var("attempts", attempts)
    needed = params.integer("succeed_on", 1)
    params.ensure_consumed()
    return ActionResult.ok() if attempts >= needed else ActionResult.fail("not yet")


def test_retries_rerun_the_step_until_it_succeeds():
    report, ctx = run(
        """
        name: t
        steps:
          - _test_flaky: {succeed_on: 3}
            retries: 5
            retry_delay: 0
        """
    )
    assert report.success
    assert ctx.variables["attempts"] == 3


def test_a_failing_step_fails_the_flow():
    report, _ = run(
        """
        name: t
        steps:
          - _test_flaky: {succeed_on: 99}
            retries: 1
            retry_delay: 0
        """
    )
    assert not report.success
    assert report.steps_failed == 1


def test_optional_steps_do_not_fail_the_flow():
    report, _ = run(
        """
        name: t
        steps:
          - _test_flaky: {succeed_on: 99}
            optional: true
        """
    )
    assert report.success


def test_disabled_steps_are_skipped():
    report, ctx = run(
        """
        name: t
        steps:
          - set: {x: 1}
            enabled: false
        """
    )
    assert report.success
    assert "x" not in ctx.variables
    assert report.steps_skipped == 1


def test_registry_exposes_aliases():
    assert get_action("tap") is get_action("find_click")
    assert get_action("nope") is None
