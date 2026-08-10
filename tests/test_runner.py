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
    return RunContext(
        config=config,
        capture=_NoScreen(),
        matcher=_NoScreen(),
        templates=_NoScreen(),
        mouse=NullMouse(),
        keyboard=NullKeyboard(),
        stop=StopKeyWatcher(None),
        **overrides,
    )


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
