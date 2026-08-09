from __future__ import annotations

import pytest
import yaml

from ttheart_sender.automation.flow import load_flow_by_name, parse_flow
from ttheart_sender.exceptions import FlowParseError


def parse(text: str):
    return parse_flow(yaml.safe_load(text), source="test.yaml")


def test_shorthand_and_explicit_forms_are_equivalent():
    flow = parse(
        """
        name: t
        steps:
          - find_click: {template: a}
          - action: find_click
            with: {template: a}
        """
    )
    assert [s.action for s in flow.steps] == ["find_click", "find_click"]
    assert flow.steps[0].params == flow.steps[1].params == {"template": "a"}


def test_scalar_shorthand_maps_to_primary_param():
    flow = parse("name: t\nsteps:\n  - wait: 1.5\n  - log: hello\n")
    assert flow.steps[0].params == {"seconds": 1.5}
    assert flow.steps[1].params == {"message": "hello"}


def test_bare_string_step():
    flow = parse("name: t\nsteps:\n  - prepare_window\n")
    assert flow.steps[0].action == "prepare_window"
    assert flow.steps[0].params == {}


def test_aliases_resolve_to_canonical_name():
    flow = parse("name: t\nsteps:\n  - tap: {template: a}\n")
    assert flow.steps[0].action == "find_click"


def test_common_keys_alongside_shorthand():
    flow = parse(
        """
        name: t
        steps:
          - find_click: {template: a}
            name: tap the thing
            optional: true
            retries: 3
            retry_delay: 1.5
            enabled: false
        """
    )
    step = flow.steps[0]
    assert step.name == "tap the thing"
    assert step.optional is True
    assert step.retries == 3
    assert step.retry_delay == 1.5
    assert step.enabled is False
    assert step.params == {"template": "a"}  # common keys are not params


def test_nested_steps_are_parsed_recursively():
    flow = parse(
        """
        name: t
        steps:
          - repeat:
              times: 3
              steps:
                - if_found:
                    template: x
                    then:
                      - wait: 1
                    else:
                      - log: nope
        """
    )
    repeat = flow.steps[0]
    assert repeat.params == {"times": 3}
    inner = repeat.children["steps"][0]
    assert inner.action == "if_found"
    assert inner.children["then"][0].action == "wait"
    assert inner.children["else"][0].action == "log"
    assert inner.children["then"][0].path == "steps[0].steps[0].then[0]"


def test_unknown_action_suggests_a_close_match():
    with pytest.raises(FlowParseError) as exc:
        parse("name: t\nsteps:\n  - find_clik: {template: a}\n")
    assert "find_click" in str(exc.value)


def test_two_actions_in_one_step_is_an_error():
    with pytest.raises(FlowParseError) as exc:
        parse("name: t\nsteps:\n  - {wait: 1, log: hi}\n")
    assert "several action keys" in str(exc.value)


def test_missing_steps_key_is_an_error():
    with pytest.raises(FlowParseError):
        parse("name: t\n")


def test_unknown_top_level_key_is_an_error():
    with pytest.raises(FlowParseError) as exc:
        parse("name: t\nstpes: []\n")
    assert "stpes" in str(exc.value)


def test_bad_common_key_type_is_an_error():
    with pytest.raises(FlowParseError):
        parse("name: t\nsteps:\n  - wait: 1\n    optional: yes-please\n")


def test_load_flow_by_name_resolves_stem(tmp_path):
    (tmp_path / "demo.yaml").write_text("name: demo\nsteps:\n  - wait: 0\n", encoding="utf-8")
    flow = load_flow_by_name(tmp_path, "demo")
    assert flow.name == "demo"
    assert flow.path == tmp_path / "demo.yaml"


def test_load_flow_by_name_lists_alternatives(tmp_path):
    (tmp_path / "demo.yaml").write_text("name: demo\nsteps: []\n", encoding="utf-8")
    with pytest.raises(FlowParseError) as exc:
        load_flow_by_name(tmp_path, "nope")
    assert "demo" in str(exc.value)


def test_bundled_example_flow_parses():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    flow = load_flow_by_name(root / "flows", "example")
    assert flow.name == "example"
    assert flow.steps
