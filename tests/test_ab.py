"""Alternating one option between rounds, so an A/B is balanced by construction.

Every comparison in the corpus so far has been one build against another,
which is not an A/B -- `docs/DATASET-FINDINGS.md` rounds 35 and 36 are both
about a switch that never reached the thing it switched. So the arm is chosen
in `play_loop`, the one place every round passes through exactly once, and
never in a flow variable that has to be declared and forwarded at three hops.

It records itself, too: `play_settings` snapshots the whole option namespace
into every sample at write time, so the arm appears in the corpus under the
alternated option's own name with nothing to remember.
"""

from __future__ import annotations

import argparse

import pytest

from ttheart_sender.game import tsum


def opts(**kw):
    base = dict(ab="", ab_off="", reject_model="models/reject.onnx",
                settle_board=True, kinds=4, step_px=12)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def reset_counter():
    tsum._AB_ROUNDS = 0
    yield
    tsum._AB_ROUNDS = 0


def arms(n, **kw):
    """The arm, and the alternated value, for `n` consecutive rounds."""
    out = []
    for _ in range(n):
        o = opts(**kw)
        said = []
        arm = tsum._ab_arm(o, said.append)
        out.append((arm, getattr(o, kw.get("ab", "reject_model"), None)))
    return out


def test_no_ab_changes_nothing():
    o = opts()
    assert tsum._ab_arm(o, lambda _m: None) == ""
    assert o.reject_model == "models/reject.onnx", "the baseline is untouched"
    assert tsum._AB_ROUNDS == 0, "a round with no A/B must not consume an arm"


def test_the_arms_alternate_round_by_round():
    got = arms(6, ab="reject_model")
    assert [a for a, _ in got] == ["on", "off", "on", "off", "on", "off"]


def test_the_off_arm_gets_the_off_value_and_the_on_arm_the_configured_one():
    got = arms(4, ab="reject_model")
    assert [v for _, v in got] == [
        "models/reject.onnx", "", "models/reject.onnx", ""]


def test_the_arms_are_balanced_over_an_even_number_of_rounds():
    got = [a for a, _ in arms(40, ab="reject_model")]
    assert got.count("on") == got.count("off") == 20


def test_a_bool_option_alternates_as_a_bool():
    got = arms(2, ab="settle_board", ab_off="false")
    assert [v for _, v in got] == [True, False]
    assert isinstance(got[1][1], bool), (
        "an unresolved string is truthy -- coercing to the live option's type "
        "is what stops the OFF arm arming the thing it is meant to disarm")


def test_an_int_option_alternates_as_an_int():
    got = arms(2, ab="kinds", ab_off="0")
    assert [v for _, v in got] == [4, 0]
    assert isinstance(got[1][1], int)


def test_an_off_value_that_is_not_the_empty_one_is_honoured():
    """`step_px`'s off-value is 8, not 0 -- the pair is stated, never guessed."""
    got = arms(2, ab="step_px", ab_off="8")
    assert [v for _, v in got] == [12, 8]


def test_an_unknown_option_is_refused_out_loud_and_alternates_nothing():
    o = opts(ab="no_such_option")
    said = []
    assert tsum._ab_arm(o, said.append) == ""
    assert any("no play option" in m for m in said), (
        "a typo must not leave every round recording `ab` set while nothing "
        "alternates -- that is an experiment that looks like it ran")
    assert tsum._AB_ROUNDS == 0


def test_identical_arms_are_refused_out_loud():
    """Both arms the same is not an A/B, and would read as one in the corpus."""
    o = opts(ab="reject_model", reject_model="", ab_off="")
    said = []
    assert tsum._ab_arm(o, said.append) == ""
    assert any("both arms" in m for m in said)
    assert tsum._AB_ROUNDS == 0


def test_the_arm_says_which_round_it_is():
    o = opts(ab="reject_model")
    said = []
    tsum._ab_arm(o, said.append)
    line = " ".join(said)
    assert "A/B round 1" in line and "ON arm" in line, (
        f"the log has to name the arm, or a corpus cannot be audited "
        f"against it; got {line!r}")


def test_the_report_carries_the_arm():
    """A round that wrote no sample still has to say which arm it was."""
    report = tsum.PlayReport()
    assert report.ab_arm == "" and report.ab == ""
    report.ab, report.ab_arm = "reject_model", "off"
    assert (report.ab, report.ab_arm) == ("reject_model", "off")


def test_the_alternated_option_is_in_the_recorded_settings():
    """`play_settings` is what puts the arm in the corpus -- no new field."""
    o = opts(ab="reject_model")
    tsum._ab_arm(o, lambda _m: None)          # ON
    on = tsum.play_settings(o)
    o = opts(ab="reject_model")
    tsum._ab_arm(o, lambda _m: None)          # OFF
    off = tsum.play_settings(o)
    assert on["reject_model"] == "models/reject.onnx"
    assert off["reject_model"] == ""
    assert on["ab"] == off["ab"] == "reject_model"
