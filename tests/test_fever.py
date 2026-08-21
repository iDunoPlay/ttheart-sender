"""FeverWatch: turning a momentary template match into a ten-second state."""

from __future__ import annotations

import pytest

from ttheart_sender.game.tsum import FEVER_CONFIDENCE, FEVER_SECONDS, FeverWatch


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Matcher:
    """Matches only the frames listed as hits, and records the confidence used."""

    def __init__(self, hits=()) -> None:
        self.hits = set(hits)
        self.confidences = []

    def find(self, frame, template, confidence=None):
        self.confidences.append(confidence)
        return object() if frame in self.hits else None


class Templates:
    """Serves `max_fever` only.

    Deliberate: everything in this file is about the trigger-plus-timer path,
    which is what runs when `templates/` has no FEVER BONUS banner. The banner
    path has its own file, `test_fever_banner.py`.
    """

    def __init__(self, have=True) -> None:
        self.have = have
        self.asked = []

    def get(self, name):
        self.asked.append(name)
        if not self.have or name != "max_fever":
            raise KeyError(name)
        return "max_fever-template"


def make(hits=(), have_template=True):
    clock = Clock()
    matcher = Matcher(hits)
    watch = FeverWatch(matcher, Templates(have_template), clock=clock)
    return watch, clock, matcher


def test_a_match_opens_a_ten_second_window():
    # The template is the meter filling -- a moment, not a state. Nothing can
    # be asked "is FEVER on" later without holding the window open.
    watch, clock, _ = make(hits={"max"})
    assert watch.update("normal") is False
    assert watch.update("max") is True

    clock.advance(FEVER_SECONDS - 0.5)
    assert watch.update("normal") is True, "still inside the window"
    clock.advance(1.0)
    assert watch.update("normal") is False


def test_the_window_re_arms_while_the_meter_stays_full():
    # The bar sits full for several frames before draining, and FEVER has not
    # started counting down until it does -- so each match pushes the deadline.
    watch, clock, _ = make(hits={"max"})
    watch.update("max")
    clock.advance(FEVER_SECONDS - 1)
    watch.update("max")
    clock.advance(FEVER_SECONDS - 1)
    assert watch.active, "the second match should have extended the window"


def test_it_matches_at_its_own_confidence_not_the_global_default():
    # Measured over 151 captured frames the real trigger scores 0.79-0.82, so
    # the shipped 0.85 default misses it entirely.
    watch, _, matcher = make()
    watch.update("anything")
    assert matcher.confidences == [FEVER_CONFIDENCE]
    assert FEVER_CONFIDENCE < 0.85


def test_a_missing_template_disables_it_rather_than_raising():
    # `templates/` predating max_fever must still run, just without the switch.
    watch, _, _ = make(hits={"max"}, have_template=False)
    assert watch.enabled is False
    assert watch.update("max") is False
    assert watch.active is False


def test_it_reports_each_transition_once():
    watch, clock, _ = make(hits={"max"})
    assert watch.took_effect() is None

    watch.update("max")
    assert watch.took_effect() == "fever"
    assert watch.took_effect() is None, "reported twice"

    clock.advance(FEVER_SECONDS + 1)
    watch.update("normal")
    assert watch.took_effect() == "normal"
    assert watch.took_effect() is None


def test_it_asks_for_the_max_fever_template():
    templates = Templates()
    FeverWatch(Matcher(), templates)
    assert templates.asked == ["max_fever"]


@pytest.mark.parametrize("seconds", [1.0, 10.0])
def test_the_window_length_is_configurable(seconds):
    clock = Clock()
    watch = FeverWatch(Matcher({"max"}), Templates(), seconds=seconds, clock=clock)
    watch.update("max")
    clock.advance(seconds - 0.1)
    assert watch.active
    clock.advance(0.2)
    assert not watch.active
