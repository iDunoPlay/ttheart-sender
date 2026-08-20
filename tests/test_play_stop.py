"""play_tsum's stop condition: several endings, one frame."""

from __future__ import annotations

from types import SimpleNamespace

from ttheart_sender.automation.tsum_actions import _stop_checker


class Matcher:
    """Matches when the frame names the template. Records every lookup."""

    def __init__(self) -> None:
        self.asked = []

    def find(self, frame, template, confidence=None):
        self.asked.append((frame, template, confidence))
        return object() if template in frame else None


class Templates:
    def get(self, name):
        return name


def make():
    matcher = Matcher()
    return SimpleNamespace(matcher=matcher, templates=Templates()), matcher


def test_no_condition_means_no_checker():
    # The loop skips the call entirely rather than paying for a match that
    # can never fire.
    ctx, _ = make()
    assert _stop_checker(ctx, [], [], None) is None


def test_until_found_stops_on_any_of_the_listed_templates():
    # The point of a list: `timeup` flashes and can be missed between drags,
    # so the frame that only has the clock at zero still ends the round.
    ctx, _ = make()
    check = _stop_checker(ctx, ["timeup", "gameplay_clock_zero"], [], None)

    assert check(["board"]) == ""
    assert check(["gameplay_clock_zero"]) == "gameplay_clock_zero appeared -- round over"
    assert check(["timeup"]) == "timeup appeared -- round over"


def test_until_found_stops_asking_once_one_matches():
    # Each name is another match per iteration, so the first hit short-circuits.
    ctx, matcher = make()
    check = _stop_checker(ctx, ["timeup", "gameplay_clock_zero"], [], 0.9)

    check(["timeup"])
    assert [name for _, name, _ in matcher.asked] == ["timeup"]
    assert matcher.asked[0][2] == 0.9, "the step's confidence reaches the matcher"


def test_until_gone_needs_every_template_gone():
    # Inverted and ANDed: one template flickering out on a live board is a
    # miss, not the end of the round.
    ctx, _ = make()
    check = _stop_checker(ctx, [], ["gameplay_footer", "combo_items"], None)

    assert check(["gameplay_footer", "combo_items"]) == ""
    assert check(["gameplay_footer"]) == ""
    assert check(["scoreboard"]) == "gameplay_footer, combo_items is gone -- round over"


def test_found_and_gone_can_be_combined():
    ctx, _ = make()
    check = _stop_checker(ctx, ["timeup"], ["gameplay_footer"], None)

    assert check(["gameplay_footer"]) == ""
    assert check(["gameplay_footer", "timeup"]) == "timeup appeared -- round over"
    assert check(["scoreboard"]) == "gameplay_footer is gone -- round over"
