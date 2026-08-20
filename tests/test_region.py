"""`region`: turning two marked corners into a --board rect."""

from __future__ import annotations

import pytest

from ttheart_sender.cli import corners_to_region
from ttheart_sender.game.tsum import _board_rect
from ttheart_sender.geometry import Point


def test_corners_marked_top_left_first():
    assert corners_to_region(Point(8, 265), Point(530, 800)) == (8, 265, 522, 535)


@pytest.mark.parametrize("a,b", [
    (Point(530, 800), Point(8, 265)),   # bottom-right first
    (Point(8, 800), Point(530, 265)),   # bottom-left first
    (Point(530, 265), Point(8, 800)),   # top-right first
])
def test_any_diagonal_gives_the_same_box(a, b):
    # Marking by hand, the second corner is as likely to be the "first" one.
    # A negative width here would crop to nothing rather than fail loudly.
    assert corners_to_region(a, b) == (8, 265, 522, 535)


def test_the_string_it_prints_is_what_board_parses():
    """The whole point of the command: paste the output straight into --board.

    Checked against the real parser rather than by eye, because a board rect
    that is slightly wrong is not obviously wrong -- detection still returns
    tsums, just the wrong set.
    """
    x, y, w, h = corners_to_region(Point(20, 300), Point(500, 700))
    spec = f"{x},{y},{w},{h}"
    assert _board_rect((994, 578, 3), spec) == (20, 300, 480, 400)


def test_a_region_marked_twice_at_one_point_is_empty():
    assert corners_to_region(Point(100, 100), Point(100, 100)) == (100, 100, 0, 0)


# -- the measured layouts -------------------------------------------------
def test_the_live_layout_has_a_normal_rect_and_a_separate_fever_one():
    """Two rects, because they win in different states.

    FEVER is the hardest state to read and the pile rides up during it, so it
    borrows its own measured rect for its ~10s instead of the normal one.
    """
    from ttheart_sender.game.tsum import LAYOUTS

    assert LAYOUTS[(994, 578)]["board"] == (10, 314, 525, 456)
    assert LAYOUTS[(994, 578)]["fever_board"] == (22, 291, 502, 451)


def test_fever_selects_its_own_rect():
    assert _board_rect((994, 578, 3), None) == (10, 314, 525, 456)
    assert _board_rect((994, 578, 3), None, fever=True) == (22, 291, 502, 451)


def test_a_layout_without_a_fever_rect_keeps_its_normal_one():
    # The saved-screenshot layout has no FEVER variant, and asking for one
    # must not fall through to the fractional default.
    assert _board_rect((956, 542, 3), None, fever=True) == (8, 258, 524, 505)


def test_an_explicit_board_is_never_swapped_out_by_fever():
    # Someone who passed --board asked for that rect; changing it underneath
    # them mid-round would be surprising and unexplainable from the output.
    assert _board_rect((994, 578, 3), "1,2,3,4", fever=True) == (1, 2, 3, 4)


def test_an_explicit_board_still_beats_the_layout():
    # The layout is a default, not a lock-in -- A/B-ing a candidate rect is a
    # flag away. Deliberately not the layout's own value, or this would pass
    # whether the override worked or not.
    assert _board_rect((994, 578, 3), "8,285,522,515") == (8, 285, 522, 515)


def test_the_skill_icon_stays_outside_the_board():
    """Narrowing the board must not cost the base tsum.

    `read_base_kind` is handed the FULL frame rather than the crop precisely
    because the skill button sits below the play area — so the rect can shrink
    freely. This pins that relationship, because a board rect that grew to
    swallow the icon would break base-chain ranking in a way detection itself
    would not complain about.
    """
    from ttheart_sender.game.tsum import LAYOUTS

    for shape, layout in ((994, 578), LAYOUTS[(994, 578)]), ((956, 542), LAYOUTS[(956, 542)]):
        _, y, _, h = layout["board"]
        icon_y = int(layout["base"].split(",")[1])
        assert icon_y > y + h, f"{shape}: the skill icon is inside the board rect"
