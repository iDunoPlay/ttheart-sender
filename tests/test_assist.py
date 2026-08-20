"""Assist mode: reading the game's marks, ordering them, walking without pressing.

The parts that can be tested without an emulator. Reading the marks is exercised
against a real captured board with a synthetic glow painted on, so the geometry
(which disc is sampled at which tsum) is the real thing rather than a fixture
shaped to pass.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ttheart_sender.game.tsum import (
    Tsum,
    marks_on_board,
    tour_from,
    walk_path,
)

BOARDS = Path(__file__).resolve().parent.parent / "scratchpad"

#: The crop the numbers in this file were measured on -- the play-area rect
#: that `LAYOUTS[(994, 578)]` carried up to 2026-08-20.
#:
#: Pinned deliberately rather than read back from `LAYOUTS`. The measurements
#: below ("a single frame reports 11 false marks, three report none") describe
#: this crop of these boards. Re-measuring the live play area is a routine
#: thing to do; having it silently change what a measured test asserts is not
#: -- when the rect moved, these tests started failing on a reading that had
#: nothing to do with the code they cover.
MEASURED_BOARD = (10, 314, 525, 456)


def line(*coords):
    """Tsums at the given (x, y) points, all one radius apart in a row."""
    return [Tsum(x=x, y=y, r=25.0, kind=0, colour=(0, 0, 0)) for x, y in coords]


# -- reading the marks ----------------------------------------------------
def test_a_tsum_the_glow_never_touched_is_not_a_hit():
    before = np.zeros((200, 200, 3), np.uint8)
    tsums = line((50, 50), (150, 50))
    hits, aura_only, _ = marks_on_board(before, before.copy(), tsums, 25.0,
                                        pressed=0, threshold=8.0, aura=90.0)
    assert hits == []
    assert aura_only == []


def test_a_lit_tsum_is_a_hit_and_the_pressed_one_is_never_reported():
    before = np.zeros((200, 400, 3), np.uint8)
    held = before.copy()
    tsums = line((50, 100), (350, 100))
    # Light up both: the pressed one always changes under the finger, and
    # reporting it would double-count the tsum the path already starts on.
    for t in tsums:
        held[int(t.y) - 12:int(t.y) + 12, int(t.x) - 12:int(t.x) + 12] = 200

    hits, aura_only, _ = marks_on_board(before, held, tsums, 25.0,
                                        pressed=0, threshold=8.0, aura=90.0)
    assert hits == [1]
    assert aura_only == []


def test_a_hit_under_the_glow_is_kept_but_flagged():
    # The warthog case from `hold`: it reacted only because it sat inside the
    # ~90px aura. Still dragged over -- that costs a detour, not a broken
    # chain -- but counted separately so a bad reading is visible.
    before = np.zeros((200, 200, 3), np.uint8)
    held = before.copy()
    tsums = line((60, 100), (110, 100))
    held[88:112, 98:122] = 200

    hits, aura_only, _ = marks_on_board(before, held, tsums, 25.0,
                                        pressed=0, threshold=8.0, aura=90.0)
    assert hits == [1]
    assert aura_only == [1], "50px away is inside a 90px aura"


def test_the_threshold_is_what_separates_a_mark_from_noise():
    before = np.zeros((200, 400, 3), np.uint8)
    faint, bright = before.copy(), before.copy()
    tsums = line((50, 100), (350, 100))
    faint[88:112, 338:362] = 4
    bright[88:112, 338:362] = 60

    assert marks_on_board(before, faint, tsums, 25.0, pressed=0,
                          threshold=8.0, aura=90.0)[0] == []
    assert marks_on_board(before, bright, tsums, 25.0, pressed=0,
                          threshold=8.0, aura=90.0)[0] == [1]


@pytest.mark.skipif(not (BOARDS / "board1.png").exists(),
                    reason="needs a captured board in scratchpad/")
def test_marks_are_read_off_a_real_board():
    """Same test, but the discs are sampled at real detected tsum positions."""
    import cv2

    from ttheart_sender.game.tsum import detect

    img = cv2.imdecode(np.fromfile(str(BOARDS / "board1.png"), np.uint8),
                       cv2.IMREAD_COLOR)
    bx, by, bw, bh = MEASURED_BOARD
    crop = img[by:by + bh, bx:bx + bw]
    tsums, radius, _ = detect(crop, k=12)
    assert len(tsums) > 10, "detection is broken, not the marks reader"

    # Paint a glow on three of them, well away from the pressed one so the
    # aura cannot be what explains the hit.
    pressed = 0
    far = sorted(range(1, len(tsums)),
                 key=lambda i: -math.hypot(tsums[i].x - tsums[pressed].x,
                                           tsums[i].y - tsums[pressed].y))[:3]
    held = crop.copy()
    for i in far:
        # Paint away from whatever colour the tsum already is. A white glow on
        # a white tsum barely moves the diff -- which is true of the real thing
        # too, and is the reader's genuine limit, not the thing under test
        # here.
        centre = crop[int(tsums[i].y), int(tsums[i].x)]
        glow = (0, 0, 0) if int(centre.mean()) > 127 else (255, 255, 255)
        cv2.circle(held, (int(tsums[i].x), int(tsums[i].y)),
                   int(radius * 0.6), glow, -1)

    hits, aura_only, _ = marks_on_board(crop, held, tsums, radius,
                                        pressed=pressed, threshold=8.0, aura=90.0)
    assert sorted(hits) == sorted(far)
    assert aura_only == []


def test_one_frame_or_several_are_both_accepted():
    before = np.zeros((200, 400, 3), np.uint8)
    held = before.copy()
    tsums = line((50, 100), (350, 100))
    held[88:112, 338:362] = 200

    assert marks_on_board(before, held, tsums, 25.0, pressed=0,
                          threshold=8.0, aura=90.0)[0] == [1]
    assert marks_on_board(before, [held, held, held], tsums, 25.0, pressed=0,
                          threshold=8.0, aura=90.0)[0] == [1]


def test_a_mark_must_persist_across_every_frame():
    # A tsum that reacted in only one of the frames did not hold still, so it
    # was something moving over the board rather than the game marking it.
    before = np.zeros((200, 400, 3), np.uint8)
    steady, flash = before.copy(), before.copy()
    tsums = line((50, 100), (350, 100))
    steady[88:112, 338:362] = 200
    flash[88:112, 338:362] = 200
    flash[88:112, 38:62] = 200  # only in this frame

    hits, _, _ = marks_on_board(before, [flash, steady, steady], tsums, 25.0,
                                pressed=1, threshold=8.0, aura=1.0)
    assert hits == [], "the transient was kept"

    hits, _, _ = marks_on_board(before, [steady, steady], tsums, 25.0,
                                pressed=0, threshold=8.0, aura=90.0)
    assert hits == [1], "the persistent mark was dropped"


@pytest.mark.skipif(not (BOARDS / "board1.png").exists(),
                    reason="needs a captured board in scratchpad/")
def test_moving_sparkles_are_rejected_but_real_marks_survive():
    """FEVER: the board animates for 10s, so a two-frame diff reads the animation.

    Measured on this board with 45 moving sparkles painted over it: a single
    frame reports 11 false marks, three frames report none, and all three
    planted marks survive either way. That gap is why `--mark-frames`
    defaults to 3.
    """
    import cv2

    from ttheart_sender.game.tsum import detect

    rng = np.random.default_rng(7)
    img = cv2.imdecode(np.fromfile(str(BOARDS / "board1.png"), np.uint8),
                       cv2.IMREAD_COLOR)
    bx, by, bw, bh = MEASURED_BOARD
    crop = img[by:by + bh, bx:bx + bw]
    tsums, radius, _ = detect(crop, k=12)
    pressed = 0
    far = sorted(range(1, len(tsums)),
                 key=lambda i: -math.hypot(tsums[i].x - tsums[pressed].x,
                                           tsums[i].y - tsums[pressed].y))[:3]

    def with_marks(base):
        out = base.copy()
        for i in far:
            centre = crop[int(tsums[i].y), int(tsums[i].x)]
            glow = (0, 0, 0) if int(centre.mean()) > 127 else (255, 255, 255)
            cv2.circle(out, (int(tsums[i].x), int(tsums[i].y)),
                       int(radius * 0.6), glow, -1)
        return out

    def with_sparkles(base, n=45):
        out = base.copy()
        for _ in range(n):
            cv2.circle(out, (int(rng.integers(0, bw)), int(rng.integers(0, bh))),
                       int(rng.integers(6, 16)), (255, 255, 255), -1)
        return out

    frames = [with_sparkles(with_marks(crop)) for _ in range(3)]

    one, _, _ = marks_on_board(crop, frames[:1], tsums, radius, pressed=pressed,
                               threshold=8.0, aura=90.0)
    three, _, _ = marks_on_board(crop, frames, tsums, radius, pressed=pressed,
                                 threshold=8.0, aura=90.0)

    assert set(far) <= set(one) and set(far) <= set(three), "lost a real mark"
    assert len(set(one) - set(far)) > 5, "the sparkles were meant to fool one frame"
    assert set(three) == set(far), f"three frames still kept {set(three) - set(far)}"


@pytest.mark.skipif(not (BOARDS / "board1.png").exists(),
                    reason="needs a captured board in scratchpad/")
def test_a_board_that_moved_lights_up_almost_everything():
    """Why `--max-marked` exists, measured rather than asserted from theory.

    A pile still settling between the baseline and the hold differs from
    itself everywhere, so nearly every tsum clears the threshold. That is
    indistinguishable from marks tsum by tsum, but obvious in the aggregate: a
    real press marks the tsums of one character, never most of the board.

    Where the line sits is measured, not guessed. Good live reads came in at
    15% and 32% of the board; a press caught mid-settle reported 86%; and the
    4px shift below lands at 59%, which is what pulled the default down to
    50% from the 60% first tried -- 60% let this very case through.
    """
    import cv2

    from ttheart_sender.game.tsum import detect

    img = cv2.imdecode(np.fromfile(str(BOARDS / "board1.png"), np.uint8),
                       cv2.IMREAD_COLOR)
    bx, by, bw, bh = MEASURED_BOARD
    crop = img[by:by + bh, bx:bx + bw]
    tsums, radius, _ = detect(crop, k=12)

    # The pile dropping a few pixels, which is what a press mid-settle sees.
    moved = np.roll(crop, 4, axis=0)
    hits, _, _ = marks_on_board(crop, moved, tsums, radius,
                                pressed=0, threshold=8.0, aura=90.0)

    share = len(hits) / (len(tsums) - 1)
    assert share > 0.5, f"only {share:.0%} lit up; the guard would let this through"


# -- ordering -------------------------------------------------------------
def test_the_tour_starts_on_the_tsum_under_the_finger():
    # The whole point: the existing routers choose their own start, or reverse
    # the path to shorten the opening hop. Neither is available once a finger
    # is already down on a particular tsum.
    tsums = line((10, 10), (200, 10), (400, 10))
    assert tour_from(2, [0, 1, 2], tsums)[0] == 2
    assert tour_from(1, [0, 1, 2], tsums)[0] == 1


def test_the_tour_visits_every_member_once_by_nearest_neighbour():
    tsums = line((0, 0), (300, 0), (100, 0), (200, 0))
    order = tour_from(0, [0, 1, 2, 3], tsums)
    assert order == [0, 2, 3, 1]


def test_a_lone_press_tours_to_itself():
    assert tour_from(0, [0], line((5, 5))) == [0]


# -- walking --------------------------------------------------------------
class FakeMouse:
    """Stands in for pyautogui, recording every position it is warped to."""

    def __init__(self, down_for=None):
        self.PAUSE = 0.0
        self.moves: list[tuple[float, float]] = []
        self._down_for = down_for

    def moveTo(self, x, y, duration=None):  # noqa: N802 - pyautogui's name
        self.moves.append((x, y))

    def still_down(self) -> bool:
        if self._down_for is None:
            return True
        return len(self.moves) < self._down_for


@pytest.fixture
def fake_mouse(monkeypatch):
    def install(**kwargs):
        mouse = FakeMouse(**kwargs)
        monkeypatch.setitem(sys.modules, "pyautogui", mouse)
        return mouse

    yield install


def test_walking_never_presses_or_releases(fake_mouse):
    # The user's finger owns the button. A mouseDown here would double the
    # touch, and a mouseUp would end their chain early -- so the stand-in has
    # neither method, and calling one is an AttributeError rather than a
    # silently wrong stroke.
    mouse = fake_mouse()
    walk_path([(0, 0), (100, 0)], per_step=0.0)
    assert not hasattr(mouse, "mouseDown")
    assert mouse.moves, "it did move the cursor"


def test_walking_steps_in_small_hops_rather_than_jumping(fake_mouse):
    # The chain is built from what the touch passes over, so a jump straight to
    # the next tsum would teleport past everything between them.
    mouse = fake_mouse()
    walk_path([(0, 0), (80, 0)], step_px=8.0, per_step=0.0)
    assert len(mouse.moves) == 10
    gaps = [math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip([(0, 0)] + mouse.moves, mouse.moves)]
    assert max(gaps) <= 8.0 + 1e-6


def test_walking_reports_every_leg_it_completed(fake_mouse):
    fake_mouse()
    assert walk_path([(0, 0), (80, 0), (160, 0)], step_px=8.0, per_step=0.0) == 2


def test_walking_stops_the_moment_the_user_lets_go(fake_mouse):
    # Every move after the release is a bare hover that starts nothing, so the
    # walk has to notice mid-leg rather than at the next waypoint.
    mouse = fake_mouse(down_for=5)
    walked = walk_path([(0, 0), (800, 0)], step_px=8.0, per_step=0.0,
                       still_down=mouse.still_down)
    assert walked == 0, "the leg never finished"
    assert len(mouse.moves) == 5


def test_walking_honours_the_stop_key(fake_mouse):
    from ttheart_sender.exceptions import StopRequested

    mouse = fake_mouse()

    def check_stop():
        if len(mouse.moves) >= 3:
            raise StopRequested("stop key")

    with pytest.raises(StopRequested):
        walk_path([(0, 0), (800, 0)], step_px=8.0, per_step=0.0, check_stop=check_stop)
    assert len(mouse.moves) == 3
