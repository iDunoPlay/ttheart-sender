"""Waiting for the board to stop moving -- the biggest number in a round.

Measured over 26 logged rounds, the gap between one chain and the next averages
839ms, of which **144ms is thinking and 695ms is the stroke and this wait**. So
detection is 17% of a round and everything downstream of the decision is 83%,
which is the opposite of where every previous round of tuning went.

It matters because of what the score turns on. Over the five rounds with both a
score and telemetry, score tracks CHAINS PLAYED (r=+0.91) and not mean chain
length (r=+0.06) -- so a round's throughput is the thing being optimised, and
this wait is the throughput.

`--settle-board` restricts the comparison to the board rect. The score counter,
the round timer and the FEVER meter animate continuously and every one of them
sits outside the board, so a whole-frame wait can sit at its cap long after the
tsums have stopped falling. `cleared_by_drag` already documents exactly this
about whole-frame diffs; this applies it to the wait as well.
"""

from __future__ import annotations

import numpy as np

from ttheart_sender.game import tsum


class _Frames:
    """A driver whose screen changes only where the test says it does.

    Unbounded on purpose: a fixed list runs out, and then two identical grabs
    in a row read as a settled board and the wait returns early for a reason
    that has nothing to do with what is being tested.
    """

    def __init__(self, make):
        self.make = make
        self.grabs = 0

    def grab(self, *_a, **_k):
        f = self.make(self.grabs)
        self.grabs += 1
        return f

    def check_stop(self):
        pass


def _frame(board_val, chrome_val):
    """A frame with a quiet board and a noisy strip of UI above it."""
    f = np.zeros((80, 60, 3), np.uint8)
    f[0:20] = chrome_val      # score counter / timer / fever meter
    f[20:80] = board_val      # the board
    return f


BOARD = (0, 20, 60, 60)       # x, y, w, h


def test_it_returns_as_soon_as_the_whole_screen_is_still():
    drv = _Frames(lambda i: _frame(100, 50))
    out: dict = {}
    tsum._settle(drv, max_wait=1.0, out=out)

    assert out["timed_out"] is False
    assert out["waited"] >= 0.0


def test_animated_chrome_holds_a_whole_frame_wait_at_its_cap():
    """The failure this is about: the board is still, the screen is not."""
    drv = _Frames(lambda i: _frame(100, (13 * i) % 250))
    out: dict = {}
    tsum._settle(drv, max_wait=0.25, out=out)

    assert out["timed_out"] is True, (
        "a ticking score counter must be what keeps this waiting -- if this "
        "passes, the test frames are not animating")


def test_scoping_to_the_board_ignores_the_chrome():
    """Same frames, same cap, and it returns instead of timing out."""
    drv = _Frames(lambda i: _frame(100, (13 * i) % 250))
    out: dict = {}
    tsum._settle(drv, max_wait=0.25, region=BOARD, out=out)

    assert out["timed_out"] is False, (
        "the board never moved, so scoping to it must settle immediately")


def test_a_moving_board_still_waits_even_when_scoped():
    """The rule must not become 'never wait'. Falling tsums are the reason it
    exists: detecting mid-fall gives coordinates that are stale by the time the
    drag runs."""
    drv = _Frames(lambda i: _frame((13 * i) % 250, 50))
    out: dict = {}
    tsum._settle(drv, max_wait=0.2, region=BOARD, out=out)

    assert out["timed_out"] is True


def test_the_wait_is_reported():
    """A cost nobody prints is a cost nobody tunes."""
    r = tsum.PlayReport()
    r.played = 3
    r.settles = 4
    r.settle_s = 2.0
    r.settle_timeouts = 1
    line = r.describe() if hasattr(r, "summary") else str(r)

    assert "settle" in line.lower() or "waited" in line.lower()


def test_it_is_off_by_default():
    assert tsum.play_defaults().settle_board is False, "a new rule ships off"


def test_the_flow_action_accepts_it():
    from ttheart_sender.automation import tsum_actions
    assert "settle_board" in tsum_actions._TUNABLES


def test_the_fever_share_is_reported():
    """The number the score actually turns on.

    Over 15 rounds carrying both a score and telemetry, score tracks the share
    of the round spent in FEVER at r=+0.81 and mean chain length at r=-0.01.
    The two catastrophic rounds in that set -- 9,841 and 25,065 against a
    normal 600,000-900,000 -- are exactly the two that never entered FEVER.
    Nothing in the loop reported it, so a round could collapse by two orders of
    magnitude with every other number on the line looking ordinary.
    """
    r = tsum.PlayReport()
    r.played = 40
    r.frames = 100
    r.fever_frames = 45
    line = r.describe()

    assert "FEVER" in line
    assert "45%" in line


def test_a_round_with_no_frames_says_nothing_about_fever():
    """Absent beats a 0% that reads as 'never entered FEVER'."""
    r = tsum.PlayReport()
    r.played = 3
    assert "FEVER" not in r.describe()
