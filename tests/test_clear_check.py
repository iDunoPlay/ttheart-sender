"""Did the drag clear anything?

`--verify` answers "did the emulator see the stroke", which a live board
passes on animation alone -- the score counter, the timer, the FEVER meter and
idle tsums jiggling. These cover the narrower question the report was
previously guessing at: did *these* tsums leave the board.

Synthetic frames rather than saved boards: the fixtures under scratchpad/ are
not in the repo, and the effect being measured is a pixel one that does not
need a real Mickey to demonstrate.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.tsum import PlayReport, cleared_by_drag


def _board(seed: int = 0):
    """A board of coloured discs on a dark bowl, plus its tsum list."""
    rng = np.random.default_rng(seed)
    frame = np.full((300, 300, 3), 30, np.uint8)
    tsums = []
    for index, (x, y) in enumerate([(60, 60), (140, 60), (220, 60), (60, 160), (140, 160)]):
        colour = tuple(int(v) for v in rng.integers(80, 255, 3))
        cv2.circle(frame, (x, y), 24, colour, -1)
        tsums.append(tsum.Tsum(x=float(x), y=float(y), r=24.0, kind=index, colour=colour))
    # Flat discs would make a jiggle read as exactly zero, which is the one
    # thing a real sprite's face never does. A smooth field stands in for the
    # features a tsum actually carries.
    texture = cv2.GaussianBlur(rng.integers(0, 255, (300, 300)).astype(np.uint8), (0, 0), 2)
    frame = np.clip(frame.astype(np.int16) + ((texture.astype(np.int16) - 128) // 2)[..., None],
                    0, 255).astype(np.uint8)
    return frame, tsums


def _shift(frame, by):
    matrix = np.float32([[1, 0, by], [0, 1, by]])
    return cv2.warpAffine(frame, matrix, frame.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)


def test_a_board_that_only_jiggles_cleared_nothing(): 
    """The failure mode this exists for.

    A settling pile moves a pixel or two, which is enough for the whole-crop
    mean to clear --change-tol while nothing has popped at all.
    """
    before, tsums = _board()

    gone, values, idle = cleared_by_drag(before, _shift(before, 2), tsums, [0, 1, 2], tol=20.0)

    assert gone == []
    assert max(values) < 20.0, f"jiggle reads as {max(values):.0f}, at or above the threshold"
    assert idle < 20.0


def test_tsums_that_were_replaced_read_as_gone():
    before, tsums = _board()
    after = before.copy()
    for node in (0, 1):
        cv2.circle(after, (int(tsums[node].x), int(tsums[node].y)), 24, (10, 200, 10), -1)

    gone, values, idle = cleared_by_drag(before, after, tsums, [0, 1, 2], tol=20.0)

    assert gone == [0, 1]
    assert values[2] == pytest.approx(0.0), "the member that stayed put reads as nothing"


def test_the_idle_baseline_comes_from_the_tsums_that_were_not_dragged():
    """Reported beside every reading, so the threshold is judged against the
    board's own noise rather than guessed at.

    A median, so one neighbour knocked about by the stroke does not drag the
    yardstick up with it -- it takes most of the board moving to do that.
    """
    before, tsums = _board()
    one_neighbour = before.copy()
    cv2.circle(one_neighbour, (int(tsums[3].x), int(tsums[3].y)), 24, (10, 200, 10), -1)
    assert cleared_by_drag(before, one_neighbour, tsums, [0, 1], tol=20.0)[2] == 0.0

    whole_board = _shift(before, 2)
    assert cleared_by_drag(before, whole_board, tsums, [0, 1], tol=20.0)[2] > 0.0


# -- what the report says --------------------------------------------------
def test_an_unmeasured_round_does_not_claim_to_have_cleared_anything():
    report = PlayReport(played=2, dragged=7)
    assert "dragged 7 tsums" in report.describe()
    assert "cleared" not in report.describe()


def test_a_measured_round_reports_both_numbers():
    report = PlayReport(played=2, dragged=7, cleared=4, checked=True, rejected=1)
    described = report.describe()
    assert "dragged 7 tsums" in described
    assert "cleared 4 (57%)" in described
    assert "1 the game would not accept" in described


# -- the loop --------------------------------------------------------------
class Templates:
    def get(self, name):
        raise KeyError(name)


def _run_loop(monkeypatch, before, after, tsums, nodes, **over):
    chain = tsum.Chain(kind=1, colour=(0, 0, 0), nodes=list(nodes))
    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0: before)
    monkeypatch.setattr(tsum, "detect",
                        lambda crop, **kw: (tsums, 24.0, np.zeros((12, 3), np.float32)))
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [chain])
    monkeypatch.setattr(tsum, "purity_filter", lambda bgr, ts, ns, r, tol: list(ns))
    # The loop grabs rather than settles while the palette is still unfit, so
    # the board only becomes the after-frame once the drag has run.
    state = {"dragged": False}

    def fake_drag(screen, **kw):
        if kw.get("after_press"):
            kw["after_press"]()
        state["dragged"] = True

    monkeypatch.setattr(tsum, "drag_chain", fake_drag)

    opts = tsum.play_defaults()
    opts.duration, opts.settle = 30.0, 0.0
    opts.skill, opts.bubble, opts.use_base = "", "", False
    opts.min_tsums = 3
    opts.verify_clears = True
    # The synthetic frame is all board; without this the loop would crop to
    # the fractional default and the tsum coordinates would not line up.
    opts.board = "0,0,300,300"
    for key, value in over.items():
        setattr(opts, key, value)

    said = []
    drv = SimpleNamespace(capture=None, matcher=None, templates=Templates(), rect=None,
                          grab=lambda: after if state["dragged"] else before,
                          to_screen=lambda x, y: (int(x), int(y)),
                          check_stop=lambda: None, say=said.append)
    played = []
    report = tsum.play_loop(
        drv, opts, stop_when=lambda _f: "done" if played else played.append(1))
    return report, said


def test_a_drag_the_game_refuses_is_counted_apart_from_a_missed_stroke(monkeypatch):
    """Two different failures that used to look identical in the report.

    The board moved, so the stroke was delivered -- but none of the dragged
    tsums went, which means the chain was not really one character. Slowing
    the drag down would not help; not offering that kind again might.
    """
    before, tsums = _board()
    after = _shift(before, 2)          # the board animates, nothing pops

    report, said = _run_loop(monkeypatch, before, after, tsums, [0, 1, 2])

    assert report.checked and report.cleared == 0
    assert report.rejected == 1
    assert report.stalled == 0, "a refused chain is not a stroke the emulator missed"
    assert report.dragged == 3, "it was still dragged; that is what dragged counts"
    assert any("popped 0/3" in line for line in said)


def test_a_drag_that_clears_is_measured_not_assumed(monkeypatch):
    before, tsums = _board()
    after = before.copy()
    for node in (0, 1, 2):
        cv2.circle(after, (int(tsums[node].x), int(tsums[node].y)), 24, (10, 200, 10), -1)

    report, _said = _run_loop(monkeypatch, before, after, tsums, [0, 1, 2])

    assert report.cleared == 3 and report.rejected == 0
