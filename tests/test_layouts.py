"""The measured layouts, and the guard for when a capture size is not one.

`LAYOUTS` maps a captured frame size to the board rect and tsum radius someone
measured for it. Everything downstream leans on it: given a layout, `detect` is
told the radius instead of inferring it and the `--radius-lock` warm-up is
skipped entirely.

Without one the loop still runs, and that is the danger. The board rect falls
back to a fraction of the whole frame -- which on LDPlayer includes the score
bar and the FEVER strip, and detects them -- and the radius falls back to an
estimator measured at 8-38px on a board whose faces are 25. Neither failure
trips a gate: the detection count stays inside `--min-tsums`..`--max-tsums`,
so nothing complains and the round just plays badly.

That is not hypothetical. Changing the emulator's resolution moved the capture
from 578x994 to 598x1031, and every measured number silently stopped applying.
"""

from __future__ import annotations

import pytest

from ttheart_sender.game import tsum


def test_a_measured_layout_is_used_instead_of_estimating():
    shape = (994, 578, 3)
    assert tsum._layout(shape), "the live LDPlayer size must stay measured"
    assert tsum._layout_radius(shape) == pytest.approx(25.0)


def test_an_unmeasured_size_reports_nothing_rather_than_guessing():
    """It must not interpolate. A rect scaled from another capture's chrome is
    arithmetic on an unmeasured offset, and reads as a measurement afterwards."""
    assert tsum._layout((1234, 567, 3)) == {}
    assert tsum._layout_radius((1234, 567, 3)) is None


@pytest.mark.parametrize("shape", sorted(tsum.LAYOUTS))
def test_every_layout_is_self_consistent(shape):
    """A typo in the table is otherwise invisible until a round plays badly."""
    h, w = shape
    lay = tsum.LAYOUTS[shape]

    for key in ("board", "fever_board"):
        rect = lay.get(key)
        if rect is None:
            continue
        x, y, bw, bh = rect
        assert 0 <= x and 0 <= y, f"{key} starts off the frame"
        assert x + bw <= w, f"{key} runs past the right edge ({x}+{bw} > {w})"
        assert y + bh <= h, f"{key} runs past the bottom ({y}+{bh} > {h})"
        # A board rect that covers most of the frame has swallowed the score
        # bar and the FEVER strip, which is the fallback's failure, not a
        # measurement.
        assert bh < h * 0.75, f"{key} is too tall to be the pile alone"

    radius = lay.get("radius")
    if radius is not None:
        # Tsums are a fixed fraction of the screen: ~25px of a 578-wide
        # capture. Anything far off that is a typo or a different game.
        assert 0.02 * w < radius < 0.09 * w, f"radius {radius} implausible for {w}px wide"

    base = lay.get("base")
    if base:
        cx, cy, r = (float(v) for v in base.split(","))
        assert 0 < cx < w and 0 < cy < h, "the skill icon must be on the frame"
        assert 0 < r < 0.15 * w


def test_the_board_rect_falls_back_rather_than_failing():
    """An unmeasured size must still play -- badly, and loudly, but not crash."""
    rect = tsum._board_rect((1031, 598, 3), None)
    x, y, w, h = rect
    assert w > 0 and h > 0
    assert x + w <= 598 and y + h <= 1031
