"""The play loop's `misses` streak: every dead end has to reach the shuffle.

A board that keeps offering chains the purity gate rejects is stuck in exactly
the way `--max-misses` exists to break, and the log shows it happening for
thirty frames at a time. It only breaks if the counter survives the frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ttheart_sender.game import tsum


class Templates:
    """No max_fever template -- FeverWatch degrades to 'never FEVER'."""

    def get(self, name):
        raise KeyError(name)


def _driver(frames):
    return SimpleNamespace(
        capture=None,
        matcher=None,
        templates=Templates(),
        rect=None,
        grab=lambda: frames(),
        to_screen=lambda x, y: (int(x), int(y)),
        check_stop=lambda: None,
        say=lambda msg: None,
    )


def _options(**over):
    opts = tsum.play_defaults()
    opts.duration = 30.0
    opts.countdown = 0.0
    opts.dry_run = False
    opts.settle = 0.0
    opts.move_time = 0.0
    opts.hold = 0.0
    opts.shuffle_delay = 0.0
    opts.skill = ""          # no skill button to read
    opts.bubble = ""         # no bubble templates to load
    opts.use_base = False
    for key, value in over.items():
        setattr(opts, key, value)
    return opts


@pytest.fixture
def board(monkeypatch):
    """A frame that always detects a chain, and never a clean one."""
    frame = np.zeros((994, 578, 3), np.uint8)
    tsums = [tsum.Tsum(x=10.0 * i, y=10.0, r=25.0, kind=1, colour=(0, 0, 0))
             for i in range(30)]
    chain = tsum.Chain(kind=1, colour=(0, 0, 0), nodes=[0, 1, 2])

    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0: frame)
    monkeypatch.setattr(tsum, "detect",
                        lambda crop, **kw: (tsums, 25.0, np.zeros((12, 3), np.float32)))
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [chain])
    # Every member reads as the wrong colour, so no chain survives the gate.
    monkeypatch.setattr(tsum, "purity_filter", lambda *a, **kw: [])
    return frame


def test_a_purity_fail_streak_taps_the_shuffle(board, monkeypatch):
    """The bug this pins: `misses` used to reset on any frame that had chains.

    The purity gate runs *after* that reset and is the thing incrementing the
    counter, so a board whose chains all failed it sat at 1 forever and the
    recovery below max_misses never ran -- thirty-odd identical skips in the
    log and no shuffle.
    """
    taps = []
    monkeypatch.setattr(tsum, "_click_shuffle",
                        lambda drv, spec, times, delay, hold, move_time: taps.append(spec))

    frames = []
    def stop_when(_frame):
        frames.append(1)
        return "stopping" if taps else ""

    opts = _options(max_misses=6, shuffle="459,859")
    tsum.play_loop(_driver(lambda: board), opts, stop_when=stop_when)

    assert taps == ["459,859"], "the fan is tapped once the streak hits max_misses"
    assert len(frames) == 7, "six unplayable frames, then the next frame stops the run"


def test_the_streak_survives_a_frame_that_finds_chains(board, monkeypatch):
    """Not just purity: any mix of dead ends counts toward the same streak.

    A no-chain frame and an impure-chain frame are the same problem from the
    board's point of view, so they must not clear each other's count.
    """
    taps = []
    monkeypatch.setattr(tsum, "_click_shuffle",
                        lambda drv, spec, times, delay, hold, move_time: taps.append(spec))

    seen = {"n": 0}
    def alternating(*a, **kw):
        # Odd frames offer nothing at all; even frames offer an impure chain.
        seen["n"] += 1
        return [] if seen["n"] % 2 else [tsum.Chain(1, (0, 0, 0), [0, 1, 2])]
    monkeypatch.setattr(tsum, "find_chains", alternating)

    opts = _options(max_misses=6, shuffle="459,859")
    tsum.play_loop(_driver(lambda: board), opts,
                   stop_when=lambda _f: "stopping" if taps else "")

    assert taps == ["459,859"]
