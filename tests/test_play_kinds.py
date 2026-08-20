"""`--kinds` inside the play loop, which is the wiring with a silent failure.

`fixed_kinds` renumbers the board every frame it fits. Two things in the loop
outlive a frame and are indexed by kind: the base tsum, and the skip list. Get
the base wrong and `--base-only` chases a character the player is not equipped
with -- no error, no stall, just a worse round, which is exactly the kind of
bug this module's history says to test rather than reason about.

Runs the real `detect` over a real `synth()` board rather than stubbing it: the
whole question is whether the ids the loop passes around index the same set,
and a stub would have to invent that set to answer it.
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


def _driver(frame):
    return SimpleNamespace(
        capture=None,
        matcher=None,
        templates=Templates(),
        rect=None,
        grab=lambda: frame,
        to_screen=lambda x, y: (int(x), int(y)),
        check_stop=lambda: None,
        say=lambda msg: None,
    )


def _options(**over):
    opts = tsum.play_defaults()
    opts.countdown = 0.0
    opts.settle = 0.0
    opts.move_time = 0.0
    opts.hold = 0.0
    opts.dry_run = True          # decide everything, touch nothing
    opts.skill = ""
    opts.bubble = ""
    for key, value in over.items():
        setattr(opts, key, value)
    return opts


@pytest.fixture(scope="module")
def board():
    return tsum.synth()


def _run(frame, **over):
    """One iteration of the real loop, returning what it said and did."""
    said = []
    drv = _driver(frame)
    drv.say = said.append
    opts = _options(**over)
    seen = {"n": 0}

    def stop_after_one(_frame):
        seen["n"] += 1
        return "" if seen["n"] < 2 else "enough"

    report = tsum.play_loop(drv, opts, stop_when=stop_after_one)
    return report, said


def test_the_loop_runs_with_kinds_on(board):
    report, _ = _run(board, kinds=5, use_base=False)
    assert report.reason


def test_the_base_id_is_in_range(board):
    """A shape check only, and deliberately labelled as one.

    It is tempting to read this as proving the base was matched against the
    face centres rather than the 12-entry pixel palette. It does not: measured
    on this board the wrong wiring returns #4, which is inside `range(5)` and
    passes. `test_play_kinds_base.py` makes that distinction properly, and
    carries its own guard against the fixture quietly losing the ability to.
    """
    _, said = _run(board, kinds=5, use_base=True)
    lines = [m for m in said if m.startswith("base tsum:")]
    assert lines, f"expected a base reading, got {said}"

    read = int(lines[0].split("#")[1].split()[0])
    assert 0 <= read < 5, f"base #{read} does not index the five characters"


def test_a_weak_base_is_reported_once_not_every_frame(board):
    """`--kinds` re-reads the base per frame, so the log must not repeat it.

    A weak reading leaves `base` at None, and the obvious "announce when base
    is None" rule then fires on every single frame of a 60-second round.
    """
    said = []
    drv = _driver(board)
    drv.say = said.append
    seen = {"n": 0}

    def stop_after_five(_frame):
        seen["n"] += 1
        return "" if seen["n"] < 6 else "enough"

    tsum.play_loop(drv, _options(kinds=5, use_base=True), stop_when=stop_after_five)
    lines = [m for m in said if m.startswith("base tsum:")]
    assert len(lines) == 1, f"base announced {len(lines)} times: {lines}"


def test_without_kinds_the_base_still_indexes_the_pixel_palette(board):
    """The other half: turning the feature off must change nothing."""
    _, said = _run(board, kinds=0, use_base=True, k=12)
    lines = [m for m in said if m.startswith("base tsum:")]
    assert lines
    read = int(lines[0].split("#")[1].split()[0])
    assert 0 <= read < 12


def test_kinds_are_renumbered_into_the_range_asked_for(board):
    """Whatever the pixel clustering produced, the loop sees 0..n-1."""
    bx, by, bw, bh = tsum._board_rect(board.shape, None)
    crop = board[by:by + bh, bx:bx + bw]
    tsums, radius, palette = tsum.detect(crop)
    assert max(t.kind for t in tsums) >= 5, "synth board should over-split first"

    fixed, centres = tsum.fixed_kinds(crop, tsums, radius, 5, palette=palette)
    assert centres is not None and len(centres) == 5
    assert {t.kind for t in fixed} <= set(range(5))
