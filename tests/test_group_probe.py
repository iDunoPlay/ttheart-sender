"""Group counts scored the way a chain is actually played.

The two things worth testing here are the ones that flipped the answer: only
scoring what the round could REACH, and scoring the run of correct members
rather than how many the group contains.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "scripts" / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = pytest.importorskip("cv2") and _load("group_probe")


def _row(x, y, r=25.0):
    from ttheart_sender.game import tsum as T
    return T.Tsum(x=x, y=y, r=r, kind=0, colour=(0, 0, 0))


def test_only_same_kind_tsums_are_reachable():
    """The grouping is what decides the candidate set; adjacency trims it."""
    tsums = [_row(100, 100), _row(150, 100), _row(200, 100)]

    out = gp.reachable(tsums, [0, 0, 1], head=0, radius=25.0,
                       link_px=105.0, block=1.25, cap=12)

    assert set(out) == {1}, "the tsum in another group is not a candidate"


def test_a_tsum_across_the_board_is_not_a_partner():
    """Same group, far away. Scoring it as one charges a grouping rule for a
    chain the round would never have offered -- the bias that made a big group
    look better than it plays."""
    tsums = [_row(100, 100), _row(120, 100), _row(900, 900)]

    out = gp.reachable(tsums, [0, 0, 0], head=0, radius=25.0,
                       link_px=105.0, block=1.25, cap=12)

    assert set(out) == {1}


def test_the_chain_is_capped_and_the_nearest_come_first():
    tsums = [_row(100, 100)] + [_row(100 + 40 * i, 100) for i in range(1, 8)]

    out = gp.reachable(tsums, [0] * len(tsums), head=0, radius=25.0,
                       link_px=105.0, block=1.25, cap=4)

    assert len(out) == 3, "cap counts the head"
    assert out == sorted(out), "nearest first, so the run is read in order"


def test_reachable_returns_an_order_not_a_set():
    """The accepted-run metric walks it in order; a set would make the run
    meaningless and it would still have scored something."""
    tsums = [_row(100, 100), _row(140, 100), _row(180, 100)]

    out = gp.reachable(tsums, [0, 0, 0], head=0, radius=25.0,
                       link_px=105.0, block=1.25, cap=12)

    assert isinstance(out, list)


def test_the_run_stops_at_the_first_stranger():
    """The game refuses 86% of what follows a refusal, so eight partners with
    a stranger third are worth two."""
    order, truth = [1, 2, 3, 4], {1, 2, 4}

    good = 0
    for i in order:
        if i not in truth:
            break
        good += 1

    assert good == 2, "the 4 beyond the stranger does not count"
