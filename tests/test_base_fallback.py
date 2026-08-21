"""What happens when the board has none of the tsum you are equipped with.

The worry is that `play` hunts for the equipped ("base") character and taps
shuffle when it cannot find one, wasting the fan on a board full of perfectly
playable chains of something else. It does not, and these pin that down:
the base only *sorts* chains, and the fan is reserved for a board with no
chain of `min_chain`+ of ANY kind.

Three separate mechanisms have to hold for that to stay true, so there is a
test for each rather than one that would pass if any of them carried it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.tsum import Tsum, find_chains


def _row(kind, y, n=4, x0=40.0, step=45.0):
    """`n` tsums of one kind in a row, close enough to link."""
    return [Tsum(x=x0 + step * i, y=float(y), r=22.0, kind=kind, colour=(0, 0, 0))
            for i in range(n)]


def test_a_base_with_no_chain_does_not_hide_the_other_chains():
    """The equipped character is on the board but never three in a row."""
    tsums = _row(1, 60) + _row(1, 400, n=1) + [Tsum(x=300.0, y=300.0, r=22.0,
                                                    kind=9, colour=(0, 0, 0))]
    chains = find_chains(tsums, 22.0, link_px=105, block=1.25, base_kind=9)

    assert chains, "a chain of the other character should still be offered"
    assert all(c.kind == 1 for c in chains)
    assert not any(c.is_base for c in chains)


def test_a_base_absent_from_the_board_entirely_is_harmless():
    """Equipped with something the board was not dealt at all."""
    tsums = _row(1, 60) + _row(2, 200)
    chains = find_chains(tsums, 22.0, link_px=105, block=1.25, base_kind=7)

    assert {c.kind for c in chains} == {1, 2}
    assert len(chains[0]) >= 3


def test_the_base_only_sorts_it_does_not_filter():
    """A short base chain outranks a long one, but the long one survives."""
    tsums = _row(1, 60, n=6) + _row(2, 300, n=3)
    chains = find_chains(tsums, 22.0, link_px=105, block=1.25, base_kind=2)

    assert chains[0].kind == 2 and chains[0].is_base, "base should sort first"
    assert any(c.kind == 1 and len(c) > len(chains[0]) for c in chains), \
        "the longer non-base chain must still be offered"


def test_base_only_is_the_flag_that_filters_and_it_is_off_by_default():
    tsums = _row(1, 60) + _row(2, 300)
    assert not find_chains(tsums, 22.0, link_px=105, block=1.25,
                           base_kind=7, base_only=True)
    assert tsum.play_defaults().base_only is False


def test_no_base_reading_at_all_ranks_by_length():
    """With `base_kind` None the ordering is by chain length alone.

    `--no-base` is one way to get here; a board read before the icon has been
    matched is the other. Either way nothing may be promoted for being the
    base, because there is no base to promote.
    """
    tsums = _row(1, 60, n=6) + _row(2, 300, n=3)
    chains = find_chains(tsums, 22.0, link_px=105, block=1.25, base_kind=None)
    assert chains[0].kind == 1 and len(chains[0]) == 6
    assert not any(c.is_base for c in chains)
