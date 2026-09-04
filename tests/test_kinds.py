"""`--kinds`: sort the board into a fixed number of identities.

The game allows at most 5 characters on a board, 4 with an item. That bound is
what `--recolour` never had: it merges by a Lab *distance*, nothing stops the
merging, and following that gradient produced chains of 27 and 31 tsums. A
count is bounded by construction.

What is pinned here is not the accuracy -- that lives in the table
`scripts/group_eval.py` prints, measured against the game's own marks -- but
the properties the rule owes:

* it is inert unless asked for;
* it produces AT MOST the number asked for, whatever the board looks like,
  because that bound is the entire argument for preferring it to `--recolour`;
* every tsum keeps an id, so nothing becomes unchainable;
* two tsums of one colour end up in one group, which is the point;
* the equipped tsum is re-derived afterwards, because renumbering `kind`
  invalidates the palette index `read_base_kind` returns -- the same landmine
  `--recolour` and `--character` both carry.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttheart_sender.game import tsum


def _board(colours, radius=20.0):
    """A frame with one flat disk per colour, and the Tsums that name them."""
    n = len(colours)
    bgr = np.zeros((160, 80 * n + 80, 3), np.uint8)
    tsums = []
    for i, c in enumerate(colours):
        x, y = 60 + 80 * i, 80
        cv = tsum.cv2
        cv.circle(bgr, (x, y), int(radius), tuple(int(v) for v in c), -1)
        tsums.append(tsum.Tsum(x=float(x), y=float(y), r=radius, kind=i,
                               colour=(0, 0, 0)))
    return bgr, tsums


def test_it_is_off_by_default():
    assert tsum.play_defaults().kinds == 0, "a new rule ships off"


def test_the_flow_action_accepts_it():
    from ttheart_sender.automation import tsum_actions
    assert "kinds" in tsum_actions._TUNABLES


@pytest.mark.parametrize("groups", [2, 3, 4, 5])
def test_it_never_makes_more_groups_than_asked(groups):
    """The bound is the whole reason this exists rather than `--recolour`."""
    # Eight visibly different colours: pixel k-means would happily call these
    # eight kinds, and a board can only hold five characters.
    colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30),
               (200, 30, 200), (30, 200, 200), (240, 240, 240), (60, 60, 60)]
    bgr, tsums = _board(colours)
    out = tsum._regroup(bgr, tsums, 20.0, groups)

    assert len({t.kind for t in out}) <= groups
    assert len(out) == len(colours), "regrouping must not drop a tsum"


def test_every_tsum_keeps_an_id():
    """A tsum with no id is unchainable, which is the disease not the cure.

    The hybrid `--character` rule left half a board on colour ids and renamed
    the other half, so one character routinely carried a name AND two
    clusters. Whatever this rule decides, it must decide it for everyone.
    """
    bgr, tsums = _board([(200, 30, 30), (30, 200, 30), (30, 30, 200)])
    out = tsum._regroup(bgr, tsums, 20.0, 2)

    assert all(isinstance(t.kind, int) for t in out)
    assert all(t.kind >= 0 for t in out)


def test_two_tsums_of_one_colour_land_together():
    """The point of the exercise, and what `adjacency` then acts on."""
    bgr, tsums = _board([(200, 30, 30), (30, 30, 200), (198, 32, 28)])
    out = tsum._regroup(bgr, tsums, 20.0, 2)

    assert out[0].kind == out[2].kind, "near-identical faces must group"
    assert out[1].kind != out[0].kind, "a clearly different face must not"


def test_a_board_smaller_than_the_group_count_is_left_alone():
    bgr, tsums = _board([(200, 30, 30), (30, 200, 30)])
    before = [t.kind for t in tsums]
    out = tsum._regroup(bgr, tsums, 20.0, 5)
    assert [t.kind for t in out] == before


def test_detect_applies_it():
    """Wired into `detect`, not merely present."""
    import inspect
    sig = inspect.signature(tsum.detect)
    assert "kinds" in sig.parameters
    assert sig.parameters["kinds"].default == 0


def test_the_base_tsum_is_re_derived_when_regrouping():
    """`read_base_kind` returns a PALETTE index; this rule throws those away.

    Left alone the bot would keep preferring an arbitrary group, stop charging
    the skill, and say nothing about it -- the exact failure `--recolour` and
    `--character` each had to guard.
    """
    import re
    src = inspect_source()
    assert re.search(r"opts\.recolour > 0 or opts\.kinds > 0", src), (
        "turning `kinds` on must trigger the same base re-derivation "
        "`recolour` does")


def inspect_source() -> str:
    import inspect
    return inspect.getsource(tsum.play_loop)
