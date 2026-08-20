"""The base tsum has to name the character the player is actually equipped with.

Split out from `test_play_kinds` because the obvious assertion there -- that
the id is in `range(kinds)` -- was measured to have no teeth: on the synth
board a base read against the *wrong* set (the 12-entry pixel palette) comes
back as #4, which is inside `range(5)` and passes a test that should fail.

So this checks the thing that actually matters: the id names the character
whose sprites are on the board, not merely a number of the right size.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttheart_sender.game import tsum


#: A pale character, so `include_dark` keeps it and it really is among the
#: faces. `synth` equips near-black Mickey by default, which detection drops
#: on purpose -- a fine board for testing detection, useless for this.
EQUIPPED = 1  # SYNTH_PALETTE index: donald, (248, 245, 245)


@pytest.fixture
def board(monkeypatch):
    monkeypatch.setattr(tsum, "BASE_INDEX", EQUIPPED)
    return tsum.synth()


def _kind_of_the_equipped_character(crop, tsums, radius):
    """Which kind holds the sprites drawn in the equipped colour."""
    want = cv2.cvtColor(np.uint8([[tsum.SYNTH_PALETTE[EQUIPPED][0]]]),
                        cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    feats = tsum._face_lab(crop, tsums, radius)
    kinds = np.array([t.kind for t in tsums])
    best, best_d = None, float("inf")
    for k in np.unique(kinds):
        d = float(np.linalg.norm(np.median(feats[kinds == k], axis=0) - want))
        if d < best_d:
            best, best_d = int(k), d
    return best, best_d


import cv2  # noqa: E402  (after the docstring, beside the other heavy imports)


def test_base_names_the_equipped_character(board):
    bx, by, bw, bh = tsum._board_rect(board.shape, None)
    crop = board[by:by + bh, bx:bx + bw]
    tsums, radius, palette = tsum.detect(crop)
    fixed, centres = tsum.fixed_kinds(crop, tsums, radius, 5, palette=palette)

    want, want_d = _kind_of_the_equipped_character(crop, fixed, radius)
    assert want_d < 40, "the equipped character is not on the board -- fix the fixture"

    base, dist = tsum.read_base_kind(board, centres)
    assert base == want, (
        f"base came back #{base}, but the equipped character is kind #{want}")
    assert dist < tsum.BASE_WEAK, f"a correct match should not read as weak ({dist:.1f})"


def test_reading_the_base_against_the_pixel_palette_is_caught(board):
    """The teeth: the wrong wiring must fail this test, not sneak through."""
    bx, by, bw, bh = tsum._board_rect(board.shape, None)
    crop = board[by:by + bh, bx:bx + bw]
    tsums, radius, palette = tsum.detect(crop)
    fixed, centres = tsum.fixed_kinds(crop, tsums, radius, 5, palette=palette)

    want, _ = _kind_of_the_equipped_character(crop, fixed, radius)
    wrong, _ = tsum.read_base_kind(board, palette)   # the bug: 12-entry palette
    assert wrong != want or len(palette) <= 5, (
        "this fixture can no longer tell the two wirings apart -- "
        "the test above is passing for the wrong reason")
