"""The face crop that does not throw a third of the board away.

`CharacterModel._crop` refuses a crop the frame edge would clip, and for that
model it is right: its training set excluded clipped crops, so a clipped one is
a picture it has never seen an example of. The cost is that **31.5% of the
faces on a real board are clipped by the board rect** and go unidentified --
and an unidentified tsum keeps a different id from its own partners, which is
the exact split `--character` was built to close and instead doubled.

`embed_net.cut` takes the other road: pad by replication, and put padded crops
in the training set so inference on one is in distribution. What is pinned here
is that it always returns a usable square, never a stretched sliver, and never
refuses -- because the whole argument for the embedding is that every tsum on
the board gets a vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from embed_net import CROP, WINDOW, cut  # noqa: E402


def _frame(w=400, h=300):
    """A frame with a gradient, so a stretched crop is visible in the pixels."""
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    bgr[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return bgr


def test_the_window_is_one_radius():
    """1.5 reached into the neighbour: tsums sit about 2.44 radii apart, so a
    1.5r window makes every crop a picture of two characters."""
    assert WINDOW == 1.0


@pytest.mark.parametrize("x,y", [(200, 150), (0, 0), (399, 299), (5, 150),
                                 (200, 2), (398, 1)])
def test_it_always_returns_a_square_crop(x, y):
    out = cut(_frame(), x, y, 25.0)
    assert out is not None, "a crop at the frame edge must be padded, not refused"
    assert out.shape == (CROP, CROP, 3)


def test_an_edge_crop_is_padded_not_stretched():
    """The failure this replaces: clamping to the frame and resizing.

    Clamping a 51px window at x=0 leaves a 26px-wide sliver, and resizing that
    to a square doubles its width -- the face comes out stretched, and the net
    is asked about a shape no tsum has. Replication keeps the geometry: the
    centre of the padded crop still sits where the tsum's centre sits.
    """
    frame = _frame()
    edge = cut(frame, 3, 150, 25.0)
    middle = cut(frame, 200, 150, 25.0)
    assert edge.shape == middle.shape

    # The gradient runs left-to-right, so a stretched crop would show the full
    # sweep across its width. A padded one is flat over the padded columns.
    left_half = edge[:, :CROP // 3, 0].astype(float)
    assert left_half.std() < 3.0, (
        "the padded side must be flat; a spread here means the visible sliver "
        "was stretched across it")


def test_a_crop_entirely_outside_the_frame_is_refused():
    """Padding a crop with nothing in it would invent a face."""
    assert cut(_frame(), -500, -500, 25.0) is None


def test_the_centre_of_the_crop_is_the_tsum():
    """Whatever the padding does, the tsum has to stay in the middle."""
    frame = np.zeros((300, 400, 3), np.uint8)
    import cv2
    cv2.circle(frame, (8, 150), 20, (0, 0, 255), -1)   # clipped by the left edge
    out = cut(frame, 8, 150, 20.0)
    mid = out[CROP // 2, CROP // 2]
    assert int(mid[2]) > 200 and int(mid[0]) < 60, (
        f"the tsum should be at the centre of its own crop, got {mid}")
