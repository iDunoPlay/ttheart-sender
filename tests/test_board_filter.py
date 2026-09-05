"""A detection that is board is not a tsum, and is dropped before it is chained.

The player's report: some detections "shouldn't be detected as it is background
or unknown object, which is invalid for the gameplay". A false detection
carries a `kind` like any other, so `adjacency` chains it and the stroke passes
over empty bowl -- a slot spent on nothing, and one more `dragged` that can
never clear.

The subtle failure these tests pin down is the one that took three attempts to
find: the board is printed with empty tsum-shaped SLOTS, and a tsum the game
has linked is drawn as a flat SILHOUETTE with the same outline. A filter
trained on `board` alone scored 1.000 AUC and threw away game-confirmed tsums
at 20.8% against 16.6% for the ones it kept.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttheart_sender.game import tsum as T


class FakeNet:
    """Returns "not a tsum" for every crop whose centre pixel is blue."""

    def __init__(self):
        self.passes = 0

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        self.passes += 1
        n = len(self.blob)
        out = np.zeros((n, 2), np.float32)
        for i in range(n):
            # channel 0 is B after the NCHW transpose; the fixture paints
            # board crops blue and tsum crops red.
            blue = float(self.blob[i, 0].mean()) > float(self.blob[i, 2].mean())
            out[i] = (8.0, -8.0) if blue else (-8.0, 8.0)
        return out


def model(floor=T.RejectModel.FLOOR):
    m = T.RejectModel.__new__(T.RejectModel)
    m.net = FakeNet()
    m.size = 96
    m.mean = np.zeros(3, np.float32)
    m.std = np.ones(3, np.float32)
    m.floor = floor
    m.seen = m.dropped = 0
    m.failed = ""
    return m


def board(kinds):
    """A frame and its detections. `kinds` is one of "board" or "tsum"."""
    bgr = np.zeros((600, 600, 3), np.uint8)
    tsums = []
    for i, kind in enumerate(kinds):
        x, y = 80.0 + 70 * (i % 6), 80.0 + 70 * (i // 6)
        colour = (200, 20, 20) if kind == "board" else (20, 20, 200)  # BGR
        bgr[int(y) - 30:int(y) + 30, int(x) - 30:int(x) + 30] = colour
        tsums.append(T.Tsum(x=x, y=y, r=20.0, kind=i, colour=(0, 0, 0)))
    return bgr, tsums, 25.0


def test_board_detections_are_dropped():
    m = model()
    bgr, tsums, radius = board(["tsum", "board", "tsum", "board"])
    kept = m.keep(bgr, tsums, radius)
    assert [t.kind for t in kept] == [0, 2]
    assert m.dropped == 2
    assert m.seen == 4


def test_the_surviving_order_is_unchanged():
    """Chain building indexes into this list, so a filter that reorders it
    silently repoints every index that was computed from it."""
    m = model()
    bgr, tsums, radius = board(["tsum", "board", "tsum", "tsum", "board"])
    kept = m.keep(bgr, tsums, radius)
    assert [t.kind for t in kept] == [0, 2, 3]


def test_a_crop_the_frame_edge_clips_is_kept():
    """The model has never seen one. Refusing a tsum at the board's rim because
    the picture of it is awkward would quietly shrink the playable board -- the
    mistake CharacterModel made by being asked about crops its training set had
    excluded."""
    m = model()
    bgr = np.zeros((600, 600, 3), np.uint8)
    edge = T.Tsum(x=3.0, y=3.0, r=20.0, kind=99, colour=(0, 0, 0))
    kept = m.keep(bgr, [edge], 25.0)
    assert [t.kind for t in kept] == [99]


def test_an_empty_board_is_returned_untouched():
    m = model()
    assert m.keep(np.zeros((10, 10, 3), np.uint8), [], 25.0) == []


def test_a_failed_net_keeps_every_detection():
    """Off for the rest of the round rather than taking the round with it, and
    never by silently dropping the whole board."""
    import cv2

    class Broken(FakeNet):
        def forward(self):
            raise cv2.error("boom")

    m = model()
    m.net = Broken()
    bgr, tsums, radius = board(["tsum", "board", "tsum"])
    kept = m.keep(bgr, tsums, radius)
    assert len(kept) == 3
    assert m.net is None
    assert "boom" in m.failed
    assert "SWITCHED OFF" in m.summary()


def test_a_wrong_shaped_answer_keeps_every_detection():
    class Wrong(FakeNet):
        def forward(self):
            return np.zeros((3, 2), np.float32)

    m = model()
    m.net = Wrong()
    bgr, tsums, radius = board(["tsum"] * 5)
    assert len(m.keep(bgr, tsums, radius)) == 5
    assert m.net is None


def test_the_summary_reports_what_it_dropped():
    m = model()
    bgr, tsums, radius = board(["tsum", "board", "board", "tsum"])
    m.keep(bgr, tsums, radius)
    assert "dropped 2/4" in m.summary()


def test_the_filter_runs_before_identity():
    """Order matters: a board detection is not a tsum of any character, and
    grouping it first only produces a group with board in it."""
    import re
    from pathlib import Path
    src = Path(T.__file__).read_text(encoding="utf-8")
    body = src[src.index("def play_loop"):]
    assert body.index("rejector.keep(") < body.index("characters.apply(")


def test_it_ships_off():
    import yaml
    from pathlib import Path
    flow = yaml.safe_load(
        (Path(T.__file__).resolve().parents[2] / "flows" / "play.yaml")
        .read_text(encoding="utf-8"))
    assert flow["vars"]["reject_model"] == ""
    assert flow["vars"]["reject_floor"] == T.RejectModel.FLOOR


def test_both_models_cut_the_same_picture():
    """RejectModel and CharacterModel are trained on crops from one extractor,
    so a change to the window, the 64px round trip or the edge refusal must
    reach both or neither."""
    bgr = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    t = T.Tsum(x=200.0, y=200.0, r=20.0, kind=0, colour=(0, 0, 0))
    shared = T._character_crop(bgr, t, 25.0, 96)
    cm = T.CharacterModel.__new__(T.CharacterModel)
    cm.size = 96
    assert np.array_equal(cm._crop(bgr, t, 25.0), shared)
