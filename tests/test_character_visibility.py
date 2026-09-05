"""The character model is not asked about a tsum it cannot see.

The labelled crops were written by `scripts/crops.py` at `v >= 0.55` and every
one of the 3,791 of them is above that line. The median detection on a real
board shows 0.41. So most of the board is, to this model, a kind of picture it
has never seen one example of -- and measured against the game's own marks it
agrees with the confirmed character 74% of the time above the line and 0-28%
below it, against a 12.5% chance rate.

The trap is that it stays CONFIDENT down there: 0.75-0.88 mean softmax the
whole way down. `--character-confidence` therefore does not catch this, and no
setting of it can, which is why the floor is a separate rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from pathlib import Path

from ttheart_sender.game import tsum as T


class FakeNet:
    """A net that names everything class 0 at full confidence -- the failure
    mode under test is exactly a model that is sure about a crop it cannot
    read, so the stand-in has to be sure about everything."""

    def __init__(self):
        self.passes = 0

    def setInput(self, blob):
        self.batch = len(blob)

    def forward(self):
        self.passes += 1
        out = np.zeros((self.batch, 3), np.float32)
        out[:, 0] = 50.0
        return out


def model(min_visible=T.CHARACTER_MIN_VISIBLE):
    m = T.CharacterModel.__new__(T.CharacterModel)
    m.net = FakeNet()
    m.classes = ["A", "B", "C"]
    m.size = 96
    m.mean = np.zeros(3, np.float32)
    m.std = np.ones(3, np.float32)
    m.confidence = 0.85
    m.min_visible = min_visible
    m.named = m.seen = m.buried = 0
    m.failed = ""
    return m


def board(visibles, radius=25.0):
    """One frame and its tsums, laid out so no crop touches the frame edge."""
    bgr = np.full((900, 900, 3), 120, np.uint8)
    tsums = [T.Tsum(x=60.0 + 60 * (i % 13), y=60.0 + 60 * (i // 13),
                    r=radius * v, kind=0, colour=(0, 0, 0))
             for i, v in enumerate(visibles)]
    return bgr, tsums, radius


def test_a_buried_tsum_is_never_offered_to_the_model():
    m = model()
    bgr, tsums, radius = board([0.30, 0.41, 0.90])
    usable, prob = m.probabilities(bgr, tsums, radius)
    assert usable == [2], "only the tsum above the floor should be scored"
    assert len(prob) == 1
    assert m.buried == 2


def test_the_floor_can_be_switched_off():
    m = model(min_visible=0.0)
    bgr, tsums, radius = board([0.30, 0.41, 0.90])
    usable, _ = m.probabilities(bgr, tsums, radius)
    assert usable == [0, 1, 2]
    assert m.buried == 0


def test_a_buried_tsum_keeps_its_colour_cluster():
    """Not renamed to a character id -- k-means agrees with the game 37% of
    the time down there and the model manages 0-28%, so the worse answer is
    the one that arrives looking certain."""
    m = model()
    bgr, tsums, radius = board([0.30, 0.90])
    before = tsums[0].kind
    m.apply(bgr, tsums, radius)
    assert tsums[0].kind == before
    assert tsums[1].kind == T.CHARACTER_KIND + 0


def test_the_default_floor_is_the_training_sets_own_rule():
    assert T.CHARACTER_MIN_VISIBLE == 0.55, (
        "crops.py wrote the training set at v >= 0.55; if that changes, this "
        "must change with it or the model is asked about unseen pictures")


def test_buried_crops_never_reach_the_net_at_all():
    """Refused before the crop is cut, not after its answer is discarded.

    Counted in inference passes rather than crops, because the batch is padded
    to a fixed CHARACTER_BATCH either way -- a crop that is batched and then
    thrown away still costs a slot. A board of 130 detections with 10 visible
    ones is a single padded pass if the floor holds, and three if it does not.
    """
    vis = [0.30] * 120 + [0.95] * 10
    m, m_off = model(), model(min_visible=0.0)
    for mm in (m, m_off):
        bgr, tsums, radius = board(vis)
        mm.probabilities(bgr, tsums, radius)
    assert m.net.passes == 1
    assert m_off.net.passes == 3
    assert m.buried == 120


def test_the_summary_says_how_much_of_the_board_was_never_asked():
    m = model()
    bgr, tsums, radius = board([0.30, 0.41, 0.90])
    m.apply(bgr, tsums, radius)
    text = m.summary()
    assert "never asked" in text
    assert "0.55 visible" in text


def test_seen_still_counts_the_whole_board():
    """`seen` is the board, `buried` is the part of it that was skipped. If
    `seen` quietly dropped to the crops that were scored, the named-share in
    the round line would climb by refusing to look."""
    m = model()
    bgr, tsums, radius = board([0.30, 0.41, 0.90])
    m.apply(bgr, tsums, radius)
    assert m.seen == 3
    assert m.named == 1


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_a_missing_radius_does_not_silence_the_model(radius):
    """Visibility is `r / radius`; with no radius there is no such ratio, and
    refusing the whole board would switch identity off on exactly the frames
    where the layout could not be read."""
    m = model()
    bgr, tsums, _ = board([0.30, 0.90])
    usable, _ = m.probabilities(bgr, tsums, radius)
    assert usable == [0, 1]
    assert m.buried == 0


def test_the_character_model_ships_off():
    """A new detection rule ships OFF and the revert is one line -- the house
    rule at the top of `flows/play.yaml`. This one has never been played, and
    the thirtieth round found score turns on FEVER share rather than on clear
    rate, so a better identity may not move the score at all."""
    flow = yaml.safe_load(_play_yaml().read_text(encoding="utf-8"))
    assert flow["vars"]["character"] == ""


def test_the_flow_pins_the_visibility_floor():
    """Not left to the CLI default. The floor is the only thing standing
    between the model and the 78% of the board where it is at the chance rate,
    so it belongs in the file that records every value and its revert."""
    flow = yaml.safe_load(_play_yaml().read_text(encoding="utf-8"))
    assert flow["vars"]["character_min_visible"] == T.CHARACTER_MIN_VISIBLE
    assert "${character_min_visible}" in _play_yaml().read_text(encoding="utf-8"), (
        "declared in vars but never passed to the play step is a value that "
        "documents a rule the bot is not running")


def _play_yaml():
    return Path(__file__).resolve().parents[1] / "flows" / "play.yaml"
