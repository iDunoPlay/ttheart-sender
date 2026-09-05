"""Pick the chain the game will accept, not the longest one.

The bot has always sorted candidates by ``(is_base, len)``. Over 7,352
proposed members the game accepts 97.9% of first members and 19.1% of
sevenths, and once it refuses one it refuses 86% of what follows -- so
longest-first optimises the end of the chain that does not survive.

These pin the mechanics of the replacement, not its value. Its value is
unmeasured until a round says so, which is why it ships off.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ttheart_sender.game import tsum as T


class FakeNet:
    """Scores a member by how close it is to the one before it.

    A tight hop is likely to be accepted and a long one is not, which is the
    real shape of the signal. Deliberately NOT a function of `position`: see
    `test_expected_accepted_cannot_truncate` for why position alone could
    never make a shorter chain win.
    """

    def __init__(self, features):
        self.features = features
        self.batches = []

    def setInput(self, blob):
        self.blob = np.asarray(blob)
        self.batches.append(int(self.blob.shape[0]))

    def forward(self):
        d = self.blob[:, self.features.index("dist_prev_r")]
        return (4.0 - 2.0 * d).reshape(-1, 1).astype(np.float32)


FEATURES = ["position", "dist_prev_r", "dist_head_r", "abs_dx_r", "dy_r",
            "turn_cos", "blockers_prev", "blockers_head", "density",
            "visible", "board_n", "lab_prev", "lab_head", "same_kind_head",
            "chain_len", "path_so_far_r", "fever", "is_base"]


def model(bonus=0.0):
    m = T.ChainModel.__new__(T.ChainModel)
    m.net = FakeNet(FEATURES)
    m.features = list(FEATURES)
    m.mu = np.zeros(len(FEATURES), np.float32)
    m.sd = np.ones(len(FEATURES), np.float32)
    m.bonus = bonus
    m.seen = m.moved = 0
    m.failed = ""
    return m


def board():
    """Two runs: 0-5 stretched far apart, 6-11 packed tight."""
    far = [T.Tsum(x=float(60 + 90 * i), y=100.0, r=20.0, kind=1,
                  colour=(40, 80, 120)) for i in range(6)]
    near = [T.Tsum(x=float(60 + 22 * i), y=300.0, r=20.0, kind=1,
                   colour=(40, 80, 120)) for i in range(6)]
    return far + near


def chain(nodes, is_base=False):
    return T.Chain(1, (40, 80, 120), list(nodes), is_base)


def test_a_shorter_chain_can_beat_a_longer_one():
    """The whole point: a tight 3 outranks a stretched 5.

    Not because it is shorter -- because each of its members is individually
    likelier to be taken, and three near-certainties beat five long shots.
    """
    ts = board()
    ranked = model().rank([chain([0, 1, 2, 3, 4]), chain([6, 7, 8])],
                          ts, 20.0, False)
    assert list(ranked[0].nodes) == [6, 7, 8], (
        "expected accepted members must beat raw length")


def test_expected_accepted_cannot_truncate():
    """A property of the objective, and it is the right one.

    `expected accepted = sum of probabilities`, and a probability is never
    negative, so ADDING a member can only raise a chain's score. This rule
    can therefore never prefer a prefix of a chain to the chain itself -- it
    re-picks, it does not shorten.

    That is not a limitation, it is agreement with the measurement. Cutting
    the played chain at its first predicted refusal was tested on 597
    held-out presses with the game's REAL answers and lost 0.427 accepted
    members a press, because the game skips a refused member and keeps
    linking, so a long-shot member is nearly free to attempt.
    """
    ts = board()
    m = model()
    ranked = m.rank([chain([6, 7, 8]), chain([6, 7, 8, 9, 10])], ts, 20.0, False)
    assert list(ranked[0].nodes) == [6, 7, 8, 9, 10], (
        "a superset of a chain always scores at least as well")


def test_the_base_chain_still_leads():
    """Clearing the equipped character charges the skill, which is a rule
    about the ROUND rather than about this press."""
    ts = board()
    ranked = model().rank(
        [chain([0, 1, 2]), chain([6, 7, 8, 9, 10, 11], is_base=True)],
        ts, 20.0, False)
    assert ranked[0].is_base, "is_base outranks expected accepted"


def test_a_length_bonus_can_buy_length_back():
    ts = board()
    long_, short = chain([0, 1, 2, 3, 4]), chain([6, 7, 8])
    assert list(model(0.0).rank([long_, short], ts, 20.0, False)[0].nodes) == [6, 7, 8]
    assert list(model(9.0).rank([long_, short], ts, 20.0, False)[0].nodes) == [0, 1, 2, 3, 4]


def test_one_chain_is_returned_untouched():
    ts = board()
    only = [chain([0, 1, 2])]
    assert model().rank(only, ts, 20.0, False) is only


def test_a_failed_net_keeps_the_original_order():
    """A ranker that quietly stops ranking looks exactly like one that was
    never on."""
    import cv2
    m = model()

    class Boom(FakeNet):
        def forward(self):
            raise cv2.error("boom")

    m.net = Boom(FEATURES)
    chains = [chain([0, 1, 2, 3, 4, 5]), chain([6, 7, 8])]
    assert m.rank(chains, board(), 20.0, False) == chains
    assert m.net is None, "switched off for the rest of the round"
    assert "boom" in m.failed


def test_it_counts_how_often_it_changed_the_pick():
    ts = board()
    m = model()
    m.rank([chain([0, 1, 2, 3, 4, 5]), chain([6, 7, 8])], ts, 20.0, False)  # moves
    m.rank([chain([6, 7, 8, 9]), chain([0, 1, 2])], ts, 20.0, False)        # already right
    assert m.seen == 2
    assert m.moved == 1
    assert "1/2" in m.summary()


def test_the_feature_order_comes_from_the_model_not_from_here():
    """A model trained on different columns must not be fed this one's.

    The rows are built against the sidecar's own list, so a reordering or a
    dropped column changes what is sent rather than silently mislabelling it.
    """
    ts = board()
    pts = np.array([[t.x, t.y] for t in ts], float)
    lab = T._cluster_lab(ts)
    want = ["visible", "position"]
    rows = T._chain_rows(ts, pts, lab, 20.0, [0, 1, 2], False, False, want)
    assert rows.shape == (2, 2)
    assert rows[0, 1] == 1.0 and rows[1, 1] == 2.0, "column 1 is `position`"


def test_the_shipped_model_matches_the_feature_builder():
    """The sidecar and the runtime must agree about the columns.

    `models/chain.json` is written by `scripts/rank_replay.py --export` from
    the same list `proposal_dataset` defines. If the play loop cannot build
    every column the model asks for, it would feed zeros under a real name --
    which is worse than failing, because the round would look normal.
    """
    meta = json.loads(open("models/chain.json", encoding="utf-8").read())
    ts = board()
    pts = np.array([[t.x, t.y] for t in ts], float)
    rows = T._chain_rows(ts, pts, T._cluster_lab(ts), 20.0, [0, 1, 2],
                         False, False, meta["features"])
    assert rows.shape == (2, len(meta["features"]))
    assert np.isfinite(rows).all(), "every column the model wants is real"
    assert "prev_accepted" not in meta["features"], (
        "the previous member's LABEL is not known at decision time")


def test_it_ships_off():
    """Never played. The board filter is the reminder that a good offline
    number is not a good round."""
    import yaml
    from ttheart_sender.config import Config
    flow = yaml.safe_load(
        (Config().flows_dir / "play.yaml").read_text(encoding="utf-8"))
    assert flow["vars"]["chain_model"] == ""
    assert flow["vars"]["chain_bonus"] == 0.0
