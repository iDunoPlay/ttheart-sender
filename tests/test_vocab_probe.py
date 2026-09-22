"""The per-round vocabulary: a round holds five characters, so cap the names.

The rule comes from the game, not from the data: every round is the equipped
tsum plus four random others. What the data decides is whether telling the
model helps, and the tests here are about the mechanics of applying it.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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


vocab = pytest.importorskip("cv2") and _load("vocab_probe")


def test_the_vocabulary_is_the_most_voted_names():
    votes = Counter({"Beast": 40, "Dory": 30, "Sulley": 12,
                     "Rex": 2, "Genie": 1})

    allowed = {n for n, _ in votes.most_common(3)}

    assert allowed == {"Beast", "Dory", "Sulley"}


def test_a_name_outside_the_vocabulary_keeps_its_colour_cluster():
    """Not renamed to something else, and not deleted -- handed back to the
    grouping it already had, which is the hybrid rule the round plays."""
    import ttheart_sender.game.tsum as T

    class FakeModel:
        classes = ["Beast", "Dory", "Genie"]

    before = [3, 3, 7]
    # Only #0 is inside the vocabulary; #1 was called Genie and refused.
    out = vocab._apply(before, {0: "Beast"}, FakeModel(), "replace",
                       _load("naming_gain"))

    assert out[0] == T.CHARACTER_KIND + 0
    assert out[1] == 3, "a refused tsum keeps the cluster it had"
    assert out[2] == 7


def test_the_merge_rule_only_ever_joins():
    """Same guarantee as `naming_gain --rule merge`: a name may pull two
    clusters together and may never split one."""
    gain = _load("naming_gain")

    class FakeModel:
        classes = ["Beast", "Dory"]

    # #0 and #2 are in different clusters and share a name.
    out = vocab._apply([1, 1, 5], {0: "Beast", 2: "Beast"},
                       FakeModel(), "merge", gain)

    assert out[0] == out[2]
    assert out[0] == out[1], "the unnamed neighbour must not be left behind"


def test_an_empty_vocabulary_means_no_limit():
    """`--vocab 0` is the baseline arm: whatever the model liked, unfiltered."""
    votes = Counter({"Beast": 3, "Dory": 1})
    allowed = {n for n, _ in votes.most_common(0)} if 0 else set(votes)

    assert allowed == {"Beast", "Dory"}
