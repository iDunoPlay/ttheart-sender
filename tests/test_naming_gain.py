"""The grouping comparison: does naming change what the round plays on?

The two rules under test differ in one way that decides the whole result --
`replace` lets a name pull a tsum OUT of the cluster its partners are still in,
`merge` never does -- so the tests are about exactly that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "naming_gain", HERE / "scripts" / "naming_gain.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["naming_gain"] = mod
    spec.loader.exec_module(mod)
    return mod


gain = pytest.importorskip("cv2") and _load()


# -- score ----------------------------------------------------------------
def test_recall_and_precision_of_a_grouping():
    # kinds:   0 0 0 1     head 0, the game marked 1 and 3.
    recall, precision, offered = gain.score([0, 0, 0, 1], head=0,
                                            truth={1, 3})

    # Of the two marked partners, only #1 shares the head's cluster.
    assert recall == 0.5
    # Of the two tsums sharing it, only #1 was marked.
    assert precision == 0.5
    assert offered == 2


def test_a_grouping_that_offers_nobody_is_not_scored_as_wrong():
    """No partners is a missed chain, not a false one. Recall carries it."""
    recall, precision, offered = gain.score([0, 1, 1], head=0, truth={1})

    assert (recall, precision, offered) == (0.0, 1.0, 0)


# -- the merge rule -------------------------------------------------------
def test_a_name_joins_two_clusters_the_colours_split():
    """The case the model is FOR: one character read as two colours."""
    kinds = [0, 1, 2]
    # The model called #0 and #1 the same character.
    out = gain.merged(kinds, {0: 7, 1: 7})

    assert out[0] == out[1]
    assert out[2] != out[0]


def test_a_name_never_splits_a_cluster_the_colours_joined():
    """The failure of the shipped rule, in one assertion. The head here is
    unnamed and its partner is named; they must stay together."""
    kinds = [0, 0, 0]
    out = gain.merged(kinds, {1: 7})

    assert out[0] == out[1] == out[2]


def test_naming_two_members_of_one_cluster_changes_nothing():
    kinds = [0, 0, 1]
    assert gain.merged(kinds, {0: 7, 1: 7}) == gain.merged(kinds, {})


def test_merging_is_transitive_across_a_chain_of_names():
    # 0-1 share a name, 1-2 share a cluster: all three end up together.
    out = gain.merged([0, 1, 1], {0: 7, 1: 7})

    assert out[0] == out[1] == out[2]


def test_an_unnamed_board_is_left_exactly_as_the_clusters_had_it():
    kinds = [0, 1, 2, 1]
    assert len(set(gain.merged(kinds, {}))) == len(set(kinds))
