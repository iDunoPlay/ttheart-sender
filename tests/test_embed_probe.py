"""Grouping the board by the learned face rather than by median colour.

The bug worth a permanent test is the one that made the first run meaningless:
a tsum left out of the clustering kept the id it already had, so five new piles
plus eight leftovers came to ELEVEN groups -- more fragmented than the rule
being replaced.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "scripts" / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ep = pytest.importorskip("cv2") and _load("embed_probe")


def test_every_tsum_lands_in_one_of_the_groups():
    """THE BUG. Leaving the unreadable ones in their old cluster turned five
    piles into eleven, which is worse than the rule being replaced."""
    feats = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 9.0], [9.1, 9.0],
                      [0.0, 0.0], [0.0, 0.0]], np.float32)
    ok = np.array([True, True, True, True, False, False])
    before = [3, 3, 7, 7, 5, 6]          # four different old ids
    lab = np.array([[0, 0, 0], [0, 0, 0], [90, 90, 90], [90, 90, 90],
                    [1, 1, 1], [89, 89, 89]], np.float32)

    out = ep.cluster(feats, ok, 2, before, lab)

    assert len(set(out)) == 2, "a refused tsum must not keep its own id"
    assert out[4] == out[0], "placed by colour with the pile it is nearest"
    assert out[5] == out[2]


def test_the_clustering_separates_what_the_feature_separates():
    feats = np.array([[0.0, 0.0], [0.1, 0.1], [9.0, 9.0], [9.1, 9.1]],
                     np.float32)
    ok = np.ones(4, bool)

    out = ep.cluster(feats, ok, 2, [0, 0, 0, 0])

    assert out[0] == out[1]
    assert out[2] == out[3]
    assert out[0] != out[2]


def test_too_few_tsums_to_split_is_left_alone():
    feats = np.zeros((2, 2), np.float32)

    assert ep.cluster(feats, np.ones(2, bool), 5, [1, 2]) == [1, 2]


def test_a_prototype_arm_only_uses_the_names_it_is_allowed():
    """The round's cast is five. Offering sixty is the guessing the player
    asked to remove."""
    protos = np.eye(4, dtype=np.float32)
    feats = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    ok = np.ones(2, bool)

    out = ep.assign(feats, ok, protos, np.array([0, 1]), [9, 9])

    # Neither tsum's true nearest (2 and 3) is on offer, so both land inside
    # the allowed set rather than inventing a sixth group.
    assert set(out) <= {0, 1}


def test_the_prototype_arm_places_the_unreadable_ones_too():
    protos = np.eye(2, dtype=np.float32)
    feats = np.array([[1, 0], [0, 1], [0, 0]], np.float32)
    ok = np.array([True, True, False])
    lab = np.array([[10, 10, 10], [90, 90, 90], [12, 12, 12]], np.float32)

    out = ep.assign(feats, ok, protos, np.array([0, 1]), [5, 5, 5], lab)

    assert out[2] == out[0], "nearest pile by face colour"
    assert len(set(out)) == 2
