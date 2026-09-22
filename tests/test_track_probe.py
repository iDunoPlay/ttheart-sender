"""The tracker's matcher, and the band it reports in.

`scripts/track_probe.py` answers one question -- can the same physical tsum be
found again in the next frame -- and the whole answer rests on `match`. These
pin the two things it has to get right: it must refuse a match that is too far
to be the same tsum, and it must SAY when the runner-up was nearly as close,
because that is the case a tracker silently gets wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multiframe_eval  # noqa: E402
import track_probe  # noqa: E402


def test_a_tsum_that_barely_moved_is_matched():
    stored = np.array([[100.0, 100.0]])
    found = np.array([[101.0, 102.0]])
    pairs, ambiguous = track_probe.match(stored, found, tol=25.0)
    assert len(pairs) == 1 and not ambiguous
    assert pairs[0][2] < 3.0


def test_a_tsum_that_moved_too_far_is_not_matched():
    stored = np.array([[100.0, 100.0]])
    found = np.array([[200.0, 100.0]])
    pairs, _ = track_probe.match(stored, found, tol=25.0)
    assert pairs == [], "100px is four radii -- that is a different tsum"


def test_two_equally_close_candidates_are_reported_as_ambiguous():
    """The failure a tracker cannot see in its own output.

    Both candidates are 10px away. Something is returned either way, so a
    match rate alone would call this a success; only the ambiguity count says
    the answer was a coin flip.
    """
    stored = np.array([[100.0, 100.0]])
    found = np.array([[110.0, 100.0], [90.0, 100.0]])
    pairs, ambiguous = track_probe.match(stored, found, tol=25.0)
    assert len(pairs) == 1
    assert ambiguous == 1


def test_a_clear_winner_is_not_called_ambiguous():
    stored = np.array([[100.0, 100.0]])
    found = np.array([[101.0, 100.0], [124.0, 100.0]])
    pairs, ambiguous = track_probe.match(stored, found, tol=25.0)
    assert len(pairs) == 1 and ambiguous == 0


def test_an_empty_frame_matches_nothing_rather_than_raising():
    stored = np.array([[100.0, 100.0]])
    pairs, ambiguous = track_probe.match(stored, np.zeros((0, 2)), tol=25.0)
    assert pairs == [] and ambiguous == 0


def test_the_visibility_bands_do_not_overlap_or_leave_a_gap():
    assert multiframe_eval.band(0.55) == multiframe_eval.band(0.59)
    assert multiframe_eval.band(0.60) != multiframe_eval.band(0.59)
    assert multiframe_eval.band(0.70) != multiframe_eval.band(0.69)
    assert multiframe_eval.band(1.00) == multiframe_eval.band(0.70)
