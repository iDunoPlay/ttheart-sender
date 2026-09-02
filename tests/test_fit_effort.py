"""The per-frame colour fit, and the one thing that must not change silently.

`fit_effort` exists because the fit is k-means with a seed: measured over 60
collected boards, two reads of ONE frame at level 1 agree on 73% of the tsums
they find and their counts differ by nine. Level 3 reads 91% and three. See
`FIT_EFFORT` and `flows/play.yaml`.

What is pinned here is not that stability -- that needs real boards -- but the
two things a refactor could quietly undo: the default staying at the settings
every measurement in `docs/DATASET-FINDINGS.md` was taken under, and the level
actually reaching the fit.
"""

import numpy as np
import pytest

from ttheart_sender.game import tsum as T


def _board(seed: int = 0) -> np.ndarray:
    """A small synthetic frame with a few distinct colour regions."""
    rng = np.random.default_rng(seed)
    img = np.zeros((90, 120, 3), np.uint8)
    for i, colour in enumerate([(40, 60, 200), (200, 120, 40), (60, 190, 90)]):
        img[:, i * 40:(i + 1) * 40] = colour
    return np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)


def test_level_one_is_what_every_measurement_was_taken_under():
    """4 restarts, 20 iterations, epsilon 1.0. Changing it re-prices the corpus."""
    assert T.FIT_EFFORT[1] == (4, 20, 1.0)
    assert T.FIT_EFFORT[2] == (8, 40, 0.5)
    assert T.FIT_EFFORT[3] == (16, 60, 0.25)


def test_default_is_level_one():
    import inspect
    assert inspect.signature(T._quantise).parameters["effort"].default == 1
    assert inspect.signature(T.detect).parameters["fit_effort"].default == 1


def test_effort_reaches_the_fit(monkeypatch):
    """detect must hand the level down, not quietly fit at its own."""
    seen = {}
    real = T.cv2.kmeans

    def spy(data, k, best, criteria, attempts, flags):
        seen["criteria"], seen["attempts"] = criteria, attempts
        return real(data, k, best, criteria, attempts, flags)

    monkeypatch.setattr(T.cv2, "kmeans", spy)
    T.detect(_board(), k=3, radius=8.0, fit_effort=3)
    assert seen["attempts"] == 16
    assert seen["criteria"][1:] == (60, 0.25)


def test_a_reused_palette_never_refits(monkeypatch):
    """The whole reason the cost is affordable: a cached palette skips the fit."""
    def boom(*_a, **_kw):
        raise AssertionError("fitted when a palette was supplied")

    monkeypatch.setattr(T.cv2, "kmeans", boom)
    centres = np.array([[40, 128, 128], [200, 128, 128], [120, 128, 128]], np.float32)
    T.detect(_board(), k=3, radius=8.0, palette=centres, fit_effort=3)


@pytest.mark.parametrize("effort", [0, 9, "x"])
def test_an_unknown_level_falls_back_to_today(effort, monkeypatch):
    """A bad value must play at level 1, not crash a round mid-board."""
    seen = {}
    real = T.cv2.kmeans

    def spy(data, k, best, criteria, attempts, flags):
        seen["attempts"] = attempts
        return real(data, k, best, criteria, attempts, flags)

    monkeypatch.setattr(T.cv2, "kmeans", spy)
    T._quantise(_board(), 3, None, effort=effort)
    assert seen["attempts"] == 4
