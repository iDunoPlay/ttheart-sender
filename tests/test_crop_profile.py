"""One crop rule, shared by the trainer and the runtime, named by the model.

This is the test that would have caught the bug that cost this project 312
rounds of play. That one was a colour feature computed two different ways --
once in the trainer, once in the play loop -- which agreed on the day it was
written and drifted apart afterwards. A crop is the same shape of hazard and a
bigger one: it decides every pixel a model ever sees.

So: `scripts/crops.py::_cut` and `ttheart_sender.game.tsum::_character_crop`
must be the same function underneath, a model must record which profile it was
trained under, and a profile this build does not have must RAISE rather than
quietly fall back to the default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crops import _cut  # noqa: E402
from ttheart_sender.game import crop as C  # noqa: E402
from ttheart_sender.game import tsum as T  # noqa: E402


def frame(w=220, h=200, seed=0):
    return np.random.default_rng(seed).integers(
        0, 255, (h, w, 3), dtype=np.uint16).astype(np.uint8)


# ------------------------------------------------------------------- parity
def test_the_trainer_and_the_runtime_cut_the_same_crop():
    """The load-bearing test of this module."""
    img = frame()
    for x, y, r in ((100, 100, 25.0), (60, 140, 18.0), (150, 90, 30.0)):
        a = _cut(img, x, y, r)
        b = C.for_model(img, x, y, r, C.PLAIN.store, C.PLAIN)
        assert a is not None and b is not None
        assert np.array_equal(a, b), (x, y, r)


def test_the_runtime_upsamples_from_the_stored_size_not_the_frame():
    """Going straight to the model's size would hand it a sharper picture than
    the training set ever held."""
    img = frame()
    small = _cut(img, 100, 100, 25.0)
    served = C.for_model(img, 100, 100, 25.0, 96, C.PLAIN)
    assert small.shape[0] == 64 and served.shape[0] == 96
    import cv2
    expect = cv2.resize(small, (96, 96), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(served, expect)


# -------------------------------------------------------------- the profiles
def test_plain_refuses_a_crop_the_edge_would_clip():
    assert C.cut(frame(), 3, 100, 25.0, C.PLAIN) is None


def test_padded_returns_a_full_square_where_plain_refused():
    got = C.cut(frame(), 3, 100, 25.0, C.PADDED_V1)
    assert got is not None and got.shape[:2] == (64, 64)


def test_padding_changes_nothing_away_from_the_edge():
    """A profile that also moved the crops it did not need to would make every
    before/after comparison meaningless."""
    img = frame()
    assert np.array_equal(C.cut(img, 100, 100, 25.0, C.PLAIN),
                          C.cut(img, 100, 100, 25.0, C.PADDED_V1))


def test_a_centre_off_the_frame_is_refused_even_when_padding():
    """Padding fills in missing pixels around a tsum that is there. It is not
    a licence to invent one that is not."""
    assert C.cut(frame(), -80, 100, 25.0, C.PADDED_V1) is None


def test_the_border_is_replicated_not_filled_flat():
    """A flat border is a strong, learnable "this one is at the edge" cue, and
    the model would read that instead of the tsum."""
    assert C.PADDED_V1.border != 0  # cv2.BORDER_CONSTANT


# ------------------------------------------------------- naming and refusal
def test_a_model_that_names_no_profile_gets_the_historical_rule():
    assert C.profile(None) is C.PLAIN
    assert C.profile("") is C.PLAIN


def test_a_profile_this_build_does_not_have_raises(monkeypatch):
    """The failure this whole module exists to prevent: a model trained with
    padding, served without it, quietly reading as a worse model."""
    with pytest.raises(ValueError) as e:
        C.profile("padded_v99")
    assert "padded_v99" in str(e.value) and "plain" in str(e.value)


def test_every_profile_is_reachable_by_its_own_name():
    for name, prof in C.PROFILES.items():
        assert C.profile(name) is prof and prof.name == name


# ---------------------------------------------------------------- the models
def test_a_model_stub_without_init_still_crops(monkeypatch):
    """Several tests build a `CharacterModel` without running `__init__`. The
    profile is a class attribute so those keep working and read as PLAIN."""
    assert T.CharacterModel.profile is None
    assert T.RejectModel.profile is None
    t = T.Tsum(100.0, 100.0, 20.0, 0, (0, 0, 0))
    assert T._character_crop(frame(), t, 25.0, 96) is not None


def test_the_runtime_honours_the_profile_it_is_handed():
    img = frame()
    rim = T.Tsum(3.0, 100.0, 20.0, 0, (0, 0, 0))
    assert T._character_crop(img, rim, 25.0, 96) is None
    assert T._character_crop(img, rim, 25.0, 96, C.PADDED_V1) is not None


def test_the_shipped_model_declares_a_profile_this_build_knows():
    import json
    meta = json.loads(Path("models/character.json").read_text(encoding="utf-8"))
    C.profile(meta.get("crop_profile"))     # raises if it does not
