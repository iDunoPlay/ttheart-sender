"""`--colour-aug`: stop the classifier reading a character's hue as its name.

The measured problem, from `scripts/colour_probe.py` against the shipped
model: 97.9% on held-out crops, **44.5% on the same crops in greyscale**, with
two dozen classes -- WhiteRabbit, Sulley, Pooh, Hades, Eeyore -- going from
100% to 0%. The geometry rows of the same probe stay at ~98%, which is what
says the collapse is the model rather than the harness: it is invariant to
exactly what `aug` perturbs and fragile to exactly what `aug` leaves alone.

What is tested here is not the robustness, which lives in the probe's table.
It is the properties the rule owes:

* it is inert unless asked for -- a model built without the flag must be bit
  for bit the model that was built before the flag existed;
* greying is really greying, in the BGR channel order the tensors are in, so
  the augmentation cannot smuggle in a colour change of its own;
* the jitter moves colour WITHOUT moving pixels, so it cannot be confused
  with the geometric augmentation whose job is a different invariance;
* the exported manifest records whether it was used, because a model trained
  with it and one trained without are not comparable on a clean test set.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _classify():
    spec = importlib.util.spec_from_file_location(
        "classify_mod", ROOT / "scripts" / "classify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(x):
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s


def _denorm(b):
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return b * s + m


def test_it_is_inert_when_it_does_nothing():
    """hue 0, saturation 1, no greying, no light: the batch comes back."""
    c = _classify()
    x = torch.rand(8, 3, 16, 16)
    out = c.colour_jitter(_norm(x), MEAN, STD, hue=0.0, sat=(1.0, 1.0),
                          grey_p=0.0, val=(1.0, 1.0), con=(1.0, 1.0))
    assert torch.allclose(_denorm(out), x, atol=1e-5)


def test_greying_uses_the_channel_order_the_tensors_are_in():
    """`cv2.imread` gives BGR and nothing swaps it.

    Greying with RGB weights would darken every red tsum and lighten every
    blue one -- a colour change wearing the costume of a colour removal, and
    the one bug that would make this augmentation actively misleading.
    """
    c = _classify()
    x = torch.rand(8, 3, 16, 16)
    out = _denorm(c.colour_jitter(_norm(x), MEAN, STD, hue=0.0, sat=(1.0, 1.0),
                                  grey_p=1.0, val=(1.0, 1.0), con=(1.0, 1.0)))
    # every channel equal ...
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5)
    assert torch.allclose(out[:, 1], out[:, 2], atol=1e-5)
    # ... at the BGR luminance, not the RGB one.
    w = torch.tensor(c._LUMA_BGR).view(1, 3, 1, 1)
    assert torch.allclose(out[:, 0], (x * w).sum(1), atol=1e-5)


def test_it_moves_colour_without_moving_pixels():
    """A hue rotation is not a geometric transform.

    Structure is what the model is supposed to fall back on, so the thing
    that removes colour must leave structure provably untouched: the greyed
    batch's luminance has to equal the original's, pixel for pixel.
    """
    c = _classify()
    x = torch.rand(4, 3, 12, 12)
    w = torch.tensor(c._LUMA_BGR).view(1, 3, 1, 1)
    before = (x * w).sum(1)
    out = _denorm(c.colour_jitter(_norm(x), MEAN, STD, hue=0.5, sat=(0.3, 1.6),
                                  grey_p=1.0, val=(1.0, 1.0), con=(1.0, 1.0)))
    assert torch.allclose((out * w).sum(1), before, atol=1e-5)


def test_hue_rotation_actually_rotates_hue():
    """Guard against a jitter that silently does nothing and reads as a win."""
    c = _classify()
    torch.manual_seed(0)
    x = torch.rand(32, 3, 12, 12)
    out = _denorm(c.colour_jitter(_norm(x), MEAN, STD, hue=0.5, sat=(1.0, 1.0),
                                  grey_p=0.0, val=(1.0, 1.0), con=(1.0, 1.0)))
    assert (out - x).abs().mean() > 0.05
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_the_flag_is_off_by_default_and_is_recorded():
    """Off by default, and the manifest says so.

    Both halves matter. Off, because every model in `models/` was trained
    without it and must stay reproducible. Recorded, because a model that saw
    jitter gives up ~1.7 points on clean crops and buys robustness only the
    probe can see -- comparing the two on accuracy alone would retire the
    better model.
    """
    src = (ROOT / "scripts" / "classify.py").read_text(encoding="utf-8")
    assert '"--colour-aug", action="store_true"' in src, (
        "--colour-aug must be a store_true flag, so omitting it is the "
        "historical behaviour")
    assert '"colour_aug": bool(args.colour_aug)' in src, (
        "the exported .json must record whether colour augmentation was used")
