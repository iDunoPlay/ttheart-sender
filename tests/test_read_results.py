"""Reading the round's score off the results screen.

What is pinned here is not the OCR accuracy -- `read_results.py --verify`
checks that against hand-read frames, and it reads 15/15 -- but the two rules
that took several attempts to get right, and one refusal that matters more
than either.

The refusal first: a misread score is worse than no score, because it enters
the corpus looking like data and there is nothing downstream to catch it. So a
glyph that matches nothing must produce None, never a nearest guess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from read_results import (  # noqa: E402
    GLYPH, SCORE_FROM_BONUS, glyphs, read_number,
)

import cv2  # noqa: E402


def _row(text, *, rule=False, size=1.0):
    """A number drawn the way the panel draws one: light ink on a dark strip."""
    img = np.zeros((60, 320, 3), np.uint8)
    img[:] = (40, 30, 20)
    cv2.putText(img, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, size,
                (255, 255, 255), 2, cv2.LINE_AA)
    if rule:
        # The bonus row has a dotted rule under it, and the crop catches its
        # left end -- clear of the digits, which sit at the right. That is
        # what broke measuring the digit height from the region's total ink
        # extent: the extent ran from the top of the digits down to the rule,
        # so every digit looked short and the number was thrown away.
        for x in range(200, 315, 8):
            cv2.line(img, (x, 57), (x + 3, 57), (200, 200, 200), 1)
    return img


REGION = (0, 60, 0, 320)


def test_commas_are_dropped():
    assert len(glyphs(_row("135,430"), REGION)) == 6


def test_a_one_is_not_mistaken_for_a_comma():
    """The case that defeats every width rule.

    `1` is about as narrow as `,`, so "1,186" cut on width either kept the
    separator or split the `8` in half -- it read as 7 glyphs for a 4-digit
    number. Height separates them cleanly: a digit is full height, a comma is
    a third of it on the baseline.
    """
    assert len(glyphs(_row("1,186"), REGION)) == 4


def test_a_rule_under_the_row_does_not_swallow_the_digits():
    """Measuring height from the region's total ink extent counted from the
    top of the digits to the dotted rule, so every digit looked short and the
    whole number was thrown away. The reference is the median run instead."""
    assert len(glyphs(_row("188,063", rule=True), REGION)) == 6


def test_an_empty_region_reads_nothing_rather_than_raising():
    assert glyphs(np.zeros((10, 10, 3), np.uint8), REGION) == []


def test_a_glyph_matching_nothing_is_refused_not_guessed():
    """The property that keeps a bad reading out of the corpus."""
    templates = {"7": np.zeros(GLYPH[::-1], np.uint8)}      # nothing like a digit
    value, worst = read_number(_row("135,430"), REGION, templates, max_dist=0.05)

    assert value is None, "an unmatched glyph must not fall back to the nearest"
    assert worst > 0.05


def test_no_templates_at_all_reads_nothing():
    assert read_number(_row("700"), REGION, {}, max_dist=0.5)[0] is None


def test_the_derived_score_matches_the_rounds_it_was_fitted_on():
    """`score = bonus * k`, checked against what the screens actually showed.

    Five rounds, hand-read. If this drifts, the equipped tsum's bonus rate
    changed and every derived score since is wrong by a constant factor.
    """
    seen = [(135430, 837141), (80195, 495716), (70435, 435384),
            (188063, 1162483), (123542, 763657)]
    for bonus, score in seen:
        assert abs(round(bonus * SCORE_FROM_BONUS) - score) <= 2, (
            f"{bonus} -> {round(bonus * SCORE_FROM_BONUS)}, screen said {score}")


@pytest.mark.parametrize("text,count", [("700", 3), ("250", 3), ("449", 3),
                                        ("27,374,010", 8)])
def test_digit_counts(text, count):
    assert len(glyphs(_row(text), REGION)) == count
