"""Promotion, and the three ways a model gets into service wrongly.

Copying a `.onnx` over the shipped one is two commands and no thought, which is
the problem: the checks that make a new model trustworthy are the ones easiest
to skip when you are pleased with a number. These pin that the checks exist, and
that the tool cannot change anything on a dry run -- the first test run of
`promote.py` swapped a shipped model while merely being tried.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import promote  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts" / "promote.py").read_text(encoding="utf-8")


def test_a_candidate_no_better_than_what_is_in_service_is_refused():
    """Tolerance is zero on purpose: a candidate that is not at least as good
    has no argument for replacing something that works."""
    assert promote.TOLERANCE == 0.0
    assert "the candidate is worse on the golden set" in SRC


def test_a_candidate_that_trained_on_golden_sessions_is_refused():
    assert "trained on %d golden session(s)" in SRC


def test_an_unknown_crop_profile_is_refused():
    """A model trained on padded crops and served unpadded ones does not
    error. It just reads as a worse model."""
    assert "crop_rules.profile(meta.get(\"crop_profile\"))" in SRC


def test_a_dry_run_returns_before_anything_is_copied():
    body = SRC[SRC.index("def main("):]
    dry = body.index("if args.dry_run:")
    first_copy = body.index("shutil.copy2")
    assert dry < first_copy, "the dry-run guard must come first"


def test_the_model_in_service_is_backed_up_before_it_is_replaced():
    assert '".prev" + suffix' in SRC
    body = SRC[SRC.index("def main("):]
    backup = body.index('".prev" + suffix')
    overwrite = body.rindex("shutil.copy2")
    assert backup < overwrite, "back up first, overwrite second"


def test_forcing_past_a_check_says_which_one():
    assert "FORCED past" in SRC


def test_the_scorer_will_not_score_a_model_on_sessions_it_trained_on():
    """`score` returns None rather than a number, so a leak can never be
    reported as a result."""
    assert "if sessions & trained:" in SRC
    assert "return None, len(sessions & trained), meta" in SRC
