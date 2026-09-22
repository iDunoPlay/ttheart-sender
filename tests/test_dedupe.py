"""Resolving a crop that is filed under two classes.

Keeping both teaches the classifier a contradiction; deleting one at random
leaves a label that is wrong half the time. So there is a rule for the case
that has one, and the case that does not gives up both copies rather than
guessing.

Nothing is deleted -- losing copies move to `crops/duplicates/`, so a wrong
call costs a move back rather than a re-labelling session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dedupe  # noqa: E402


def test_a_negative_beats_a_character():
    """The asymmetry that makes this decidable: `reject_net.py` re-checks its
    negatives against the game and hands back anything the game marked or
    cleared. A wrong junk call self-corrects; a wrong character label does
    not."""
    plan = dedupe.resolve({"a.png": ["CheshireCat", "junk"]}, Path("."))
    name, keep, drop, why = plan[0]
    assert keep == "junk" and drop == ["CheshireCat"]
    assert "re-checked against the game" in why


def test_two_character_labels_give_up_both():
    plan = dedupe.resolve({"a.png": ["22", "Sulley"]}, Path("."))
    _, keep, drop, why = plan[0]
    assert keep is None
    assert drop == ["22", "Sulley"]
    assert "no rule" in why


def test_two_negatives_keep_one():
    plan = dedupe.resolve({"a.png": ["board", "junk"]}, Path("."))
    _, keep, drop, _ = plan[0]
    assert keep == "board" and drop == ["junk"]


def test_board_and_score_count_as_negatives_too():
    plan = dedupe.resolve({"a.png": ["coin", "score"]}, Path("."))
    _, keep, drop, _ = plan[0]
    assert keep == "score" and drop == ["coin"]


def test_nothing_is_deleted_only_moved():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "dedupe.py"
           ).read_text(encoding="utf-8")
    assert "shutil.move" in src
    assert ".unlink(" not in src and "os.remove" not in src


def test_it_reports_before_it_moves_anything():
    """`--apply` is opt-in: a bare run says what would happen."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "dedupe.py"
           ).read_text(encoding="utf-8")
    assert 'if args.apply:' in src
    assert 'WOULD move' in src


def test_the_corpus_has_no_crop_in_two_classes():
    """The state this leaves behind, so a later tool change cannot reintroduce
    contradictory labels unnoticed."""
    from retrain import duplicates
    root = Path(__file__).resolve().parents[1] / "crops" / "labelled"
    if not root.is_dir():
        return
    assert duplicates(root) == {}
