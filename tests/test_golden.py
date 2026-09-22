"""The frozen test set, and the two ways it can silently stop meaning anything.

A golden set exists so that "did this model get better?" has an answer. It only
does that while two things hold: the sessions never change, and no model is
trained on them. Both failures are silent -- a model trained on part of the
golden set simply scores well, and a re-frozen set simply reports different
numbers -- so both are guarded rather than documented.

The motivating case is on record: a retrained classifier read **97.9% against
the shipped 94.9%** and was, on a common set, exactly as good. The difference
was the split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import golden  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def crop(root: Path, cls: str, sess: str, i: int, vis: float = 0.64):
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sess}_{i:04d}_00_v{vis}.png").write_bytes(b"")


def test_crops_are_counted_per_session_and_class(tmp_path):
    crop(tmp_path, "Pascal", "s1", 1)
    crop(tmp_path, "Pascal", "s1", 2)
    crop(tmp_path, "Grim", "s2", 1)
    got = golden.crops_by_session(tmp_path)
    assert got["s1"]["Pascal"] == 2 and got["s2"]["Grim"] == 1


def test_the_non_character_folders_are_not_frozen(tmp_path):
    crop(tmp_path, "board", "s1", 1)
    crop(tmp_path, "score", "s1", 2)
    crop(tmp_path, "Pascal", "s2", 1)
    assert set(golden.crops_by_session(tmp_path)) == {"s2"}


def test_every_class_with_two_sessions_gets_one_frozen(tmp_path):
    """A class with no golden crops can never be scored, which is how Cleo
    came to be measured at 0% having never been taught."""
    for s in ("s1", "s2"):
        crop(tmp_path, "Pascal", s, 1)
        crop(tmp_path, "Grim", s, 2)
    chosen, _, covered = golden.choose(golden.crops_by_session(tmp_path), 0.5, 0)
    assert covered["Pascal"] and covered["Grim"]


def test_a_class_in_only_one_session_is_left_trainable(tmp_path):
    """Freezing its only session would make the class impossible to LEARN.
    Unscored is bad; untrainable is worse, and it poisons the classes its
    crops land on -- Cleo's fell on Beast and took its precision to 16%."""
    crop(tmp_path, "Pascal", "s1", 1)
    crop(tmp_path, "Pascal", "s2", 2)
    crop(tmp_path, "Cleo", "s3", 1)
    chosen, _, covered = golden.choose(golden.crops_by_session(tmp_path), 0.3, 0)
    assert "s3" not in chosen, "Cleo's only session must stay trainable"


def test_freezing_is_deterministic_for_a_seed(tmp_path):
    for s in ("s1", "s2", "s3", "s4"):
        crop(tmp_path, "Pascal", s, 1)
        crop(tmp_path, "Grim", s, 2)
    by = golden.crops_by_session(tmp_path)
    assert golden.choose(by, 0.5, 7)[0] == golden.choose(by, 0.5, 7)[0]


def test_the_share_is_honoured_once_coverage_is_satisfied(tmp_path):
    for i in range(10):
        crop(tmp_path, "Pascal", "s%d" % i, 1)
        crop(tmp_path, "Grim", "s%d" % i, 2)
    chosen, _, _ = golden.choose(golden.crops_by_session(tmp_path), 0.5, 0)
    assert len(chosen) == 5


# ------------------------------------------------------------- the shipped set
def test_the_shipped_golden_set_exists_and_is_frozen():
    body = json.loads((ROOT / "models" / "golden.json").read_text(encoding="utf-8"))
    assert body["sessions"] and body.get("frozen")
    assert len(body["sessions"]) == len(set(body["sessions"]))


def test_the_trainer_refuses_to_train_on_the_golden_set():
    """The guard, read out of the source rather than assumed: `classify.py`
    must subtract the golden sessions before it splits anything."""
    src = (ROOT / "scripts" / "classify.py").read_text(encoding="utf-8")
    assert "set(sess.tolist()) - golden" in src
    assert "s not in golden" in src


def test_the_scorer_refuses_a_model_that_trained_on_golden_sessions():
    src = (ROOT / "scripts" / "recog_eval.py").read_text(encoding="utf-8")
    assert "REFUSED" in src and "held & trained" in src


def test_a_candidate_trained_after_the_freeze_records_the_set_it_owes():
    """The artifact carries the golden list, so a later comparison cannot
    quietly be run against a different one."""
    meta = ROOT / "models" / "character_candidate.json"
    if not meta.exists():
        return
    body = json.loads(meta.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "models" / "golden.json").read_text(encoding="utf-8"))
    assert body.get("golden") == frozen["sessions"]
    assert not set(body["train_sessions"]) & set(frozen["sessions"])
    assert body.get("crop_profile")
