"""The recognition reader: a class nobody taught must not read as a failure.

`scripts/recog_eval.py` scores the shipped ONNX on the sessions baked into
`models/character.json`. The trap it exists to avoid is arithmetic rather than
modelling, and it bit on the first run: **Cleo has 41 held-out crops and zero
training crops**, so it scored 0% at something it was never shown -- and its
crops landed on Beast, dragging Beast's PRECISION to 16% and making Beast read
as the worst class the model was taught. It is not; its recall is 83%.

These pin both halves: an untaught class is named as untaught, and the
taught-only headline is recomputed on the taught subset rather than merely
dropping the untaught rows from the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recog_eval  # noqa: E402


def crop(root: Path, cls: str, sess: str, i: int, vis: float = 0.64):
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sess}_{i:04d}_00_v{vis}.png").write_bytes(b"")


def test_a_class_with_no_held_out_crop_is_not_averaged_in():
    names = ["A", "B", "C"]
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    rows, macro = recog_eval.per_class(y, pred, names)
    assert macro == 1.0, "C has no crops and must not pull the mean down"
    assert [r[0] for r in rows if r[1]] == ["A", "B"]


def test_a_class_it_always_gets_wrong_scores_zero_not_nan():
    names = ["A", "B"]
    y = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 1, 1])
    rows, macro = recog_eval.per_class(y, pred, names)
    assert dict((r[0], r[4]) for r in rows)["A"] == 0.0
    assert macro < 1.0


def test_train_support_counts_only_the_sessions_the_model_saw(tmp_path):
    crop(tmp_path, "Cleo", "s_late", 1)
    crop(tmp_path, "Cleo", "s_late", 2)
    crop(tmp_path, "Beast", "s_early", 1)
    shown = recog_eval.train_support(tmp_path, {"s_early"})
    assert shown["Beast"] == 1
    assert not shown["Cleo"], "Cleo's only session was held out -- never taught"


def test_the_non_character_folders_are_not_scored_as_identities(tmp_path):
    crop(tmp_path, "board", "s_early", 1)
    crop(tmp_path, "score", "s_early", 1)
    crop(tmp_path, "Beast", "s_early", 1)
    assert set(recog_eval.train_support(tmp_path, {"s_early"})) == {"Beast"}
    paths, labels, _, _ = recog_eval.load(tmp_path, None)
    assert labels == ["Beast"]


def test_load_can_restrict_to_the_held_out_sessions(tmp_path):
    crop(tmp_path, "Beast", "s_early", 1)
    crop(tmp_path, "Beast", "s_late", 2)
    _, labels, sess, _ = recog_eval.load(tmp_path, {"s_late"})
    assert labels == ["Beast"] and sess == ["s_late"]


def test_an_untaught_class_wrecks_another_classes_precision():
    """The exact shape of the first run, in miniature.

    Cleo was never trained, so every Cleo crop is called something else. Judged
    over everything, Beast looks broken. Judged over the crops of classes the
    model was actually taught, it does not -- and the second is the honest
    reading, because the first is measuring a gap in the label set.
    """
    names = ["Beast", "Cleo"]
    y = np.array([0, 0, 1, 1, 1, 1])          # 2 Beast, 4 Cleo
    pred = np.array([0, 0, 0, 0, 0, 0])       # everything called Beast

    rows, _ = recog_eval.per_class(y, pred, names)
    beast = dict((r[0], r) for r in rows)["Beast"]
    assert beast[3] == 1.0, "recall is perfect"
    assert beast[2] < 0.5, "precision is destroyed by the untaught class"

    taught = np.array([n == 0 for n in y])
    trows, tmacro = recog_eval.per_class(y[taught], pred[taught], names)
    tbeast = dict((r[0], r) for r in trows)["Beast"]
    assert tbeast[2] == 1.0 and tbeast[4] == 1.0
    assert tmacro == 1.0
