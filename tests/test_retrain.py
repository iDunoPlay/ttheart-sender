"""The one-command retrain, and the cut that silently undid a labelling session.

`--min-class` defaults to 20, so an evening spent adding Dory (9 crops),
Maleficent (6) and Eeyore (5) trained none of them. The only sign was one line
in the middle of several hundred. So the survey runs FIRST, says what is about
to be dropped, and says how many more crops each one needs.

And `--min-class` has to mean something when it is lowered: `classify.py` used
to abort on any class under 20 regardless, which made the flag a lie.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import retrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def crop(root: Path, cls: str, n: int):
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / ("s_%04d_00_v0.64.png" % i)).write_bytes(b"")


def test_the_survey_counts_every_class(tmp_path):
    crop(tmp_path, "Pascal", 30)
    crop(tmp_path, "Dory", 9)
    assert retrain.counts(tmp_path) == {"Pascal": 30, "Dory": 9}


def test_the_survey_names_what_will_be_dropped_and_by_how_much(capsys, tmp_path):
    crop(tmp_path, "Pascal", 30)
    crop(tmp_path, "Dory", 9)
    retrain.survey(tmp_path, 20)
    out = capsys.readouterr().out
    assert "THIN" in out
    assert "Dory" in out and "+11" in out, "say how many more crops it needs"
    # Only the THIN block -- Pascal appears later under the too-few-rounds
    # warning, correctly: this fixture puts all 30 of its crops in one session.
    assert "Pascal" not in out.split("THIN")[1].split("MORE CROPS")[0]


def test_the_survey_says_nothing_alarming_when_nothing_is_dropped(capsys, tmp_path):
    crop(tmp_path, "Pascal", 30)
    retrain.survey(tmp_path, 20)
    assert "DROPPED" not in capsys.readouterr().out


def test_the_non_character_folders_are_not_counted_as_trainable(tmp_path):
    crop(tmp_path, "board", 700)
    crop(tmp_path, "Pascal", 30)
    assert "board" in retrain.counts(tmp_path)
    assert "board" in retrain.NOT_CHARACTERS


def test_lowering_min_class_no_longer_aborts_the_trainer():
    """It used to abort on any class under 20 whatever the flag said, which
    made `--min-class` a lie: the flag exists so a newly labelled class can be
    trained before it reaches 20 crops."""
    src = (ROOT / "scripts" / "classify.py").read_text(encoding="utf-8")
    assert "if thin and args.min_class >= 20:" in src
    assert "were kept because" in src


def test_the_trainer_still_stops_when_the_filter_leaks():
    """A class under 20 getting through while the cut is at 20 or above is a
    bug, not a choice, and still stops."""
    src = (ROOT / "scripts" / "classify.py").read_text(encoding="utf-8")
    i = src.index("if thin and args.min_class >= 20:")
    assert "return 1" in src[i:i + 300]


def test_the_same_crop_in_two_classes_is_caught(tmp_path):
    """Moving a crop between folders is the supported way to fix a label.
    COPYING one is the mistake, and it is silent -- the picture then trains as
    two different characters at once."""
    for cls in ("Beast", "BeastIdle"):
        d = tmp_path / cls
        d.mkdir(parents=True)
        (d / "s_0001_00_v0.64.png").write_bytes(b"")
    got = retrain.duplicates(tmp_path)
    assert "s_0001_00_v0.64.png" in got
    assert sorted(got["s_0001_00_v0.64.png"]) == ["Beast", "BeastIdle"]


def test_a_crop_moved_rather_than_copied_is_not_flagged(tmp_path):
    (tmp_path / "Beast").mkdir(parents=True)
    (tmp_path / "BeastIdle").mkdir(parents=True)
    (tmp_path / "Beast" / "s_0001_00_v0.64.png").write_bytes(b"")
    assert retrain.duplicates(tmp_path) == {}


def test_negatives_are_not_listed_as_homework(capsys, tmp_path):
    """`score`, `board` and `junk` train the REJECT model, where "thin" means
    nothing. Listing them beside the characters made `score` read as 8 crops
    of work it is not."""
    crop(tmp_path, "Pascal", 30)
    crop(tmp_path, "Dory", 9)
    crop(tmp_path, "score", 12)
    retrain.survey(tmp_path, 20)
    out = capsys.readouterr().out
    thin = out.split("THIN")[1].split("MORE CROPS")[0]
    assert "Dory" in thin and "score" not in thin
    assert "REJECT model instead" in out and "score 12" in out


def test_an_empty_folder_is_called_out_separately(capsys, tmp_path):
    """A name typed before its crops arrived. It trains nothing and is not
    20 crops of work either."""
    crop(tmp_path, "Pascal", 30)
    (tmp_path / "Scrump").mkdir(parents=True)
    retrain.survey(tmp_path, 20)
    out = capsys.readouterr().out
    assert "EMPTY folder" in out and "Scrump" in out


def test_the_survey_totals_the_work_left(capsys, tmp_path):
    crop(tmp_path, "Pascal", 30)
    crop(tmp_path, "Dory", 9)
    crop(tmp_path, "Eeyore", 8)
    retrain.survey(tmp_path, 20)
    assert "23 MORE CROPS" in capsys.readouterr().out


def test_sessions_are_counted_per_class(tmp_path):
    d = tmp_path / "Dory"
    d.mkdir(parents=True)
    for s, i in (("sA", 0), ("sA", 1), ("sB", 0)):
        (d / ("%s_0001_%02d_v0.64.png" % (s, i))).write_bytes(b"")
    assert retrain.sessions_per_class(tmp_path)["Dory"] == {"sA", "sB"}


def test_a_class_from_one_round_is_flagged_even_with_plenty_of_crops(capsys, tmp_path):
    """20 crops from one board are 20 near-duplicates. The split is by
    session, so such a class is trained and never scored, or scored having
    never been trained -- which is how Cleo read 0% recall."""
    d = tmp_path / "Flounder"
    d.mkdir(parents=True)
    for i in range(37):
        (d / ("sOnly_0001_%02d_v0.64.png" % i)).write_bytes(b"")
    retrain.survey(tmp_path, 20)
    out = capsys.readouterr().out
    assert "TOO FEW ROUNDS" in out
    assert "Flounder" in out and "one round only" in out
    assert "DIFFERENT board" in out


def test_a_class_spread_over_rounds_is_not_flagged(capsys, tmp_path):
    d = tmp_path / "Pascal"
    d.mkdir(parents=True)
    for i in range(30):
        (d / ("s%02d_0001_00_v0.64.png" % i)).write_bytes(b"")
    retrain.survey(tmp_path, 20)
    assert "TOO FEW ROUNDS" not in capsys.readouterr().out
