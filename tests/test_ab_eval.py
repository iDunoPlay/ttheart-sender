"""The A/B reader: refuses to pool builds, and refuses to decide underpowered.

`scripts/ab_eval.py` exists because every comparison in this project so far has
been one build against another. These pin the two things that makes it worth
having: it will not silently mix builds, and it will not call a result on a
metric the corpus cannot resolve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ab_eval  # noqa: E402


def write_round(root: Path, name: str, *, arm: str, version: str,
                cleared: int, dragged: int = 360, fever: int = 50,
                frames: int = 110, rejected: int = 1):
    d = root / name
    d.mkdir(parents=True)
    (d / "round.json").write_text(json.dumps({
        "version": version, "session": name,
        "started": 1000.0, "ended": 1074.0,
        "cleared": cleared, "dragged": dragged, "rejected": rejected,
        "frames": frames, "fever_frames": fever,
        "ab": "reject_model", "ab_arm": arm,
    }), encoding="utf-8")
    (d / "samples.jsonl").write_text(json.dumps({
        "version": version,
        "options": {"ab": "reject_model",
                    "reject_model": "models/reject.onnx" if arm == "on" else ""},
    }) + "\n", encoding="utf-8")


def corpus(tmp_path, n=20, lift=0, version="1.11.6"):
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(n):
        arm = "on" if i % 2 == 0 else "off"
        write_round(root, f"2026090{i // 9}_{100000 + i}_1", arm=arm,
                    version=version,
                    cleared=250 + lift * (arm == "on") + (i % 5) * 4)
    return root


def test_it_reads_the_arms_off_the_round_record(tmp_path):
    rows = ab_eval.load(corpus(tmp_path), Path("nope.jsonl"))
    assert len(rows) == 20
    assert sum(r["arm"] == "on" for r in rows) == 10
    assert sum(r["arm"] == "off" for r in rows) == 10
    assert {r["option"] for r in rows} == {"reject_model"}


def test_an_older_round_recovers_its_arm_from_the_option_value(tmp_path):
    """A build that alternated before `ab_arm` existed must not be dropped."""
    root = tmp_path / "dataset"
    root.mkdir()
    write_round(root, "20260905_100000_1", arm="on", version="1.11.6", cleared=250)
    j = json.loads((root / "20260905_100000_1" / "round.json").read_text())
    del j["ab_arm"]
    (root / "20260905_100000_1" / "round.json").write_text(json.dumps(j),
                                                           encoding="utf-8")
    rows = ab_eval.load(root, Path("nope.jsonl"))
    assert rows[0]["arm"] == "on", "recovered from reject_model being set"


def test_it_will_not_pool_builds_unless_told(tmp_path, capsys):
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(12):
        write_round(root, f"20260905_1000{i:02d}_1", arm="on" if i % 2 == 0
                    else "off", version="1.11.6" if i < 8 else "1.11.5",
                    cleared=250 + i)
    ab_eval.main_argv = None
    rc = _run(["--dir", str(root), "--rounds", str(tmp_path / "none.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reading 1.11.6 only" in out
    assert "--pool-builds" in out
    assert "ON  n=4    OFF n=4" in out, out


def test_underpowered_is_need_more_data_not_a_verdict(tmp_path, capsys):
    root = corpus(tmp_path, n=6)
    rc = _run(["--dir", str(root), "--rounds", str(tmp_path / "none.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NEED MORE DATA" in out, out


def test_a_null_effect_at_power_is_rejected_not_shipped(tmp_path, capsys):
    root = corpus(tmp_path, n=200, lift=0)
    rc = _run(["--dir", str(root), "--rounds", str(tmp_path / "none.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "REJECT (no effect)" in out, out


def test_a_primary_that_fell_is_a_revert_even_when_a_guardrail_also_fell(
        tmp_path, capsys):
    """The wording bug the first real A/B caught.

    The guardrail vetoes used to be tested BEFORE the primary's direction, so a
    run where both went down printed "cleared improved, but FEVER share fell"
    -- on the run where cleared had dropped 18.4. The decision was one branch
    further down, under a sentence saying the opposite of the number beside it.
    """
    root = tmp_path / "dataset"
    root.mkdir()
    for i in range(120):
        arm = "on" if i % 2 == 0 else "off"
        # ON clears less AND fevers less: both the primary and a guardrail down.
        write_round(root, f"2026090{i // 40}_{100000 + i}_1", arm=arm,
                    version="1.11.6c",
                    cleared=(240 if arm == "on" else 275) + (i % 7),
                    fever=(30 if arm == "on" else 55) + (i % 5))
    rc = _run(["--dir", str(root), "--rounds", str(tmp_path / "none.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "REVERT" in out, out
    assert "improved" not in out, (
        "a primary that fell must never be reported as an improvement")


def test_no_ab_rounds_says_how_to_collect_some(tmp_path, capsys):
    root = tmp_path / "dataset"
    root.mkdir()
    rc = _run(["--dir", str(root), "--rounds", str(tmp_path / "none.jsonl")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No A/B rounds" in out
    assert "ab: reject_model" in out, "it has to say how to start one"


def _run(argv):
    old = sys.argv
    sys.argv = ["ab_eval.py"] + argv
    try:
        return ab_eval.main()
    finally:
        sys.argv = old
