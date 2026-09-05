"""`round.json` records the chains the round actually played.

The assignment `report.played = played` used to sit AFTER the `finally:` block
that closes the sample writer, and `close()` is what writes `round.json` from
the report. So every round record ever written said `played: 0` while the log
line beside it said 89 chains -- and mean chain length, chains per second, and
the throughput half of every score correlation were all taken against a zero.

The ordering is the whole test. A record written from a report that has not
been finished yet is worse than no record: it looks like data.
"""

from __future__ import annotations

import json
from pathlib import Path

import ttheart_sender.game.tsum as T


def test_played_is_set_before_the_samples_are_closed():
    src = Path(T.__file__).read_text(encoding="utf-8")
    body = src[src.index("def play_loop"):src.index("def _play(args)")]
    assign = body.index("report.played = played")
    close = body.index("samples.close(report)")
    assert assign < close, (
        "report.played must be assigned before close() writes round.json, or "
        "every round record says played: 0")


def test_the_assignment_is_inside_the_finally():
    """Not merely earlier -- inside the `finally`, so an aborted round records
    the chains it managed rather than a zero."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    body = src[src.index("def play_loop"):src.index("def _play(args)")]
    tail = body[body.rindex("    finally:"):]
    assert "report.played = played" in tail
    assert "samples.close(report)" in tail


def test_the_round_record_carries_played(tmp_path):
    from ttheart_sender.game.dataset import DatasetWriter

    class Report:
        played, dragged, cleared = 87, 404, 301
        settle_s, settles = 15.682, 114
        fever_frames, frames = 55, 116
        reason = "timeup appeared -- round over"

    w = DatasetWriter(tmp_path, per_round=5)
    assert w._open()
    w.close(Report())
    row = json.loads((w.dir / "round.json").read_text(encoding="utf-8"))
    assert row["played"] == 87
    assert row["dragged"] == 404
    assert row["cleared"] == 301


def test_a_round_that_played_nothing_still_records_zero(tmp_path):
    """Zero is a real answer -- a round that never found a chain. What the bug
    made indistinguishable from it was a round that played 89."""
    from ttheart_sender.game.dataset import DatasetWriter

    class Report:
        played, dragged, cleared = 0, 0, 0
        reason = "no board"

    w = DatasetWriter(tmp_path, per_round=5)
    assert w._open()
    w.close(Report())
    row = json.loads((w.dir / "round.json").read_text(encoding="utf-8"))
    assert row["played"] == 0
