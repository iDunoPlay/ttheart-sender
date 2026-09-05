"""Every round writes a record that a score can be joined to.

The two halves of the ground truth have existed separately for a while and
nothing connected them: `flows/play.yaml` saves the results screen into the
debug directory named by wall-clock time, and `DatasetWriter` writes a round's
samples into its own session folder. Recovering what a round scored meant
matching timestamps by eye.

That is what `round.json` fixes -- the round's own totals and the times that
bound it, beside the samples it produced, so `scripts/rounds_table.py` can make
the join mechanically and report the gap it matched on.
"""

from __future__ import annotations

import json

from ttheart_sender.game.dataset import DatasetWriter


class _Report:
    played = 42
    dragged = 170
    cleared = 140
    checked = True
    settle_s = 12.5
    settles = 60
    settle_timeouts = 2
    reason = "timeup appeared"


def _writer(tmp_path):
    w = DatasetWriter(tmp_path, per_round=5)
    assert w._open(), "the session folder should open"
    return w


def test_a_round_leaves_a_record_beside_its_samples(tmp_path):
    w = _writer(tmp_path)
    w.close(_Report())

    record = w.dir / "round.json"
    assert record.exists(), "a round with no record cannot be joined to a score"
    row = json.loads(record.read_text(encoding="utf-8"))
    assert row["session"] == w.dir.name
    assert row["played"] == 42
    assert row["cleared"] == 140


def test_the_record_carries_the_times_that_bound_the_round(tmp_path):
    """The join is by time, so the times are the load-bearing part."""
    w = _writer(tmp_path)
    w.close(_Report())
    row = json.loads((w.dir / "round.json").read_text(encoding="utf-8"))

    assert row["ended"] >= row["started"]
    assert row["started"] > 1_700_000_000, "should be a unix timestamp"


def test_the_settle_cost_survives_into_the_record(tmp_path):
    """It is the number the throughput work is judged on."""
    w = _writer(tmp_path)
    w.close(_Report())
    row = json.loads((w.dir / "round.json").read_text(encoding="utf-8"))

    assert row["settles"] == 60
    assert row["settle_timeouts"] == 2
    assert abs(row["settle_s"] - 12.5) < 1e-6


def test_closing_without_a_report_still_writes_a_record(tmp_path):
    """An aborted round is still a round, and still had a score on screen."""
    w = _writer(tmp_path)
    w.close()
    row = json.loads((w.dir / "round.json").read_text(encoding="utf-8"))

    assert row["session"] == w.dir.name
    assert "played" not in row, "absent is better than a zero that reads as data"


def test_a_session_that_never_opened_writes_nothing(tmp_path):
    """`enabled` in config should cost nothing until something is collected."""
    w = DatasetWriter(tmp_path, per_round=5)
    w.close(_Report())

    assert not list(tmp_path.glob("*/round.json"))
