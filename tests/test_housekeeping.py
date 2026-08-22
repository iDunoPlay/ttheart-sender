"""Clearing the two directories the app fills up on its own."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

from ttheart_sender.housekeeping import ClearResult, clear_dataset, clear_logs, empty_dir


class FakeConfig:
    def __init__(self, log_dir: Path, dataset_dir: Path) -> None:
        self.log_dir = log_dir
        self.dataset_dir = dataset_dir


@pytest.fixture
def dirs(tmp_path):
    logs = tmp_path / "logs"
    data = tmp_path / "dataset"
    (logs / "debug").mkdir(parents=True)
    data.mkdir()
    return FakeConfig(logs, data)


@pytest.fixture
def file_handler(dirs):
    """A real rotating handler on the root logger, removed again afterwards."""
    root = logging.getLogger()
    saved = list(root.handlers)
    handler = logging.handlers.RotatingFileHandler(
        dirs.log_dir / "ttheart.log", maxBytes=10_000, backupCount=2, encoding="utf-8"
    )
    root.handlers = [handler]
    try:
        yield handler
    finally:
        handler.close()
        root.handlers = saved


def test_empty_dir_keeps_the_directory_itself(dirs):
    (dirs.dataset_dir / "session").mkdir()
    (dirs.dataset_dir / "session" / "a.jpg").write_bytes(b"x" * 10)
    (dirs.dataset_dir / "README.txt").write_text("note")

    result = empty_dir(dirs.dataset_dir)

    assert result.removed == 2, "the nested sample counts, the folder itself does not"
    assert result.freed == 14
    assert result.ok
    assert dirs.dataset_dir.is_dir(), "the collector's own space check needs it"
    assert list(dirs.dataset_dir.iterdir()) == []


def test_empty_dir_on_a_missing_directory_is_not_an_error(tmp_path):
    assert empty_dir(tmp_path / "never-existed") == ClearResult()


def test_clear_logs_removes_the_file_the_handler_is_holding_open(dirs, file_handler):
    """The whole point: Windows will not unlink a file that is still open."""
    log = logging.getLogger("test.housekeeping")
    log.warning("something worth %d bytes", 100)
    file_handler.flush()
    active = dirs.log_dir / "ttheart.log"
    assert active.stat().st_size > 0
    (dirs.log_dir / "ttheart.log.1").write_bytes(b"old" * 100)
    (dirs.log_dir / "debug" / "miss.png").write_bytes(b"png")

    result = clear_logs(dirs)

    assert result.ok, "nothing should have been left behind"
    assert result.removed == 3, "active log, one rotation, one debug capture"
    assert dirs.log_dir.is_dir()


def test_logging_keeps_working_after_a_clear(dirs, file_handler):
    logging.getLogger("test.housekeeping").warning("before")
    file_handler.flush()

    clear_logs(dirs)

    logging.getLogger("test.housekeeping").warning("after the clear")
    file_handler.flush()
    active = dirs.log_dir / "ttheart.log"
    assert active.exists(), "emit() must reopen the stream it was left without"
    text = active.read_text(encoding="utf-8")
    assert "after the clear" in text
    assert "before" not in text, "the old records must be gone, not just hidden"
    assert "\x00" not in text, "truncating under the handler would pad with NULs"


def test_clear_dataset_empties_every_session(dirs):
    for stamp in ("20260820_100255_19728", "20260821_090000_1234"):
        session = dirs.dataset_dir / stamp
        session.mkdir()
        (session / "samples.jsonl").write_text("{}")
    (dirs.dataset_dir / "README.txt").write_text("regenerated on next sample")

    result = clear_dataset(dirs)

    assert result.removed == 3
    assert result.ok
    assert list(dirs.dataset_dir.iterdir()) == []
