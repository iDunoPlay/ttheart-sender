"""Emptying the two directories the app fills up on its own.

Both are pure disk operations with no window in them, so the panel buttons stay
thin and this can be tested without one.

The awkward part is the log directory: the running process holds ``ttheart.log``
open through a :class:`~logging.handlers.RotatingFileHandler`, and Windows will
not unlink a file that is open. Truncating it underneath the handler is worse
than it sounds -- the handler keeps its own file position, so the next record
lands at the old offset and pads everything before it with NUL bytes. So the
stream is closed first and left closed: :meth:`logging.FileHandler.emit` reopens
it on the next record, which puts the file back with nothing in it.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClearResult:
    """What one clear actually managed to remove."""

    removed: int = 0
    freed: int = 0
    failed: int = 0

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def describe(self) -> str:
        if not self.removed and not self.failed:
            return "Nothing to clear"
        text = f"Cleared {self.removed} file(s), {_size(self.freed)}"
        if self.failed:
            text += f" -- {self.failed} still in use"
        return text


def _size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _tally(path: Path) -> tuple:
    """(files, bytes) under ``path``, counting the file itself if it is one."""
    if path.is_file():
        try:
            return 1, path.stat().st_size
        except OSError:
            return 1, 0
    files = size = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                size += item.stat().st_size
            except OSError:
                pass
    return files, size


def empty_dir(directory: Path) -> ClearResult:
    """Remove everything *inside* ``directory``, keeping the directory itself.

    Keeping it matters: the log directory is created once at startup and the
    handler writes straight into it, and the dataset directory is what the
    collector's own free-space check looks at.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return ClearResult()

    removed = freed = failed = 0
    for item in sorted(directory.iterdir()):
        files, size = _tally(item)
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError as exc:
            # A file another process still holds open is the ordinary case
            # here, not a bug -- report it rather than failing the whole clear.
            log.warning("Could not remove %s: %s", item, exc)
            failed += files
            continue
        removed += files
        freed += size
    return ClearResult(removed=removed, freed=freed, failed=failed)


def _file_handlers() -> List[logging.FileHandler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]


def clear_logs(config) -> ClearResult:
    """Empty the log directory, including this session's own log file.

    The handlers are held shut for the whole sweep rather than closed and
    released one at a time, so a record logged by another thread cannot recreate
    the file half way through and leave it behind.
    """
    handlers = _file_handlers()
    for handler in handlers:
        handler.acquire()
    try:
        for handler in handlers:
            if handler.stream is not None:
                handler.stream.close()
                handler.stream = None
        result = empty_dir(config.log_dir)
    finally:
        for handler in handlers:
            handler.release()

    log.info("Clear logs: %s", result.describe())  # reopens the file
    return result


def clear_dataset(config) -> ClearResult:
    """Empty the dataset directory -- every collected session.

    ``README.txt`` goes with them; the collector writes a fresh one the next
    time it saves a sample, so there is nothing to preserve.
    """
    result = empty_dir(config.dataset_dir)
    log.info("Clear data collection: %s", result.describe())
    return result
