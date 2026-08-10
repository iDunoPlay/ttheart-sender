"""Console + rotating file logging."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_TIME_FORMAT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    *,
    file_name: str = "ttheart.log",
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure the root logger. Idempotent -- existing handlers are replaced."""
    numeric_level = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)

    # A windowed (--noconsole) build has no stderr at all, and a StreamHandler
    # wrapped around None swallows every record into a logging-internal error.
    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _TIME_FORMAT))
        root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / file_name,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    # Third-party chatter we never care about.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("mss").setLevel(logging.WARNING)

    return logging.getLogger("ttheart_sender")
