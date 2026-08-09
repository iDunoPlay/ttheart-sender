"""Convenience entry point so the app can be started with ``python main.py``."""

from __future__ import annotations

import sys

from ttheart_sender.cli import main

if __name__ == "__main__":
    sys.exit(main())
