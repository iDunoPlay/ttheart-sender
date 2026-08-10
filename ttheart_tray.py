"""Windowed entry point -- this is what the .exe is built from.

Launching it with no arguments drops straight into the system tray. Every CLI
subcommand still works (``ttheart-sender.exe flows``), but a windowed build has
no console to print to, so those are only really useful with ``--console``
builds or when redirecting output.
"""

from __future__ import annotations

import sys

from ttheart_sender.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["tray"]))
