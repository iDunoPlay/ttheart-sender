"""The one place the version number is written.

Everything else reads it from here: ``__init__`` re-exports it, ``main.py
--version`` prints it, the tray shows it, and ``pyproject.toml`` picks it up via
``[tool.setuptools.dynamic]`` so a release is a one-line edit rather than a hunt
through the tree for copies that have drifted apart.

Keep this module import-free -- setuptools reads the assignment below without
executing the file, which only works while it stays a plain literal.
"""

from __future__ import annotations

__version__ = "1.7.0"
