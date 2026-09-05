"""The one-file build carries its version in the file name.

A folder build ships as a directory, so the version can live in the directory
or in the binary and either is findable. A `--onefile` build is a single .exe
that gets copied to another machine on its own -- and two of them side by side
in a downloads folder are indistinguishable.

That is not a hypothetical tidiness point. Rounds were collected on a build
that predated a flow fix, came back missing the one measurement every rule is
priced against, and nothing about the file said which build had produced them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build  # noqa: E402

from ttheart_sender.version import __version__  # noqa: E402


def test_the_onefile_name_carries_the_version():
    assert build.ONEFILE_NAME == f"{build.NAME}_v{__version__}"
    assert __version__ in build.ONEFILE_NAME


def test_the_version_is_read_from_the_one_place_it_is_written():
    """Not duplicated into build.py, where it would drift silently."""
    assert build._version() == __version__


def test_the_version_is_read_as_text_not_imported():
    """`version.py` is import-free so setuptools can read the literal without
    executing it. Importing the package here would drag its dependencies into
    a build script that only needs one string."""
    src = Path(build.__file__).read_text(encoding="utf-8")
    assert "read_text" in src
    assert not re.search(r"^\s*from ttheart_sender", src, re.M), (
        "build.py must not import the package to learn its version")


def test_the_folder_build_name_is_unchanged():
    """Only the one-file name gains the version -- the folder build is shipped
    as a directory and renaming it would break `_internal\\` expectations."""
    assert build.NAME == "ttheart-sender"
