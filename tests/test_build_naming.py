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

import pytest
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


def test_the_models_directory_ships():
    """A flow can name `models/reject.onnx`. A build without it made the tray
    die the moment a round started."""
    assert "models" in build.DATA_DIRS


def test_editable_copies_are_replaced_not_merged():
    """Copies beside the .exe WIN over the bundle, so a stale one silently
    replaces every flow in the build -- and one did, for a month. Three
    experiments were selected, played, and came back as baseline rounds
    because the app was reading August's flows."""
    src = Path(build.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _copy_editable_data"):src.index("def _report")]
    assert "rmtree" in body, (
        "copytree(dirs_exist_ok=True) leaves files the source no longer has")
    code = [l for l in body.splitlines()
            if "dirs_exist_ok" in l and not l.lstrip().startswith("#")
            and '"""' not in l and "`" not in l]
    assert not code, code


def test_an_existing_editable_folder_is_always_refreshed():
    """Even for --onefile without --with-data. Emitting the folder is opt-in;
    leaving a stale one behind is not."""
    src = Path(build.__file__).read_text(encoding="utf-8")
    assert "(target / DATA_DIRS[0]).exists()" in src


def test_only_runtime_model_files_ship():
    """`.pt` files are torch checkpoints kept so a model can be retrained.
    Torch is not a runtime dependency at all, and the .exe is copied to another
    machine by hand -- 12MB of dead weight in that file is 12MB of transfer."""
    assert set(build.MODEL_SUFFIXES) == {".onnx", ".json"}
    src = Path(build.__file__).read_text(encoding="utf-8")
    assert "MODEL_SUFFIXES" in src.split("def build(")[1].split("def _copy")[0]


def test_the_shipped_bundle_carries_the_board_filter():
    """Both halves of it. A model is two files and the app needs both."""
    models = Path(build.ROOT) / "models"
    if not (models / "reject.onnx").exists():
        pytest.skip("no reject model built yet")
    for suffix in build.MODEL_SUFFIXES:
        assert (models / f"reject{suffix}").exists(), suffix
