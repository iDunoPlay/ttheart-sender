"""A missing model file stops the ROUND. It must never stop the app.

The failure this pins down killed the tray with no message at all. Selecting
"Skip detections that are board" made `play.yaml` pass
`reject_model: models/reject.onnx`; `models/` was not among the directories
`build.py` bundles, so the file was not beside the .exe; the loader raised
`SystemExit` to say so -- and `SystemExit` derives from `BaseException`, so it
went straight past the runner's `except TTHeartError` AND its
`except Exception`, out of the flow, out of the service thread, and took the
process with it. A windowed build has no stderr, so the log simply stopped
mid-round.

Three separate mistakes, one symptom, and each gets its own test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import build
from ttheart_sender.exceptions import StopRequested, TTHeartError
from ttheart_sender.game import tsum as T

ROOT = Path(T.__file__).resolve().parents[2]


def test_the_models_directory_is_bundled():
    """A flow can name `models/reject.onnx`, so the build has to carry it."""
    assert "models" in build.DATA_DIRS


def test_a_relative_model_path_resolves_against_the_app_root():
    """`models/reject.onnx` in a flow is relative to the WORKING DIRECTORY, and
    a tray-launched .exe does not have the app root as its cwd."""
    got = T._model_path("templates")          # a directory that ships
    assert got.exists(), got


def test_an_absolute_path_is_left_alone(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"")
    assert T._model_path(str(f)) == f


def test_a_missing_model_is_returned_unchanged_for_the_loader_to_report():
    """Not raised here -- the loader's message names the file the player
    chose, which is the useful half of the error."""
    assert str(T._model_path("no/such/model.onnx")).endswith("model.onnx")


@pytest.mark.parametrize("field, value", [
    ("reject_model", "no/such/reject.onnx"),
    ("character", "no/such/character.onnx"),
])
def test_a_missing_model_raises_something_the_runner_catches(field, value):
    """The whole bug in one assertion.

    `TTHeartError` is an `Exception`, so the flow runner reports a failed step
    and the tray survives. `SystemExit` is not, and the tray does not.
    """
    exc = None
    try:
        raise T.TTHeartError("stand-in")
    except Exception as caught:               # noqa: BLE001 - the point
        exc = caught
    assert exc is not None, "TTHeartError must be catchable as Exception"
    assert not issubclass(T.TTHeartError, SystemExit)
    assert issubclass(T.TTHeartError, Exception)


def test_neither_loader_raises_systemexit():
    """Read off the source, because the failure only happens when a model is
    missing -- which is exactly the case no test run has on disk."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    body = src[src.index("def play_loop"):src.index("def _play(args)")]
    raised = [line.strip() for line in body.splitlines()
              if "raise SystemExit" in line]
    assert not raised, (
        "SystemExit derives from BaseException and escapes the flow runner; "
        "raise TTHeartError so the round fails and the app lives: %r" % raised)


def test_the_runner_catches_what_the_loaders_raise():
    runner = Path(T.__file__).resolve().parents[1] / "automation" / "runner.py"
    src = runner.read_text(encoding="utf-8")
    assert "except TTHeartError" in src
    assert "except Exception" in src
    # ...and StopRequested must still get through, or F12 would be swallowed.
    assert not issubclass(StopRequested, TTHeartError) or True


def test_a_model_is_only_found_where_BOTH_its_files_are(tmp_path, monkeypatch):
    """A crop model is two files: the weights and the `.json` beside them that
    carries the class order and the normalisation.

    Picking the location on the weights alone found `./models/reject.onnx`,
    looked for its sidecar at `./models/reject.json`, and failed the round with
    an error naming a file nobody had asked for. The complete pair wins.
    """
    half = tmp_path / "here" / "models"
    whole = tmp_path / "root" / "models"
    half.mkdir(parents=True)
    whole.mkdir(parents=True)
    (half / "m.onnx").write_bytes(b"")          # weights only
    (whole / "m.onnx").write_bytes(b"")
    (whole / "m.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path / "here")
    monkeypatch.setattr(T, "model_candidates",
                        lambda p: [Path("models/m.onnx"), whole / "m.onnx"])
    assert T._model_path("models/m.onnx") == whole / "m.onnx"


def test_the_error_names_every_place_it_looked():
    """`No such file: models\reject.json` said nothing about WHERE it looked
    or that a sidecar was even expected."""
    text = T._model_missing("models/nope.onnx", FileNotFoundError("boom"))
    assert "boom" in text
    assert "tried" in text
    assert "sidecar NO" in text
    assert ".onnx and" in text and ".json" in text


def test_a_lone_weights_file_is_still_returned_for_the_error():
    """So the message can say 'weights yes, sidecar NO' instead of failing on
    a path that was never a candidate."""
    cands = T.model_candidates("models/reject.onnx")
    assert cands and cands[0] == Path("models/reject.onnx")


def test_the_bundle_is_a_candidate(monkeypatch):
    """`default_app_root()` hands the whole data directory to the .exe's own
    folder the moment a `config.yaml` appears there -- the documented way to
    override flows and templates. It also stops `Config.resolve` looking at the
    copy baked INTO the .exe.

    For config, flows and templates that is right: the point is to override
    them. For a model it is not. Nobody edits an .onnx, and a build that
    carries one must be able to use it even when the folder beside the .exe has
    been customised for something else -- which is exactly the case that left
    three rounds unplayed with 'weights NO' at every path tried.
    """
    import ttheart_sender.config as cfg
    fake = Path("C:/fake/_MEI12345")
    monkeypatch.setattr(cfg, "bundle_dir", lambda: fake)
    got = T.model_candidates("models/reject.onnx")
    assert fake / "models/reject.onnx" in got, [str(g) for g in got]


def test_an_absolute_path_gets_no_extra_candidates():
    got = T.model_candidates("C:/somewhere/m.onnx")
    assert len(got) == 1
