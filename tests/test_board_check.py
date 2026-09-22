"""Whole-board recognition validation, offline, on stored screenshots.

The tool exists because every other scorer starts partway down the pipeline:
`recog_eval.py` reads crops that were cut and saved long ago, so it can only
ever see the crops that succeeded. That is survivorship, and it hid the finding
this tool produced on its first run -- **40.7% of tsums above the visibility
floor are never offered to the model at all**, because the crop window runs off
the frame edge.

So the property that matters most here is the one that made that visible:
ASKED and NAMED are counted separately. A tsum refused by the crop and a tsum
the model declined are different failures with different fixes, and a single
"named %" cannot tell them apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import board_check  # noqa: E402
from ttheart_sender.game import tsum as T  # noqa: E402


def test_the_bands_start_at_zero_so_the_never_asked_are_counted():
    """The buried four fifths of a board must appear in the table, not be
    filtered out of it. A tool that only reported the crops it could score
    would report 99% and say nothing about coverage."""
    assert board_check.BANDS[0][0] == 0.0
    assert board_check.BANDS[0][1] == T.CHARACTER_MIN_VISIBLE


def test_the_bands_tile_the_range_without_gap_or_overlap():
    for i in range(len(board_check.BANDS) - 1):
        assert board_check.BANDS[i][1] == board_check.BANDS[i + 1][0]
    assert board_check.BANDS[-1][1] > 1.0


def test_band_of_places_every_visibility_somewhere():
    for v in (0.0, 0.3, 0.549, 0.55, 0.64, 0.65, 0.74, 0.75, 1.0, 1.4):
        assert board_check.band_of(v) in board_check.BANDS


def test_a_labelled_crop_is_keyed_back_to_its_exact_detection(tmp_path):
    """The filename is a primary key into `samples.jsonl`. Losing that mapping
    would silently score the wrong tsum against the wrong label."""
    d = tmp_path / "Pascal"
    d.mkdir(parents=True)
    (d / "20260906_101010_1_0007_03_v0.64.png").write_bytes(b"")
    got = board_check.labels(tmp_path)
    assert got == {("20260906_101010_1", 7, 3): "Pascal"}


def test_the_non_character_folders_are_not_treated_as_identities(tmp_path):
    for name in ("board", "score", "Pascal"):
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "s_0001_00_v0.60.png").write_bytes(b"")
    assert set(board_check.labels(tmp_path).values()) == {"Pascal"}


def test_rounds_that_already_ran_the_model_are_excluded(tmp_path):
    """`character` on means `kind` was overwritten in the saved row, and
    `--recolour`/`--kinds` renumber it. Neither board can be compared."""
    d = tmp_path / "s"
    d.mkdir(parents=True)
    rows = [
        {"index": 1, "tsums": [{"x": 1, "y": 1, "r": 1, "kind": 0}], "options": {}},
        {"index": 2, "tsums": [{"x": 1, "y": 1, "r": 1, "kind": 0}],
         "options": {"character": "models/character.onnx"}},
        {"index": 3, "tsums": [{"x": 1, "y": 1, "r": 1, "kind": 0}],
         "options": {"kinds": 4}},
    ]
    (d / "samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    for i in (1, 2, 3):
        (d / ("%04d_before.jpg" % i)).write_bytes(b"")
    got = board_check.samples(tmp_path, 0, 0, "", {})
    assert [r[1]["index"] for r in got] == [1]


def test_a_crop_the_frame_edge_would_clip_is_refused():
    """The refusal the tool measures. It is deliberate -- the training set
    excluded clipped crops -- but it costs 40.7% of the readable board, and
    that only becomes visible if ASKED is counted apart from NAMED."""
    img = np.zeros((200, 200, 3), np.uint8)
    middle = T.Tsum(100.0, 100.0, 20.0, 0, (0, 0, 0))
    edge = T.Tsum(3.0, 100.0, 20.0, 0, (0, 0, 0))
    assert T._character_crop(img, middle, 25.0, 96) is not None
    assert T._character_crop(img, edge, 25.0, 96) is None


def test_a_fully_visible_tsum_at_the_rim_is_the_one_that_gets_refused():
    """Not a corner case: the clearest tsums sit at the board's rim, which is
    why the refusal rate RISES with visibility (52% in the top band)."""
    img = np.zeros((200, 200, 3), np.uint8)
    clear_but_at_the_rim = T.Tsum(10.0, 100.0, 25.0, 0, (0, 0, 0))
    assert T._character_crop(img, clear_but_at_the_rim, 25.0, 96) is None
