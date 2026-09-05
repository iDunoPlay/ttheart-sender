"""One net per process, not one per round.

The tray plays rounds back to back for hours. Building a `cv2.dnn` net per
round means a fresh native parse and a fresh set of layer buffers each time,
with the old ones released only if and when Python gets around to it -- and the
reported failure of that shape was two rounds fine and the third killing the
process with no traceback and no error box.

`CharacterModel` was given a cache for exactly that reason. `RejectModel` was
constructed per round from the day it shipped, and it had never been noticed
because it had never played a long session: it first ran on 2026-09-05, for 14
rounds. These pin both.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ttheart_sender.game import tsum


class _CountingNet:
    """A net that records how many times the module asked cv2 to parse one."""

    parses = 0

    def __init__(self):
        _CountingNet.parses += 1
        self._last = 0

    def setInput(self, blob):
        self._last = int(blob.shape[0])

    def forward(self):
        out = np.zeros((self._last, 2), np.float32)
        out[:, 1] = 20.0
        return out


@pytest.fixture
def onnx(tmp_path, monkeypatch):
    """A model file whose loader counts parses instead of reading ONNX."""
    path = tmp_path / "m.onnx"
    path.write_bytes(b"x")
    path.with_suffix(".json").write_text(
        json.dumps({"classes": ["not_a_tsum", "tsum"], "size": 96,
                    "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}),
        encoding="utf-8")
    _CountingNet.parses = 0
    tsum._CHARACTER_CACHE.clear()
    monkeypatch.setattr(tsum.cv2.dnn, "readNetFromONNX",
                        lambda _p: _CountingNet())
    yield path
    tsum._CHARACTER_CACHE.clear()


def test_the_board_filter_is_parsed_once_for_many_rounds(onnx):
    first = tsum.load_reject_model(onnx, 0.10)
    for _ in range(9):
        tsum.load_reject_model(onnx, 0.10)
    assert _CountingNet.parses == 1, (
        f"ten rounds must share one net, not build "
        f"{_CountingNet.parses}")
    assert tsum.load_reject_model(onnx, 0.10) is first


def test_the_character_model_is_parsed_once_for_many_rounds(onnx):
    tsum.load_character_model(onnx, 0.85)
    for _ in range(9):
        tsum.load_character_model(onnx, 0.85)
    assert _CountingNet.parses == 1


def test_a_different_floor_is_a_different_model(onnx):
    a = tsum.load_reject_model(onnx, 0.10)
    b = tsum.load_reject_model(onnx, 0.50)
    assert a is not b
    assert _CountingNet.parses == 2
    assert (a.floor, b.floor) == (0.10, 0.50)


def test_the_counters_reset_each_round_but_the_net_does_not(onnx):
    model = tsum.load_reject_model(onnx, 0.10)
    model.seen, model.dropped, model.failed = 40, 3, "something went wrong"
    again = tsum.load_reject_model(onnx, 0.10)
    assert again is model, "the net is per process"
    assert (again.seen, again.dropped, again.failed) == (0, 0, ""), \
        "the counters are per round"


def test_a_filter_that_switched_itself_off_is_rebuilt(onnx):
    """`net is None` means it failed mid-round and stopped filtering.

    Handed back as it is, the counter reset below would clear `failed` too and
    the rest of the process would run with the filter silently doing nothing --
    which is the one state this project refuses to ship.
    """
    dead = tsum.load_reject_model(onnx, 0.10)
    dead.net = None
    fresh = tsum.load_reject_model(onnx, 0.10)
    assert fresh.net is not None
    assert _CountingNet.parses == 2


def test_the_character_model_is_rebuilt_after_it_switches_itself_off(onnx):
    dead = tsum.load_character_model(onnx, 0.85)
    dead.net = None
    fresh = tsum.load_character_model(onnx, 0.85)
    assert fresh.net is not None
