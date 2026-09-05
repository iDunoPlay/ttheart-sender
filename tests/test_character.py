"""`--character`: name the tsum instead of trusting its colour cluster.

`kind` is a per-frame k-means id. It agrees with the game about a confirmed
partner 37% of the time, and restricting to the clearest tsums on the board
only reaches 46.6% -- face colour does not identify a Tsum Tsum character,
because many of them share one. A classifier over the face crop reads ~95% on
held-out sessions.

What is tested here is not the accuracy, which lives in the confusion matrix
`scripts/classify.py` prints. It is the two properties every opt-in rule in
this project owes, plus the one that makes this rule safe:

* it is inert unless asked for;
* a crop the model is unsure of KEEPS its colour cluster, so an unlabelled
  character plays exactly as it does today rather than becoming unchainable;
* a named tsum cannot collide with a colour cluster, or `adjacency` would
  join two characters that merely share a number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ttheart_sender.game import tsum


class _FakeNet:
    """Stands in for cv2.dnn: returns the logits it was handed, in order."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, np.float32)
        self.blob = None

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.logits[:len(self.blob)]


def _model(monkeypatch, tmp_path, logits, confidence=0.85, classes=("A", "B")):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"not really onnx")
    onnx.with_suffix(".json").write_text(
        json.dumps({"classes": list(classes), "size": 96,
                    "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}),
        encoding="utf-8")
    monkeypatch.setattr(tsum.cv2.dnn, "readNetFromONNX",
                        lambda _p: _FakeNet(logits))
    return tsum.CharacterModel(onnx, confidence)


def _tsums(n, kind=7):
    return [tsum.Tsum(x=60.0 + 60 * i, y=60.0, r=20.0, kind=kind, colour=(0, 0, 0))
            for i in range(n)]


def test_it_is_off_by_default():
    assert tsum.play_defaults().character == "", "a new rule ships off"


def test_the_flow_action_accepts_it():
    from ttheart_sender.automation import tsum_actions
    assert "character" in tsum_actions._TUNABLES
    assert "character_confidence" in tsum_actions._TUNABLES


def test_a_confident_crop_is_named(monkeypatch, tmp_path):
    board = np.zeros((200, 400, 3), np.uint8)
    ts = _tsums(2)
    # Class 1, overwhelmingly.
    model = _model(monkeypatch, tmp_path, [[0.0, 20.0], [0.0, 20.0]])
    named = model.apply(board, ts, 20.0)

    assert named == 2
    assert ts[0].kind == tsum.CHARACTER_KIND + 1
    assert ts[1].kind == tsum.CHARACTER_KIND + 1


def test_an_unsure_crop_keeps_its_colour_cluster(monkeypatch, tmp_path):
    """The property that makes the rule safe.

    The model knows the characters somebody labelled and no others. If an
    unsure crop lost its `kind` it would become unchainable, and the bot would
    stop playing every character nobody has labelled yet -- most of the board.
    """
    board = np.zeros((200, 400, 3), np.uint8)
    ts = _tsums(2, kind=5)
    # Nearly a coin flip: 0.5/0.5 is far under the 0.85 floor.
    model = _model(monkeypatch, tmp_path, [[1.0, 1.0], [1.0, 1.0]])
    named = model.apply(board, ts, 20.0)

    assert named == 0
    assert [t.kind for t in ts] == [5, 5], "an unsure tsum must keep its cluster"


def test_a_named_tsum_cannot_collide_with_a_colour_cluster(monkeypatch, tmp_path):
    """`kind` carries both, so character 3 must not equal colour cluster 3."""
    board = np.zeros((200, 400, 3), np.uint8)
    ts = _tsums(1, kind=1)
    model = _model(monkeypatch, tmp_path, [[0.0, 20.0]])
    model.apply(board, ts, 20.0)

    assert ts[0].kind != 1
    assert ts[0].kind >= tsum.CHARACTER_KIND
    assert tsum.CHARACTER_KIND > 200, "must clear any plausible k-means k"


def test_a_crop_the_frame_edge_would_clip_is_left_alone(monkeypatch, tmp_path):
    """Training refused clipped crops, so inference must not ask about one.

    A clipped crop is a sliver stretched to a square -- a picture the model has
    never seen an example of, and one it would still answer confidently.
    """
    board = np.zeros((200, 400, 3), np.uint8)
    edge = [tsum.Tsum(x=3.0, y=3.0, r=20.0, kind=9, colour=(0, 0, 0))]
    model = _model(monkeypatch, tmp_path, [[0.0, 20.0]])
    named = model.apply(board, edge, 20.0)

    assert named == 0
    assert edge[0].kind == 9


def test_named_tsums_of_one_character_can_chain(monkeypatch, tmp_path):
    """The point of the whole exercise: two tsums the model agrees on link.

    `adjacency` only ever asks whether two `kind`s are equal, which is why a
    character id can be carried in the same field -- but it is worth pinning,
    because the value of naming a tsum is entirely that the graph then joins it
    to its own character.
    """
    board = np.zeros((200, 400, 3), np.uint8)
    ts = [tsum.Tsum(x=60.0, y=60.0, r=20.0, kind=4, colour=(0, 0, 0)),
          tsum.Tsum(x=120.0, y=60.0, r=20.0, kind=8, colour=(0, 0, 0))]
    # Different colour clusters, same character.
    model = _model(monkeypatch, tmp_path, [[0.0, 20.0], [0.0, 20.0]])
    model.apply(board, ts, 20.0)

    adj = tsum.adjacency(ts, 20.0, link_px=105.0, block=1.25)
    assert 1 in adj[0], "two tsums named the same character must link"


def test_a_missing_model_is_refused_rather_than_ignored(tmp_path):
    """A round played with the model silently off looks exactly like one
    played with it on and doing nothing."""
    with pytest.raises((OSError, ValueError, KeyError)):
        tsum.CharacterModel(tmp_path / "nope.onnx", 0.85)


class _SizeSpy:
    """Records the batch size of every forward, the way cv2 sees it."""

    def __init__(self, n_classes=2):
        self.sizes = []
        self.n = n_classes
        self._last = 0

    def setInput(self, blob):
        self.sizes.append(int(blob.shape[0]))
        self._last = int(blob.shape[0])

    def forward(self):
        # Confident on class 1, so every crop gets named.
        out = np.zeros((self._last, self.n), np.float32)
        out[:, 1] = 20.0
        return out


def test_the_buffers_are_sized_at_the_cap_before_any_board(monkeypatch, tmp_path):
    """The workaround for the crash that took the app down mid-round.

    OpenCV 5.0.0 sizes a dnn net's buffers on the FIRST batch and does not grow
    them; a later larger batch overruns and Windows kills the process with
    0xc0000409 inside cv2.pyd -- no traceback, no error box. Reproduced: a
    batch of 1 then 46 dies on the 46.

    Live this is guaranteed: a round's first frame is the board still filling
    (~11 tsums) and a settled board is ~45. So the buffers must be sized at the
    maximum BEFORE the first board, and no board may ever exceed it. Those two
    are what this pins -- not that inference is correct, but that the net is
    never asked to GROW.

    It does not pin that the batch never varies. It used to: every forward was
    padded to 64. Measured on cv2 5.0.0, a net warmed at 64 then fed 41, 33,
    47, 12, 58, 1, 64, 40, 5 and 44 survives all of them and answers
    bit-identically to the padded version, and the padding was 43% of every
    forward on a ~41-detection board.
    """
    spy = _SizeSpy()
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    onnx.with_suffix(".json").write_text(
        json.dumps({"classes": ["A", "B"], "size": 96,
                    "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}),
        encoding="utf-8")
    monkeypatch.setattr(tsum.cv2.dnn, "readNetFromONNX", lambda _p: spy)
    model = tsum.CharacterModel(onnx, 0.5)

    board = np.zeros((600, 900, 3), np.uint8)
    for n in (11, 45, 46, 9, 64, 70):
        ts = [tsum.Tsum(x=float(60 + 60 * (i % 12)), y=float(60 + 60 * (i // 12)),
                        r=20.0, kind=1, colour=(0, 0, 0)) for i in range(n)]
        model.apply(board, ts, 20.0)

    assert spy.sizes, "the model must actually have run"
    assert spy.sizes[0] == tsum.CHARACTER_BATCH, (
        f"the buffers must be sized at {tsum.CHARACTER_BATCH} by the warm-up "
        f"before any board is read; the first forward was {spy.sizes[0]}")
    assert max(spy.sizes) <= tsum.CHARACTER_BATCH, (
        f"no forward may exceed {tsum.CHARACTER_BATCH} crops -- that is the "
        f"grow that kills the process; got {sorted(set(spy.sizes))}")
    assert 70 not in spy.sizes, "a board over the cap must go round again"


def test_a_short_board_is_sent_at_its_own_size(monkeypatch, tmp_path):
    """No padding: a five-tsum board costs five rows, and names five."""
    spy = _SizeSpy()
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    onnx.with_suffix(".json").write_text(
        json.dumps({"classes": ["A", "B"], "size": 96,
                    "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}),
        encoding="utf-8")
    monkeypatch.setattr(tsum.cv2.dnn, "readNetFromONNX", lambda _p: spy)
    model = tsum.CharacterModel(onnx, 0.5)

    board = np.zeros((600, 900, 3), np.uint8)
    ts = [tsum.Tsum(x=float(60 + 60 * i), y=60.0, r=20.0, kind=3, colour=(0, 0, 0))
          for i in range(5)]
    named = model.apply(board, ts, 20.0)

    assert named == 5
    assert model.seen == 5
    # sizes[0] is the warm-up; the board itself is one forward of exactly 5.
    assert spy.sizes[1:] == [5], (
        f"a 5-crop board must be one forward of 5, not padded; "
        f"got {spy.sizes[1:]}")
