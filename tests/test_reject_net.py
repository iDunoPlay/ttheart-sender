"""The negative class, and the one check that keeps it honest.

`reject_net.py` trains "is this a tsum at all" against the folders in
`NOT_TSUM`. A folder a person filled on purpose is a label -- but a person can
be wrong, and the game itself is not: a detection it marked or cleared is a
tsum whatever folder it ended up in.

That rule existed and only ran on the leftover pile in `crops/all`. It did not
reach the negative FOLDER, where 37 of 418 checkable `board` crops (8.9%) are
game-confirmed and were training as "not a tsum". It also happens to be the
guard against marking a whole class junk that the game plays with -- press `j`
on coins and they arrive in `board/`, 76.2% confirmed, and are handed back
rather than poisoning the negatives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reject_net as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def crop(root: Path, cls: str, sess: str, sample: int, idx: int, vis=0.64):
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("%s_%04d_%02d_v%.2f.png" % (sess, sample, idx, vis))
    p.write_bytes(b"")
    return p


def test_a_character_folder_trains_as_a_tsum(tmp_path):
    crop(tmp_path / "lab", "Pascal", "s", 1, 0)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0, set())
    assert list(y) == [1]


def test_the_negative_folder_trains_as_not_a_tsum(tmp_path):
    crop(tmp_path / "lab", "board", "s", 1, 0)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0, set())
    assert list(y) == [0]


def test_a_game_confirmed_crop_in_the_negative_folder_is_rescued(tmp_path):
    """The fix. Without it, a tsum the game dragged and cleared trains the
    model to throw that detection away."""
    crop(tmp_path / "lab", "board", "s", 1, 0)
    crop(tmp_path / "lab", "board", "s", 1, 1)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0,
                     {("s", 1, 0)})
    assert sorted(y) == [0, 1], "the confirmed one must come back as a tsum"


def test_the_rescue_does_not_touch_the_positive_folders(tmp_path):
    """A character crop is already a tsum; confirming it changes nothing, and
    the rescue must not double-count or relabel it."""
    crop(tmp_path / "lab", "Pascal", "s", 1, 0)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0, {("s", 1, 0)})
    assert list(y) == [1]


def test_without_the_game_s_answers_nothing_is_rescued(tmp_path):
    """`keep` empty means the dataset was not available. The folders are then
    taken at face value rather than silently half-checked."""
    crop(tmp_path / "lab", "board", "s", 1, 0)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0, None)
    assert list(y) == [0]


def test_coin_is_not_in_the_negative_list():
    """Measured: coin crops are marked by the game 76.2% of the time and 42.9%
    were cleared by a drag that worked -- more than any real character."""
    assert "coin" not in R.NOT_TSUM
    assert "unknown_lightball" not in R.NOT_TSUM
    assert "board" in R.NOT_TSUM


def test_the_junk_folder_trains_as_not_a_tsum(tmp_path):
    """A folder the player asked for. It needs no separate justification
    because the rescue covers it: anything dropped in there that the game
    marked or cleared is handed back at train time."""
    crop(tmp_path / "lab", "junk", "s", 1, 0)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0, set())
    assert list(y) == [0]
    assert "junk" in R.NOT_TSUM


def test_a_mistaken_junk_call_is_handed_back(tmp_path):
    """The guarantee `coin` did not have when it was added on appearance."""
    crop(tmp_path / "lab", "junk", "s", 1, 0)
    crop(tmp_path / "lab", "junk", "s", 1, 1)
    _, y, _ = R.load(tmp_path / "lab", tmp_path / "none", 0.0, {("s", 1, 0)})
    assert sorted(y) == [0, 1]
