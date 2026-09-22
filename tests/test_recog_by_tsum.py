"""Per-equipped-tsum recognition scoring.

The measurement this replaces was the obvious one -- split the labelled crops
by equipped tsum -- and it is impossible: every labelled crop whose session
still exists came from Beast rounds, and the 43 rounds on the second tsum have
none. So the comparison runs on the game's own marks instead, which need no
labels at all.

What has to hold: a session is attributed to the tsum its skill icon says, an
unrecognised icon is named as such rather than quietly folded into a profile,
and a round played with the character model ON is never mixed in -- its `kind`
was already overwritten, so its marks describe a different board.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recog_by_tsum  # noqa: E402


def session(root: Path, name: str, labs):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples.jsonl").write_text("\n".join(
        json.dumps({"index": i + 1, "base": {"lab": lab, "kind": 0},
                    "tsums": [], "options": {}})
        for i, lab in enumerate(labs)), encoding="utf-8")


def test_a_session_is_attributed_to_the_tsum_its_icon_names(tmp_path):
    session(tmp_path, "s_beast", [[199, 134, 165], [200, 135, 165]])
    session(tmp_path, "s_other", [[223, 137, 138], [223, 136, 138]])
    got = recog_by_tsum.equipped(tmp_path)
    assert got["s_beast"] == "beast"
    assert got["s_other"] not in ("beast", "unprofiled")


def test_an_icon_no_profile_claims_is_named_unprofiled_not_guessed(tmp_path):
    """Folding an unknown tsum into the nearest profile would silently pool two
    tsums' rounds, which is the exact mistake the base investigation made once
    and had to correct."""
    session(tmp_path, "s_weird", [[10, 200, 30]])
    assert recog_by_tsum.equipped(tmp_path)["s_weird"] == "unprofiled"


def test_the_median_icon_survives_a_few_bad_readings(tmp_path):
    """The icon reads to ~1.0 Lab unit within a session, but a frame caught
    mid-animation can be far off. A median absorbs that; a mean would not."""
    session(tmp_path, "s", [[199, 134, 165]] * 9 + [[10, 250, 20]])
    assert recog_by_tsum.equipped(tmp_path)["s"] == "beast"


def test_a_session_with_no_icon_reading_is_left_out(tmp_path):
    d = tmp_path / "s"
    d.mkdir(parents=True)
    (d / "samples.jsonl").write_text(
        json.dumps({"index": 1, "base": {}, "tsums": []}), encoding="utf-8")
    assert recog_by_tsum.equipped(tmp_path) == {}


def test_the_visibility_bands_cover_the_model_s_range_without_overlap():
    lows = [lo for lo, _ in recog_by_tsum.BANDS]
    highs = [hi for _, hi in recog_by_tsum.BANDS]
    assert lows[0] == 0.55, "the model is never asked below its training floor"
    assert highs[-1] > 1.0, "a fully visible tsum must land in the top band"
    for i in range(len(lows) - 1):
        assert highs[i] == lows[i + 1], "no gap and no overlap between bands"
