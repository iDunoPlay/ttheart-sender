"""The crop filename is a primary key, and it has to parse.

`<session>_<sample>_<index>_v<visible>.png` is how every tool here gets from a
crop back to the detection it came from -- which session it belongs to (the
split), which sample, and how visible it was. A filename that does not parse
does not error: `classify.py` files it under session "unknown", so every such
crop lands on ONE side of the split together, cannot be excluded by the golden
set, and cannot be filtered by visibility.

That happened. `label_board.py` numbers hand-added circles from 90, so adding
more than ten on one board produced index 100+, and the pattern accepted exactly
two digits. 26 crops went in invisible to the split.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

ROOT = Path(__file__).resolve().parents[1]
GOOD = [
    "20260904_235003_8872_0002_07_v0.64",     # a real detection
    "20260904_235003_8872_0002_91_v1.00",     # hand-added, two digits
    "20260905_061026_8872_0005_108_v1.00",    # hand-added, three digits
]


def patterns():
    """Every copy of the filename pattern in the scripts, by file."""
    out = {}
    for p in sorted((ROOT / "scripts").glob("*.py")):
        src = p.read_text(encoding="utf-8")
        for line in src.splitlines():
            if "P<idx>" in line and "re.compile" in line:
                out[p.name] = line
    return out


def test_every_tool_parses_a_three_digit_index():
    """The bug: 26 crops with index 100+ were filed under session 'unknown'."""
    found = patterns()
    assert found, "no filename patterns found at all"
    for name, line in found.items():
        pat = re.compile(re.search(r'r"(.+?)"', line).group(1))
        for stem in GOOD:
            assert pat.match(stem), "%s cannot parse %s" % (name, stem)


def test_the_session_is_recovered_whole():
    from classify import NAME
    m = NAME.match("20260905_061026_8872_0005_108_v1.00")
    assert m.group("sess") == "20260905_061026_8872"
    assert m.group("sample") == "0005"
    assert m.group("idx") == "108"
    assert m.group("vis") == "1.00"


def test_a_hand_dropped_file_does_not_parse_and_that_is_visible():
    """A crop dragged in by hand has no session, so it cannot join the split.
    It still TRAINS -- it just lands under 'unknown' with every other such
    crop, which is worth knowing before relying on the held-out number."""
    from classify import NAME
    for stem in ("Dory_1", "my photo", "screenshot (3)"):
        assert NAME.match(stem) is None


def test_every_shipped_crop_parses():
    """The corpus itself, so a future tool change cannot orphan crops."""
    from classify import NAME
    root = ROOT / "crops" / "labelled"
    if not root.is_dir():
        return
    bad = [p.name for d in root.iterdir() if d.is_dir()
           for p in d.glob("*.png") if not NAME.match(p.stem)]
    assert not bad, "%d crop(s) do not parse, e.g. %s" % (len(bad), bad[:3])
