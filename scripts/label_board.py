"""Click a tsum, name it. Mark the junk. Add what the detector missed.

Labelling here has only ever worked one way: `crops.py cluster` groups crops by
an embedding, emits a contact sheet per cluster, and `crops.py assign 03 Mickey`
names the WHOLE cluster. That made 3,958 labels affordable -- twenty decisions
instead of two thousand -- but it has no way to fix ONE crop, no way to say
"that is not a tsum at all", and no way to add a tsum the detector never found.

This is the other half. One board at a time, every detection circled, and three
kinds of edit:

    NAME     click a tsum, press a number or type a name
    JUNK     press `j` -- an empty bowl, board graphics, an effect flash
    ADD      click empty space where a tsum plainly is, and name it

    python scripts/label_board.py
    python scripts/label_board.py --profile beans_camo_vil
    python scripts/label_board.py --only-wrong     # boards the model disagrees on

Keys
----

    LEFT click   select the nearest detection; or click a NAME in the panel
                 to apply it to the selected tsum, without typing
    RIGHT click  ADD a circle where the detector missed one
    1-9          accept a suggestion for the selection
    /            start typing a name; TAB completes to the top match,
                 Enter commits, Backspace edits, Esc cancels
    /            while typing, LOCKS what you have typed as a filter that
                 survives labelling -- so the same two letters are not retyped
                 for every tsum of one character. A locked filter is not text
                 entry, so 1-9 pick from it again
    l            clears a locked filter (it cannot lock ONE while typing: the
                 typing branch appends every printable key, so `l` there is
                 the letter l)
    j            mark as junk -- not a tsum (works on rim half-circles too)
    d            remove the label, or the junk verdict
    h            shrink junk-marked circles to dots, to get them out of the way
    x            remove an ADDED circle (never touches a real detection)
    - / =        shrink / grow an added circle
    n / p        next / previous board
    u            undo the last write
    q            quit

Single letters are commands, so typing a name starts with `/`. The first version
started typing on any a-z and `j` could never mean junk.

Where the edits go
------------------

**Names** -> `crops/labelled/<Name>/<session>_<sample>_<index>_v<visible>.png`,
the exact layout and filename `crops.py assign` produces, cut by **crops.py's
own `_cut`**, imported rather than copied. `classify.py` picks them up unchanged.

**Junk** -> `crops/labelled/board/`, and that is not a dumping ground: it is the
class `scripts/reject_net.py` trains its "is this a tsum at all" model against
(`NOT_TSUM`). Its docstring is worth reading before using this key -- 763 crops
were once treated as junk merely for being unlabelled, and 21% of them turned
out to be tsums the game had drawn its highlight over. **An absence of a label
is not a negative label. This key is a label**, so use it only when you can see
it is bowl, board or effect.

**Added circles** -> the same crop folders, with an index from 90 up so it can
never collide with a real detection, plus a line in `<out>/_manual.jsonl` so the
circle is still there when you come back to the board.

Half circles at the rim
-----------------------

A detection at the board's edge is a half circle, it is usually not a tsum, and
`j` marks it junk like any other -- but **no crop is written for it**. `_cut`
refuses a crop the frame edge would clip, and both ways of forcing one make a
model worse: clipping and stretching teaches `reject_net.py` that "distorted =
junk", and padding the frame teaches it that "padded = junk". Either is
train/serve skew, which is the bug that cost this project 312 rounds. And
`RejectModel` reads its crops through `_character_crop` too, so at play time it
is **never asked about an edge detection anyway**.

So the verdict is recorded in `<out>/_junk.jsonl` and the crop is not. The
circle turns junk-coloured, the call is kept, nothing is poisoned -- and the
count is the evidence for whether padding the frame is worth doing, because
those are exactly the detections it would make trainable. **40.7% of tsums above
the visibility floor are refused this way, and 52% of the clearest ones** -- see
`scripts/board_check.py`.

One more refusal
----------------

**The model's guess is never written as a label.** It is shown; a person presses
the key. `docs/IDENTITY.md` section 9 has been the rule since the first
mark-harvest.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crops import _cut  # noqa: E402  -- ONE crop rule, shared with the extractor
from ttheart_sender.game import crop as crop_rules  # noqa: E402
from ttheart_sender.game import profiles  # noqa: E402
from ttheart_sender.game import tsum as T  # noqa: E402

NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2,})_v(?P<vis>[\d.]+)$")

#: Folders that are not characters. `board` is the junk class `reject_net.py`
#: trains against; `score` joined it once this tool made per-crop junk marking
#: possible and the folder was looked at.
NOT_CHARACTERS = {"board", "score", "junk"}
#: Where `j` files a crop. `board` holds the 705 crops a person filled by hand
#: before this tool existed and stays a negative class; new junk goes to its own
#: folder so the two populations remain separable -- `board` is verified empty
#: bowl and board graphics, `junk` is whatever you point at.
JUNK_CLASS = "junk"

#: Added detections start here. Real detections on a board run to about 50, and
#: an index is part of the crop's filename and therefore its identity -- so the
#: two ranges must not be able to meet.
MANUAL_BASE = 90
PANEL = 250

LABELLED = (70, 210, 70)      # a person named this
JUNK = (60, 60, 235)          # a person said it is not a tsum
GUESS = (0, 190, 235)         # the model's guess, nobody has confirmed it
UNASKED = (120, 120, 120)     # never offered to the model
REFUSED = (150, 90, 200)      # the frame edge clips its crop
ADDED = (235, 180, 60)        # a circle a person put there
PICKED = (255, 255, 255)


def labels(root: Path) -> dict:
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            m = NAME.match(p.stem)
            if m:
                out[(m.group("sess"), int(m.group("sample")),
                     int(m.group("idx")))] = (d.name, p)
    return out


def equipped(root: Path) -> dict:
    out = {}
    for jl in sorted(root.glob("*/samples.jsonl")):
        labs = []
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            b = d.get("base") or {}
            if b.get("lab"):
                labs.append(b["lab"])
        if labs:
            med = np.median(np.asarray(labs, float), axis=0)
            out[jl.parent.name] = (profiles.identify([float(v) for v in med])
                                   or "unprofiled")
    return out


def boards(root: Path, profile: str, who: dict, seed: int):
    rows = []
    for jl in sorted(root.glob("*/samples.jsonl")):
        sess = jl.parent.name
        if profile and who.get(sess) != profile:
            continue
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            o = d.get("options") or {}
            if o.get("character") or o.get("recolour") or o.get("kinds"):
                continue
            img = jl.parent / ("%04d_before.jpg" % d.get("index", -1))
            if d.get("tsums") and img.exists():
                rows.append((sess, d, img))
    random.Random(seed).shuffle(rows)
    return rows


class Manual:
    """Circles a person added, and the file that remembers them.

    A crop written to `crops/labelled/` is already the durable artifact -- the
    training set has it either way. This exists so the CIRCLE is still on the
    board when you come back to it, which is what makes a second pass over the
    same board possible.
    """

    def __init__(self, path: Path):
        self.path = path
        self.rows = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                self.rows[(d["sess"], d["sample"], d["idx"])] = d

    def on(self, sess, sample):
        return sorted((d for (s, sm, _), d in self.rows.items()
                       if s == sess and sm == sample),
                      key=lambda d: d["idx"])

    def next_index(self, sess, sample) -> int:
        used = {d["idx"] for d in self.on(sess, sample)}
        i = MANUAL_BASE
        while i in used:
            i += 1
        return i

    def add(self, sess, sample, idx, x, y, r):
        self.rows[(sess, sample, idx)] = {
            "sess": sess, "sample": sample, "idx": idx,
            "x": float(x), "y": float(y), "r": float(r)}
        self.save()

    def resize(self, sess, sample, idx, r):
        d = self.rows.get((sess, sample, idx))
        if d:
            d["r"] = max(4.0, float(r))
            self.save()

    def remove(self, sess, sample, idx):
        self.rows.pop((sess, sample, idx), None)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "\n".join(json.dumps(d) for d in self.rows.values()),
            encoding="utf-8")


class Verdicts:
    """Junk calls on detections that CANNOT be cropped, kept without a crop.

    A half circle at the board's rim is plainly not something to chain, and
    saying so must be possible. But no crop can be written for it, and the two
    obvious ways to force one are both worse than keeping none:

    * **Clip and stretch it.** A sliver resized to a square is a distorted
      picture, and `reject_net.py` would learn "distorted = junk" rather than
      anything about the board.
    * **Pad the frame and cut a full square.** Then every junk crop is padded
      and every other crop is not, so the model learns "padded = junk". That is
      train/serve skew, which is the bug that cost this project 312 rounds.

    And there is a third reason, which settles it: `RejectModel` reads its crops
    through `_character_crop` too (tsum.py:851), so **at play time the reject
    model is never asked about an edge detection either.** A crop written here
    would train it for a question it is never posed.

    So the verdict is recorded and the crop is not. It costs nothing, it cannot
    poison a model, and it is the evidence for whether padding the frame is
    worth doing: these are the detections that would become trainable.
    """

    def __init__(self, path: Path):
        self.path = path
        self.rows = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                self.rows.add((d["sess"], d["sample"], d["idx"]))

    def has(self, key) -> bool:
        return key in self.rows

    def add(self, key):
        self.rows.add(key)
        self.save()

    def remove(self, key):
        self.rows.discard(key)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(
            json.dumps({"sess": s, "sample": sm, "idx": i, "verdict": "junk"})
            for s, sm, i in sorted(self.rows)), encoding="utf-8")


class Session:
    """One labelling run: what is on screen, and every write it has made."""

    def __init__(self, out: Path, truth: dict):
        self.out = out
        self.truth = truth
        self.history = []          # (kind, key, from_path, to_path)
        self.written = 0

    def classes(self):
        seen = {n for n, _ in self.truth.values()} - NOT_CHARACTERS
        if self.out.is_dir():
            seen |= {d.name for d in self.out.iterdir()
                     if d.is_dir() and d.name not in NOT_CHARACTERS}
        return sorted(seen)

    def assign(self, key, name: str, img, item, radius) -> str:
        """Write or move one crop. Returns a line for the status bar."""
        crop = _cut(img, item["x"], item["y"], radius)
        if crop is None:
            return "REFUSED: frame edge clips this crop"
        sess, sample, idx = key
        vis = min(item["r"] / radius if radius else 0.0, 9.99)
        fname = "%s_%04d_%02d_v%.2f.png" % (sess, sample, idx, vis)
        dest_dir = self.out / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname

        old = self.truth.get(key)
        if old and old[1].exists() and old[1] != dest:
            # A CORRECTION moves the file. Leaving the old one behind would
            # train the model on both answers at once.
            old[1].unlink()
            self.history.append(("move", key, old[1], dest))
            note = "corrected %s -> %s" % (old[0], name)
        else:
            self.history.append(("new", key, None, dest))
            note = ("marked JUNK" if name == JUNK_CLASS
                    else "labelled %s" % name)
        cv2.imwrite(str(dest), crop)
        self.truth[key] = (name, dest)
        self.written += 1
        return note

    def delete(self, key) -> str:
        old = self.truth.get(key)
        if not old or not old[1].exists():
            return "nothing to delete here"
        old[1].unlink()
        self.history.append(("delete", key, old[1], None))
        del self.truth[key]
        return "removed the %s label" % old[0]

    def undo(self, img_of) -> str:
        if not self.history:
            return "nothing to undo"
        kind, key, src, dest = self.history.pop()
        if dest and dest.exists():
            dest.unlink()
        if kind in ("move", "delete") and src is not None:
            got = img_of(key)
            if got is not None:
                src.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(src), got)
                self.truth[key] = (src.parent.name, src)
        elif kind == "new":
            self.truth.pop(key, None)
        return "undone"


def items_for(ts, manual_rows):
    """Real detections then added ones, each carrying its own index."""
    out = [{"x": t["x"], "y": t["y"], "r": t["r"], "idx": i, "manual": False}
           for i, t in enumerate(ts)]
    out += [{"x": d["x"], "y": d["y"], "r": d["r"], "idx": d["idx"],
             "manual": True} for d in manual_rows]
    return out


#: How far a click may land from a detection's centre and still select it,
#: in BOARD pixels.
#:
#: Not the detection's own radius, which was the first version's rule and made
#: a fifth of the board unselectable: `r` is the VISIBLE radius, and measured
#: over the corpus it is under 8px for 20% of detections and under 12px for
#: 59%. A rim half-circle -- exactly the thing most worth marking junk -- has
#: the smallest `r` of all, so the tsums hardest to click were the ones you
#: most needed to reach.
#:
#: 14px against a median centre-to-centre spacing of ~61px, so the nearest
#: detection is still unambiguous.
SLACK = 14.0


def pick(items, x, y, slack: float = SLACK):
    """The nearest detection to the cursor, or None if the click is nowhere."""
    best, bd = None, 1e9
    for n, it in enumerate(items):
        d = float(np.hypot(it["x"] - x, it["y"] - y))
        if d < max(slack, it["r"]) and d < bd:
            best, bd = n, d
    return best


def outlined(dst, text, org, scale, colour, thick=1):
    """Text with a dark stroke behind it.

    A board is busy, bright and multicoloured, so plain thin text on it is
    unreadable over roughly half the tsums -- which is the entire reason the
    first version's labels "could not be seen clearly". The stroke costs one
    extra draw and makes the label legible over anything.
    """
    cv2.putText(dst, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(dst, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def render(img, items, radius, sess, sample, state, sel, guesses, typed,
           status, idx_of_board, n_boards, zoom=1.0, hide_junk=False,
           hits=None, locked=""):
    """The board, magnified, with what is known about each detection on it.

    `zoom` scales the BOARD, not the text: the picture is enlarged first and
    the labels are drawn on top at a readable size afterwards, so they stay
    crisp instead of being blown up with the pixels.
    """
    h0, w0 = img.shape[:2]
    if zoom != 1.0:
        shown = cv2.resize(img, None, fx=zoom, fy=zoom,
                           interpolation=cv2.INTER_LINEAR)
    else:
        shown = img.copy()
    h, w = shown.shape[:2]
    panel = max(PANEL, int(PANEL * min(zoom, 1.5)))
    # The PANEL does not scale with zoom and the board labels do. Zoom exists
    # to make a 40px tsum readable; scaling the panel with it only ate the
    # space the name list needs -- at 1.8x it left room for ONE name out of 67.
    fs = 0.44                                     # panel font, fixed
    ls = 0.42 * min(max(zoom, 1.0), 1.7)          # label font on the board
    row = 19

    # The canvas is at least as tall as the PANEL needs, not an arbitrary
    # number. At zoom 1.0 a 456px board gave a 520px canvas, which left room
    # for exactly ONE name -- the list was sized by the board, which has
    # nothing to do with how many classes there are.
    want_rows = 18 + 10                  # header block, then ten names
    canvas_h = max(h, 22 + want_rows * row + int(9.5 * row) + 16)
    canvas = np.zeros((canvas_h, w + panel, 3), np.uint8)
    canvas[:h, :w] = shown
    # Draw into a VIEW of the board area. A tsum at the rim has its centre
    # within a radius of the edge -- the whole 40.7% finding -- so its circle
    # would otherwise be drawn across the legend. A numpy slice is a view.
    board = canvas[:h, :w]

    for n, it in enumerate(items):
        x, y = int(it["x"] * zoom), int(it["y"] * zoom)
        r = max(3, int(it["r"] * zoom))
        key = (sess, sample, it["idx"])
        named = state["truth"].get(key)
        junked = state["verdict"](key) or (named and named[0] in NOT_CHARACTERS)
        if junked:
            colour = JUNK
        elif named:
            colour = LABELLED
        elif it["manual"]:
            colour = ADDED
        elif it["idx"] in state["refused"]:
            colour = REFUSED
        elif it["idx"] in state["guess"]:
            colour = GUESS
        else:
            colour = UNASKED
        if hide_junk and junked and n != sel:
            # Shrunk to a dot, NOT removed. A circle you can no longer see is
            # one you can no longer take back, and `j` is a training label
            # rather than a delete key -- it must stay reachable to undo.
            cv2.circle(board, (x, y), 2, colour, -1)
            continue
        thick = 2 if (named or it["manual"] or it["idx"] in state["guess"]) else 1
        if n == sel:
            cv2.circle(board, (x, y), r + 4, PICKED, 3)
        cv2.circle(board, (x, y), r, colour, thick)
        text = "junk" if junked else (
            named[0] if named else state["guess"].get(it["idx"], ("", 0))[0])
        if text:
            outlined(board, text[:14], (x - int(26 * zoom), y - r - 4),
                     ls, colour)

    px = w + 8

    def line(n, s, colour=(235, 235, 235), scale=None):
        outlined(canvas, s, (px, 22 + n * row), scale or fs, colour)

    line(0, "board %d/%d" % (idx_of_board + 1, n_boards))
    line(1, "%s #%d" % (sess[-13:], sample), (170, 170, 170))
    line(3, "LEGEND")
    for n, (colour, what) in enumerate((
            (LABELLED, "you named it"),
            (JUNK, "you marked junk"),
            (ADDED, "you added it"),
            (GUESS, "model's guess"),
            (UNASKED, "never asked"),
            (REFUSED, "edge clips crop"))):
        cv2.circle(canvas, (px + 7, 17 + (4 + n) * row), 6, colour, 2)
        line(4 + n, "      " + what, colour)

    if sel is None:
        line(11, "SELECTED")
        line(12, "  click a tsum, or bare", (170, 170, 170))
        line(13, "  board to ADD a circle", (170, 170, 170))
    else:
        it = items[sel]
        key = (sess, sample, it["idx"])
        named = state["truth"].get(key)
        recorded = state["verdict"](key)
        is_junk = recorded or (named and named[0] in NOT_CHARACTERS)
        # The current name, big and first. Knowing what a circle says NOW is
        # the whole prerequisite for deciding to change it, and at the old
        # size it was unreadable both on the board and here.
        now = ("junk" if is_junk else named[0] if named else "not named")
        line(11, "NOW:", (170, 170, 170))
        line(12, "  " + now[:16],
             JUNK if is_junk else LABELLED if named else (170, 170, 170),
             scale=fs * 1.45)
        vis = it["r"] / radius if radius else 0.0
        sub = ("recorded, no crop" if recorded else
               "#%d%s" % (it["idx"], " (added)" if it["manual"] else ""))
        line(13, "  " + sub, (150, 150, 150), scale=fs * 0.85)
        # Visibility is written into the crop's FILENAME and every downstream
        # tool bands by it. For an added circle it is not measured -- it is
        # whatever radius you left, defaulting to the full board radius -- so
        # it is shown in the added colour and named an estimate, because 58 of
        # 91 crops from the first labelling session went in claiming v1.00.
        line(14, "  visible %.2f%s" % (vis, "  (estimate, -/=)"
                                       if it["manual"] else ""),
             ADDED if it["manual"] else (150, 150, 150), scale=fs * 0.85)
        head = ("MATCHING \"%s\"" % typed if typed else
                "LOCKED \"%s\"  (l clears)" % locked if locked else
                "CHANGE IT TO:" if named else "NAME IT:")
        line(16, head, (235, 235, 235))
        if typed is not None:
            line(17, "  > %s_    (/ locks)" % typed, (0, 235, 235))
        first = 18 if typed is not None else 17

        # Every name is CLICKABLE, and how many fit is a property of the
        # window rather than of the keyboard. Nine was the number of digit
        # keys, which stopped being the right limit at 69 classes.
        # The footer is eight lines tall and drawn from the bottom up, so the
        # list has to stop above it. Reserving a flat 60px let the names run
        # straight through the key hints at 1.8x zoom.
        footer = int(9.5 * row) + 16
        room = max(1, (canvas.shape[0] - footer - (22 + first * row)) // row)
        for n, (nm, pr) in enumerate(guesses[:room]):
            mark = " <- now" if named and nm == named[0] else ""
            key = "%d" % (n + 1) if n < 9 else " "
            y = 22 + (first + n) * row
            line(first + n, "  %s %s %s%s"
                 % (key, nm[:15], ("%.2f" % pr) if pr else "", mark),
                 LABELLED if mark else GUESS)
            if hits is not None:
                # The rectangle this row was actually drawn in, handed back so
                # the click handler never has to re-derive the geometry. Two
                # copies of that arithmetic is how a list stops matching what
                # it selects.
                hits.append((px, y - row + 4, canvas.shape[1], y + 4, nm))
        if len(guesses) > room:
            line(first + room, "  ...%d more, type / to filter"
                 % (len(guesses) - room), (150, 150, 150))

    ch = canvas.shape[0]

    def foot(up, s, colour=(170, 170, 170)):
        outlined(canvas, s, (px, ch - 10 - up * int(row * 0.95)),
                 fs * 0.92, colour)

    foot(0, "written: %d" % state["written"])
    if status:
        foot(1, status[:32], (120, 235, 255))
    foot(3, "n/p board  u undo  q quit")
    foot(4, "j junk  d unlabel  x un-add")
    foot(5, "/ type, / again LOCKS it")
    foot(6, "1-9 pick or CHANGE name")
    foot(7, "click a NAME to apply it")
    foot(8, "[ ] zoom out / in")
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--profile", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.85)
    ap.add_argument("--min-visible", type=float, default=0.0,
                    help="ask the model this far down. 0 by default, NOT the "
                         "play loop's 0.55: down here its guess is near chance "
                         "and the point is for you to correct it")
    ap.add_argument("--needs", default="",
                    help="hunt for one class: order the boards by how strongly "
                         "the model thinks it is on them. 10,747 boards is not "
                         "a labelling job; the 159 crops that would make every "
                         "existing class trainable is, once you can find them")
    ap.add_argument("--unsure", action="store_true",
                    help="order boards by how much of them the model cannot "
                         "name. Where the classes it does not know live, and "
                         "the only way to hunt a class that has never been "
                         "trained")
    ap.add_argument("--scan", type=int, default=1200,
                    help="boards to score when hunting. Each costs about 5ms")
    ap.add_argument("--board", type=int, default=0,
                    help="open at this board number -- the one the panel shows "
                         "as 'board N/10747'. The order is fixed by --seed, so "
                         "the same number reopens the same board")
    ap.add_argument("--session", default="",
                    help="open the board from this session; with --sample, "
                         "exactly one board. Survives a change of --seed, which "
                         "--board does not")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--zoom", type=float, default=1.8,
                    help="magnify the board. A stored board is 525x456, which "
                         "puts a tsum at about 40px and its label under that -- "
                         "too small to read. Adjustable live with [ and ]")
    ap.add_argument("--crop-profile", default="",
                    help="cut crops under this profile instead of the "
                         "one the model records. `padded_v1` extends "
                         "the frame so a tsum at the rim yields a crop "
                         "instead of nothing -- 40.7%% of tsums above "
                         "the visibility floor are refused without it")
    ap.add_argument("--only-wrong", action="store_true",
                    help="only boards where the model disagrees with a label "
                         "you already wrote")
    args = ap.parse_args()

    model = T.load_character_model(args.model, args.confidence, args.min_visible)
    truth = labels(args.out)
    manual = Manual(args.out / "_manual.jsonl")
    verdicts = Verdicts(args.out / "_junk.jsonl")
    who = equipped(args.dir)
    rows = boards(args.dir, args.profile, who, args.seed)
    if not rows:
        print("no boards found")
        return 2
    ses = Session(args.out, truth)

    print("%d boards. Click a tsum to name it, bare board to add one." % len(rows))
    print("j = junk (goes to %s/, which reject_net.py trains against)."
          % JUNK_CLASS)
    print("Crops are written to %s in crops.py's own format." % args.out)

    win = "label board"
    cv2.namedWindow(win)
    click = [None]

    def on_mouse(event, x, y, flags, param):
        # LEFT selects, RIGHT adds. They were one button, and the two wanted
        # opposite things from a near-miss: selection wants to be forgiving,
        # adding wants to be sure you did not mean an existing circle. Split,
        # both can be right.
        if event == cv2.EVENT_LBUTTONDOWN:
            click[0] = (x, y, False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click[0] = (x, y, True)

    cv2.setMouseCallback(win, on_mouse)

    if args.unsure:
        # No class needed. Boards the model is least sure about are where the
        # characters it cannot name are, which is exactly where a NEW class
        # comes from -- and a class it has never seen cannot be hunted by name.
        print("scanning %d board(s) for the ones the model is least sure about ..."
              % min(args.scan, len(rows)))
        scored = []
        for n, (sess_, d_, path_) in enumerate(rows[:args.scan]):
            im = cv2.imread(str(path_))
            if im is None:
                continue
            ts_ = d_["tsums"]
            rad_ = float(d_.get("radius") or 25.0)
            u_, pr_ = model.probabilities(
                im, [T.Tsum(t["x"], t["y"], t["r"], 0, (0, 0, 0)) for t in ts_],
                rad_)
            if not len(u_):
                continue
            # The share of readable tsums it will not name at the shipped
            # floor. High means a board full of something it does not know.
            scored.append((float((pr_.max(axis=1) < args.confidence).mean()), n))
        scored.sort(reverse=True)
        rows = [rows[n] for _, n in scored] + rows[args.scan:]
        print("  worst board: %.0f%% of its readable tsums unnamed."
              % (100 * scored[0][0]) if scored else "  nothing scored.")

    if args.needs:
        # Rank boards by the model's best guess that this class is present.
        # The model is only ~19% sure of a board at all, so this is a HINT and
        # not a filter: it puts the likely boards first and leaves the rest
        # reachable with `n`.
        want = None
        for n, cls in enumerate(model.classes):
            if cls.lower() == args.needs.lower():
                want = n
                break
        if want is None:
            # The bootstrap problem, stated rather than hidden: you need crops
            # to train a class, and a trained class to find crops. `--min-class
            # 5` breaks it -- a class with 9 crops trains badly but well enough
            # to point at boards, which is all this needs.
            print("this model has no class %r, so it cannot hunt for it."
                  % args.needs)
            print("")
            print("  Break the loop by training one that knows it:")
            print("    scripts/retrain.py --min-class 5 --dry-run")
            print("    scripts/label_board.py --needs %s "
                  "--model models/character_candidate.onnx" % args.needs)
            print("")
            print("  Or find the boards it is UNSURE about, which is where")
            print("  the classes it cannot name live:")
            print("    scripts/label_board.py --unsure")
            return 2
        print("scanning %d board(s) for %s ..."
              % (min(args.scan, len(rows)), model.classes[want]))
        scored = []
        for n, (sess_, d_, path_) in enumerate(rows[:args.scan]):
            im = cv2.imread(str(path_))
            if im is None:
                continue
            ts_ = d_["tsums"]
            rad_ = float(d_.get("radius") or 25.0)
            u_, pr_ = model.probabilities(
                im, [T.Tsum(t["x"], t["y"], t["r"], 0, (0, 0, 0)) for t in ts_],
                rad_)
            best = float(pr_[:, want].max()) if len(u_) else 0.0
            scored.append((best, n))
        scored.sort(reverse=True)
        rows = ([rows[n] for _, n in scored]
                + rows[args.scan:])
        top = sum(1 for b, _ in scored if b >= 0.5)
        print("  %d board(s) where it is over 0.50 sure. Those come first."
              % top)

    i = 0
    if args.session:
        # An exact board, by name. `--board` is an index into an order that
        # only holds while `--seed` does; a session and sample name one frame
        # for good.
        want = [n for n, (sess_, d_, _) in enumerate(rows)
                if sess_ == args.session
                and (not args.sample or d_["index"] == args.sample)]
        if not want:
            print("no board for session %s%s"
                  % (args.session,
                     " sample %d" % args.sample if args.sample else ""))
            return 2
        i = want[0]
    elif args.board:
        i = max(0, min(len(rows) - 1, args.board - 1))
    typed = [None]
    status = [""]
    zoom = [max(0.5, min(4.0, args.zoom))]
    hits = []          # (x0, y0, x1, y1, name) for each clickable name row
    #: A filter that SURVIVES a label. Typing narrows the list and then clears
    #: on commit, which means retyping the same two letters for every tsum of
    #: the same character -- the common case, because a board holds five
    #: characters and about forty tsums. `l` locks the current text here.
    #:
    #: It also frees the number keys: while TYPING, `1` has to mean the
    #: character "1" (there is a class called `22`), but a locked filter is not
    #: text entry, so 1-9 go back to picking from the list.
    filt = [""]
    hide = [False]
    while 0 <= i < len(rows):
        sess, d, path = rows[i]
        sample = d["index"]
        img = cv2.imread(str(path))
        if img is None:
            i += 1
            continue
        ts = d["tsums"]
        radius = float(d.get("radius") or 25.0)
        usable, prob = model.probabilities(
            img, [T.Tsum(t["x"], t["y"], t["r"], 0, (0, 0, 0)) for t in ts],
            radius)
        guess = {}
        for n, gi in enumerate(usable):
            b = int(prob[n].argmax())
            guess[gi] = (model.classes[b], float(prob[n][b]))
        refused = {n for n, t in enumerate(ts)
                   if _cut(img, t["x"], t["y"], radius) is None}

        if args.only_wrong:
            bad = any((sess, sample, k) in ses.truth
                      and ses.truth[(sess, sample, k)][0] != v[0]
                      for k, v in guess.items())
            if not bad:
                i += 1
                continue

        sel = [None]
        typed[0] = None
        step = 0
        while True:
            items = items_for(ts, manual.on(sess, sample))
            if click[0] is not None:
                x, y, adding = click[0]
                click[0] = None
                if x >= img.shape[1] * zoom[0]:
                    # A click in the PANEL. If it landed on a name, that name
                    # is the answer -- no typing, no digit to remember.
                    if sel[0] is not None:
                        for x0, y0, x1, y1, nm in hits:
                            if x0 <= x <= x1 and y0 <= y <= y1:
                                status[0] = ses.assign(
                                    (sess, sample, items[sel[0]]["idx"]),
                                    nm, img, items[sel[0]], radius)
                                # `typed` clears, `filt` does NOT -- that is
                                # the whole point of locking it.
                                typed[0] = None
                                break
                    else:
                        status[0] = "click a tsum first, then a name"
                    x = -1
                # Back into BOARD coordinates, and only AFTER the panel test:
                # the panel's own hit rectangles are in window pixels, so
                # scaling first would have made every name row unclickable.
                # The window shows a magnified copy; every detection, crop and
                # filename is in the original frame's pixels.
                x, y = x / zoom[0], y / zoom[0]
                if 0 <= x < img.shape[1]:
                    hit = pick(items, x, y)
                    if adding:
                        # RIGHT click: the detector missed a tsum here.
                        idx = manual.next_index(sess, sample)
                        manual.add(sess, sample, idx, x, y, radius)
                        items = items_for(ts, manual.on(sess, sample))
                        sel[0] = next(n for n, it in enumerate(items)
                                      if it["idx"] == idx)
                        status[0] = ("added #%d at v1.00 -- -/= if buried"
                                     % idx)
                    elif hit is not None:
                        sel[0] = hit
                    else:
                        status[0] = "nothing there -- right-click to add one"

            state = {"truth": ses.truth, "guess": guess, "refused": refused,
                     "written": ses.written, "verdict": verdicts.has}
            gl = []
            if sel[0] is not None and sel[0] < len(items):
                it = items[sel[0]]
                if not it["manual"] and it["idx"] in usable:
                    n = usable.index(it["idx"])
                    order = np.argsort(-prob[n])[:3]
                    gl = [(model.classes[int(c)], float(prob[n][int(c)]))
                          for c in order]
                recent = [c for c in ses.classes()
                          if c not in [g[0] for g in gl]]
                # EVERY remaining class, not a handful: the panel decides how
                # many fit and the rest are reachable by typing to filter.
                gl += [(c, 0.0) for c in recent]
                pre = (typed[0] or filt[0] or "").lower()
                if pre:
                    # The list becomes the classes that MATCH, so a name is
                    # spelled the way the folder already is. Same rule whether
                    # the text is being typed or is a locked filter.
                    gl = [(c, 0.0) for c in ses.classes()
                          if c.lower().startswith(pre)]
            else:
                sel[0] = None
            hits.clear()
            cv2.imshow(win, render(img, items, radius, sess, sample, state,
                                   sel[0], gl, typed[0], status[0],
                                   i, len(rows), zoom[0], hide[0], hits,
                                   filt[0]))
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            if typed[0] is not None:
                if k == 9 and gl:
                    # TAB completes to the first match. Digits cannot do it --
                    # `22` is a real class name, so a number key while typing
                    # has to mean the character 2.
                    typed[0] = gl[0][0]
                    continue
                if k in (13, 10):
                    name = typed[0].strip().replace(" ", "")
                    # Case-insensitive match against an existing class wins.
                    # Typing "marshmallow" for Marshmallow would otherwise
                    # create a SECOND folder that trains as a different
                    # character, and nothing downstream could tell them apart.
                    for known in ses.classes():
                        if known.lower() == name.lower():
                            name = known
                            break
                    typed[0] = None
                    if name and sel[0] is not None:
                        status[0] = ses.assign((sess, sample, items[sel[0]]["idx"]),
                                               name, img, items[sel[0]], radius)
                    continue
                if k == ord("/"):
                    # LOCK, and it has to be a key no class name contains.
                    # `l` cannot do it here: the branch below appends every
                    # printable character, so typing "mike" then `l` gave
                    # "mikel" rather than a lock. `/` is the key that STARTED
                    # typing, so it reads as a toggle, and no class name has a
                    # slash in it.
                    if typed[0]:
                        filt[0] = typed[0]
                        typed[0] = None
                        status[0] = ("filter LOCKED to %r -- 1-9 work again"
                                     % filt[0])
                    else:
                        typed[0] = None
                        status[0] = "nothing typed to lock"
                    continue
                if k == 27:
                    typed[0] = None
                    continue
                if k == 8:
                    typed[0] = typed[0][:-1]
                    continue
                if 32 <= k < 127:
                    typed[0] += chr(k)
                    continue
            if k == ord("q"):
                cv2.destroyAllWindows()
                print("wrote %d crop(s); %d edge detection(s) recorded as "
                      "junk without one."
                      % (ses.written, len(verdicts.rows)))
                return 0
            if k == ord("n"):
                i += 1
                break
            if k == ord("p"):
                i = max(0, i - 1)
                break
            if k == ord("l"):
                # UNLOCK only. Locking happens with `/` while typing, because
                # this branch is never reached mid-typing: the typing handler
                # appends every printable key first, so `l` there is the letter
                # l -- "mike" + l gave "mikel".
                if filt[0]:
                    filt[0] = ""
                    status[0] = "filter cleared"
                else:
                    status[0] = "no filter set -- type / name, then / to lock"
                continue
            if k == ord("h"):
                hide[0] = not hide[0]
                status[0] = ("junk shrunk to dots" if hide[0]
                             else "showing everything")
                continue
            if k in (ord("["), ord("]")):
                zoom[0] = max(0.5, min(4.0, zoom[0] + (0.2 if k == ord("]")
                                                       else -0.2)))
                status[0] = "zoom %.1fx" % zoom[0]
                continue
            if k == ord("u"):
                status[0] = ses.undo(lambda key: None)
                continue
            if sel[0] is None:
                continue
            it = items[sel[0]]
            key = (sess, sample, it["idx"])
            if k == ord("/"):
                typed[0] = ""
                continue
            if k == ord("j"):
                if _cut(img, it["x"], it["y"], radius) is None:
                    # A half circle at the rim. No crop can be written for it
                    # that would not skew a model -- see `Verdicts` -- so the
                    # verdict is recorded and the crop is not.
                    verdicts.add(key)
                    status[0] = "junk recorded (no crop: edge clips it)"
                else:
                    status[0] = ses.assign(key, JUNK_CLASS, img, it, radius)
                continue
            if k == ord("d"):
                if verdicts.has(key):
                    verdicts.remove(key)
                    status[0] = "junk verdict removed"
                else:
                    status[0] = ses.delete(key)
                continue
            if k == ord("x"):
                if not it["manual"]:
                    status[0] = "that is a real detection, not an added one"
                else:
                    ses.delete(key)
                    verdicts.remove(key)
                    manual.remove(sess, sample, it["idx"])
                    sel[0] = None
                    status[0] = "removed the added circle"
                continue
            if k in (ord("-"), ord("=")) and it["manual"]:
                manual.resize(sess, sample, it["idx"],
                              it["r"] + (2.0 if k == ord("=") else -2.0))
                continue
            if ord("1") <= k <= ord("9"):
                n = k - ord("1")
                if n < len(gl):
                    status[0] = ses.assign(key, gl[n][0], img, it, radius)
                continue
    cv2.destroyAllWindows()
    print("wrote %d crop(s); %d edge detection(s) recorded as junk without "
          "one." % (ses.written, len(verdicts.rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
