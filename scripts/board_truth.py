"""Three questions the saved dataset can answer without playing a round.

The bot reports a `cleared %` and it reads 80%+. That number is a real
measurement -- `cleared_by_drag` looks inside each dragged tsum's own disk and
asks whether it changed more than the board's idle noise -- but it is a proxy
for "this tsum popped", and there is one known way for a proxy like that to
read high. When the front of a chain clears, the tsums above the gap **fall**,
and a tsum that moved changes its disk exactly like one that popped. So a
refused member sitting above a cleared one can be counted as cleared.

That artefact is visible in the shape of the answer, which is why this script
exists. The game refuses a chain at its first bad link, so a truthful reading
is always a **prefix**: the first N members go and the rest stay. A reading
with a **hole** -- member 3 stayed but member 5 went -- cannot be the game
refusing, because member 5 was only reachable through member 3. Holes are the
artefact, and counting them bounds how inflated the headline is.

The second and third questions come from a rule of the game itself: a board
holds at most **5 different characters** (4 if an item is used). That is a
much smaller number than it sounds, and it turns two open-ended problems into
closed ones.

* **Over-splitting is measurable without labels.** If detection reports 9
  distinct `kind`s on a settled board, at least 4 of them are the same
  character split in two, and the bot will never chain across the split. No
  ground truth needed -- the count alone is the error.

* **Naming is the wrong goal; consistency is the right one.** The classifier
  knows 15 characters, and the tsum most common on the board is usually the
  equipped one, which may not be among them. But an unknown character called
  the same wrong name every time still chains with itself perfectly. It only
  costs a round when it is called two different names, or when it shares a
  name with a second character that is actually on the board. So the thing to
  measure is not accuracy against a label we do not have -- it is agreement
  with the game's own marks, which we do have on every sample.

Run:  python scripts/board_truth.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CHARACTER_KIND = 1000


def rows(root: Path):
    """Every sample, tagged with the session it came from."""
    for d in sorted(root.glob("*/")):
        f = d / "samples.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["_session"] = d.name
            yield r


def pct(a, b):
    return f"{a / b:.1%}" if b else "n/a"


# --------------------------------------------------------------- question 1

def clears_are_prefixes(data):
    """Is `cleared` the front of `dragged`, as a refusal would make it?"""
    full = prefix = hole = nothing = 0
    dragged_n = cleared_n = 0
    hole_lost = 0          # members counted as cleared that sit past a gap
    for r in data:
        d, c = r.get("dragged"), r.get("cleared")
        if not d or c is None:
            continue
        dragged_n += len(d)
        cleared_n += len(c)
        pos = [d.index(i) for i in c if i in d]
        if not pos:
            nothing += 1
        elif len(c) == len(d):
            full += 1
        elif pos == list(range(len(pos))):
            prefix += 1
        else:
            hole += 1
            # everything after the first gap is suspect
            first_gap = next(i for i, p in enumerate(pos) if p != i)
            hole_lost += len(pos) - first_gap

    n = full + prefix + hole + nothing
    print("1. IS THE CLEARED % HONEST?")
    print(f"   {n} verified drags, {dragged_n} tsums attempted, "
          f"{cleared_n} read as cleared ({pct(cleared_n, dragged_n)})\n")
    print(f"   whole chain cleared   {full:>5}  ({pct(full, n)})")
    print(f"   clean prefix cleared  {prefix:>5}  ({pct(prefix, n)})   <- a refusal, as expected")
    print(f"   nothing cleared       {nothing:>5}  ({pct(nothing, n)})")
    print(f"   HOLE in the middle    {hole:>5}  ({pct(hole, n)})   <- impossible; falling tsums")
    if cleared_n:
        lo = cleared_n - hole_lost
        print(f"\n   {hole_lost} of the {cleared_n} cleared readings sit past a hole.")
        print(f"   So the true clear rate is between {pct(lo, dragged_n)} "
              f"and {pct(cleared_n, dragged_n)}.")
    print()


# --------------------------------------------------------------- question 2

def kinds_per_board(data, min_tsums):
    """A board holds <=5 characters. How many does detection think it sees?"""
    plain, named = Counter(), Counter()
    for r in data:
        ts = r.get("tsums") or []
        if len(ts) < min_tsums:
            continue
        ks = {t.get("kind") for t in ts}
        (named if any(k >= CHARACTER_KIND for k in ks) else plain)[len(ks)] += 1

    print(f"2. HOW MANY IDENTITIES DOES IT SEE?  (boards of {min_tsums}+ tsums)")
    print("   The game allows at most 5 (4 with an item). More than that is")
    print("   one character split across several kinds, which cannot chain.\n")
    for tag, c in (("colour clustering only", plain), ("character model on", named)):
        if not c:
            continue
        tot = sum(c.values())
        mean = sum(k * v for k, v in c.items()) / tot
        over = sum(v for k, v in c.items() if k > 5)
        print(f"   {tag}: {tot} boards, mean {mean:.1f} kinds, "
              f"{pct(over, tot)} see more than 5")
        print("     " + "  ".join(f"{k}:{v}" for k, v in sorted(c.items())))
    print()


# --------------------------------------------------------------- question 3

def agreement_with_marks(data):
    """The game marks what it will link. Does our `kind` agree?

    This is the measurement that does not need a label, and so is the one that
    works for a character nobody has trained. `marked` is the game's own
    answer: same character *and* reachable. If our id agrees with it, the
    chain is real whatever the character is called.

    Unmarked is NOT a clean negative -- a tsum can be the same character and
    simply out of reach -- so this reports only the agreement side.
    """
    def score(sel):
        agree = tot = 0
        for r in data:
            ts, head, marked = r.get("tsums"), r.get("head"), r.get("marked")
            if not ts or head is None or not marked or head >= len(ts):
                continue
            hk = ts[head].get("kind")
            if not sel(hk):
                continue
            for i in marked:
                if i == head or i >= len(ts):
                    continue
                tot += 1
                agree += ts[i].get("kind") == hk
        return agree, tot

    named_a, named_t = score(lambda k: k >= CHARACTER_KIND)
    plain_a, plain_t = score(lambda k: k < CHARACTER_KIND)

    print("3. DOES OUR ID AGREE WITH THE GAME?")
    print("   Of the tsums the game lit up when we pressed one, how many did")
    print("   we give the same id? No labels needed -- works for any character.\n")
    print(f"   head named by the model   {named_a:>5}/{named_t:<6} {pct(named_a, named_t)}")
    print(f"   head from colour cluster  {plain_a:>5}/{plain_t:<6} {pct(plain_a, plain_t)}")
    print("\n   Higher is better. This is the number the whole exercise moves.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--min-tsums", type=int, default=30,
                    help="a board still filling has fewer identities than it will")
    args = ap.parse_args()

    data = list(rows(args.dir))
    if not data:
        print(f"no samples under {args.dir}")
        return 1
    sess = len({r["_session"] for r in data})
    print(f"{len(data)} samples over {sess} sessions\n" + "=" * 66 + "\n")

    clears_are_prefixes(data)
    kinds_per_board(data, args.min_tsums)
    agreement_with_marks(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
