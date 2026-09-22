"""Resolve crops that are filed under two classes at once.

The same picture in two folders trains as two characters at once, and nothing
downstream can tell which was meant. `retrain.py --survey-only` finds them;
this settles them.

    python scripts/dedupe.py            # say what would happen
    python scripts/dedupe.py --apply    # do it

**Nothing is deleted.** Losing copies are MOVED to `crops/duplicates/<class>/`,
so a wrong call costs a `move` back rather than a re-labelling session.

Two cases, and only one of them can be decided by rule
------------------------------------------------------

**A negative against a character** -- `junk`/`board`/`score` on one side and a
character on the other. **The negative wins**, and the asymmetry is the reason:
a negative is checked against the game at train time. `reject_net.py` rescues
anything in a negative folder that the game MARKED or CLEARED and counts it back
as a tsum. So a wrong junk call self-corrects and a wrong character label does
not, which makes keeping the negative the side with a safety net under it.

**A character against a character** -- `22` against `Sulley`, `CheshireCat`
against `Sebastian`. **Neither can be trusted and both are moved out.** There is
no rule here: the file name says which detection it is and nothing says which
name is right. Deleting one at random would leave a label that is wrong half the
time; keeping both guarantees the classifier is taught a contradiction. Removing
both costs one crop of training data and leaves the detection free to be
labelled again in `label_board.py`.

Timestamps are not used, and were checked before being rejected: most of these
pairs carry the *same* mtime to the second, because a move on one volume
preserves it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain import NOT_CHARACTERS, duplicates  # noqa: E402


def resolve(dupes, root: Path):
    """(name, keep_or_None, [classes to move out], reason) per duplicate."""
    out = []
    for name, where in sorted(dupes.items()):
        negatives = [c for c in where if c in NOT_CHARACTERS]
        characters = [c for c in where if c not in NOT_CHARACTERS]
        if len(negatives) == 1 and characters:
            out.append((name, negatives[0], characters,
                        "negative wins -- it is re-checked against the game"))
        elif not characters:
            # Two negative folders. Either is a negative, so keep one and be
            # done: the reject model sees the same label from both.
            out.append((name, sorted(negatives)[0], sorted(negatives)[1:],
                        "both are negatives -- one is enough"))
        else:
            out.append((name, None, sorted(where),
                        "two character labels -- no rule can say which"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--out", type=Path, default=Path("crops/duplicates"),
                    help="where losing copies go. Moved, never deleted")
    ap.add_argument("--apply", action="store_true",
                    help="actually move them. Without this it only reports")
    args = ap.parse_args()

    dupes = duplicates(args.crops)
    if not dupes:
        print("no crop is filed under two classes.")
        return 0

    plan = resolve(dupes, args.crops)
    kept = Counter()
    moved = 0
    print("%d crop(s) filed under two or more classes\n" % len(plan))
    for name, keep, drop, why in plan:
        print("  %s" % name)
        print("     keep  %s" % (keep or "-- NOTHING, both are moved out"))
        print("     move  %s" % ", ".join(drop))
        print("     why   %s" % why)
        kept[keep or "(none)"] += 1
        for cls in drop:
            src = args.crops / cls / name
            if not src.exists():
                continue
            moved += 1
            if args.apply:
                dest = args.out / cls
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest / name))
    print()
    for k, n in kept.most_common():
        print("  kept in %-18s %d" % (k, n))
    if args.apply:
        print("\nmoved %d file(s) to %s -- nothing was deleted." % (moved, args.out))
        print("To undo one: move it back into crops/labelled/<class>/")
        # Taking contradictory crops out of a class can drop it under the
        # training cut, and that is worth saying rather than discovering at the
        # next retrain. `22` lost 14 of its 51 crops here -- over a quarter of
        # the class was also filed as Sulley, which is most of why
        # `22 -> Sulley` sat at the top of the confusion list.
        from retrain import counts
        touched = {c for _, _, drop, _ in plan for c in drop}
        thin = sorted((n, k) for k, n in counts(args.crops).items()
                      if 0 < n < 20 and k in touched)
        if thin:
            print("\n  These lost crops and are now UNDER the 20-crop training"
                  "\n  cut. They need re-labelling, not just a retrain:")
            for n, k in thin:
                print("    %-18s %3d crops   (+%d to be trainable again)"
                      % (k, n, 20 - n))
    else:
        print("\n%d file(s) WOULD move. Nothing changed -- pass --apply."
              % moved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
