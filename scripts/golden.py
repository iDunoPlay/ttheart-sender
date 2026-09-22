"""Freeze a test set, once, so "did it improve?" has an answer.

Every model so far has been scored on its own held-out split. Two models
trained with different splits cannot be compared that way at all -- the
retrained classifier read 97.9% against the shipped 94.9% and was, on a common
set, **exactly as good**. The difference was the split, not the model.

A golden set fixes that: a list of SESSIONS, frozen to a file, never trained
on, and used to score every candidate. It is a list of session names rather
than a copy of the crops, because a crop can be relabelled and a relabelling
should reach the golden set too -- what must not change is *which boards* are
being asked about.

    python scripts/golden.py freeze --share 0.25
    python scripts/golden.py status

Once frozen it is not re-frozen casually. `freeze` refuses to overwrite an
existing set without `--force`, and prints what would change if it did.

Held out by SESSION, never by crop: boards inside one round are near
duplicates and the same physical tsum appears in consecutive frames, so a
random split over crops reports memory rather than skill.

**Every class that can be scored should be scored.** The chooser walks classes
from rarest to commonest and gives each one at least one session, before
topping up to the requested share. A class whose every session lands in the
training half is invisible to the golden set forever, which is how Cleo came to
be measured at 0% having never been taught.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT = Path("models/golden.json")
NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2,})_v(?P<vis>[\d.]+)$")
NOT_CHARACTERS = {"board", "score", "junk"}


def crops_by_session(root: Path):
    """session -> Counter of class -> crops, over the labelled tree."""
    out = defaultdict(Counter)
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir() or d.name in NOT_CHARACTERS:
            continue
        for p in d.glob("*.png"):
            m = NAME.match(p.stem)
            if m:
                out[m.group("sess")][d.name] += 1
    return out


def choose(by_session, share: float, seed: int):
    """The sessions to freeze: every class represented, then topped up.

    Rarest class first, because a class with two sessions has to be handled
    before a class with eighty takes the sessions it needed.
    """
    totals = Counter()
    for c in by_session.values():
        totals.update(c)
    rng = random.Random(seed)
    chosen, covered = set(), Counter()

    for cls, _ in sorted(totals.items(), key=lambda kv: kv[1]):
        if covered[cls]:
            continue
        have = [s for s, c in by_session.items() if c.get(cls)]
        if len(have) < 2:
            # Only one session has this class. Freezing it would leave the
            # class untrainable, which is worse than leaving it unscored.
            continue
        pick = rng.choice(sorted(s for s in have if s not in chosen)
                          or sorted(have))
        chosen.add(pick)
        covered.update(by_session[pick])

    want = max(1, int(len(by_session) * share))
    rest = sorted(s for s in by_session if s not in chosen)
    rng.shuffle(rest)
    while len(chosen) < want and rest:
        chosen.add(rest.pop())
    return sorted(chosen), totals, covered


def report(path: Path, root: Path):
    if not path.exists():
        print("no golden set at %s -- run `golden.py freeze`" % path)
        return 2
    g = json.loads(path.read_text(encoding="utf-8"))
    by_session = crops_by_session(root)
    sessions = set(g["sessions"])
    have = Counter()
    for s in sessions:
        have.update(by_session.get(s, {}))
    totals = Counter()
    for c in by_session.values():
        totals.update(c)
    print("golden set %s  frozen %s" % (path, g.get("frozen", "?")))
    print("  %d session(s), %d crop(s) today, %d class(es) scoreable"
          % (len(sessions), sum(have.values()), len(have)))
    missing = [c for c in totals if not have.get(c)]
    if missing:
        print("  %d class(es) have NO golden crops and cannot be scored: %s"
              % (len(missing), ", ".join(sorted(missing))))
    thin = sorted((n, c) for c, n in have.items() if n < 5)
    if thin:
        print("  thin in the golden set (under 5 crops): "
              + ", ".join("%s %d" % (c, n) for n, c in thin))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="status",
                    choices=["freeze", "status"])
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--share", type=float, default=0.25,
                    help="share of sessions to freeze, after every class has "
                         "been given one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="re-freeze over an existing set. Every number ever "
                         "reported against the old one becomes incomparable, "
                         "so this is deliberately awkward")
    args = ap.parse_args()

    if args.mode == "status":
        return report(args.out, args.crops)

    by_session = crops_by_session(args.crops)
    if not by_session:
        print("no labelled crops under %s" % args.crops)
        return 2
    sessions, totals, covered = choose(by_session, args.share, args.seed)

    if args.out.exists() and not args.force:
        old = set(json.loads(args.out.read_text(encoding="utf-8"))["sessions"])
        print("a golden set already exists at %s (%d sessions)."
              % (args.out, len(old)))
        print("  it would change by +%d / -%d sessions."
              % (len(set(sessions) - old), len(old - set(sessions))))
        print("  Re-freezing makes every number ever reported against the old "
              "set incomparable.\n  Pass --force if that is what you mean.")
        return 1

    import datetime
    body = {
        "frozen": datetime.date.today().isoformat(),
        "seed": args.seed,
        "share": args.share,
        "sessions": sessions,
        "note": ("Sessions held out of training FOREVER, so candidate models "
                 "can be compared on identical boards. Held out by session, "
                 "not by crop: boards inside one round are near duplicates."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    have = Counter()
    for s in sessions:
        have.update(by_session[s])
    print("froze %d session(s), %d crop(s), %d class(es) -> %s"
          % (len(sessions), sum(have.values()), len(have), args.out))
    unscoreable = [c for c in totals if not have.get(c)]
    if unscoreable:
        print("  %d class(es) cannot be scored by it -- they appear in only "
              "one session, and freezing that session would leave them "
              "untrainable: %s" % (len(unscoreable), ", ".join(sorted(unscoreable))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
