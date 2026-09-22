"""How many colour groups should a board be sorted into?

THE GAME SUPPLIES THE ANSWER AND IT IS A HARD ONE. Every round holds the
equipped tsum plus four random others -- **five characters** -- and an in-game
item cuts that to **four, the base included**. Nothing about the board is
open-ended, so identity is not "name a character out of sixty" but "sort these
forty tsums into five piles", which needs no labels, works on characters nobody
has trained, and works on a base tsum the classifier has never seen.

WHAT THE DETECTOR ACTUALLY DOES. Over 5,528 stored boards, per-frame k-means
lands on 7 or 8 groups:

    7 x1366   8 x1342   6 x927   9 x831   5 x484   10 x266   4 x185

So the average character is split across about one and a half groups before any
model runs, and `find_chains` groups by `kind` -- half of every character is
already unreachable from the other half. That is the same finding as the base
cluster holding a median 13% of a board on which Beast is guaranteed.

WHY THIS SCRIPT AND NOT `group_eval.py`. That one scores in simulated tsums
cleared per drag, and the twenty-sixth round disqualified the metric when an
ORACLE grouping scored BELOW the shipped rule on it. Its table -- groups=4 best
at +4.6%, groups=5 at +2.4% -- has never been re-checked with anything better.

This scores the way `naming_gain.py` does: against the game's own marks, which
say which tsums really are one character and reachable.

    python scripts/group_probe.py
    python scripts/group_probe.py --groups 0,3,4,5,6,7 --samples 600

`0` is the shipped rule -- whatever per-frame k-means produced, read straight
out of the corpus. Everything else re-sorts the SAME detections with
`tsum._regroup`, which only rewrites `kind`, so every index still lines up with
the marks it is scored against.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics as st
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ttheart_sender.game import tsum as T  # noqa: E402


def _score():
    """`naming_gain.score`, so the numbers sit beside its table unchanged."""
    spec = importlib.util.spec_from_file_location(
        "naming_gain", HERE / "naming_gain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score


def boards(root: Path, want: int, seed: int):
    rows = []
    for jl in sorted(root.glob("*/samples.jsonl")):
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            o = d.get("options") or {}
            # A round that already forced a group count, or renamed by model,
            # has no baseline to compare against.
            if o.get("character") or o.get("recolour") or o.get("kinds"):
                continue
            head = d.get("head")
            if (not d.get("tsums") or not d.get("marked")
                    or head is None or not (0 <= head < len(d["tsums"]))):
                continue
            img = jl.parent / ("%04d_before.jpg" % d.get("index", -1))
            if img.exists():
                rows.append((d, img))
    random.Random(seed).shuffle(rows)
    return rows[:want] if want else rows


def reachable(tsums, kinds, head, radius, link_px, block, cap):
    """The tsums the round could actually put in a chain from `head`.

    `find_chains` does not offer the whole same-kind group -- it walks
    `adjacency`, which refuses a pair with another tsum sitting between them,
    and then takes the longest PATH and truncates it. Scoring the group instead
    charges a grouping rule for same-character tsums buried across the board,
    which the round would never have reached either. That bias grows with group
    size, so it lands hardest on exactly the rule under test.
    """
    for t, k in zip(tsums, kinds):
        t.kind = k
    adj = T.adjacency(tsums, radius, block=block, link_px=link_px)
    seen, stack = {head}, [head]
    while stack:
        for j in adj[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    seen.discard(head)
    # The chain is capped, and a longer reachable set than the cap is not a
    # longer chain. Nearest first, which is the order `orient_chain` favours.
    out = sorted(seen, key=lambda i: (tsums[i].x - tsums[head].x) ** 2
                 + (tsums[i].y - tsums[head].y) ** 2)
    return out[:max(0, cap - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="0,3,4,5,6,7",
                    help="group counts to try. 0 is the shipped per-frame "
                         "k-means, read out of the corpus unchanged")
    ap.add_argument("--raw", action="store_true",
                    help="score the whole same-kind GROUP instead of the chain "
                         "the round could actually drag. Biased against big "
                         "groups -- see `reachable`")
    ap.add_argument("--link-px", type=float, default=105.0)
    ap.add_argument("--block", type=float, default=1.25)
    ap.add_argument("--max-chain", type=int, default=12)
    args = ap.parse_args()

    score = _score()
    want = [int(g) for g in args.groups.split(",") if g.strip()]
    rows = boards(args.dir, args.samples, args.seed)
    if not rows:
        print("no drags with marks were collected under the shipped rule")
        return 2

    recall = {g: [] for g in want}
    prec = {g: [] for g in want}
    offered = {g: [] for g in want}
    run = {g: [] for g in want}
    groups_made = {g: [] for g in want}
    n = 0

    for d, img_path in rows:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        radius = float(d["radius"])
        head = int(d["head"])
        raw = d["tsums"]
        truth = {int(i) for i in d["marked"]
                 if 0 <= int(i) < len(raw) and int(i) != head}
        if not truth:
            continue
        n += 1
        for g in want:
            if g == 0:
                kinds = [int(t["kind"]) for t in raw]
            else:
                # Fresh Tsum objects each time: `_regroup` writes `kind` in
                # place, and one shared list would carry the last count's
                # answer into the next.
                ts = [T.Tsum(x=float(t["x"]), y=float(t["y"]),
                             r=float(t["r"]), kind=0, colour=(0, 0, 0))
                      for t in raw]
                kinds = [t.kind for t in T._regroup(frame, ts, radius, g)]
            groups_made[g].append(len(set(kinds)))
            if args.raw:
                got = score(kinds, head, truth)
                if got is None:
                    continue
                recall[g].append(got[0])
                prec[g].append(got[1])
                offered[g].append(got[2])
                continue
            ts = [T.Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                         kind=0, colour=(0, 0, 0)) for t in raw]
            order = reachable(ts, kinds, head, radius, args.link_px,
                              args.block, args.max_chain)
            same = set(order)
            hit = len(same & truth)
            recall[g].append(hit / len(truth))
            prec[g].append(hit / len(same) if same else 1.0)
            offered[g].append(len(same))
            # The run of correct members before the first wrong one. See the
            # module docstring: the game refuses 86% of what follows a refusal,
            # so a chain is worth its leading run and not its length.
            good = 0
            for i in order:
                if i not in truth:
                    break
                good += 1
            run[g].append(good)

    if not n:
        print("no scorable drags")
        return 2

    print("%d drags with marks, all collected under the shipped rule" % n)
    print("the round holds 5 characters -- 4 with the in-game item")
    print("scoring: %s"
          % ("the whole same-kind group (biased against big groups)"
             if args.raw else
             "the chain the round could drag -- adjacency from the head, "
             "capped at %d" % args.max_chain))
    print()
    print("  groups   made   recall   precision   offered   ACCEPTED RUN")
    base = st.mean(recall[want[0]]) if recall.get(want[0]) else 0.0
    for g in want:
        if not recall[g]:
            continue
        label = "k-means" if g == 0 else str(g)
        base_run = st.mean(run[want[0]]) if run.get(want[0]) else 0.0
        print("  %-7s %5.1f   %5.1f%%      %5.1f%%   %6.2f   %6.2f%s"
              % (label, st.mean(groups_made[g]), 100 * st.mean(recall[g]),
                 100 * st.mean(prec[g]), st.mean(offered[g]),
                 st.mean(run[g]),
                 "" if g == want[0]
                 else "  %+.0f%%" % (100 * (st.mean(run[g]) / base_run - 1)
                                     if base_run else 0.0)))

    # Paired against the shipped rule, on the same drag: the boards differ far
    # more from each other than the rules differ on one board.
    if want[0] == 0:
        print()
        for g in want[1:]:
            if not recall[g]:
                continue
            d_r = [a - b for a, b in zip(run[g], run[0])]
            better = sum(1 for v in d_r if v > 1e-9)
            worse = sum(1 for v in d_r if v < -1e-9)
            print("  %d groups vs k-means, on the ACCEPTED RUN: better on "
                  "%d, worse on %d, unchanged on %d"
                  % (g, better, worse, len(d_r) - better - worse))
    print()
    print("READ THE LAST COLUMN. Recall up with precision down is not a")
    print("verdict -- a bigger group offers more partners and more strangers")
    print("at once. The game breaks the tie: it accepts 97.9% of first members")
    print("and 19.1% of sevenths, and once it refuses one it refuses 86% of")
    print("what follows. So what a chain is worth is the RUN of correct")
    print("members before the first wrong one, which is the last column.")
    print("Only a played round settles what that is worth in score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
