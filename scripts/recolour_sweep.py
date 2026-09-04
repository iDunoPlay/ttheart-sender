"""Calibrate `_recolour` against the game's own marks -- the measurement it was left off for.

`Tsum.kind` is a pixel-level k-means cluster id, and it is a coin flip about
identity: over the collected corpus it links 38% of the partners the game
confirms and 24% of the tsums it does not. `_recolour` is the fix already
written for this -- sample each face once, in the middle where lighting is
most even, and merge agglomeratively -- and it ships OFF, for a reason its
docstring states plainly:

    "Tuning `thresh` against labelled links looked like a win -- 80% -> 89% of
    drawn links got the same kind. But labels only record pairs that DO belong
    together, and a positive-only score always improves by merging more, right
    up to 'everything is one character'. Calibrating this needs negative
    examples ... which the current labelling flow does not collect."

The collection flow now does collect them. Holding a tsum makes the game light
up every tsum that is the same character *and* reachable, so per sample:

  * **positives** -- (head, marked) pairs, skipping the 90px glow where a
    reaction means proximity rather than identity. The game's word.
  * **negatives** -- (head, unmarked) tsums near enough that reach is unlikely
    to be the reason. Noisy in one direction only: a same-character tsum that
    was merely unreachable lands here and is scored as a miss, so every `split`
    below is a LOWER bound.

Scored the way `learn.py` scores a palette, deliberately, so the two are
comparable and so the merge trap is closed:

    agreement -- share of positive pairs given the same id
    split     -- share of negative pairs given different ids
    balanced  -- the mean of the two

`balanced` is what a threshold has to beat. Merging everything drives
agreement to 100% and split to 0%, so it cannot win by collapsing the board,
which is exactly how the first attempt at this fooled itself.

It prices the SHIPPED function rather than a copy: `_recolour` is imported and
called. Run it over a collection:

    python scripts/recolour_sweep.py --dir dataset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttheart_sender.game import tsum as T  # noqa: E402


def pairs(row: dict, aura: float, neg_max: float):
    """(positive, negative) member indices for one sample."""
    ts = row["tsums"]
    head = row["head"]
    if head >= len(ts):
        return [], []
    marked = set(row.get("marked") or [])
    if not marked:
        return [], []
    hx, hy = ts[head]["x"], ts[head]["y"]
    pos, neg = [], []
    for i, t in enumerate(ts):
        if i == head:
            continue
        d = math.hypot(t["x"] - hx, t["y"] - hy)
        if d <= aura:
            continue
        if i in marked:
            pos.append(i)
        elif d <= neg_max:
            neg.append(i)
    return pos, neg


def score(kinds, head, pos, neg):
    """(agreement hits, positives, split hits, negatives) for one sample."""
    hk = kinds[head]
    agree = sum(1 for i in pos if kinds[i] == hk)
    split = sum(1 for i in neg if kinds[i] != hk)
    return agree, len(pos), split, len(neg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dataset", type=Path)
    ap.add_argument("--aura", type=float, default=90.0)
    ap.add_argument("--neg-max", type=float, default=260.0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N samples")
    ap.add_argument("--thresholds", default="0,10,15,20,25,30,35,40,50,60,80",
                    help="Lab merge distances to sweep")
    args = ap.parse_args()

    files = sorted(args.dir.glob("*/samples.jsonl"))
    if not files:
        print(f"no sessions under {args.dir}")
        return 1
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]

    # (agree, npos, split, nneg) per rule
    tally = {"k-means (today)": [0, 0, 0, 0]}
    for t in thresholds:
        tally[f"recolour {t:g}"] = [0, 0, 0, 0]

    n = skipped = 0
    t0 = time.perf_counter()
    for f in files:
        folder = f.parent
        for line in f.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pos, neg = pairs(row, args.aura, args.neg_max)
            if not pos or not neg:
                skipped += 1
                continue
            img = cv2.imread(str(folder / ("%04d_before.jpg" % row["index"])))
            if img is None:
                skipped += 1
                continue

            head = row["head"]
            base = [int(t["kind"]) for t in row["tsums"]]
            for k, v in zip(tally["k-means (today)"],
                            score(base, head, pos, neg)):
                pass
            a, np_, s, nn = score(base, head, pos, neg)
            acc = tally["k-means (today)"]
            acc[0] += a; acc[1] += np_; acc[2] += s; acc[3] += nn

            radius = float(row.get("radius", 25.0))
            for t in thresholds:
                # A fresh Tsum list each time: `_recolour` rewrites `kind` in
                # place, so a shared list would carry one threshold's answer
                # into the next.
                ts = [T.Tsum(float(d["x"]), float(d["y"]), float(d["r"]),
                             int(d["kind"]), (0, 0, 0)) for d in row["tsums"]]
                out = T._recolour(img, ts, radius, t) if t > 0 else ts
                kinds = [x.kind for x in out]
                a, np_, s, nn = score(kinds, head, pos, neg)
                acc = tally[f"recolour {t:g}"]
                acc[0] += a; acc[1] += np_; acc[2] += s; acc[3] += nn

            n += 1
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break

    print(f"{n} samples over {len(files)} sessions ({skipped} unusable), "
          f"{time.perf_counter() - t0:.0f}s")
    print(f"positives {tally['k-means (today)'][1]}  "
          f"negatives {tally['k-means (today)'][3]}")
    print()
    print("agreement -- positive pairs given the same id (the game says same character)")
    print("split     -- negative pairs given different ids (lower bound: an unmarked")
    print("             tsum may be the same character and merely unreachable)")
    print("balanced  -- the mean. Merging the whole board scores 100/0, so it cannot win.")
    print()
    print(f"{'rule':>20}{'agreement':>11}{'split':>9}{'balanced':>10}")
    best = None
    for name, (a, npos, s, nneg) in tally.items():
        ag = a / max(npos, 1)
        sp = s / max(nneg, 1)
        bal = (ag + sp) / 2
        star = ""
        if name != "k-means (today)":
            base_bal = ((tally["k-means (today)"][0] / max(npos, 1)
                         + tally["k-means (today)"][2] / max(nneg, 1)) / 2)
            if best is None or bal > best[1]:
                best = (name, bal)
            star = "  <-- beats k-means" if bal > base_bal else ""
        print(f"{name:>20}{ag:10.1%}{sp:9.1%}{bal:10.1%}{star}")

    if best:
        base_bal = ((tally["k-means (today)"][0] / max(tally["k-means (today)"][1], 1)
                     + tally["k-means (today)"][2] / max(tally["k-means (today)"][3], 1)) / 2)
        print()
        print(f"best: {best[0]} at {best[1]:.1%} balanced, against "
              f"{base_bal:.1%} for k-means "
              f"({best[1] - base_bal:+.1%})")
        print("A win here is an OFFLINE win on identity. It says nothing about")
        print("clears: a better id changes which chains exist, and only a played")
        print("round prices that. See docs/IMPROVEMENT-LOOP.md step 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
