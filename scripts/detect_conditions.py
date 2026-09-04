"""Score detection separately in the conditions that are not alike.

`sweep_detect.py` scores a setting over a whole collection. That pools two
boards the detector does not see alike -- normal play, and FEVER, which
repaints the pile in neon and animates it -- and a setting that helps one can
be paid for by the other without either showing up.

It also cannot test the two switches that decide how much of the board is
looked at in the first place:

* ``include_dark`` -- a black-faced tsum (classic Mickey) and every tsum's
  outline are the same colour, so the dark clusters are either processed under
  their own rules or thrown away whole. `detect`'s own docstring calls the
  pass "best-effort, not solved" and records up to 13 phantoms on one real
  frame, with real black-faced detections never outnumbering false ones.
* ``bowl_reject`` -- drops a detection whose face colour sits too near the
  board's. During FEVER the board is repainted and everything washes toward
  the same neon, so the same threshold should throw away more real tsums
  there than in normal play. That is a hypothesis this script can settle.

The label is the game's, as everywhere else here: a tsum the game lit up while
another was held is a real tsum at a real position. So the question asked per
setting is **what share of the game-confirmed tsums does it still find**,
reported beside detections per board, because this corpus cannot tell a newly
found tsum from a phantom -- only a lost one from a kept one.

Biased in the usual direction and worth restating: the confirmed positions come
from the run that was played, so the live setting scores near 100% for free.
**This catches a regression; it cannot prove an improvement.** What it can do
that nothing else here does is show a setting costing recall in one condition
and paying it back in the other.

    python scripts/detect_conditions.py --dir dataset --samples 150
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttheart_sender.game import tsum as T  # noqa: E402


def load(root: Path, limit: int, seed: int):
    rows = []
    for f in sorted(root.glob("*/samples.jsonl")):
        folder = f.parent
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (r.get("marked") or []):
                continue
            r["_img"] = folder / ("%04d_before.jpg" % r["index"])
            rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def confirmed(row: dict) -> list[tuple[float, float]]:
    """Positions the game itself confirmed hold a tsum."""
    ts = row["tsums"]
    return [(ts[i]["x"], ts[i]["y"]) for i in row["marked"] if i < len(ts)]


def recall(found, truth, tol: float) -> tuple[int, int]:
    """How many confirmed positions have a detection on them."""
    if not truth:
        return 0, 0
    hit = 0
    used = set()
    for tx, ty in truth:
        best, bd = None, tol * tol
        for j, t in enumerate(found):
            if j in used:
                continue
            d = (t.x - tx) ** 2 + (t.y - ty) ** 2
            if d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            hit += 1
    return hit, len(truth)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dataset", type=Path)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-effort", type=int, default=3,
                    help="pass the level the round was played at")
    ap.add_argument("--tol", type=float, default=0.6,
                    help="match distance, in detected radii")
    args = ap.parse_args()

    rows = load(args.dir, args.samples, args.seed)
    if not rows:
        print(f"no usable samples under {args.dir}")
        return 1

    # (include_dark, bowl_reject) -- the live setting first.
    combos = [(True, 40.0), (False, 40.0), (True, 0.0), (False, 0.0)]
    # hits, truth, detections, frames -- per condition per combo
    acc = {(c, cond): [0, 0, 0, 0] for c in combos for cond in ("normal", "FEVER")}

    t0 = time.perf_counter()
    for n, r in enumerate(rows, 1):
        img = cv2.imread(str(r["_img"]))
        if img is None:
            continue
        truth = confirmed(r)
        radius = float(r.get("radius", 25.0))
        cond = "FEVER" if r.get("fever") else "normal"
        o = r["options"]
        for combo in combos:
            dark, bowl = combo
            found, _, _ = T.detect(img, k=int(o.get("k", 12)), radius=radius,
                                   include_dark=dark, bowl_reject=bowl,
                                   merge=bool(o.get("merge", False)),
                                   fit_effort=args.fit_effort)
            hit, tot = recall(found, truth, args.tol * radius)
            a = acc[(combo, cond)]
            a[0] += hit; a[1] += tot; a[2] += len(found); a[3] += 1
        if n % 25 == 0:
            print(f"  ...{n}/{len(rows)}", file=sys.stderr)

    print(f"{len(rows)} samples, fit_effort {args.fit_effort}, "
          f"{time.perf_counter() - t0:.0f}s\n")
    print("`kept` = share of the tsums the GAME confirmed that this setting still")
    print("finds. The live setting is first and scores near 100% by construction --")
    print("read the DIFFERENCES between rows, and read the two conditions apart.\n")
    print(f"{'include_dark':>13}{'bowl_reject':>13}"
          f"{'normal kept':>13}{'found':>8}{'FEVER kept':>12}{'found':>8}")
    for combo in combos:
        dark, bowl = combo
        cells = ""
        for cond in ("normal", "FEVER"):
            hit, tot, det, fr = acc[(combo, cond)]
            cells += f"{hit / max(tot, 1):12.1%}{det / max(fr, 1):8.1f}"
        live = "  <- live" if combo == (True, 40.0) else ""
        print(f"{str(dark):>13}{bowl:>13.0f}{cells}{live}")

    print()
    print("A row that keeps more of the confirmed tsums AND finds more per board")
    print("is finding real ones. A row that keeps more while finding far more may")
    print("just be finding phantoms -- this corpus cannot tell those apart, which")
    print("is why `found` is printed and why only a played round decides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
