"""Can the same physical tsum be matched across two frames? And is it worth it?

The multi-frame idea is: a tsum that is buried now may be well visible a
moment later, so track it and identify it from its best observation. Before
building any of that, two things have to be true, and this measures both on
data already on disk.

**There is only one pair of consecutive frames in this corpus**, and it is not
the one the idea assumes. `dataset/*/NNNN_before.jpg` and `NNNN_marked.jpg` are
the same board about 0.25s apart, taken either side of a hold. Consecutive
SAMPLES are 3.87s apart at the median and never closer than 1.98s, with a
round's worth of clearing in between, so they are not a sequence at all.

Both sides of every comparison are detected by THIS pipeline, on the round's
own recorded options:

    control : re-detect `before`  vs `before`  -> the detector's own noise
    test    : detect    `marked`  vs `before`  -> noise + 0.25s of motion

Comparing against the detections the bot stored at play time was tried first
and is wrong: the loop caches its palette across frames, so that measures the
difference between two pipelines rather than motion. It pinned the control at
72% and biased every radius downward.

The control comes out at 100% and 0.00px, which is itself worth stating: given
one frame and one palette the detector is deterministic, so everything the
test moves is the frame's content changing and none of it is seed noise.

**The `marked` frame is contaminated by the game's own highlight**, which is
painted over the tsums the game marked. `samples.jsonl` records exactly which
those are, so every appearance-sensitive number here is computed on the
UNMARKED tsums only, and the marked ones are reported separately as the size
of the contamination rather than quietly averaged in.

    python scripts/track_probe.py --samples 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game import tsum as T  # noqa: E402


def samples(root: Path, limit: int, seed: int):
    """(session, row, before, marked) for a random spread of samples."""
    rows = []
    for jl in sorted(root.glob("*/samples.jsonl")):
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            i = d.get("index")
            b = jl.parent / f"{i:04d}_before.jpg"
            m = jl.parent / f"{i:04d}_marked.jpg"
            if d.get("tsums") and b.exists() and m.exists():
                rows.append((jl.parent.name, d, b, m))
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def match(stored, found, tol):
    """Nearest-neighbour match stored -> found, and how ambiguous it was.

    Returns (pairs, ambiguous). A match is ambiguous when the runner-up is
    within 1.5x the winner's distance: that is the case a tracker would get
    wrong, and counting it is the difference between "matching works" and
    "matching works where it was never going to be hard".
    """
    pairs, ambiguous = [], 0
    if not len(found):
        return pairs, ambiguous
    for i, (sx, sy) in enumerate(stored):
        d = np.hypot(found[:, 0] - sx, found[:, 1] - sy)
        order = np.argsort(d)
        best = int(order[0])
        if d[best] > tol:
            continue
        if len(order) > 1 and d[order[1]] < d[best] * 1.5:
            ambiguous += 1
        pairs.append((i, best, float(d[best])))
    return pairs, ambiguous


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", type=int, default=3,
                    help="fit_effort. 3 is what plays; 1 is the code default "
                         "and is far too unstable to measure motion through")
    args = ap.parse_args()

    rows = samples(args.dir, args.samples, args.seed)
    if not rows:
        print("no samples with both frames found")
        return 2
    print("%d samples from %d sessions, fit_effort=%d"
          % (len(rows), len({r[0] for r in rows}), args.effort))

    ctrl_d, test_d = [], []
    ctrl_hit, test_hit, n_stored = 0, 0, 0
    ctrl_amb, test_amb = 0, 0
    dv_all, dv_unmarked = [], []
    rose, rose_n = 0, 0
    marked_share = []
    near_stats = [0, 0]
    away_stats = [0, 0]
    t_detect = []

    for sess, d, bp, mp in rows:
        radius = float(d.get("radius") or 25.0)
        stored = np.array([[t["x"], t["y"]] for t in d["tsums"]], np.float64)
        stored_r = np.array([t["r"] for t in d["tsums"]], np.float64)
        marked = set(d.get("marked") or [])
        marked_share.append(len(marked) / max(1, len(stored)))
        tol = radius * 1.0

        # ONE palette, fitted on the first frame and reused for the second --
        # which is what the play loop does, and without it the comparison is
        # swamped by k-means seed noise: a bare re-detect of the SAME frame
        # recovers only 72% of its own stored detections. The palette is the
        # expensive half of detection and the loop refits it about five times
        # a round, not once a frame.
        first = cv2.imread(str(bp))
        second = cv2.imread(str(mp))
        if first is None or second is None:
            continue
        # The round's OWN detection settings, not the function defaults. This
        # matters more than it looks: rounds run `include_dark=True`, and the
        # default False drops every black tsum, which alone cost 28% of the
        # control's match rate before it was noticed.
        o = d.get("options") or {}
        kw = dict(k=int(o.get("k", 12)),
                  include_dark=bool(o.get("include_dark", False)),
                  merge=bool(o.get("merge", False)),
                  bowl_reject=float(o.get("bowl_reject", 0.0) or 0.0),
                  kinds=int(o.get("kinds", 0) or 0),
                  recolour=float(o.get("recolour", 0.0) or 0.0),
                  scale=float(o.get("scale", 1.0) or 1.0),
                  fit_effort=args.effort)
        # BOTH sides of every comparison are detected by this same offline
        # pipeline. Comparing a play-time detection against an offline one
        # measures the difference between two pipelines, not motion: it held
        # the "control" at 72% and biased every radius, because the stored
        # rows were produced by a loop that caches its palette across frames.
        def _run(img, pal):
            t0 = time.perf_counter()
            ts, _, _ = T.detect(img, radius=radius, palette=pal, **kw)
            t_detect.append(time.perf_counter() - t0)
            return (np.array([[t.x, t.y] for t in ts], np.float64),
                    np.array([t.r for t in ts], np.float64))

        _, pal = T._quantise(first, kw["k"], None, effort=args.effort)
        base_pts, base_r = _run(first, pal)
        if not len(base_pts):
            continue
        n_stored += len(base_pts)

        # Control: the SAME frame read again through an independently fitted
        # palette. Whatever this moves is the detector's own instability, and
        # the motion figure has to beat it to mean anything.
        _, pal2 = T._quantise(first, kw["k"], None, effort=args.effort)
        ctrl_pts, ctrl_rr = _run(first, pal2)
        pairs, amb = match(base_pts, ctrl_pts, tol)
        ctrl_hit += len(pairs); ctrl_amb += amb
        ctrl_d += [p[2] for p in pairs]

        # Test: the next frame, 0.25s later, read through the SAME palette.
        test_pts, test_rr = _run(second, pal)
        pairs, amb = match(base_pts, test_pts, tol)
        test_hit += len(pairs); test_amb += amb
        test_d += [p[2] for p in pairs]

        # A failed match is not necessarily a tsum that moved. The game paints
        # its highlight over the marked tsums in this very frame, which changes
        # the colours the segmentation is built from -- so the match rate has
        # to be split by whether a tsum is anywhere near a mark, or the
        # highlight gets reported as detector churn.
        got = {p[0] for p in pairs}
        mk = np.array([[stored[i][0], stored[i][1]] for i in marked
                       if i < len(stored)], np.float64)
        for si in range(len(base_pts)):
            if len(mk):
                near = np.hypot(mk[:, 0] - base_pts[si][0],
                                mk[:, 1] - base_pts[si][1]).min() <= radius * 2.0
            else:
                near = False
            bucket = near_stats if near else away_stats
            bucket[0] += 1
            bucket[1] += 1 if si in got else 0

        # Which of the base detections the game had marked, so the highlight
        # can be excluded from every appearance-sensitive number. `marked`
        # indexes the STORED rows, so it is carried over by position.
        marked_pts = np.array([[stored[i][0], stored[i][1]] for i in marked
                               if i < len(stored)], np.float64)
        for si, fi, _ in pairs:
            dv = (test_rr[fi] - base_r[si]) / radius
            dv_all.append(dv)
            if len(marked_pts):
                dm = np.hypot(marked_pts[:, 0] - base_pts[si][0],
                              marked_pts[:, 1] - base_pts[si][1]).min()
                if dm <= radius * 0.5:
                    continue
            dv_unmarked.append(dv)
            if base_r[si] / radius < T.CHARACTER_MIN_VISIBLE:
                rose_n += 1
                if test_rr[fi] / radius >= T.CHARACTER_MIN_VISIBLE:
                    rose += 1

    c, t = np.array(ctrl_d), np.array(test_d)
    print("\n-- matching stored detections into a re-detected frame --")
    print("%-34s%10s%10s" % ("", "control", "+0.25s"))
    print("%-34s%10s%10s" % ("", "(before)", "(marked)"))
    print("%-34s%9.1f%%%9.1f%%" % ("matched within 1.0 radius",
                                   100 * ctrl_hit / max(1, n_stored),
                                   100 * test_hit / max(1, n_stored)))
    print("%-34s%10.2f%10.2f" % ("median displacement (px)",
                                 np.median(c) if len(c) else float("nan"),
                                 np.median(t) if len(t) else float("nan")))
    print("%-34s%10.2f%10.2f" % ("p90 displacement (px)",
                                 np.percentile(c, 90) if len(c) else float("nan"),
                                 np.percentile(t, 90) if len(t) else float("nan")))
    print("%-34s%9.1f%%%9.1f%%" % ("ambiguous (runner-up < 1.5x)",
                                   100 * ctrl_amb / max(1, ctrl_hit),
                                   100 * test_amb / max(1, test_hit)))

    print("\n-- and how much of the miss is the game's highlight --")
    print("  tsums WITHIN 2 radii of a marked tsum : %.1f%% matched  (n=%d)"
          % (100 * near_stats[1] / max(1, near_stats[0]), near_stats[0]))
    print("  tsums AWAY from every mark            : %.1f%% matched  (n=%d)"
          % (100 * away_stats[1] / max(1, away_stats[0]), away_stats[0]))

    dv = np.array(dv_unmarked)
    print("\n-- visibility change over 0.25s, UNMARKED tsums only --")
    print("  n = %d  (%d marked tsums excluded: the game paints its highlight "
          "over them)" % (len(dv), len(dv_all) - len(dv)))
    if len(dv):
        print("  change in r/radius:  p10 %+.3f   median %+.3f   p90 %+.3f"
              % (np.percentile(dv, 10), np.median(dv), np.percentile(dv, 90)))
        print("  got MORE visible: %.1f%%    less: %.1f%%"
              % (100 * (dv > 0).mean(), 100 * (dv < 0).mean()))
    print("\n  the question that decides the idea:")
    print("  buried tsums (below %.2f visible) that rose ABOVE it 0.25s later:"
          % T.CHARACTER_MIN_VISIBLE)
    print("      %d of %d  =  %.1f%%"
          % (rose, rose_n, 100 * rose / max(1, rose_n)))

    print("\n-- cost --")
    print("  detect(): %.0f ms per frame at fit_effort=%d  (%d fits timed)"
          % (1000 * float(np.mean(t_detect)), args.effort, len(t_detect)))
    print("  a round captures ~1.46 frames/s, so one extra observation per "
          "decision\n  costs one extra detect plus the wait for the frame.")
    print("\n  marked share of a board: median %.1f%%"
          % (100 * float(np.median(marked_share))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
