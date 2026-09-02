"""Score detection and colour settings against the game's own marks.

`tsum eval` scores detection against ten boards somebody reviewed by hand.
This scores it against **every board the game answered on** -- a tsum the game
lit up while another was held is a real tsum at a real position, stated by the
game rather than by a person, and a collection holds thousands of them.

Three questions, and they must not be mixed:

* **detection -- does the setting still find the tsums the game confirmed?**
  Reported as `kept`, and biased by construction: the confirmed positions come
  from the run that was played, so `live` scores 100% for free and a candidate
  is only ever measured on tsums the live setting already found. It catches a
  regression. It cannot prove an improvement, and `found` -- detections per
  board -- sits beside it because this corpus cannot tell a newly found tsum
  from a phantom.

* **stability -- does the setting read the same board the same way twice?**
  (`--stability`.) The per-frame colour fit is k-means with a seed, so the
  same frame at the same `k` and the same radius can be read twice and
  disagree. A fixed palette has no fit and is deterministic, which is the
  whole of its case here.

* **colour -- given the tsums, does `kind` agree with the game about which are
  the same character?** `agreement` is the share of game-confirmed
  same-character pairs given one id, `split` the share of weak negatives given
  different ids, `balanced` the mean. **Read `balanced` alone**: agreement
  rises as split falls across the whole range of `k`, so either by itself
  measures the trade rather than the quality. That is the ninth round in
  `docs/DATASET-FINDINGS.md`, and reading agreement alone nearly shipped a
  merge-everything palette as a win.

Each candidate is scored **against live on the tsums both of them found**, and
live's score on that same population is printed beside it. A setting that
loses tsums is therefore never flattered by being scored on an easier subset --
which is the same mistake in a new dress. Rows are comparable to live, not to
each other; `pairs` says how much population each comparison had.

    python scripts/sweep_detect.py --dir dataset --samples 150
    python scripts/sweep_detect.py --dir dataset -k 8,12,16 --stability
    python scripts/sweep_detect.py --dir dataset --palette models/palette.json

About 0.3s per frame per setting, tripled for the settings `--stability`
re-fits. A coffee-length run, not a loop step. See `docs/IMPROVEMENT-LOOP.md`.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttheart_sender.game import tsum as T  # noqa: E402
from ttheart_sender.game.learn import AURA, Palette, crop_of, iter_rows, usable  # noqa: E402

#: Seeds the stability check re-fits at. The live loop never sets a seed, so
#: which one it gets is an accident of the build -- that is the point.
SEEDS = (0, 1, 2)


def confirmed(row: dict) -> list[int]:
    """Indices the game confirmed are the pressed character, outside the glow.

    Inside the glow a reaction means proximity rather than identity -- the same
    exclusion `tsum dataset` and `tsum learn` apply, for the same reason.
    """
    head = int(row["head"])
    ts = row["tsums"]
    hx, hy = ts[head]["x"], ts[head]["y"]
    return [i for i in (row.get("marked") or [])
            if 0 <= i < len(ts) and i != head
            and math.hypot(ts[i]["x"] - hx, ts[i]["y"] - hy) > AURA]


def negatives(row: dict, want: int) -> list[int]:
    """Weak negatives: far tsums the game left dark, sampled down to `want`.

    Weak, because the game marks same-character AND reachable -- a same
    character on the far side of the board is dark, and telling it apart from
    the pressed one is not really an error. Sampled down so a board of sixty
    does not drown five real pairs, and seeded off the row so the figure
    repeats run to run.
    """
    head = int(row["head"])
    ts = row["tsums"]
    hx, hy = ts[head]["x"], ts[head]["y"]
    lit = {int(i) for i in (row.get("marked") or [])} | {head}
    dark = [i for i in range(len(ts))
            if i not in lit
            and math.hypot(ts[i]["x"] - hx, ts[i]["y"] - hy) > AURA]
    return random.Random(head).sample(dark, min(want, len(dark)))


def nearest(point, found, tol: float):
    """The detection at `point`, or None if this setting lost it."""
    best, best_d = None, tol
    for t in found:
        d = math.hypot(t.x - point[0], t.y - point[1])
        if d < best_d:
            best, best_d = t, d
    return best


def overlap(a, b, tol: float) -> float:
    """Share of `a`'s detections that `b` also found."""
    if not a:
        return 0.0
    return sum(1 for p in a if nearest((p.x, p.y), b, tol) is not None) / len(a)


class Recorded:
    """A detection read back from the row, shaped like a live `Tsum`.

    So the live setting goes through exactly the same scoring path as every
    candidate rather than a shortcut of its own.
    """

    __slots__ = ("x", "y", "kind")

    def __init__(self, t):
        self.x, self.y, self.kind = float(t["x"]), float(t["y"]), int(t["kind"])


class Setting:
    """One thing to score: a name and a way of detecting on a frame."""

    def __init__(self, name, run, *, refits: bool):
        self.name = name
        self.run = run
        #: Whether the colours are re-fitted per frame. A fixed palette is
        #: deterministic, so its stability is 100% by construction and is
        #: reported rather than measured.
        self.refits = refits


def build_settings(args) -> list[Setting]:
    out = []
    for k in [int(v) for v in args.k.split(",") if v.strip()]:
        out.append(Setting(
            f"k {k}",
            lambda crop, radius, k=k, seed=None: T.detect(
                crop, k=k, radius=radius, scale=args.scale,
                palette=(T._quantise(crop, k, None, seed=seed)[1]
                         if seed is not None else None))[0],
            refits=True))
    for path in args.palette or []:
        centres = Palette.load(path).centres
        out.append(Setting(
            f"palette {Path(path).stem}",
            lambda crop, radius, centres=centres, seed=None: T.detect(
                crop, k=centres.shape[0], radius=radius, scale=args.scale,
                palette=centres)[0],
            refits=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dataset", type=Path)
    ap.add_argument("--samples", type=int, default=150,
                    help="frames to score, drawn across the whole collection")
    ap.add_argument("-k", default="12",
                    help="comma-separated colour cluster counts to try. Empty "
                         "to score only a palette")
    ap.add_argument("--palette", action="append",
                    help="also score a learned palette (repeatable)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="run detection downscaled, as `play --scale` does")
    ap.add_argument("--tol", type=float, default=0.6,
                    help="a confirmed tsum counts as found when a detection "
                         "lands within this fraction of the board radius")
    ap.add_argument("--stability", action="store_true",
                    help="re-fit each per-frame setting at three seeds and "
                         "report how much two reads of one frame agree")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [(f, r) for f, r in iter_rows(args.dir)
            if usable(r) and r.get("schema", 1) >= 2 and confirmed(r)]
    if not rows:
        print(f"no scorable samples under {args.dir}")
        return 1
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.samples]
    settings = build_settings(args)
    if not settings:
        print("nothing to score -- pass -k and/or --palette")
        return 1

    print(f"{len(rows)} frames from {len({f for f, _ in rows})} session(s), "
          f"{sum(len(confirmed(r)) for _, r in rows)} game-confirmed tsums")
    live_opts = rows[0][1]["options"]
    print(f"  live settings: k {live_opts['k']}  scale {live_opts.get('scale', 1.0)}"
          f"  (every candidate detects at the frame's own locked radius)")

    stat = {s.name: {"kept": 0, "wanted": 0, "found": [], "stable": [],
                     "pairs": 0, "agree": 0, "negs": 0, "split": 0,
                     "live_agree": 0, "live_split": 0} for s in settings}
    live_kept = live_wanted = 0
    live_found: list[int] = []
    t0 = time.time()

    for done, (folder, row) in enumerate(rows, 1):
        crop = crop_of(folder, row)
        if crop is None:
            continue
        radius = float(row.get("radius", 25.0))
        tol = radius * args.tol
        ts = row["tsums"]
        pos, neg_idx = confirmed(row), None

        live = [Recorded(t) for t in row["tsums"]]
        live_found.append(len(live))
        live_head = nearest((ts[row["head"]]["x"], ts[row["head"]]["y"]), live, tol)
        live_wanted += 1 + len(pos)
        live_kept += (live_head is not None) + len(pos)  # true by construction
        neg_idx = negatives(row, len(pos))

        for s in settings:
            found = s.run(crop, radius)
            st_row = stat[s.name]
            st_row["found"].append(len(found))
            if args.stability and s.refits:
                runs = [s.run(crop, radius, seed=sd) for sd in SEEDS]
                st_row["stable"].append(st.mean(
                    [overlap(runs[i], runs[j], tol)
                     for i in range(len(runs)) for j in range(len(runs)) if i != j]))
            head = nearest((ts[row["head"]]["x"], ts[row["head"]]["y"]), found, tol)
            st_row["wanted"] += 1 + len(pos)
            st_row["kept"] += head is not None
            hits = {i: nearest((ts[i]["x"], ts[i]["y"]), found, tol) for i in pos}
            st_row["kept"] += sum(1 for h in hits.values() if h is not None)
            if head is None or live_head is None:
                # `kept` is counted above either way -- losing the pressed tsum
                # is a miss like any other. What it costs is the identity
                # score: there is no head to compare a partner's id against.
                continue
            for i, hit in hits.items():
                if hit is None:
                    continue
                st_row["pairs"] += 1
                st_row["agree"] += int(hit.kind == head.kind)
                st_row["live_agree"] += int(ts[i]["kind"] == ts[row["head"]]["kind"])
            for i in neg_idx:
                hit = nearest((ts[i]["x"], ts[i]["y"]), found, tol)
                if hit is None:
                    continue
                st_row["negs"] += 1
                st_row["split"] += int(hit.kind != head.kind)
                st_row["live_split"] += int(ts[i]["kind"] != ts[row["head"]]["kind"])
        if done % 25 == 0:
            print(f"  ...{done}/{len(rows)} frames ({time.time() - t0:.0f}s)")

    print(f"\n{'setting':>20} {'kept':>6} {'found':>6} {'stable':>7} {'pairs':>6} "
          f"{'balanced':>9} {'live here':>10} {'delta':>7}")
    print(f"{'live (as played)':>20} {live_kept/max(1,live_wanted):6.0%} "
          f"{st.mean(live_found) if live_found else 0:6.1f} {'-':>7} {'-':>6} "
          f"{'-':>9} {'-':>10} {'-':>7}")
    for s in settings:
        v = stat[s.name]
        if not v["pairs"] or not v["negs"]:
            print(f"{s.name:>20}   scored no comparable pairs")
            continue
        bal = (v["agree"] / v["pairs"] + v["split"] / v["negs"]) / 2
        live_bal = (v["live_agree"] / v["pairs"] + v["live_split"] / v["negs"]) / 2
        stable = (f"{st.mean(v['stable']):.0%}" if v["stable"]
                  else ("100%" if not s.refits else "-"))
        note = ""
        if v["kept"] / v["wanted"] < 0.95:
            note = "  <- loses confirmed tsums"
        elif bal > live_bal + 0.01:
            note = "  <- better than live"
        print(f"{s.name:>20} {v['kept']/v['wanted']:6.0%} {st.mean(v['found']):6.1f} "
              f"{stable:>7} {v['pairs']:6d} {bal:9.1%} {live_bal:10.1%} "
              f"{bal - live_bal:+7.1%}{note}")

    print("\n  `balanced` is the verdict, and only against `live here` -- live's own "
          "score\n  on the very same pairs. `kept` is a floor: live is 100% by "
          "construction,\n  because the confirmed positions are the ones it found. "
          "A fixed palette\n  reads `stable` 100% because it does not fit anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
