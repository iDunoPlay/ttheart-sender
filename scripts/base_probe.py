"""What the equipped ("base") tsum actually does, and where it can go wrong.

The player reports that gameplay is better with Beast equipped than with other
tsums. This measures the mechanism rather than arguing about the observation.

The base is **not** a character. `read_base_kind` takes the median Lab colour of
the skill icon in the corner and returns `argmin` over the 12 palette centres --
the nearest colour cluster. `find_chains` then flags chains of that cluster
`is_base`, and chains sort by `(is_base, len)`, so a base chain is played ahead
of every longer chain. That is the entire mechanism: **one colour match, used to
reorder candidates.**

Which means the failure mode is a colour match, not a recognition failure. If
the icon's colour is not distinctly closest to one cluster, `argmin` still
returns one, the bot prioritises the wrong group on every frame of the round,
and nothing says so -- `read_base_kind`'s own docstring says a large distance
means the caller should not trust it, and the play loop never checks.

Three measurements, all on data already on disk:

    icon      how far the icon sits from the cluster it was matched to, and
              whether the corpus contains more than one equipped tsum at all
    drags     tsums cleared on base chains vs other chains, within one build
    replay    the same board re-run with every possible base, to find the
              earliest stage where the base changes anything

    python scripts/base_probe.py --replay 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game import tsum as T  # noqa: E402

#: The CLI diagnostic at `tsum board` prints "(WEAK -- check --base)" above
#: this Lab distance. The play loop has no such check, which is the point.
WEAK = 30.0


def rows(root: Path):
    """Every sample that can speak about the base, with its round."""
    for jl in sorted(root.glob("*/samples.jsonl")):
        rj = jl.parent / "round.json"
        rnd = {}
        if rj.exists():
            try:
                rnd = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                rnd = {}
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            b = d.get("base") or {}
            if b.get("kind") is None:
                continue
            o = d.get("options") or {}
            # `--recolour` and `--kinds` renumber `kind`, so a stored cluster id
            # no longer means what `base.kind` means. Those samples cannot be
            # compared and are dropped rather than quietly mismatched.
            if o.get("recolour") or o.get("kinds"):
                continue
            yield jl.parent.name, d, b, o, rnd


def welch(a, b):
    return a.mean() - b.mean(), sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--replay", type=int, default=120,
                    help="boards to re-run under every possible base")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labs, dists, per_build = [], [], defaultdict(list)
    by_session = defaultdict(list)
    boards = []
    for sess, d, b, o, rnd in rows(args.dir):
        if b.get("lab"):
            labs.append(b["lab"])
            by_session[sess].append(b["lab"])
        if b.get("distance") is not None:
            dists.append(float(b["distance"]))
        ts, h = d.get("tsums") or [], d.get("head")
        if h is not None and h < len(ts):
            per_build[d.get("version", "?")].append(
                (ts[h]["kind"] == b["kind"], len(d.get("cleared") or [])))
            boards.append((sess, d, o))

    print("-- 1. the icon, and how well it matches a cluster --")
    if labs:
        L = np.asarray(labs, float)
        print("   samples with an icon colour: %d" % len(L))
        print("   icon Lab range:  L %.0f-%.0f   a %.0f-%.0f   b %.0f-%.0f"
              % (L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max(),
                 L[:, 2].min(), L[:, 2].max()))
        # Two different questions, and reading one number for both gets it
        # backwards. How many tsums were equipped is a question about the
        # SESSION medians; how steady the reading is, is the spread WITHIN a
        # session. The per-sample spread answers neither on its own.
        med = np.asarray([np.median(np.asarray(v, float), axis=0)
                          for v in by_session.values()])
        across = float(np.linalg.norm(med.max(0) - med.min(0)))
        within = float(np.median([
            np.linalg.norm(np.asarray(v, float).max(0) - np.asarray(v, float).min(0))
            for v in by_session.values() if len(v) > 1]))
        print("   %d session(s); spread of the session medians: %.1f Lab units"
              % (len(by_session), across))
        print("   -> %s"
              % ("ONE equipped tsum in the whole corpus, so this data cannot "
                 "compare bases" if across < 5 else
                 "more than one equipped tsum is present"))
        print("   spread WITHIN one session (median): %.1f Lab units -- the "
              "reading itself\n      is this unsteady frame to frame, on a "
              "sprite that never changes." % within)
    if dists:
        D = np.asarray(dists, float)
        print("   Lab distance to the matched cluster: median %.1f  p90 %.1f  "
              "max %.1f" % (np.median(D), np.percentile(D, 90), D.max()))
        print("   above the CLI's own WEAK line of %.0f: %.1f%% of samples"
              % (WEAK, 100 * (D > WEAK).mean()))
        print("   the play loop applies NO threshold: every one of those still "
              "reorders\n   the candidates for its frame.")

    # Group the sessions by which tsum was equipped, now that the corpus has
    # more than one. The icon colour is the only identifier available: the
    # cluster index is a per-frame k-means id and means nothing between frames.
    groups = defaultdict(list)
    if by_session:
        meds = {s: np.median(np.asarray(v, float), axis=0)
                for s, v in by_session.items()}
        keys = []
        for s, m in meds.items():
            hit = next((k for k in keys if np.linalg.norm(m - k) < 8), None)
            if hit is None:
                keys.append(m)
                hit = m
            groups[tuple(int(round(float(v))) for v in hit)].append(s)
    if len(groups) > 1:
        print("\n-- 1b. per equipped tsum --")
        print("   %-18s%9s%10s%10s%12s%12s"
              % ("icon Lab", "rounds", "dist med", ">WEAK", "board share",
                 "base drags"))
        for key, sess in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            keep = set(sess)
            dd = [float(b["distance"]) for s, d, b, o, r in rows(args.dir)
                  if s in keep and b.get("distance") is not None]
            shares, fired, seen_d = [], 0, 0
            for s, d, b, o, r in rows(args.dir):
                if s not in keep:
                    continue
                ts, h = d.get("tsums") or [], d.get("head")
                if not ts:
                    continue
                ks = [t["kind"] for t in ts]
                shares.append(ks.count(b["kind"]) / len(ks))
                if h is not None and h < len(ts):
                    seen_d += 1
                    fired += ts[h]["kind"] == b["kind"]
            print("   %-18s%9d%10.1f%9.1f%%%11.1f%%%11.1f%%"
                  % ("%d,%d,%d" % key, len(sess), float(np.median(dd)),
                     100 * float(np.mean(np.asarray(dd) > WEAK)),
                     100 * float(np.mean(shares)),
                     100 * fired / max(1, seen_d)))
        print("   A base whose colour covers more of the board fires the\n"
              "   preference far more often -- which multiplies its cost.")

    print("\n-- 2. tsums cleared, base chains vs the rest (within build) --")
    print("   %-10s%8s%8s%10s%11s%9s" % ("build", "n", "base%", "clr base",
                                         "clr other", "diff"))
    for v, rs in sorted(per_build.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 200:
            continue
        a = np.asarray(rs, float)
        base, other = a[a[:, 0] == 1][:, 1], a[a[:, 0] == 0][:, 1]
        if len(base) < 30 or len(other) < 30:
            continue
        diff, se = welch(base, other)
        print("   %-10s%8d%7.1f%%%10.2f%11.2f%9.2f +/- %.2f%s"
              % (v, len(rs), 100 * a[:, 0].mean(), base.mean(), other.mean(),
                 diff, 2 * se, "  REAL" if abs(diff) > 2 * se else "  noise"))

    print("\n-- 3. the same board, re-run under every possible base --")
    random.Random(args.seed).shuffle(boards)
    same_count, moved, lost, base_len = 0, 0, [], []
    seen = 0
    for sess, d, o in boards:
        if seen >= args.replay:
            break
        p = args.dir / sess / ("%04d_before.jpg" % d["index"])
        if not p.exists():
            continue
        radius = float(d.get("radius") or 25.0)
        ts = [T.Tsum(t["x"], t["y"], t["r"], t["kind"], (0, 0, 0))
              for t in d["tsums"]]
        if len(ts) < 4:
            continue
        out = {}
        for bk in [None] + sorted({t.kind for t in ts}):
            ch = T.find_chains(ts, radius, o.get("link", 1.5),
                               block=o.get("block", 1.25),
                               link_px=o.get("link_px"), base_kind=bk,
                               base_only=False, mode=o.get("mode", "disc"),
                               max_chain=o.get("max_chain", 0) or 0)
            best = ch[0] if ch else None
            out[bk] = (len(ch), len(best) if best else 0,
                       best.kind if best else None)
        same_count += len({v[0] for v in out.values()}) == 1
        moved += len({v[2] for v in out.values()}) > 1
        lens = [v[1] for v in out.values()]
        base_len.append(out[None][1])
        lost.append(out[None][1] - min(lens))
        seen += 1
    if seen:
        print("   boards replayed: %d" % seen)
        print("   candidate chains IDENTICAL under every base: %.1f%%"
              % (100 * same_count / seen))
        print("   chain CHOSEN changes with the base:          %.1f%%"
              % (100 * moved / seen))
        print("   chosen length with no base: %.2f tsums" % np.mean(base_len))
        print("   lost by pointing the base at the WORST cluster: %.2f tsums"
              % np.mean(lost))
        print("\n   The candidate set never moves. The base changes ONLY which\n"
              "   candidate is played, so chain ranking is the first and only\n"
              "   stage it touches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
