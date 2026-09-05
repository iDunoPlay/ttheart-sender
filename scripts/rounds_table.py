"""Join each round's telemetry to the score it actually earned.

The two halves have existed separately for a while and nothing connected them.
`flows/play.yaml` saves the results screen to the debug directory, named by
wall-clock time; `game/dataset.py` writes a round's samples into its own
session folder. Recovering "what did this round score" meant matching
timestamps by eye, which is fine for five rounds and is not a pipeline.

`DatasetWriter` now writes `round.json` beside the samples, carrying the times
that bound the round and the totals the play loop finished with. This joins a
results frame to a round by asking which round had just ended when the frame
was taken, and **reports the gap it matched on** rather than assuming the
match. A join nobody can audit is worse than no join.

UNKNOWN, not invented
---------------------

Every field that cannot be read comes out as ``null`` and is counted. A score
this could not read is a score that must not enter the corpus looking like
data -- `read_results.py` already refuses a digit it is unsure of, and this
carries that refusal through rather than filling a gap with a guess.

    .venv/Scripts/python scripts/rounds_table.py
    .venv/Scripts/python scripts/rounds_table.py --dir dataset --frames logs/debug
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_results import (  # noqa: E402
    REGIONS, SCORE_FROM_BONUS, load, read_number,
)

import cv2  # noqa: E402

#: A results frame belongs to the round that ended just before it. The flow
#: taps through a banner or two on the way, so allow a window rather than an
#: instant -- but a small one, because rounds are ~2 minutes apart and a match
#: outside this is more likely a missing frame than a slow tap.
MAX_GAP_S = 90.0

#: How close two end times have to be to be the same round seen twice.
#: `round.json` is written in `play_loop`'s `finally` and the log line is
#: emitted just after it, so the real offset is milliseconds; rounds are ~2
#: minutes apart, so anything in between is unambiguous.
LOG_DUP_S = 5.0


#: The end-of-round line the play loop already writes. Parsed as a FALLBACK
#: source of rounds, so every round ever played becomes joinable rather than
#: only those from builds that write `round.json`. The log is less complete --
#: it has no session id, so a round from here cannot be tied back to its
#: samples -- but it has the totals and the time, which is what a score needs.
ROUND_LINE = re.compile(
    r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) INFO.*play_tsum -> '
    r'played (\d+) chains, dragged (\d+) tsums \(mean ([\d.]+)/chain\)'
    r'(?:, cleared (\d+))?')


def rounds_from_log(path: Path):
    """Rounds recovered from the play log, for sessions with no round.json."""
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = ROUND_LINE.match(line)
        if not m:
            continue
        ended = (datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                 + int(m.group(2)) / 1000)
        out.append({
            "session": None,               # the log cannot name the session
            "source": f"log:{path.name}",
            "ended": ended,
            "played": int(m.group(3)),
            "dragged": int(m.group(4)),
            "cleared": int(m.group(6)) if m.group(6) else None,
        })
    return out


def frame_time(name: str):
    """The wall-clock time in a debug frame's file name."""
    try:
        stamp, ms = name[:15], name[16:19]
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp() + int(ms) / 1000
    except (ValueError, IndexError):
        return None


def read_frame(path: Path, templates, max_dist: float):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return {}
    out = {}
    for field, region in REGIONS.items():
        out[field], _ = read_number(bgr, region, templates, max_dist)
    out["score"] = (round(out["bonus"] * SCORE_FROM_BONUS)
                    if out.get("bonus") else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--frames", type=Path, nargs="*",
                    default=[Path("logs/debug"), Path("dataset/logs/debug")])
    ap.add_argument("--templates", type=Path, default=Path("models/digits.npz"))
    ap.add_argument("--max-dist", type=float, default=0.16)
    ap.add_argument("--logs", nargs="*",
                    default=["logs/ttheart.log", "dataset/logs/ttheart.log"],
                    help="play logs to recover rounds from when a session has "
                         "no round.json")
    ap.add_argument("--out", type=Path, default=Path("dataset/rounds.jsonl"))
    args = ap.parse_args()

    if not args.templates.exists():
        print(f"no digit templates at {args.templates} -- "
              f"run `read_results.py build` first")
        return 1
    templates = load(args.templates)

    shots = []
    for d in args.frames:
        for f in sorted(Path(d).glob("*_results.png")):
            t = frame_time(f.name)
            if t is not None:
                shots.append((t, f))
    shots.sort()

    rounds = []
    for rj in sorted(args.dir.glob("*/round.json")):
        try:
            rounds.append(json.loads(rj.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    from_json = len(rounds)
    # The log is a FALLBACK, not a second source. A round that wrote a
    # round.json also wrote a line in the log, and taking both put the same
    # round in the table twice -- once with its score, and once as UNKNOWN
    # because the first copy had already claimed the results frame. 284 rows
    # for 248 rounds, with 36 phantom failures.
    #
    # Matched on the end time, which both sources record independently, within
    # a window far tighter than the gap between rounds.
    by_end = sorted(((r["ended"], r) for r in rounds if r.get("ended")),
                    key=lambda kv: kv[0])
    have = [t for t, _ in by_end]
    merged = 0
    for log in args.logs:
        for r in rounds_from_log(Path(log)):
            i = bisect.bisect_left(have, r["ended"] - LOG_DUP_S)
            if i < len(have) and have[i] <= r["ended"] + LOG_DUP_S:
                # MERGED, not discarded. Rounds recorded before the
                # `report.played` ordering fix all wrote `played: 0`, and the
                # log line beside them holds the real count -- so the log is
                # read as a repair for fields the record is missing, and never
                # as an override of one it has.
                into = by_end[i][1]
                for key in ("played", "dragged", "cleared"):
                    if r.get(key) and not into.get(key):
                        into[key] = r[key]
                        into["repaired"] = True
                merged += 1
                continue
            rounds.append(r)
    rounds.sort(key=lambda r: r.get("ended") or 0)
    if rounds:
        print(f"{from_json} round(s) from round.json, "
              f"{len(rounds) - from_json} recovered from the play log"
              + (f", {merged} merged into a round.json that was missing "
                 f"fields" if merged else ""))

    if not rounds:
        print(f"no round.json under {args.dir} -- those are written by rounds "
              f"played on a build that has the round record. Older sessions "
              f"have samples but no round-level totals.")
        return 1

    used, out = set(), []
    for r in rounds:
        ended = r.get("ended")
        best, best_gap = None, None
        for t, f in shots:
            if f in used or ended is None:
                continue
            gap = t - ended
            if 0 <= gap <= MAX_GAP_S and (best_gap is None or gap < best_gap):
                best, best_gap = f, gap
        vals = {}
        if best is not None:
            used.add(best)
            vals = read_frame(best, templates, args.max_dist)

        dragged, cleared = r.get("dragged"), r.get("cleared")
        out.append({
            "round_id": r.get("session") or r.get("source"),
            "version": r.get("version"),
            "started": r.get("started"),
            "duration": (round(r["ended"] - r["started"], 1)
                         if r.get("ended") and r.get("started") else None),
            "played": r.get("played"),
            "dragged": dragged,
            "cleared": cleared,
            "clear_rate": (round(cleared / dragged, 4)
                           if dragged and cleared is not None else None),
            "settle_ms_per_drag": (round(1000 * r["settle_s"] / r["settles"])
                                   if r.get("settles") else None),
            "final_score": vals.get("score"),
            "score_is_derived": True if vals.get("score") else None,
            "final_coins": vals.get("coins"),
            "bonus": vals.get("bonus"),
            "star": vals.get("star"),
            "results_frame": best.name if best else None,
            "join_gap_s": round(best_gap, 1) if best_gap is not None else None,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")

    have = sum(1 for r in out if r["final_score"] is not None)
    print(f"{len(out)} rounds, {len(shots)} results frames, "
          f"{have} joined with a score\n")
    print(f"{'round':>22}{'played':>7}{'clr%':>7}{'score':>11}{'coins':>7}"
          f"{'gap s':>7}")
    for r in out:
        score = f"{r['final_score']:,}" if r["final_score"] else "UNKNOWN"
        rate = f"{r['clear_rate']:.0%}" if r["clear_rate"] is not None else "--"
        print(f"{str(r['round_id'])[:21]:>22}{r['played'] or '--':>7}{rate:>7}"
              f"{score:>11}{r['final_coins'] or '--':>7}"
              f"{r['join_gap_s'] if r['join_gap_s'] is not None else '--':>7}")
    print(f"\n-> {args.out}")
    if have < len(out):
        print(f"   {len(out) - have} round(s) have no readable score. UNKNOWN, "
              f"not guessed -- check the frame was captured and read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
