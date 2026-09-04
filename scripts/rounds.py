"""Print the last N rounds as one table, so a result can be read at a glance.

Every experiment in `docs/IMPROVEMENT-LOOP.md` ends with "play a round and
watch the end-of-round line". That line is one entry in a log that also holds
every chain, every skip and every template match, scrolls past during play, and
rotates -- so in practice the result of a round is hard to see, and a round
whose result is not seen is a round not taken.

This reads the log back and prints one row per round:

    python scripts/rounds.py            # the last 10
    python scripts/rounds.py -n 30      # more
    python scripts/rounds.py --settings # and what each was played at

`cleared %` is the number to watch for almost every experiment here, with mean
chain length beside it: a change that lifts one while dropping the other has
not obviously won, and that has happened more than once.

The settings come from `dataset/*/samples.jsonl` rather than the log, because
the log does not record them -- so `--settings` only names a round that was
collecting. A round played without collection still shows its numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime
from pathlib import Path

ROUND = re.compile(
    r"played (\d+) chains, dragged (\d+) tsums \(mean ([\d.]+)/chain\)"
    r"(?:, cleared (\d+) \((\d+)%\))?"
    r"(?:, (\d+) drag\(s\) did not register)?")
EXTRA = {
    "trimmed": r"the game trimmed (\d+) member",
    "abandoned": r"rejected (\d+) chain\(s\) outright",
    "checks": r"checked (\d+) chain",
    "rebuilt": r"marks rebuilt (\d+) chain",
}


def read_rounds(log_dir: Path):
    out = []
    for f in sorted(log_dir.glob("ttheart.log*"), reverse=True):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "play_tsum -> played" not in line:
                continue
            m = ROUND.search(line)
            if not m:
                continue
            body = line.split("play_tsum -> ")[1]
            row = {
                "when": line[:19],
                "chains": int(m.group(1)),
                "dragged": int(m.group(2)),
                "len": float(m.group(3)),
                "cleared": int(m.group(4)) if m.group(4) else None,
                "pct": int(m.group(5)) if m.group(5) else None,
                "stalls": int(m.group(6)) if m.group(6) else 0,
            }
            for k, pat in EXTRA.items():
                mm = re.search(pat, body)
                row[k] = int(mm.group(1)) if mm else 0
            out.append(row)
    out.sort(key=lambda r: r["when"])
    return out


def settings_by_time():
    """When each collected session ran, and the settings it ran at."""
    seen = []
    for f in sorted(glob.glob("dataset/*/samples.jsonl")):
        try:
            first = json.loads(next(l for l in open(f, encoding="utf-8") if l.strip()))
        except (StopIteration, json.JSONDecodeError, OSError):
            continue
        o = first.get("options") or {}
        keys = ("verify_reach", "verify_extend", "recolour", "bowl_reject",
                "max_chain", "min_chain", "link_px", "block", "fit_effort")
        seen.append((first.get("time", 0.0),
                     " ".join(f"{k}={o[k]}" for k in keys if k in o)))
    return sorted(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=10, help="how many rounds to show")
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--settings", action="store_true",
                    help="name the settings each round was collected at")
    args = ap.parse_args()

    rounds = read_rounds(args.log_dir)
    if not rounds:
        print(f"no rounds found in {args.log_dir}/ttheart.log*")
        return 1
    shown = rounds[-args.n:]

    print(f"last {len(shown)} of {len(rounds)} rounds in the log\n")
    print(f"{'when':>19}{'chains':>8}{'dragged':>9}{'cleared':>9}{'%':>5}"
          f"{'len':>7}{'stall':>7}{'trim':>6}{'chk':>5}")
    for r in shown:
        cl = str(r["cleared"]) if r["cleared"] is not None else "--"
        pc = f"{r['pct']}%" if r["pct"] is not None else "--"
        print(f"{r['when']:>19}{r['chains']:8d}{r['dragged']:9d}{cl:>9}{pc:>5}"
              f"{r['len']:7.2f}{r['stalls']:7d}{r['trimmed']:6d}{r['checks']:5d}")

    got = [r for r in shown if r["pct"] is not None]
    if got:
        n = len(got)
        print(f"\n{'mean of these':>19}"
              f"{sum(r['chains'] for r in got) / n:8.1f}"
              f"{sum(r['dragged'] for r in got) / n:9.1f}"
              f"{sum(r['cleared'] for r in got) / n:9.1f}"
              f"{100 * sum(r['cleared'] for r in got) / sum(r['dragged'] for r in got):4.0f}%"
              f"{sum(r['len'] for r in got) / n:7.2f}")
        print("\n`cleared %` is the one to watch, with `len` beside it: a change that")
        print("lifts one and drops the other has not obviously won.")
    else:
        print("\nNo `cleared` figures: these rounds were played without "
              "`verify_clears`, so nothing measured what left the board.")

    if args.settings:
        sets = settings_by_time()
        print("\nsettings, from the collected sessions (a round played without "
              "collection is not listed):")
        last = None
        for t, s in sets:
            when = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            if s != last:
                print(f"  from {when}  {s}")
                last = s
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
