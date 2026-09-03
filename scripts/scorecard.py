"""Three numbers for a collection: colour, detection, gameplay.

The other scripts answer one question deeply. This one answers the same three
questions the same way every time, so collections can be put side by side and
the project can tell whether it is getting better rather than merely busy.

    python scripts/scorecard.py --dir dataset
    python scripts/scorecard.py --dir dataset --append

`--append` writes the row to `docs/SCORECARD.md`. One row per collection, and
the settings each was played under sit in the row, because a number without
them is not comparable to anything.

What each headline is, and what it cannot see:

* **colour** -- how much more often the game's confirmed partners share the
  pressed tsum's `kind` than a random far tsum does. 1.00x is a clustering
  that knows nothing about identity. It says nothing about whether the same
  frame would cluster the same way twice: that is `scripts/sweep_detect.py
  --stability`, which needs the images.
* **detection** -- the share of boards read at a plausible size, plus the
  median count. It cannot measure phantoms: an unmarked tsum is not a phantom,
  it is a tsum the game had no reason to light. Over-splitting is the failure
  it does catch, and it is the one that has actually happened.
* **gameplay** -- the share of proposed chain members the game refused, and
  the share of drags left under `min_chain`, which run and clear nothing.

All of it is read from `samples.jsonl`; no image is decoded, so this is a
seconds-long check, not a coffee-length one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

AURA = 90.0
HISTORY = Path("docs/SCORECARD.md")

HEADER = """# Scorecard

One row per collection, written by `python scripts/scorecard.py --append`.
Every number is measured against the game's own marks -- see
`docs/IMPROVEMENT-LOOP.md` for how a row is produced and
`docs/DATASET-FINDINGS.md` for what the rounds behind them concluded.

* **colour lift** -- how much likelier a game-confirmed partner is to share the
  pressed tsum's `kind` than a random far tsum is. 1.00x knows nothing.
* **plausible** -- share of boards read at a believable size (20-110
  detections); **found** is the median count.
* **refused** -- share of proposed chain members the game would not take;
  **dead drags** run and clear nothing.
* **cleared** -- share of DRAGGED tsums that actually left the board. The only
  column that is not a proxy, and `--` until a round is played with "Measure
  tsums cleared" on. Before schema 3 no collection could carry it.

| collected | samples | settings | colour lift | plausible | found | refused | dead drags | cleared |
|---|---:|---|---:|---:|---:|---:|---:|---:|
"""


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def load(root: Path):
    rows = []
    for f in sorted(root.glob("*/samples.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("schema", 1) >= 2:
                row["_session"] = f.parent.name
                rows.append(row)
    return rows


def colour(rows) -> dict:
    """Does `kind` agree with the game about who is the same character?"""
    pairs = same = 0
    base_n = base_same = 0
    for r in rows:
        ts, head = r["tsums"], r["tsums"][r["head"]]
        far = [i for i in range(len(ts))
               if i != r["head"] and dist(ts[i], head) > AURA]
        marked = [i for i in (r.get("marked") or []) if i in set(far)]
        if not marked:
            continue
        for i in marked:
            pairs += 1
            same += int(ts[i]["kind"] == head["kind"])
        # The base rate has to come from the same boards, or the comparison is
        # against a different population's colours. Sampled to the number of
        # positives and seeded off the row, as `tsum dataset` does.
        for i in random.Random(r["head"]).sample(far, min(len(marked), len(far))):
            base_n += 1
            base_same += int(ts[i]["kind"] == head["kind"])
    if not pairs or not base_n or not base_same:
        return {}
    agree, base = same / pairs, base_same / base_n
    return {"pairs": pairs, "agreement": agree, "base": base, "lift": agree / base}


def detection(rows) -> dict:
    counts = sorted(len(r["tsums"]) for r in rows)
    blank = sum(1 for r in rows if not r.get("marked"))
    return {
        "median": counts[len(counts) // 2],
        "p90": counts[9 * len(counts) // 10],
        "max": counts[-1],
        "oversplit": sum(1 for c in counts if c > 75) / len(counts),
        "plausible": sum(1 for c in counts if 20 <= c <= 110) / len(counts),
        "blank": blank / len(rows),
    }


def gameplay(rows) -> dict:
    min_chain = int(rows[0]["options"].get("min_chain", 3))
    proposed = sum(len(r["proposed"]) for r in rows)
    kept = sum(len(r["kept"]) for r in rows)
    dead = sum(1 for r in rows if len(r["kept"]) < min_chain)
    offered = []
    for r in rows:
        head = r["tsums"][r["head"]]
        prop = set(r["proposed"])
        offered.append(sum(1 for i in (r.get("marked") or [])
                           if i not in prop and dist(r["tsums"][i], head) > AURA))
    return {
        "proposed": proposed / len(rows),
        "kept": kept / len(rows),
        "refused": 1 - kept / proposed,
        "dead": dead / len(rows),
        "ignored": st.mean(offered),
        "min_chain": min_chain,
    }


def _defaults() -> dict:
    """The play settings at their defaults, or {} if the game module is absent.

    Imported lazily and defensively: everything else here is stdlib reading
    JSONL, and a scorecard should still print for someone holding only a
    collection. Without it the row falls back to naming what it can see.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ttheart_sender.game.tsum import play_defaults
        return {k: v for k, v in vars(play_defaults()).items()
                if isinstance(v, (bool, int, float, str)) or v is None}
    except Exception:
        return {}


#: Settings that describe the machine or the collection rather than how the
#: round was played, and would crowd out the ones that matter. Excluded from
#: the row only -- schema 3 records them, and this is a display choice.
_NOT_PLAY = ("dataset", "duration", "countdown", "dry_run", "merge",
             "no_prepare", "debug_dir")


def clears(rows) -> dict:
    """What actually left the board, when the round measured it.

    The only headline here that is not a proxy. `refused` and `dead drags`
    describe what the game SAID; this is what the board did, and it is the
    number the project is ultimately played for. Empty unless the corpus was
    collected with `verify_clears` on -- which, before schema 3, no corpus
    could carry at all: the count lived in the play log of the machine that
    played the round.
    """
    measured = [r for r in rows if isinstance(r.get("cleared"), list)]
    if not measured:
        return {}
    popped = sum(len(r["cleared"]) for r in measured)
    tried = sum(len(r.get("dragged") or r["kept"]) for r in measured)
    dead = sum(1 for r in measured if not r["cleared"])
    return {"drags": len(measured), "popped": popped, "tried": tried,
            "rate": popped / max(1, tried), "per_drag": popped / len(measured),
            "dead": dead / len(measured)}


def settings_of(rows) -> str:
    """What this collection was played at, as everything that is NOT default.

    Not a chosen list of interesting keys. That is what this function used to
    be, and the interesting key is always the one somebody forgot to add: the
    row for the eleventh round could not say whether `verify_reach` had been
    on. A diff against the defaults names the new flag the round after it is
    invented, with nobody editing this file.

    Falls back to the old hand-named few when the row predates schema 3 and
    only carries them.
    """
    o = rows[0]["options"]
    base = _defaults()
    full = rows[0].get("schema", 1) >= 3
    bits = [f"k{o.get('k')}", f"link{o.get('link_px')}"]

    if base:
        changed = []
        for key in sorted(o):
            if key in _NOT_PLAY or key.startswith("dataset_") or key in ("k", "link_px"):
                continue
            if key in base and o[key] != base[key]:
                value = o[key]
                if isinstance(value, bool):
                    changed.append(key if value else f"no-{key}")
                elif isinstance(value, float) and value.is_integer():
                    changed.append(f"{key} {int(value)}")
                else:
                    changed.append(f"{key} {value}")
        bits += changed
    else:
        # Schema 2 and older: only the curated dozen was recorded, so the row
        # can only name those, and cannot promise the rest were at defaults.
        bits.append(f"fit{o.get('fit_effort', 1)}")
        if o.get("verify_reach"):
            bits.append(f"reach{int(o['verify_reach'])}")

    bits.append(f"floor{rows[0]['capture'].get('floor_mult')}")
    who = equipped_of(rows)
    if who:
        bits.append(who)
    if not full:
        # Schema 2 recorded a curated subset, so "not in this list" means "not
        # recorded", not "left at its default". Say so in the row rather than
        # let a later reader assume the stronger thing.
        bits.append("[partial: pre-schema-3 row]")
    return " ".join(bits)


def equipped_of(rows) -> str:
    """The equipped tsum, as the colour of its skill icon.

    There is no name to be had -- see `read_base_kind` -- but the icon colour
    separates one character from another, and the equipped tsum decides how
    the board is filled and which skill fires. A row that does not say which
    character played it is not comparable to one that does, which is a thing
    this file learned three rounds late.
    """
    labs = [tuple(r["base"]["lab"]) for r in rows
            if isinstance(r.get("base"), dict) and r["base"].get("lab")]
    if not labs:
        return ""
    mid = [round(st.median(v[i] for v in labs)) for i in range(3)]
    spread = max(
        max(abs(v[i] - mid[i]) for i in range(3)) for v in labs)
    tag = f"base Lab({mid[0]},{mid[1]},{mid[2]})"
    # A corpus is meant to be one character. A wide spread means it is not,
    # and every number in the row is then an average over two games.
    return tag + ("!" if spread > 12 else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dataset", type=Path)
    ap.add_argument("--append", action="store_true",
                    help=f"append the row to {HISTORY}")
    ap.add_argument("--label", default="",
                    help="what changed since the last row, e.g. 'fit_effort 3'")
    args = ap.parse_args()

    rows = load(args.dir)
    if not rows:
        print(f"no schema 2 samples under {args.dir}")
        return 1
    col, det, play = colour(rows), detection(rows), gameplay(rows)
    if not col:
        print("no confirmed pairs -- the collection carries no usable label")
        return 1

    when = time.strftime("%Y-%m-%d", time.localtime(min(r["time"] for r in rows)))
    print(f"{len(rows)} drags over {len({r['_session'] for r in rows})} sessions, "
          f"collected {when} on v{rows[0]['version']}")
    print(f"  settings: {settings_of(rows)}")

    print(f"\ncolour     {col['lift']:.2f}x   {col['agreement']:.1%} of the game's "
          f"confirmed partners share the pressed tsum's kind,")
    print(f"                 against {col['base']:.1%} for a random far tsum "
          f"({col['pairs']} pairs)")
    print(f"detection  {det['plausible']:.0%}     of boards read at a plausible size; "
          f"median {det['median']}, p90 {det['p90']}, max {det['max']}")
    print(f"                 over-split {det['oversplit']:.1%}; "
          f"{det['blank']:.1%} of readings showed no mark at all")
    print(f"gameplay   {play['refused']:.0%}     of proposed members refused; "
          f"{play['dead']:.0%} of drags left under min_chain {play['min_chain']}")
    print(f"                 proposed {play['proposed']:.2f}, kept {play['kept']:.2f}, "
          f"and {play['ignored']:.2f} marked tsums per drag never proposed")

    clr = clears(rows)
    if clr:
        print(f"clears     {clr['rate']:.0%}     of dragged tsums actually left the "
              f"board ({clr['popped']} of {clr['tried']} over {clr['drags']} "
              f"measured drags)")
        print(f"                 {clr['per_drag']:.2f} cleared per drag; "
              f"{clr['dead']:.0%} of drags cleared nothing at all")
    else:
        print('clears     --      not measured. Tick "Measure tsums cleared" for a '
              'round and\n                 this becomes the one headline here that '
              'is not a proxy.')

    if args.append:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        if not HISTORY.exists():
            HISTORY.write_text(HEADER, encoding="utf-8")
        note = settings_of(rows) + (f" — {args.label}" if args.label else "")
        with HISTORY.open("a", encoding="utf-8") as fh:
            fh.write(f"| {when} | {len(rows)} | {note} | {col['lift']:.2f}x | "
                     f"{det['plausible']:.0%} | {det['median']} | "
                     f"{play['refused']:.0%} | {play['dead']:.0%} | "
                     f"{(format(clr['rate'], '.0%') if clr else '--')} |\n")
        print(f"\nappended to {HISTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
