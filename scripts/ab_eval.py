"""Read an alternating A/B out of the corpus, on one build, with a decision rule.

WHY THIS EXISTS, AND WHAT IT REFUSES TO DO
------------------------------------------

Every comparison this project has made was one build against another. The board
filter's only reading -- 13 rounds of 1.11.5 against 244 of 1.11.1e -- is
confounded by the build, and the three rounds of 1.11.3 that sit between them
are too few to separate anything. `--ab` fixes the collection side by
alternating the arm inside the play loop; this reads the result.

It **refuses to pool across builds** unless told to. That is the whole point:
a number that mixes builds is the number this was written to stop producing.

THE DECISION METRIC IS NOT SCORE, AND THAT IS MEASURED, NOT PREFERRED
---------------------------------------------------------------------

Score is the objective. It is also, over 277 scored rounds, a distribution with
a **63% coefficient of variation** -- p10 82k, p50 503k, p90 995k. At that
spread, detecting a 10% change in score at the usual 80% power needs **632
rounds per arm**, about 42 hours of play, and adjacent rounds are uncorrelated
at every lag tried (r = -0.03 at lag 1), so no pairing or blocking buys any of
it back.

    metric        CV      rounds/arm for +10%
    score        63.4%           632
    coins        59.0%           547
    FEVER share  36.9%           214
    cleared      21.2%            71
    clear rate   18.7%            55
    clears/sec   15.1%            36

So the design here is:

* **PRIMARY: `cleared`.** 71 rounds per arm for +10%, and `corr(cleared,
  score) = +0.72` on this corpus -- the thirtieth round validated it as the
  proxy after rounds 26 and 28 had written it off. It is a count, not a ratio.
* **GUARDRAILS: score, FEVER share, coins.** Reported with intervals and used
  only to VETO -- if the primary wins while a guardrail moves the wrong way by
  more than its own noise, that is a result to distrust, not to ship.
* **MECHANISM: dead drags, seconds per frame.** These say whether the switch
  did the thing it claims. They are cheap to move and must never be the
  decision on their own.

`clear rate` is deliberately **not** primary. It is the number that fell 14.8pp
in round 35 while the score did not move with it, because it is a ratio and its
denominator moved -- and it is the number that made "pretty bad" the wrong
verdict on 18 rounds.

    .venv/Scripts/python scripts/ab_eval.py
    .venv/Scripts/python scripts/ab_eval.py --option reject_model
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

#: Two-sided 95%, and 80% power. Spelled out because every n below is derived
#: from them and a reader should not have to guess which convention was used.
Z_ALPHA, Z_POWER = 1.959964, 0.841621

#: No verdict below this many rounds per arm, whatever the power sum says.
#:
#: The power calculation divides by the sample standard deviation, and at a
#: handful of rounds that standard deviation is itself mostly noise -- a run
#: that happens to come out tight computes a tiny requirement and then reports
#: "at adequate power" on six rounds. A floor is the cheap guard against a
#: confident answer derived from its own good luck, and 20 is where the
#: corpus's own loosest metric (clears per second, 36 per arm for +10%) stops
#: being wildly out of reach.
MIN_ROUNDS_PER_ARM = 20

#: (field, label, higher-is-better). The order is the order it prints in.
PRIMARY = [("cleared", "cleared", True)]
GUARDRAIL = [("score", "score", True),
             ("fever", "FEVER share", True),
             ("coins", "coins", True)]
MECHANISM = [("rejected", "dead drags", False),
             ("s_per_frame", "seconds/frame", False),
             ("dragged", "dragged", None),
             ("rate", "clear rate", None),
             ("frames", "frames", None)]


def welch(a: list, b: list):
    """t, two-sided p, the difference and its 95% interval. Normal-approximate.

    The p uses the normal tail rather than Student's, which is the right call
    at the tens-of-rounds this is built for and slightly optimistic below ~10.
    Reported alongside n so a reader can discount it.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return None
    t = (ma - mb) / se
    p = math.erfc(abs(t) / math.sqrt(2))
    pooled = math.sqrt((st.variance(a) + st.variance(b)) / 2)
    return {"diff": ma - mb, "se": se, "t": t, "p": p,
            "lo": (ma - mb) - Z_ALPHA * se, "hi": (ma - mb) + Z_ALPHA * se,
            "d": (ma - mb) / pooled if pooled else float("nan")}


def rounds_needed(values: list, lift: float):
    """Rounds per arm to see a `lift` fractional change at 80% power.

    None when it cannot be computed -- no rounds, or no variation to divide by.
    None is printed as "--", because a missing requirement printed as 0 reads
    as "no rounds needed", which is the opposite of what it means.
    """
    if len(values) < 2:
        return None
    m, sd = st.mean(values), st.stdev(values)
    if not m or not sd:
        return None
    d = abs(lift * m) / sd
    return math.ceil(2 * (Z_ALPHA + Z_POWER) ** 2 / d ** 2) if d else None


def load(dataset: Path, rounds_file: Path):
    """One record per round: its totals, its score, its build and its arm."""
    scores = {}
    if rounds_file.exists():
        for line in rounds_file.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(r.get("round_id", "")).startswith("log:"):
                scores[r["round_id"]] = r

    out = []
    for rj in sorted(dataset.glob("*/round.json")):
        sess = rj.parent.name
        try:
            j = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # The version and the arm both come off the round record where it has
        # them. `ab_arm` is written by builds that have --ab; older rounds fall
        # back to the samples, which carry the whole option set anyway.
        opts = {}
        sp = rj.parent / "samples.jsonl"
        if sp.exists():
            try:
                with sp.open(encoding="utf-8") as fh:
                    opts = (json.loads(fh.readline()) or {}).get("options", {})
            except (OSError, json.JSONDecodeError, TypeError):
                opts = {}
        arm = j.get("ab_arm") or ""
        option = j.get("ab") or opts.get("ab") or ""
        if not arm and option and option in opts:
            # A round from a build that alternated but did not stamp the arm:
            # recover it from the value itself rather than dropping the round.
            arm = "off" if opts.get(option) in ("", 0, False, None) else "on"
        dur = (j.get("ended") or 0) - (j.get("started") or 0)
        s = scores.get(sess, {})
        rec = {
            "sess": sess, "arm": arm, "option": option,
            "version": j.get("version") or opts.get("version"),
            "cleared": j.get("cleared"), "dragged": j.get("dragged"),
            "rejected": j.get("rejected"), "frames": j.get("frames"),
            "score": s.get("final_score"), "coins": s.get("final_coins"),
        }
        if j.get("frames") and dur > 0:
            rec["s_per_frame"] = dur / j["frames"]
            rec["fever"] = (j.get("fever_frames") or 0) / j["frames"]
        if j.get("dragged"):
            rec["rate"] = (j.get("cleared") or 0) / j["dragged"]
        out.append(rec)
    return out


def report(on: list, off: list, block: list, title: str) -> list:
    """One block of the table. Returns the rows, so the caller can decide."""
    print(f"\n{title}")
    print(f"  {'metric':14s} {'ON mean':>12s} {'OFF mean':>12s} "
          f"{'diff':>12s} {'95% interval':>26s} {'p':>7s} {'d':>6s}")
    rows = []
    for field, label, _better in block:
        a = [r[field] for r in on if r.get(field) is not None]
        b = [r[field] for r in off if r.get(field) is not None]
        w = welch(a, b)
        if w is None:
            why = ("no variation to test" if len(a) >= 2 and len(b) >= 2
                   else "too few rounds with this field")
            print(f"  {label:14s} {'--':>12s} {'--':>12s}   {why}")
            continue
        fmt = (lambda v: f"{v:12,.3f}" if abs(v) < 1000 else f"{v:12,.0f}")
        print(f"  {label:14s} {fmt(st.mean(a))} {fmt(st.mean(b))} "
              f"{fmt(w['diff'])} "
              f"{'[' + fmt(w['lo']).strip() + ', ' + fmt(w['hi']).strip() + ']':>26s} "
              f"{w['p']:7.3f} {w['d']:6.2f}")
        rows.append((field, label, w, a, b))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--rounds", type=Path, default=Path("dataset/rounds.jsonl"),
                    help="the score join, written by scripts/rounds_table.py")
    ap.add_argument("--option", default="",
                    help="which alternated option to read; default is "
                         "whichever one the corpus holds")
    ap.add_argument("--version", default="",
                    help="restrict to one build. Default is the build with "
                         "the most A/B rounds")
    ap.add_argument("--pool-builds", action="store_true",
                    help="compare across builds anyway. Do not: every "
                         "confounded reading this project has produced came "
                         "from doing exactly this")
    ap.add_argument("--lift", type=float, default=0.10,
                    help="the fractional change the plan is powered for")
    args = ap.parse_args()

    rows = load(args.dir, args.rounds)
    played = [r for r in rows if r["arm"] in ("on", "off")]
    if not played:
        print(f"No A/B rounds under {args.dir}.\n")
        print("To collect some: set `reject_model: models/reject.onnx` and")
        print("`ab: reject_model` in flows/play.yaml, then play. The arm")
        print("alternates every round and records itself.")
        return 1

    named = [r["option"] for r in played if r["option"]]
    if not args.option and not named:
        print("Rounds carry an arm but not the name of the option it "
              "alternated. Pass --option to say which.")
        return 1
    option = args.option or st.mode(named)
    played = [r for r in played if r["option"] == option]
    if not played:
        print(f"No rounds alternated `{option}`.")
        return 1

    by_build = defaultdict(list)
    for r in played:
        by_build[r["version"]].append(r)
    if args.version:
        builds = [args.version]
    elif args.pool_builds:
        builds = sorted(by_build)
    else:
        builds = [max(by_build, key=lambda v: len(by_build[v]))]

    print(f"A/B on `{option}`")
    print(f"  {len(played)} alternating rounds over "
          f"{len(by_build)} build(s): "
          + ", ".join(f"{v} n={len(g)}" for v, g in sorted(by_build.items())))
    if len(by_build) > 1 and not args.pool_builds:
        print(f"  reading {builds[0]} only. Pooling builds is what made every "
              f"previous reading unusable;")
        print(f"  pass --pool-builds if you really mean to.")

    use = [r for r in played if r["version"] in builds]
    on = [r for r in use if r["arm"] == "on"]
    off = [r for r in use if r["arm"] == "off"]
    print(f"\n  ON  n={len(on)}    OFF n={len(off)}")
    if len(on) != len(off):
        print(f"  arms are UNEVEN by {abs(len(on) - len(off))} -- the counter "
              f"restarts with the process, so a restart mid-run leaves a gap. "
              f"Not fatal; it costs power, not validity.")
    if not on or not off:
        print("\n  Only one arm has rounds. Nothing to compare yet.")
        return 1

    primary = report(on, off, PRIMARY, "PRIMARY -- the decision")
    guard = report(on, off, GUARDRAIL, "GUARDRAILS -- can only veto")
    report(on, off, MECHANISM, "MECHANISM -- did the switch do its job")

    print("\nPOWER: rounds per arm needed to see a "
          f"{args.lift:.0%} change, at this corpus's spread")
    print(f"  {'metric':14s} {'have':>6s} {'need':>8s}")
    have = min(len(on), len(off))
    for field, label, _b in PRIMARY + GUARDRAIL:
        v = [r[field] for r in use if r.get(field) is not None]
        need = rounds_needed(v, args.lift)
        if need is None:
            print(f"  {label:14s} {have:6d} {'--':>8s}   "
                  f"<- no rounds carry it")
            continue
        need = max(need, MIN_ROUNDS_PER_ARM)
        flag = "" if have >= need else "   <- not there yet"
        print(f"  {label:14s} {have:6d} {need:8d}{flag}")

    print("\nVERDICT")
    if not primary:
        print("  NEED MORE DATA -- the primary metric has no rounds.")
        return 0
    _f, label, w, a, _b = primary[0]
    need = max(rounds_needed(a + _b, args.lift) or MIN_ROUNDS_PER_ARM,
               MIN_ROUNDS_PER_ARM)
    vetoes = [(lab, ww) for _fld, lab, ww, _x, _y in guard
              if ww["p"] < 0.05 and ww["diff"] < 0]
    if min(len(on), len(off)) < need:
        print(f"  NEED MORE DATA. {label} needs {need} rounds per arm for a "
              f"{args.lift:.0%} read and has {min(len(on), len(off))}.")
        print(f"  Current best guess: {w['diff']:+,.1f} "
              f"[{w['lo']:+,.1f}, {w['hi']:+,.1f}], p={w['p']:.3f}.")
    elif w["p"] >= 0.05:
        print(f"  REJECT (no effect). {label} moved {w['diff']:+,.1f} "
              f"[{w['lo']:+,.1f}, {w['hi']:+,.1f}], p={w['p']:.3f}, at "
              f"adequate power.")
    elif w["diff"] < 0:
        # BEFORE the guardrails, and the ordering is the correction: a primary
        # that FELL is a revert whatever the guardrails did. Checking the
        # vetoes first printed "cleared improved, but ..." on a run where
        # cleared had dropped 18 -- the right decision under a sentence that
        # said the opposite of the number beside it.
        print(f"  REVERT. {label} fell {w['diff']:+,.1f} "
              f"[{w['lo']:+,.1f}, {w['hi']:+,.1f}], p={w['p']:.3f}."
              + (" Guardrails agree: "
                 + "; ".join(f"{lab} {ww['diff']:+,.3f} (p={ww['p']:.3f})"
                             for lab, ww in vetoes) if vetoes else ""))
    elif vetoes:
        print(f"  DISTRUST. {label} improved, but "
              + "; ".join(f"{lab} fell {ww['diff']:+,.1f} (p={ww['p']:.3f})"
                          for lab, ww in vetoes))
    else:
        print(f"  SHIP. {label} rose {w['diff']:+,.1f} "
              f"[{w['lo']:+,.1f}, {w['hi']:+,.1f}], p={w['p']:.3f}, "
              f"no guardrail broken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
