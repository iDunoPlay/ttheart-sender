"""Replay a collection's drags and price the play settings against the game's marks.

`tsum dataset` answers "are the labels real?". This answers the question after
it: **given real labels, was the bot right?** Every schema 2 row carries the
chain the bot proposed, the subset the game marked while the first tsum was
held, and the settings both were taken under -- a recording of the bot being
right and wrong, drag by drag, with the game as the judge.

It decodes no images: everything below is read out of ``samples.jsonl``, and
all of it is stdlib except the `verify_extend` section, which imports the
game module so it prices the shipped rule rather than a copy of it. Run it
over a directory of session folders:

    python scripts/replay_decisions.py --dir dataset

The two numbers that decide anything are `cleared` (how many tsums a rule pops)
and `clears/s` (what it costs to pop them) -- and neither prices the scoring
curve, which is why a rule that wins on clears/s can still be the wrong rule.

`cleared` is a MODEL, not a measurement, and since the fifteenth round every
table that reports it prints two: the `all-or-nothing` one this script assumed
for its first five rounds, under which a chain with any refused member clears
nothing, and the `per-member` one that round measured, under which each member
clears on its own odds. They disagree sharply about what NOT checking is worth,
which is the whole of `verify_reach`'s case. Read the per-member half; the old
half is there so no earlier round is re-scored without anyone noticing.
See ``docs/IMPROVEMENT-LOOP.md`` for where this sits in the loop and
``docs/DATASET-FINDINGS.md`` for what previous rounds concluded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

#: The stroke cost model, from `play`'s own defaults. A drag walks each leg in
#: `STEP_PX` hops pausing `PER_STEP` at each, and pays `HOLD` at both ends.
STEP_PX, PER_STEP, HOLD = 8.0, 0.004, 0.05

#: What one verify hold costs. The check waits `--verify-delay` (0.25) and then
#: grabs; the spread is one frame against three, because the collector reads
#: three and a live check need not. A rule whose ranking flips across these
#: three columns is not a finding.
CHECK_COSTS = (0.17, 0.28, 0.41)

#: What a member's fate actually is, measured in the fifteenth round.
#:
#: Every number this script printed before that round assumed a drag is
#: all-or-nothing: an unchecked chain clears in full if the game accepted
#: every member and clears NOTHING otherwise. That assumption was never
#: measured, and it is false. Over 111 sampled drags dragged exactly as
#: proposed -- the game named the refusals, the bot ignored them, and
#: `verify_clears` recorded what left the board:
#:
#:     members the game marked  : 291/324 cleared (89.8%)
#:     members the game refused :  19/83  cleared (22.9%)
#:     drags that cleared nothing: 15 of 863 (1.7%)
#:
#: So a chain with one refused member out of five still clears the others,
#: where the all-or-nothing model scored that drag as zero.
#:
#: KNOWN WRONG IN ITS SHAPE, and left in place anyway -- read this before
#: quoting it. "Per member" implies the refusals are independent, and the
#: sixteenth round says they are not: they are a SUFFIX. Of the drags with a
#: refusal, 78% have every member from the first bad one onward refused;
#: P(refused | previous refused) is 81% against 34.5% after a kept one; and
#: members past the first refusal clear 28.6% against 84.8% before it. The
#: chain dies where it first goes wrong, which is what a continuous stroke
#: through a wrong character should do.
#:
#: `P_CLEAR_REFUSED` is therefore not "a refused member's own odds" but "what
#: the dead tail still happens to clear", and this pair is an approximation
#: of a prefix rule, not a description of one. It is not re-fitted here
#: because the evidence is 18 refusing drags and 7 post-refusal members --
#: enough to know the shape is wrong, not enough to fit the right one.
#: Fix it when a corpus can carry it, and re-read the tables when you do.
#:
#: What survives either shape: trimming buys nothing. Under independence the
#: refused members still clear a little, and under a prefix they were already
#: dead -- so in both, the trim removes members that cost nothing but stroke
#: time, and the only part of the check that pays is the rebuild.
P_CLEAR_MARKED = 0.898
P_CLEAR_REFUSED = 0.229

#: Members `verify_extend`'s rebuild adds, which the same round priced at
#: 22 of 22. Kept as its own name rather than folded into P_CLEAR_MARKED: it
#: is a different population -- partners the graph could not reach -- and n=22
#: is thin enough that it should be re-measured rather than trusted forever.
#: Both models are printed side by side, always. Correcting a cost model
#: re-scores every round already decided under it -- the tenth, eleventh,
#: thirteenth and fourteenth -- and this project's own rule is not to
#: re-score a measurement in silence, so the old column stays and stays
#: labelled. Delete the old one only once every round that quoted it has
#: been re-read under the new one.
P_CLEAR_ADDED = 1.000


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/samples.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["_session"] = f.parent.name
                rows.append(row)
    return [r for r in rows if r.get("schema", 1) >= 2]


def describe_run(rows: list[dict]) -> list[str]:
    """The conditions this corpus was collected under, read off the corpus.

    Two things a replay has to be able to say and, for three rounds, could
    not: which optional rules were armed, and which tsum was equipped. Schema
    3 records both. Everything here is derived rather than listed, so a rule
    added next month is reported by this function without it being edited.
    """
    out = []
    o = rows[0]["options"]
    schema = rows[0].get("schema", 1)
    builds = sorted({r.get("version", "?") for r in rows})
    out.append(f"built by: v{', v'.join(builds)}  (sample schema {schema})")

    # What this corpus is physically unable to answer, stated up front. The
    # capabilities arrived in a specific build, and a collection from before
    # it does not fail loudly -- it just quietly has no field, which reads the
    # same as a measurement that came back empty.
    cannot = []
    if schema < 3:
        cannot.append("which switches were on (only a curated subset of the "
                      "settings was recorded, so an absent one is UNKNOWN, "
                      "not default)")
        cannot.append("which tsum was equipped")
    if not any("cleared" in r for r in rows):
        cannot.append("what any drag actually cleared -- either the round was "
                      "played without `verify_clears`, or by a build that "
                      "could not write it down (before schema 3 the count "
                      "went only to the play log)")
    if cannot:
        out.append("cannot say: " + "; ".join(cannot))

    # A switch is "armed" if it is on, whatever its type: the point is to name
    # the ones that were doing something, not to know them in advance.
    armed = [k for k, v in sorted(o.items())
             if k.startswith(("verify_", "fit_", "bowl_", "first_leg", "base_"))
             and v not in (False, 0, 0.0, "", None)]
    out.append("armed: " + (", ".join(f"{k}={o[k]}" for k in armed) or "nothing"))

    mixed = {json.dumps(r["options"], sort_keys=True) for r in rows}
    if len(mixed) > 1:
        out.append(f"WARNING: {len(mixed)} different settings across these rows -- "
                   f"this is more than one experiment and the totals mix them")

    labs = [tuple(r["base"]["lab"]) for r in rows
            if isinstance(r.get("base"), dict) and r["base"].get("lab")]
    if labs:
        mid = [round(st.median(v[i] for v in labs)) for i in range(3)]
        spread = max(max(abs(v[i] - mid[i]) for i in range(3)) for v in labs)
        note = "  WARNING: more than one character in this corpus" if spread > 12 else ""
        out.append(f"equipped tsum: skill icon Lab {tuple(mid)} "
                   f"(spread {spread:.0f}){note}")
    else:
        out.append("equipped tsum: not recorded -- ask the player, and note it "
                   "in the scorecard row")
    return out


def dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def reach(row: dict, nodes=None) -> float:
    """How far the chain gets from the tsum being pressed."""
    head = row["tsums"][row["head"]]
    return max(dist(row["tsums"][i], head) for i in (nodes or row["proposed"]))


def label_bar(row: dict) -> float:
    """The bar the *label* was read at: the board's own floor times floor_mult."""
    mult = row["capture"].get("floor_mult", 0.0)
    return max(8.0, row["baseline"] * mult) if mult else 8.0


def stroke_time(row: dict, nodes) -> float:
    t = 2 * HOLD
    for a, b in zip(nodes, nodes[1:]):
        t += max(1, int(dist(row["tsums"][a], row["tsums"][b]) / STEP_PX)) * PER_STEP
    return t


def smd(a, b) -> float:
    """Standardised mean difference -- separation in units the features share."""
    pooled = math.sqrt((st.pvariance(a) + st.pvariance(b)) / 2) or 1e-9
    return abs(st.mean(b) - st.mean(a)) / pooled


def waste(rows: list[dict], min_chain: int) -> None:
    print("\n== what the game refused ==")
    print("  `proposed` is the chain the bot chose, `kept` what survived the game's")
    print("  own answer. A drag whose chain falls under min_chain runs and clears")
    print("  nothing at all.")
    def line(lbl, sel):
        if not sel:
            return
        p = [len(r["proposed"]) for r in sel]
        k = [len(r["kept"]) for r in sel]
        print(f"  {lbl:>8} {len(sel):5d} drags  proposed {st.mean(p):4.2f}  kept {st.mean(k):4.2f}  "
              f"refused {1 - sum(k)/sum(p):5.1%}  under min_chain "
              f"{sum(1 for v in k if v < min_chain)/len(sel):5.1%}  "
              f"clean {sum(1 for a, b in zip(p, k) if a == b)/len(sel):5.1%}")
    line("all", rows)
    line("normal", [r for r in rows if not r["fever"]])
    line("FEVER", [r for r in rows if r["fever"]])


def by_reach(rows: list[dict]) -> None:
    print("\n== clean rate by reach ==")
    print("  The one feature that predicts refusal. A chain that stays near the")
    print("  press is almost always right; one that crosses the board is not.")
    print(f"  {'reach':>12} {'drags':>6} {'clean':>7} {'kept/proposed':>14}")
    for lo, hi in ((0, 90), (90, 150), (150, 220), (220, 260), (260, 300), (300, 10**9)):
        sel = [r for r in rows if lo <= reach(r) < hi]
        if not sel:
            continue
        lbl = f"{lo}-{hi}px" if hi < 10**8 else f"over {lo}px"
        print(f"  {lbl:>12} {len(sel):6d} "
              f"{sum(1 for r in sel if len(r['kept']) == len(r['proposed']))/len(sel):7.1%} "
              f"{sum(len(r['kept']) for r in sel)/sum(len(r['proposed']) for r in sel):14.1%}")


def separation(rows: list[dict], aura: float) -> None:
    print("\n== what separates a kept member from a refused one ==")
    print(f"  Members inside the {aura:.0f}px glow are kept regardless, so only the ones")
    print("  outside it are evidence. Anything under ~0.2 separates nothing.")
    feats = {k: ([], []) for k in
             ("position in chain", "distance to pressed", "leg to previous",
              "detected radius", "tsums on board")}
    for r in rows:
        head = r["tsums"][r["head"]]
        kept = set(r["kept"])
        for pos, i in enumerate(r["proposed"]):
            d = dist(r["tsums"][i], head)
            if d <= aura:
                continue
            b = 0 if i in kept else 1
            prev = r["tsums"][r["proposed"][pos - 1]] if pos else r["tsums"][i]
            feats["position in chain"][b].append(pos)
            feats["distance to pressed"][b].append(d)
            feats["leg to previous"][b].append(dist(r["tsums"][i], prev))
            feats["detected radius"][b].append(r["tsums"][i]["r"])
            feats["tsums on board"][b].append(len(r["tsums"]))
    n0 = len(feats["position in chain"][0])
    n1 = len(feats["position in chain"][1])
    if not (n0 and n1):
        print("  not enough members outside the glow to say anything")
        return
    print(f"  {n0 + n1} far members, {n1/(n0+n1):.1%} refused")
    print(f"  {'feature':>22} {'kept':>8} {'refused':>9} {'separation':>11}")
    for name, (a, b) in feats.items():
        print(f"  {name:>22} {st.mean(a):8.2f} {st.mean(b):9.2f} {smd(a, b):11.2f}")


def refusal_by_position(rows: list[dict]) -> None:
    print("\n== where in the chain the refusals are ==")
    tally: dict[int, list[int]] = {}
    for r in rows:
        kept = set(r["kept"])
        for pos, i in enumerate(r["proposed"]):
            seen = tally.setdefault(pos, [0, 0])
            seen[0] += 1
            if i not in kept:
                seen[1] += 1
    for pos in sorted(tally):
        n, ref = tally[pos]
        if n < 20:
            continue
        print(f"  member {pos:>2}: {n:5d} proposed, {ref/n:5.1%} refused")


def sweep(rows: list[dict], min_chain: int) -> None:
    print("\n== verify_reach: buying the game's opinion only where it pays ==")
    print("  A checked drag is trimmed to what the game marked and abandoned if")
    print("  that falls under min_chain. What an UNCHECKED drag clears is where")
    print("  the two models differ, and it is the whole disagreement:")
    print("    all-or-nothing -- it clears in full if the game accepted every")
    print("      member, and nothing otherwise. What this script assumed until")
    print("      the fifteenth round, and what +27%/+17%/+7% was computed under.")
    print("    per-member -- each member clears on its own: 89.8% of the ones")
    print("      the game marked, 22.9% of the ones it refused. Measured.")
    print("  The time column is shared: the drag runs either way. Read the")
    print("  per-member half; the other is kept so nothing is re-scored in")
    print("  silence.")
    for cost in CHECK_COSTS:
        print(f"\n  per-check cost {cost:.2f}s")
        print(f"  {'':>12} {'':>7} {'   all-or-nothing (old)':>26} "
              f"{'      per-member (measured)':>28}")
        print(f"  {'verify past':>12} {'holds':>7} {'time':>8} "
              f"{'cleared':>8} {'clears/s':>9} {'vs off':>8} "
              f"{'cleared':>8} {'clears/s':>9} {'vs off':>8}")
        base_old = base_new = None
        for thr in (10**9, 300, 260, 220, 180, 150, 0):
            old = new = 0.0
            holds = 0
            total = 0.0
            for r in rows:
                kept_n, prop_n = len(r["kept"]), len(r["proposed"])
                refused_n = max(0, prop_n - kept_n)
                checking = thr == 0 or reach(r) > thr
                if checking:
                    holds += 1
                    total += cost
                    nodes = r["kept"] if kept_n >= min_chain else []
                    if nodes:
                        total += stroke_time(r, nodes)
                        old += len(nodes)
                        # A trimmed chain is all marked members, so only the
                        # marked probability applies.
                        new += P_CLEAR_MARKED * len(nodes)
                else:
                    total += stroke_time(r, r["proposed"])
                    if refused_n == 0:
                        old += prop_n
                    new += P_CLEAR_MARKED * kept_n + P_CLEAR_REFUSED * refused_n
            r_old, r_new = old / total, new / total
            base_old = base_old if base_old is not None else r_old
            base_new = base_new if base_new is not None else r_new
            lbl = "never" if thr == 10**9 else ("every drag" if thr == 0 else f"{thr}px")
            print(f"  {lbl:>12} {holds:7d} {total:8.1f} "
                  f"{old:8.0f} {r_old:9.2f} {r_old/base_old - 1:+8.1%} "
                  f"{new:8.0f} {r_new:9.2f} {r_new/base_new - 1:+8.1%}")


def truncation(rows: list[dict], min_chain: int) -> None:
    print("\n== truncating the tail instead of checking it ==")
    print("  Free -- no hold to pay -- but it shortens chains, and a long chain")
    print("  scores far more than the two short ones that clear the same tsums.")
    print("  Watch the >=6 column, not clears/s.")
    print(f"  {'rule':>26} {'drags':>6} {'mean len':>9} {'cleared':>8} {'time':>7} "
          f"{'clears/s':>9} {'>=6':>6}")

    def run(pick, lbl):
        lens, cleared, total = [], 0, 0.0
        for r in rows:
            nodes = pick(r)
            if len(nodes) < min_chain:
                continue
            total += stroke_time(r, nodes)
            lens.append(len(nodes))
            if all(n in set(r["kept"]) for n in nodes):
                cleared += len(nodes)
        c = Counter(lens)
        print(f"  {lbl:>26} {len(lens):6d} {st.mean(lens):9.2f} {cleared:8d} {total:7.1f} "
              f"{cleared/total:9.2f} {sum(v for k, v in c.items() if k >= 6)/len(lens):6.1%}")

    run(lambda r: r["proposed"], "today")
    for limit in (300, 260, 220, 180):
        def cut(r, limit=limit):
            head = r["tsums"][r["head"]]
            out = []
            for n in r["proposed"]:
                if dist(r["tsums"][n], head) > limit:
                    break
                out.append(n)
            return out
        run(cut, f"truncate past {limit}px")
    for cap in (4, 6, 8):
        run(lambda r, cap=cap: r["proposed"][:cap], f"max_chain {cap}")


def rebuild(rows: list[dict], min_chain: int, verify_at: float) -> None:
    """`verify_extend`: rebuild a checked chain from the marks, or only trim it.

    The trim spends the check's answer on the members it takes AWAY. The same
    reading also names partners the proposal never held -- see `recall()` above
    for how many -- and they cost nothing, because the press is already paid
    for.

    Two things this prices that a document cannot. First, identity: rebuilding
    with the bot's own `kind` ids loses, because `adjacency()` will not link
    across a `kind` difference, and that is the whole finding. Second, the
    reading. The corpus's `marked` was read at `floor_mult` over three frames;
    a live check reads ONE frame, so the rebuild is only affordable if the
    strict bar survives a single frame -- which this cannot tell you, and a
    round can. Both rows are printed at every cost so the comparison is never
    made against a reading the candidate would not pay for.
    """
    # Imported here rather than at the top: everything else in this file is
    # stdlib, and only this section needs the graph the game is played on. It
    # calls the shipped `chain_from_marks` rather than re-implementing it, so
    # the row cannot quietly disagree with the rule it is pricing.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ttheart_sender.game import tsum as T

    print("\n== verify_extend: rebuild the checked chain, or only trim it ==")
    print("  A trimmed drag keeps the proposed members the game marked; a")
    print("  rebuilt one walks the marks themselves, believing the game about")
    print("  identity as well as about refusal. Compared at an IDENTICAL")
    print("  reading cost -- a candidate that needs a dearer read has to beat")
    print("  the incumbent's read, not its own.")

    fired = [r for r in rows if reach(r) > verify_at]
    grown: dict[int, list[int]] = {}
    for r in fired:
        marked = [int(i) for i in (r.get("marked") or [])]
        if not marked:
            continue
        ts = [T.Tsum(float(t["x"]), float(t["y"]), float(t["r"]), int(t["kind"]),
                     (0, 0, 0)) for t in r["tsums"]]
        o = r["options"]
        grown[id(r)] = T.chain_from_marks(
            ts, r["proposed"], r["kept"], marked, float(r.get("radius", 25.0)),
            link_px=float(o.get("link_px", 105)), block=float(o.get("block", 1.25)),
            max_chain=int(o.get("max_chain", 12)))

    def run(cost: float, extend: bool):
        old = new = 0.0
        holds = 0
        total = 0.0
        lens = []
        for r in rows:
            kept = r["kept"]
            if reach(r) > verify_at:
                holds += 1
                total += cost
                nodes = grown.get(id(r)) if extend else None
                nodes = nodes if nodes is not None else kept
                nodes = nodes if len(nodes) >= min_chain else []
                if nodes:
                    total += stroke_time(r, nodes)
                    old += len(nodes)
                    # A rebuild carries two populations: the marked members
                    # the trim would also have dragged, and the ones the marks
                    # handed over, which the fifteenth round priced separately.
                    added = len(set(nodes) - set(kept))
                    new += (P_CLEAR_MARKED * (len(nodes) - added)
                            + P_CLEAR_ADDED * added)
                lens.append(len(nodes))
            else:
                total += stroke_time(r, r["proposed"])
                prop_n, kept_n = len(r["proposed"]), len(kept)
                refused_n = max(0, prop_n - kept_n)
                if refused_n == 0:
                    old += prop_n
                new += P_CLEAR_MARKED * kept_n + P_CLEAR_REFUSED * refused_n
                lens.append(prop_n if refused_n == 0 else 0)
        return old, new, total, holds, lens

    print(f"\n  {'':>12} {'':>9} {'':>7} {'':>8} {'  all-or-nothing':>20}"
          f" {'   per-member':>20}")
    print(f"  {'reading':>12} {'rule':>9} {'holds':>7} {'time':>8} "
          f"{'cleared':>8} {'clears/s':>9} {'vs trim':>8} "
          f"{'cleared':>8} {'clears/s':>9} {'vs trim':>8} {'>=6':>6}")
    for cost in CHECK_COSTS:
        base_old = base_new = None
        for extend in (False, True):
            o, n, t, h, lens = run(cost, extend)
            r_old, r_new = o / t, n / t
            base_old = base_old if base_old is not None else r_old
            base_new = base_new if base_new is not None else r_new
            long = sum(v >= 6 for v in lens) / len(lens)
            print(f"  {cost:11.2f}s {'rebuild' if extend else 'trim':>9} "
                  f"{h:7d} {t:8.1f} "
                  f"{o:8.0f} {r_old:9.2f} {r_old/base_old - 1:+8.1%} "
                  f"{n:8.0f} {r_new:9.2f} {r_new/base_new - 1:+8.1%} {long:6.1%}")

    same = sum(1 for r in fired if len(grown.get(id(r), r["kept"])) > len(r["kept"]))
    print(f"\n  the marks grew {same} of {len(fired)} checked chains "
          f"({same / max(1, len(fired)):.0%})")
    print("  A rebuild that cannot beat the trim hands the trim back, so this")
    print("  rule can add clears and never costs any -- what it can cost is a")
    print("  wrong member, and only a played round prices that.")


def readability(rows: list[dict], verify_at: float, min_chain: int) -> None:
    print("\n== readings the trim should not have trusted ==")
    print("  A trim is only as good as the frame it read. A board that shows no")
    print("  mark at all has not said 'refused' -- it has said nothing.")
    empty = [r for r in rows if not r["marked"]]
    silent = [r for r in rows if r["marks"][r["head"]] < label_bar(r)]
    print(f"  no mark anywhere on the board: {len(empty):4d} ({len(empty)/len(rows):5.1%} of drags)")
    print(f"  the pressed tsum itself did not react: {len(silent):4d} ({len(silent)/len(rows):5.1%})")
    checked = [r for r in rows if reach(r) > verify_at]
    if checked:
        bad = [r for r in checked if not r["marked"]]
        cancelled = [r for r in bad if len(r["kept"]) < min_chain]
        print(f"  of the {len(checked)} drags a verify_reach {verify_at:.0f} check would fire on, "
              f"{len(bad)} read nothing,")
        print(f"  and {len(cancelled)} of those are chains the trim cancels on a frame "
              f"that said nothing.")


def recall(rows: list[dict], aura: float) -> None:
    print("\n== what the game offered that the bot never asked for ==")
    print("  Every press is the game naming its own partners. A chain far shorter")
    print("  than that set is the graph, not the game, doing the refusing.")
    marked = [len(r["marked"]) for r in rows]
    missed = []
    for r in rows:
        head = r["tsums"][r["head"]]
        prop = set(r["proposed"])
        missed.append(sum(1 for i in r["marked"]
                          if i not in prop and dist(r["tsums"][i], head) > aura))
    print(f"  tsums the game marked per press: mean {st.mean(marked):.2f}, "
          f"median {st.median(marked):.0f}")
    print(f"  marked, outside the glow, never proposed: mean {st.mean(missed):.2f}")
    print(f"  the chain the bot proposed: mean {st.mean([len(r['proposed']) for r in rows]):.2f}, "
          f"dragged after the trim: {st.mean([len(r['kept']) for r in rows]):.2f}")


def health(rows: list[dict]) -> None:
    print("\n== board and detection health ==")
    def pct(xs, q):
        xs = sorted(xs)
        return xs[int(q * (len(xs) - 1))]
    counts = [len(r["tsums"]) for r in rows]
    floors = [r["baseline"] for r in rows]
    print(f"  detections per board: p10 {pct(counts,.1)} p50 {pct(counts,.5)} "
          f"p90 {pct(counts,.9)} max {max(counts)}   (a real board holds ~50-70; "
          f"over 75 is over-splitting)")
    print(f"  over-split boards: {sum(1 for c in counts if c > 75)}    "
          f"boards under the 20-tsum gate: {sum(1 for c in counts if c < 20)}")
    print(f"  board noise floor: p10 {pct(floors,.1):.1f} p50 {pct(floors,.5):.1f} "
          f"p90 {pct(floors,.9):.1f} max {max(floors):.1f}")
    print(f"  fever drags: {sum(r['fever'] for r in rows)/len(rows):.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dataset", type=Path,
                    help="the dataset directory holding the session folders")
    ap.add_argument("--aura", type=float, default=90.0,
                    help="members this close to the press are kept regardless, "
                         "so they are excluded from every measurement here")
    ap.add_argument("--verify-at", type=float, default=260.0,
                    help="the verify_reach threshold to report readability against")
    args = ap.parse_args()

    rows = load(args.dir)
    if not rows:
        print(f"no schema 2 samples under {args.dir} -- nothing to replay")
        return 1
    opts = rows[0]["options"]
    min_chain = int(opts.get("min_chain", 3))
    print(f"{len(rows)} drags over {len({r['_session'] for r in rows})} session(s)")
    print("  play settings: " + "  ".join(
        f"{k} {opts[k]}" for k in
        ("mode", "k", "link_px", "block", "purity", "min_chain", "max_chain")
        if k in opts))
    # Every switch that was on while this was collected, named without anyone
    # keeping a list up to date -- see `describe_run`. A replay that cannot
    # state the conditions it is replaying is the eleventh round's mistake.
    for line in describe_run(rows):
        print(f"  {line}")
    print(f"  capture: {rows[0]['capture']}")
    versions = Counter(r["version"] for r in rows)
    if len(versions) > 1:
        print(f"  WARNING: mixed versions {dict(versions)} -- settings may differ between rows")

    waste(rows, min_chain)
    by_reach(rows)
    separation(rows, args.aura)
    refusal_by_position(rows)
    sweep(rows, min_chain)
    truncation(rows, min_chain)
    rebuild(rows, min_chain, args.verify_at)
    readability(rows, args.verify_at, min_chain)
    recall(rows, args.aura)
    health(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
