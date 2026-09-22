"""Does NAMING the board find partners that colour clustering misses?

THE QUESTION THIS ANSWERS, in the player's own words: "if it doesn't recognise
in the first place but the app plays the game smoothly, then the character
model isn't needed -- what is the point?"

That is the right question and nothing here had answered it. Every other
scorer asks whether the model reads a crop CORRECTLY. None asks whether being
correct changes the grouping the round actually plays on, which is the only
reason the model exists.

WHAT THE ROUND ACTUALLY PLAYS ON. `adjacency()` and `find_chains()` group by
`kind`. Without the model that is a per-frame k-means colour cluster. With it,
a tsum the model is confident about gets `CHARACTER_KIND + index` and one it is
unsure of KEEPS its cluster -- the hybrid rule. So the two groupings are
directly comparable, and this compares them.

THE TRUTH IS FREE. Holding a tsum makes the game light up everything that is
both the same character and reachable, and `marked` in `samples.jsonl` records
it. Over 4,306 collected drags the game marks a mean 6.1 tsums per press and
**4.0 of them were never proposed** -- the recall gap, and the biggest single
number in the collection. If naming closes any of that gap it is worth having.
If it does not, it is not, whatever it scores on crops.

    python scripts/naming_gain.py
    python scripts/naming_gain.py --samples 400 --min-visible 0.0

TWO NUMBERS, AND THE SECOND IS THE ONE THAT BITES
-------------------------------------------------

* **recall** -- of the partners the game marked, how many share the head's
  group. Higher is a longer chain.
* **precision** -- of the tsums sharing the head's group, how many the game
  actually marked. Lower means the grouping invents partners, and a chain
  built through one is a stroke over a tsum that will not clear.

A rule that raises recall by inventing partners has not helped. Both are
reported for both groupings, on the same boards, and the boards are the ones
the collector sampled with the model OFF -- so the baseline is what the round
really did, not a reconstruction of it.

WHAT THIS IS NOT. Marks are "same character AND reachable". A partner the game
did not mark may be the right character and simply out of reach, which counts
against precision here. That biases both groupings the same way, so the
COMPARISON stands even though neither absolute number is a clean identity
score.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import cv2  # noqa: F401 - imported for the same reason board_check does
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ttheart_sender.game import tsum as T  # noqa: E402
from ttheart_sender.game import crop as crop_rules  # noqa: E402


def boards(root: Path, want: int, seed: int):
    """Sampled drags that carry marks, played WITHOUT the model.

    A round that ran the model already has `kind` overwritten, so its baseline
    is not a baseline. `--recolour` and `--kinds` renumber it for their own
    reasons and are excluded for the same one.
    """
    rows = []
    for jl in sorted(root.glob("*/samples.jsonl")):
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            o = d.get("options") or {}
            if o.get("character") or o.get("recolour") or o.get("kinds"):
                continue
            if not d.get("tsums") or not d.get("marked"):
                continue
            head = d.get("head")
            if head is None or not (0 <= head < len(d["tsums"])):
                continue
            img = jl.parent / ("%04d_before.jpg" % d.get("index", -1))
            if img.exists():
                rows.append((jl.parent.name, d, img))
    random.Random(seed).shuffle(rows)
    return rows[:want] if want else rows


def merged(kinds: list, names: dict) -> list:
    """Grouping that lets a name MERGE two colour clusters, never split one.

    WHY THIS RULE EXISTS. The shipped rule REPLACES `kind` with the model's
    identity wherever it is confident. That is fine when the whole group is
    named and destructive when it is not -- and it usually is not: the chain
    head is named on 17% of drags, so on the other 83% a named partner is
    given an identity the unnamed head does not share, and leaves the head's
    group. Naming one member of a pair splits the pair.

    So: start from the clusters, and only ever join. Two tsums the model names
    the same way end up in one group whatever k-means thought; two tsums
    k-means grouped stay grouped whatever the model thinks of one of them.
    Recall can only go up and precision can only go down, which is the honest
    shape of a merge and is exactly what the caller measures.
    """
    parent = {k: k for k in set(kinds)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_name: dict = {}
    for i, nm in names.items():
        by_name.setdefault(nm, []).append(i)
    for members in by_name.values():
        for j in members[1:]:
            union(kinds[members[0]], kinds[j])
    return [find(k) for k in kinds]


def score(group: list, head: int, truth: set):
    """Recall and precision of one grouping against the game's marks."""
    same = {i for i, g in enumerate(group)
            if i != head and g == group[head]}
    if not truth:
        return None
    hit = len(same & truth)
    return (hit / len(truth), hit / len(same) if same else 1.0, len(same))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--samples", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.85)
    ap.add_argument("--min-visible", type=float, default=T.CHARACTER_MIN_VISIBLE)
    ap.add_argument("--crop-profile", default="")
    ap.add_argument("--rule", choices=("replace", "merge"), default="replace",
                    help="`replace` is what ships: a named tsum takes the "
                         "model's identity and leaves whatever cluster it was "
                         "in. `merge` only ever joins clusters the model names "
                         "the same way, and never splits one")
    args = ap.parse_args()

    # The RUNTIME's model object, so this scores the same pictures a round
    # does, through the same crop window and the same capped batch.
    model = T.load_character_model(args.model, args.confidence, args.min_visible)
    if args.crop_profile:
        model.profile = crop_rules.profile(args.crop_profile)

    rows = boards(args.dir, args.samples, args.seed)
    if not rows:
        print("no drags with marks were collected with the model off")
        return 2

    base_r, base_p, name_r, name_p = [], [], [], []
    base_n, name_n, moved, named_head = [], [], 0, 0
    for _sess, d, img_path in rows:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        # `_before.jpg` IS the board crop -- `board` records where that rect
        # sat in the frame, and every tsum coordinate is already relative to
        # it. Cropping again by that rect (the first version of this script
        # did) feeds the model a sliver of the board and refuses 82% of the
        # heads for being off the edge. `board_check.py` had it right.
        radius = float(d["radius"])
        crop = frame
        tsums = [T.Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                        kind=int(t["kind"]), colour=(0, 0, 0))
                 for t in d["tsums"]]
        head = int(d["head"])
        truth = {int(i) for i in d["marked"]
                 if 0 <= int(i) < len(tsums) and int(i) != head}
        if not truth:
            continue

        before = [t.kind for t in tsums]
        b = score(before, head, truth)
        if args.rule == "replace":
            # The HYBRID rule the round plays: named tsums take the model's
            # identity, the rest keep the cluster they already had.
            model.apply(crop, tsums, radius)
            after = [t.kind for t in tsums]
        else:
            usable, prob = model.probabilities(crop, tsums, radius)
            got = {}
            for n_, i in enumerate(usable):
                best = int(prob[n_].argmax())
                if float(prob[n_][best]) >= args.confidence:
                    got[i] = best
            after = merged(before, got)
        a = score(after, head, truth)
        if b is None or a is None:
            continue
        if after[head] >= T.CHARACTER_KIND:
            named_head += 1
        if after != before:
            moved += 1
        base_r.append(b[0]); base_p.append(b[1]); base_n.append(b[2])
        name_r.append(a[0]); name_p.append(a[1]); name_n.append(a[2])

    n = len(base_r)
    if not n:
        print("no scorable drags")
        return 2

    print("%d sampled drags, all played with the model OFF" % n)
    print("grouping rule: %s" % args.rule)
    print("the head was named on %d of them (%.0f%%)"
          % (named_head, 100 * named_head / n))
    print("the grouping changed at all on %d (%.0f%%)"
          % (moved, 100 * moved / n))
    print()
    print("                     recall   precision   partners offered")
    print("  colour clusters   %6.1f%%     %6.1f%%   %6.2f"
          % (100 * st.mean(base_r), 100 * st.mean(base_p), st.mean(base_n)))
    print("  + character model %6.1f%%     %6.1f%%   %6.2f"
          % (100 * st.mean(name_r), 100 * st.mean(name_p), st.mean(name_n)))
    print("  change            %+6.1fpp    %+6.1fpp   %+6.2f"
          % (100 * (st.mean(name_r) - st.mean(base_r)),
             100 * (st.mean(name_p) - st.mean(base_p)),
             st.mean(name_n) - st.mean(base_n)))
    print()

    # Paired, because the two groupings are scored on the SAME drag. An
    # unpaired comparison of these means would be swamped by how much the
    # boards differ from each other.
    d_r = [a - b for a, b in zip(name_r, base_r)]
    better = sum(1 for v in d_r if v > 1e-9)
    worse = sum(1 for v in d_r if v < -1e-9)
    print("  paired on the same drag: recall better on %d, worse on %d, "
          "unchanged on %d" % (better, worse, n - better - worse))
    if better + worse:
        # A sign test, which needs no assumption about the shape of the
        # difference -- the right tool when most drags are unchanged.
        from math import comb
        k, m = min(better, worse), better + worse
        p = sum(comb(m, i) for i in range(k + 1)) / (2 ** m) * 2
        print("  sign test p = %.4f%s" % (min(p, 1.0),
                                          "" if p < 0.05 else "  (not significant)"))
    print()
    print("READ IT LIKE THIS. Recall up is a longer chain. Precision down is a")
    print("stroke passing over a tsum that will not clear. A rule that buys")
    print("recall with precision has not helped, and only a played round can")
    print("say what the trade is worth -- but if recall does not move at all,")
    print("naming is not changing what the round plays and the labelling is")
    print("not what is holding the score back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
