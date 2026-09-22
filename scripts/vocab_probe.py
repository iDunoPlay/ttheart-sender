"""A round holds FIVE characters. Does telling the model that help?

THE PLAYER'S OWN RULE, and it is a hard one: every round is the equipped tsum
-- guaranteed present -- plus four random others. Five names, no more. The
model does not know it, and names whatever each crop looks like most.

WHAT THAT COSTS, measured over twelve Beast-equipped rounds:

    Duchess 75, Pascal 71, unknown_lightball 2, Piglet 2, Sisu 2
    Monstro 57, CheshireCat 52, Baymax 48, unknown_lightball 4, Tramp 2, Beast 1
    StreetwearStitch 36, Pascal 18, Tramp 11, Piglet 9, unknown_lightball 3,
        CheshireCat 2, Lucifer 1, Monstro 1, Rex 1

Two to four names carry the round and the rest is a tail of ones and twos. The
tail is not a character that appeared briefly -- a round's cast does not
change -- it is a misread, and every one of them takes a real tsum out of the
group it belonged to.

So: accumulate votes over the round, keep the top N names, and give anything
outside that vocabulary back to its colour cluster. Nothing is invented; the
only claim is that a name nobody voted for more than twice is wrong.

    python scripts/vocab_probe.py                  # top 5, the game's rule
    python scripts/vocab_probe.py --vocab 0        # off, for the baseline
    python scripts/vocab_probe.py --rule merge

SCORED THE WAY `naming_gain.py` SCORES, against the game's own marks, so the
number is comparable with the one that said naming costs 5.2pp of recall.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ttheart_sender.game import tsum as T  # noqa: E402


def _naming_gain():
    """`score` and `merged`, from the script that established the baseline."""
    spec = importlib.util.spec_from_file_location(
        "naming_gain", HERE / "naming_gain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rounds(root: Path, want: int, seed: int):
    """Sampled drags that carry marks, grouped by the ROUND they came from.

    Grouped, because a vocabulary is a per-round thing: it is built from what
    the model saw earlier in the same round and applied to what it sees next.
    """
    out: dict = {}
    for jl in sorted(root.glob("*/samples.jsonl")):
        rows = []
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            o = d.get("options") or {}
            if o.get("character") or o.get("recolour") or o.get("kinds"):
                continue
            head = d.get("head")
            if (not d.get("tsums") or not d.get("marked")
                    or head is None or not (0 <= head < len(d["tsums"]))):
                continue
            img = jl.parent / ("%04d_before.jpg" % d.get("index", -1))
            if img.exists():
                rows.append((d, img))
        if rows:
            out[jl.parent.name] = rows
    keys = sorted(out)
    random.Random(seed).shuffle(keys)
    picked, n = {}, 0
    for k in keys:
        picked[k] = out[k]
        n += len(out[k])
        if want and n >= want:
            break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.85)
    ap.add_argument("--min-visible", type=float, default=T.CHARACTER_MIN_VISIBLE)
    ap.add_argument("--vocab", type=int, default=5,
                    help="how many names a round may use. 5 is the game's own "
                         "rule -- the equipped tsum plus four random. 0 turns "
                         "the vocabulary off, which is what ships today")
    ap.add_argument("--rule", choices=("replace", "merge"), default="replace")
    args = ap.parse_args()

    gain = _naming_gain()
    model = T.load_character_model(args.model, args.confidence,
                                   args.min_visible)
    top_kind = T.CHARACTER_KIND + len(model.classes)

    picked = rounds(args.dir, args.samples, args.seed)
    if not picked:
        print("no drags with marks were collected with the model off")
        return 2

    base_r, base_p, vocab_r, vocab_p = [], [], [], []
    dropped, kept, vocab_sizes = 0, 0, []

    for _sess, rows in picked.items():
        # PASS ONE: what does the model think this round is made of? A real
        # round is played the same way -- the vocabulary is whatever the
        # frames so far have voted for, and by the second or third board it
        # has settled.
        votes: Counter = Counter()
        loaded = []
        for d, img_path in rows:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            radius = float(d["radius"])
            tsums = [T.Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                            kind=int(t["kind"]), colour=(0, 0, 0))
                     for t in d["tsums"]]
            usable, prob = model.probabilities(frame, tsums, radius)
            named = {}
            for n_, i in enumerate(usable):
                best = int(prob[n_].argmax())
                if float(prob[n_][best]) >= args.confidence:
                    named[i] = model.classes[best]
            votes.update(named.values())
            loaded.append((d, tsums, named))
        if not loaded:
            continue
        allowed = ({n for n, _ in votes.most_common(args.vocab)}
                   if args.vocab else set(votes))
        vocab_sizes.append(len(allowed))

        # PASS TWO: play the round twice, with and without the vocabulary.
        for d, tsums, named in loaded:
            head = int(d["head"])
            truth = {int(i) for i in d["marked"]
                     if 0 <= int(i) < len(tsums) and int(i) != head}
            if not truth:
                continue
            before = [t.kind for t in tsums]
            b = gain.score(before, head, truth)

            free = dict(named)
            limited = {i: n for i, n in named.items() if n in allowed}
            dropped += len(free) - len(limited)
            kept += len(limited)

            after = _apply(before, limited, model, args.rule, gain)
            a = gain.score(after, head, truth)
            if b is None or a is None:
                continue
            plain = _apply(before, free, model, args.rule, gain)
            p = gain.score(plain, head, truth)
            base_r.append(p[0]); base_p.append(p[1])
            vocab_r.append(a[0]); vocab_p.append(a[1])

    n = len(base_r)
    if not n:
        print("no scorable drags")
        return 2
    print("%d rounds, %d scorable drags, rule: %s"
          % (len(picked), n, args.rule))
    print("vocabulary: top %s name(s); the round holds 5 (base + 4 random)"
          % (args.vocab or "no limit"))
    if vocab_sizes:
        print("names a round actually used: median %d"
              % sorted(vocab_sizes)[len(vocab_sizes) // 2])
    print("names refused as outside it: %d of %d (%.0f%%)"
          % (dropped, dropped + kept,
             100 * dropped / max(dropped + kept, 1)))
    print()
    print("                        recall   precision")
    print("  every name it likes  %6.1f%%     %6.1f%%"
          % (100 * st.mean(base_r), 100 * st.mean(base_p)))
    print("  top-%-2d only          %6.1f%%     %6.1f%%"
          % (args.vocab, 100 * st.mean(vocab_r), 100 * st.mean(vocab_p)))
    print("  change               %+6.1fpp    %+6.1fpp"
          % (100 * (st.mean(vocab_r) - st.mean(base_r)),
             100 * (st.mean(vocab_p) - st.mean(base_p))))
    better = sum(1 for a, b in zip(vocab_r, base_r) if a > b + 1e-9)
    worse = sum(1 for a, b in zip(vocab_r, base_r) if a < b - 1e-9)
    print()
    print("  paired: recall better on %d, worse on %d, unchanged on %d"
          % (better, worse, n - better - worse))
    return 0


def _apply(before, named, model, rule, gain):
    """The grouping a round would play, given these names."""
    if rule == "merge":
        return gain.merged(before, {i: n for i, n in named.items()})
    out = list(before)
    index = {c: i for i, c in enumerate(model.classes)}
    for i, name in named.items():
        out[i] = T.CHARACTER_KIND + index[name]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
