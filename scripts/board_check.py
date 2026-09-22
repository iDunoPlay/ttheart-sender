"""Validate recognition on whole stored boards -- the app's own path, offline.

Every other scorer here starts partway down the pipeline. `recog_eval.py` reads
crops that `crops.py` cut and saved months ago, so it cannot see a regression in
detection or in the crop window. `mark_probe.py` and `recog_by_tsum.py` only
ever look at the tsums the game happened to mark. None of them shows you a
board.

This runs the pipeline a round runs -- the stored detections, the runtime's own
`CharacterModel.probabilities`, the same visibility floor and confidence floor
-- over saved `_before.jpg` screenshots, and reports what the app would have
believed about each board. It also writes the board back out with the names
drawn on it, because for recognition the fastest check a person can make is to
look.

    python scripts/board_check.py --samples 200
    python scripts/board_check.py --samples 40 --overlay out/
    python scripts/board_check.py --profile beans_camo_vil --samples 60

**Two ground truths, scored separately and never pooled**, because they answer
different questions and have different biases:

* ``human``  -- a crop a person put in a folder. Exact, but every labelled crop
  is at least 0.55 visible, so it says nothing about the buried four fifths of
  a board.
* ``marks``  -- the game lit this tsum up while the head was held, so it is the
  same character as the head. Free, reaches every visibility, but the head's
  own name comes from the model, so it is a FLOOR rather than an estimate.

Anything with neither is counted as unscored and drawn, never guessed at.

**What this does not do:** it does not re-detect. The labelled crops are keyed
by the index of a tsum in the STORED detection list, so re-detecting would
break the only exact ground truth available. Detection stability is a separate
question and `scripts/track_probe.py` measures it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game import crop as crop_rules  # noqa: E402
from ttheart_sender.game import profiles  # noqa: E402
from ttheart_sender.game import tsum as T  # noqa: E402

NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2,})_v(?P<vis>[\d.]+)$")
NOT_CHARACTERS = {"board", "score", "junk"}
BANDS = ((0.00, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01))


def labels(root: Path) -> dict:
    """(session, sample, index) -> the class a person put it in."""
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in NOT_CHARACTERS:
            continue
        for p in d.glob("*.png"):
            m = NAME.match(p.stem)
            if m:
                out[(m.group("sess"), int(m.group("sample")),
                     int(m.group("idx")))] = d.name
    return out


def equipped(root: Path) -> dict:
    """session -> profile name, from the skill icon's own colour."""
    out = {}
    for jl in sorted(root.glob("*/samples.jsonl")):
        labs = []
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            b = d.get("base") or {}
            if b.get("lab"):
                labs.append(b["lab"])
        if labs:
            med = np.median(np.asarray(labs, float), axis=0)
            out[jl.parent.name] = (profiles.identify([float(v) for v in med])
                                   or "unprofiled")
    return out


def samples(root: Path, want: int, seed: int, profile: str, who: dict):
    rows = []
    for jl in sorted(root.glob("*/samples.jsonl")):
        sess = jl.parent.name
        if profile and who.get(sess) != profile:
            continue
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            o = d.get("options") or {}
            # Rounds that ran the model already have `kind` overwritten, and
            # `--recolour`/`--kinds` renumber it. Neither can be compared.
            if o.get("character") or o.get("recolour") or o.get("kinds"):
                continue
            img = jl.parent / ("%04d_before.jpg" % d.get("index", -1))
            if d.get("tsums") and img.exists():
                rows.append((sess, d, img))
    random.Random(seed).shuffle(rows)
    return rows[:want] if want else rows


def band_of(v: float):
    for lo, hi in BANDS:
        if lo <= v < hi:
            return (lo, hi)
    return BANDS[-1]


def draw(img, tsums, radius, named, truth, out: Path):
    """The board with what the app believes written on it.

    Colour is the verdict, so a wrong name is findable at a glance:
    green right, red wrong, amber named but unscorable, grey never asked.
    """
    vis = img.copy()
    for i, t in enumerate(tsums):
        v = t["r"] / radius if radius else 0.0
        name, conf = named.get(i, (None, 0.0))
        want = truth.get(i)
        if name is None:
            colour, text = (110, 110, 110), ""
        elif want is None:
            colour, text = (0, 190, 235), f"{name} {conf:.2f}"
        elif want == name:
            colour, text = (60, 200, 60), f"{name} {conf:.2f}"
        else:
            colour, text = (60, 60, 235), f"{name}!={want}"
        cv2.circle(vis, (int(t["x"]), int(t["y"])), max(3, int(t["r"])),
                   colour, 1 if name is None else 2)
        if text:
            cv2.putText(vis, text, (int(t["x"]) - 22, int(t["y"]) - int(t["r"]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
        cv2.putText(vis, f"{v:.2f}", (int(t["x"]) - 12, int(t["y"]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), vis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="",
                    help="only boards played with this equipped tsum")
    ap.add_argument("--confidence", type=float, default=0.85,
                    help="the shipped --character-confidence")
    ap.add_argument("--min-visible", type=float, default=T.CHARACTER_MIN_VISIBLE,
                    help="the shipped --character-min-visible. 0 asks the model "
                         "about the whole board, which is what the play loop "
                         "refuses to do and why the floor exists")
    ap.add_argument("--crop-profile", default="",
                    help="cut crops under this profile instead of the "
                         "one the model records. `padded_v1` extends "
                         "the frame so a tsum at the rim yields a crop "
                         "instead of nothing -- 40.7%% of tsums above "
                         "the visibility floor are refused without it")
    ap.add_argument("--with-truth", action="store_true",
                    help="prefer boards that carry ground truth. Without it a "
                         "random draw is mostly unscorable: the 805 labelled "
                         "crops are spread over 738 sessions")
    ap.add_argument("--overlay", type=Path,
                    help="write each board back out with the names drawn on it")
    args = ap.parse_args()

    # The RUNTIME's model object, not a re-implementation. `probabilities` was
    # split out of `apply` for exactly this: so an offline check scores the
    # same pictures a round does, through the same crop window and the same
    # capped batch.
    model = T.load_character_model(args.model, args.confidence, args.min_visible)
    if args.crop_profile:
        # Overriding what the model recorded is a measurement, not a fix: it
        # asks "how much MORE of the board would this profile let us reach",
        # and the answer is about coverage. Accuracy under a profile the model
        # was not trained on means nothing.
        model.profile = crop_rules.profile(args.crop_profile)
        print("crop profile OVERRIDDEN to %s -- coverage is comparable, "
              "accuracy is not" % model.profile.name)
    classes = model.classes

    who = equipped(args.dir)
    truth_all = labels(args.crops)
    rows = samples(args.dir, 0, args.seed, args.profile, who)
    if args.with_truth:
        # A random draw is mostly unscorable: 805 labelled crops are spread
        # across 738 sessions, so 300 random boards carried 22 of them.
        have = {(s_, samp) for s_, samp, _ in truth_all}
        rows.sort(key=lambda r: (r[0], r[1]["index"]) not in have)
    rows = rows[:args.samples] if args.samples else rows
    if not rows:
        print("no comparable boards found")
        return 2
    if args.overlay:
        args.overlay.mkdir(parents=True, exist_ok=True)

    per_band = defaultdict(lambda: [0, 0, 0, 0])   # band -> [seen, asked, named, right]
    human = defaultdict(lambda: [0, 0])            # band -> [n, right]
    marks = defaultdict(lambda: [0, 0])
    detections = named_total = 0
    boards = 0
    ms = []
    confusions = Counter()
    no_truth = 0

    for sess, d, img_path in rows:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ts = d["tsums"]
        radius = float(d.get("radius") or 25.0)
        model.buried = model.named = model.seen = 0
        t0 = time.perf_counter()
        usable, prob = model.probabilities(img, [
            T.Tsum(t["x"], t["y"], t["r"], 0, (0, 0, 0)) for t in ts], radius)
        ms.append((time.perf_counter() - t0) * 1000.0)
        boards += 1
        detections += len(ts)

        named = {}
        for n, i in enumerate(usable):
            best = int(prob[n].argmax())
            if float(prob[n][best]) >= args.confidence:
                named[i] = (classes[best], float(prob[n][best]))
        named_total += len(named)

        # Ground truth, per tsum, from whichever source has it.
        truth = {}
        for i in range(len(ts)):
            got = truth_all.get((sess, d["index"], i))
            if got is not None:
                truth[i] = got
        head = d.get("head")
        marked = [i for i in (d.get("marked") or [])
                  if isinstance(i, int) and i < len(ts)]
        head_name = named.get(head, (None, 0.0))[0] if head is not None else None
        head_ok = (head is not None and head_name is not None
                   and ts[head]["r"] / radius >= 0.65)

        asked = set(usable)
        for i, t in enumerate(ts):
            v = t["r"] / radius if radius else 0.0
            b = band_of(v)
            cell = per_band[b]
            cell[0] += 1
            # ASKED and NAMED are different failures, and lumping them together
            # hides the one that matters. A tsum that never reached the net was
            # refused by the crop -- under the visibility floor, or clipped by
            # the frame edge -- and no confidence setting can rescue it. One
            # that was asked and not named is the confidence floor talking.
            if i in asked:
                cell[1] += 1
            got = named.get(i)
            if got:
                cell[2] += 1
            want = truth.get(i)
            if want is not None and got:
                cell[3] += int(got[0] == want)
                hc = human[b]
                hc[0] += 1
                hc[1] += int(got[0] == want)
                if got[0] != want:
                    confusions[(want, got[0])] += 1
            elif got and head_ok and i in marked and i != head:
                # The game says this is the head's character. A floor, and
                # kept apart from the human column for that reason.
                mc = marks[b]
                mc[0] += 1
                mc[1] += int(got[0] == head_name)
            elif got:
                no_truth += 1

        if args.overlay:
            draw(img, ts, radius, named, truth,
                 args.overlay / f"{sess}_{d['index']:04d}.png")

    print("\n%d boards, %d detections, equipped tsum: %s"
          % (boards, detections, args.profile or "all"))
    print("model floors: >=%.2f visible, >=%.2f confident   (the shipped values)"
          % (args.min_visible, args.confidence))

    print("\n%-16s%9s%8s%8s%22s%22s"
          % ("visibility", "tsums", "asked", "named", "human-labelled",
             "game's marks"))
    for b in BANDS:
        seen, ask, nm, _ = per_band[b]
        if not seen:
            continue
        row = "%-16s%9d%7.0f%%%7.0f%%" % ("%.2f-%.2f" % b, seen,
                                          100 * ask / seen, 100 * nm / seen)
        for src in (human, marks):
            n, ok = src[b]
            if not n:
                row += "%22s" % "--"
                continue
            p_ = ok / n
            se = sqrt(max(p_ * (1 - p_), 1e-9) / n)
            row += "%22s" % ("%.0f+/-%.0f%%  n=%d" % (100 * p_, 200 * se, n))
        print(row)

    tot_h = [sum(v[0] for v in human.values()), sum(v[1] for v in human.values())]
    tot_m = [sum(v[0] for v in marks.values()), sum(v[1] for v in marks.values())]
    print("\n  named %d of %d detections (%.0f%%); %d named with no ground "
          "truth of either kind" % (named_total, detections,
                                    100 * named_total / max(1, detections),
                                    no_truth))
    if tot_h[0]:
        print("  against human labels : %.1f%% of %d"
              % (100 * tot_h[1] / tot_h[0], tot_h[0]))
    if tot_m[0]:
        print("  against the marks    : %.1f%% of %d  (a FLOOR -- the head's "
              "name is the model's own)" % (100 * tot_m[1] / tot_m[0], tot_m[0]))
    print("  recognition: %.1f ms a board, %d boards timed"
          % (float(np.mean(ms)), len(ms)))

    if confusions:
        print("\n  worst confusions against human labels (true -> called):")
        for (a, b_), c in confusions.most_common(5):
            print("    %s -> %s   %d" % (a, b_, c))

    if args.overlay:
        print("\n  wrote %d board(s) to %s" % (boards, args.overlay))
        print("  green = right, red = wrong, amber = named but unscorable,")
        print("  grey = never asked (below the visibility floor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
