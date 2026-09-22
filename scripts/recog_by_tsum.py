"""Is recognition better on some equipped tsums than others? No labels needed.

The obvious way to ask this is to score the labelled crops per equipped tsum.
**That cannot be done today and the reason is worth stating first:** all 805
labelled crops whose session still exists come from Beast rounds, and the 43
rounds played with a second tsum have **zero** labelled crops. Splitting the
labelled set by equipped tsum would compare one tsum against nothing.

So this uses the ground truth that needs no labelling. When the bot holds a
tsum the game lights up every tsum that is the SAME CHARACTER and reachable,
and `marked` in `samples.jsonl` records exactly which. Name the head where the
model is on solid ground -- visible and confident -- then ask what it calls
each marked partner. **They must agree**, because the game has already said
they are the same character.

That works on any session, labelled or not, which is what makes the comparison
possible at all.

    python scripts/recog_by_tsum.py
    python scripts/recog_by_tsum.py --floor 0.90

**What this does not measure**, carried over from `scripts/mark_probe.py` and
`docs/IDENTITY.md` section 10, because it is easy to over-read:

* The head's name comes from the model, so a mistaken head mislabels a whole
  group. Every figure here is a **floor** on accuracy, not an estimate of it.
* A disagreement has two causes this cannot separate -- the model misreading
  the picture, or the picture not containing the tsum at all.
* The game draws its highlight over marked tsums, and `_before.jpg` is captured
  with the marks up, so some partner crops are pictures of the mark. Partners
  are therefore reported banded by visibility, never as one number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game import profiles  # noqa: E402
from ttheart_sender.game import tsum as T  # noqa: E402

BANDS = ((0.55, 0.65), (0.65, 0.75), (0.75, 1.01))


def equipped(root: Path):
    """session -> profile name, from the skill icon's own colour.

    The icon is the only identifier of the equipped tsum that survives between
    sessions: the cluster index is a per-frame k-means id. It reads to within
    ~1.0 Lab unit inside a session and the shipped profiles sit 36.9 apart.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--floor", type=float, default=0.85,
                    help="confidence a HEAD must clear to name its group")
    ap.add_argument("--head-visible", type=float, default=0.65,
                    help="how visible a head must be to be trusted to name a "
                         "group. Above the model's own 0.55 training floor on "
                         "purpose: a head is judging every partner")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    size = int(meta.get("size", 96))
    mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
    net = cv2.dnn.readNetFromONNX(str(args.model))
    T._warm_up(net, size)

    who = equipped(args.dir)
    tally = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # prof -> band -> [n, agree]
    heads = defaultdict(lambda: [0, 0])                        # prof -> [samples, usable]
    seen = 0

    for jl in sorted(args.dir.glob("*/samples.jsonl")):
        prof = who.get(jl.parent.name, "unprofiled")
        for line in jl.read_text(encoding="utf-8").splitlines():
            if args.limit and seen >= args.limit:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts, h = d.get("tsums") or [], d.get("head")
            marked = [i for i in (d.get("marked") or [])
                      if isinstance(i, int) and i < len(ts)]
            if h is None or h >= len(ts) or not marked:
                continue
            o = d.get("options") or {}
            if o.get("character"):
                # `kind` was overwritten by the model in these rounds, and the
                # marks were read against that board. Never mixed in.
                continue
            img = cv2.imread(str(jl.parent / ("%04d_before.jpg" % d["index"])))
            if img is None:
                continue
            radius = float(d.get("radius") or 25.0)
            heads[prof][0] += 1
            seen += 1

            want = [h] + [i for i in marked if i != h]
            crops, keep = [], []
            for i in want:
                t = ts[i]
                c = T._character_crop(
                    img, T.Tsum(t["x"], t["y"], t["r"], 0, (0, 0, 0)),
                    radius, size)
                if c is not None:
                    crops.append(c)
                    keep.append(i)
            if not crops or keep[0] != h:
                continue
            blob = np.stack(crops).astype(np.float32) / 255.0
            blob = (blob - mean) / std
            blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), np.float32)
            logits = T._forward(net, blob, "the character model")
            logits = logits - logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)

            head_v = ts[h]["r"] / radius
            if head_v < args.head_visible or prob[0].max() < args.floor:
                continue
            heads[prof][1] += 1
            name = int(prob[0].argmax())
            for n, i in enumerate(keep[1:], start=1):
                v = ts[i]["r"] / radius
                for lo, hi in BANDS:
                    if lo <= v < hi:
                        cell = tally[prof][(lo, hi)]
                        cell[0] += 1
                        cell[1] += int(prob[n].argmax() == name)
                        break

    print("\nrecognition agreement with the game's own marks, by equipped tsum")
    print("head must be >=%.2f visible and >=%.2f confident to name its group"
          % (args.head_visible, args.floor))
    print("\n%-18s%9s%9s%s" % ("equipped tsum", "samples", "usable",
                               "".join("%20s" % ("%.2f-%.2f" % b) for b in BANDS)))
    for prof in sorted(tally, key=lambda p: -heads[p][0]):
        row = "%-18s%9d%9d" % (prof, heads[prof][0], heads[prof][1])
        for b in BANDS:
            n, ok = tally[prof][b]
            if not n:
                row += "%20s" % "--"
                continue
            p_ = ok / n
            # Two standard errors, printed beside every cell rather than left
            # to the reader. 43 rounds give ~25 partners a band, and a bare
            # "55.2%" from 29 observations reads exactly like a bare "84.7%"
            # from 354 unless the interval is on the same line.
            se = sqrt(max(p_ * (1 - p_), 1e-9) / n)
            row += "%20s" % ("%.0f+/-%.0f%%  n=%d" % (100 * p_, 200 * se, n))
        print(row)

    print("\n  A FLOOR, not an estimate: the head's name comes from the model,")
    print("  so a mistaken head marks a whole group wrong. And the game paints")
    print("  its highlight over marked tsums in this very frame, so the lower")
    print("  bands carry a known bias. Compare the COLUMNS across rows, not a")
    print("  row against 100%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
