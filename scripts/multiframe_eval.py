"""Single-frame vs best-visible-frame vs multi-frame aggregation, on real crops.

The multi-frame proposal is that a second look at the same physical tsum gives
a better observation to identify it from. This scores that claim against human
labels, using the only second look this corpus contains: `NNNN_marked.jpg`,
the same board about 0.25s after `NNNN_before.jpg`.

Ground truth is the folder a person put the crop in. Every labelled crop is
named `<session>_<sample>_<index>_v<visible>.png`, so it points back at exactly
one row of `samples.jsonl` and exactly one detection in it.

Three methods, all through the shipped ONNX:

    A  single frame      the `before` crop -- what the bot does today
    B  best visible      whichever of the two observations shows more
    C  aggregation       the mean of the two probability vectors

**Tsums the game marked are excluded.** The game paints its link highlight over
them, so their second observation is a picture of the highlight rather than the
tsum -- about 14% of a board. Including them would credit or blame the method
for something that only happens while a chain is held.

    python scripts/multiframe_eval.py --samples 400
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game import tsum as T  # noqa: E402

NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2,})_v(?P<vis>[\d.]+)$")
NOT_CHARACTERS = {"board", "score", "junk"}


def labelled(root: Path):
    """(session, sample, index) -> class name, from the labelled folders."""
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in NOT_CHARACTERS:
            continue
        for p in d.glob("*.png"):
            m = NAME.match(p.stem)
            if m:
                out[(m.group("sess"), int(m.group("sample")),
                     int(m.group("idx")))] = d.name
    return out


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def run(net, crops, size, mean, std):
    blob = np.stack(crops).astype(np.float32) / 255.0
    blob = (blob - mean) / std
    blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)
    return softmax(T._forward(net, blob, "the character model"))


def band(v):
    if v >= 0.70:
        return "high  (>=0.70)"
    if v >= 0.60:
        return "medium(0.60-0.70)"
    return "low   (0.55-0.60)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--effort", type=int, default=3)
    args = ap.parse_args()

    meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    names = list(meta["classes"])
    index = {n: i for i, n in enumerate(names)}
    size = int(meta.get("size", 96))
    mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
    net = cv2.dnn.readNetFromONNX(str(args.model))
    T._warm_up(net, size)

    truth = labelled(args.crops)
    by_sample = defaultdict(list)
    for (sess, samp, idx), cls in truth.items():
        by_sample[(sess, samp)].append((idx, cls))
    print("%d labelled crops over %d samples" % (len(truth), len(by_sample)))

    rows = []
    done = 0
    for (sess, samp), items in sorted(by_sample.items()):
        if done >= args.samples:
            break
        jl = args.dir / sess / "samples.jsonl"
        bp = args.dir / sess / ("%04d_before.jpg" % samp)
        mp = args.dir / sess / ("%04d_marked.jpg" % samp)
        if not (jl.exists() and bp.exists() and mp.exists()):
            continue
        row = None
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("index") == samp:
                row = d
                break
        if not row or not row.get("tsums"):
            continue
        wanted = [(i, c) for i, c in items
                  if c in index and i < len(row["tsums"])
                  and i not in set(row.get("marked") or [])]
        if not wanted:
            continue
        first, second = cv2.imread(str(bp)), cv2.imread(str(mp))
        if first is None or second is None:
            continue
        radius = float(row.get("radius") or 25.0)
        o = row.get("options") or {}
        kw = dict(k=int(o.get("k", 12)),
                  include_dark=bool(o.get("include_dark", False)),
                  merge=bool(o.get("merge", False)),
                  bowl_reject=float(o.get("bowl_reject", 0.0) or 0.0),
                  kinds=int(o.get("kinds", 0) or 0),
                  recolour=float(o.get("recolour", 0.0) or 0.0),
                  scale=float(o.get("scale", 1.0) or 1.0),
                  fit_effort=args.effort)
        _, pal = T._quantise(first, kw["k"], None, effort=args.effort)
        later, _, _ = T.detect(second, radius=radius, palette=pal, **kw)
        if not later:
            continue
        lpts = np.array([[t.x, t.y] for t in later], np.float64)

        crops_a, crops_b, keep = [], [], []
        for i, cls in wanted:
            t = row["tsums"][i]
            ta = T.Tsum(t["x"], t["y"], t["r"], 0, (0, 0, 0))
            ca = T._character_crop(first, ta, radius, size)
            if ca is None:
                continue
            # Track by position: the probe measured a median displacement of
            # 1.41px over this interval, so nearest-neighbour is the whole
            # tracker and a more elaborate one would have nothing to fix.
            dist = np.hypot(lpts[:, 0] - t["x"], lpts[:, 1] - t["y"])
            j = int(dist.argmin())
            if dist[j] > radius:
                continue
            tb = T.Tsum(later[j].x, later[j].y, later[j].r, 0, (0, 0, 0))
            cb = T._character_crop(second, tb, radius, size)
            if cb is None:
                continue
            crops_a.append(ca)
            crops_b.append(cb)
            keep.append((index[cls], t["r"] / radius, later[j].r / radius))
        if not keep:
            continue
        pa = run(net, crops_a, size, mean, std)
        pb = run(net, crops_b, size, mean, std)
        for n, (y, va, vb) in enumerate(keep):
            rows.append((y, va, vb, pa[n], pb[n]))
        done += 1

    if not rows:
        print("no comparable crops")
        return 2

    y = np.array([r[0] for r in rows])
    va = np.array([r[1] for r in rows])
    vb = np.array([r[2] for r in rows])
    pa = np.stack([r[3] for r in rows])
    pb = np.stack([r[4] for r in rows])

    A = pa.argmax(1)
    better = vb > va
    B = np.where(better, pb.argmax(1), pa.argmax(1))
    C = ((pa + pb) / 2.0).argmax(1)

    print("\n%d crops from %d samples, marked tsums excluded" % (len(y), done))
    print("second observation was MORE visible for %.1f%% of them "
          "(median gain %+.3f)" % (100 * better.mean(), float(np.median(vb - va))))

    print("\n%-20s%9s%14s%14s%14s"
          % ("visibility band", "n", "A single", "B best-vis", "C mean"))
    bands = defaultdict(list)
    for i, v in enumerate(va):
        bands[band(v)].append(i)
    for name in ("high  (>=0.70)", "medium(0.60-0.70)", "low   (0.55-0.60)"):
        ii = np.array(bands.get(name, []), int)
        if not len(ii):
            continue
        print("%-20s%9d%13.1f%%%13.1f%%%13.1f%%"
              % (name, len(ii), 100 * (A[ii] == y[ii]).mean(),
                 100 * (B[ii] == y[ii]).mean(), 100 * (C[ii] == y[ii]).mean()))
    print("%-20s%9d%13.1f%%%13.1f%%%13.1f%%"
          % ("ALL", len(y), 100 * (A == y).mean(),
             100 * (B == y).mean(), 100 * (C == y).mean()))
    print("\n  B changed the answer on %d crop(s); C on %d."
          % (int((A != B).sum()), int((A != C).sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
