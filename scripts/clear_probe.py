"""Does the link model predict what actually CLEARS?

The twenty-sixth round ended with the gameplay metric disqualified: scored
against the game's marks, a perfect-identity oracle came out below the rule in
use, which means marks cannot rank linking rules. Everything built on them is
suspect, including the model this scores.

`cleared` is a different and better answer. It is measured after the fact from
the pixels -- did this tsum leave the board -- and the link model has never
seen it. So this is an external check on a model trained on marks, using a
target it was not fitted to and a metric that cannot be gamed by proposing
more: among the members the bot ALREADY dragged, can the model tell which ones
will pop?

That question is also directly useful. Every member of a proposed chain is
dragged whether or not it will be accepted, and a refused one costs its slot.
A score that separates the two is a trim rule -- and unlike the offline
partition scores, it needs no assumption about what a chain could have reached,
because the chain is the one the bot really drew.

The population is deliberately narrow: only tsums the current rule already put
in a chain, which are all same-`kind` and all adjacent. Whatever separates them
is something the shipped rule does not already know.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttheart_sender.game import tsum as T  # noqa: E402
from link_net import auc, blockers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/link_colour.pt"))
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    blob = torch.load(args.model, map_location="cpu", weights_only=False)
    cols, mu, sd = blob["cols"], blob["mu"], blob["sd"]
    net = nn.Sequential(nn.Linear(cols, 96), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(96, 48), nn.ReLU(),
                        nn.Linear(48, 1), nn.Flatten(0))
    net.load_state_dict(blob["state"])
    net.eval()

    sessions = sorted(d.name for d in args.dir.glob("*/")
                      if (d / "samples.jsonl").exists())
    rng = np.random.RandomState(args.seed)
    order = list(sessions)
    rng.shuffle(order)
    test_s = set(order[:max(1, int(round(len(order) * args.holdout)))])

    p_all, y_all, dist_all, lab_all, pos_all = [], [], [], [], []
    drags = 0
    for d in sorted(args.dir.glob("*/")):
        f = d / "samples.jsonl"
        if not f.exists() or d.name not in test_s:
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts, head = r.get("tsums"), r.get("head")
            dragged, cleared = r.get("dragged"), r.get("cleared")
            if not ts or head is None or not dragged or cleared is None:
                continue
            bgr = cv2.imread(str(d / f"{r['index']:04d}_before.jpg"))
            if bgr is None:
                continue
            radius = float(r.get("radius") or 25.0)
            objs = [T.Tsum(x=float(t["x"]), y=float(t["y"]),
                           r=float(t.get("r", radius)), kind=int(t["kind"]),
                           colour=(0, 0, 0)) for t in ts]
            lab = T._face_lab(bgr, objs, radius)
            gone = set(int(i) for i in cleared)

            feats, keep = [], []
            for slot, i in enumerate(dragged):
                i = int(i)
                if i >= len(ts) or i == head:
                    continue
                dx = ts[i]["x"] - ts[head]["x"]
                dy = ts[i]["y"] - ts[head]["y"]
                feats.append([float(np.hypot(dx, dy)) / radius, abs(dx) / radius,
                              dy / radius, blockers(ts, head, i, radius),
                              len(ts) / 50.0,
                              float(np.linalg.norm(lab[i] - lab[head])) / 40.0,
                              1.0 if ts[i]["kind"] == ts[head]["kind"] else 0.0])
                keep.append((i, slot))
            if not feats:
                continue
            X = np.asarray(feats, np.float32)
            with torch.no_grad():
                p = torch.sigmoid(
                    net(torch.from_numpy((X[:, :cols] - mu) / sd).float())).numpy()
            for (i, slot), q, row in zip(keep, p, X):
                p_all.append(float(q))
                y_all.append(1 if i in gone else 0)
                dist_all.append(float(row[0]))
                lab_all.append(float(row[5]))
                pos_all.append(slot)
            drags += 1

    if not p_all:
        print("no held-out drags with a clear reading")
        return 1
    p = np.asarray(p_all)
    y = np.asarray(y_all)
    dist = np.asarray(dist_all)
    labd = np.asarray(lab_all)
    slot = np.asarray(pos_all, np.float64)

    print(f"{drags} held-out drags, {len(y)} dragged members, "
          f"{y.mean():.1%} of them cleared\n")
    print(f"{'signal':>26} {'AUC':>7}")
    print(f"{'link model P(link)':>26} {auc(p, y):>7.3f}")
    print(f"{'-distance from head':>26} {auc(-dist, y):>7.3f}")
    print(f"{'-face colour distance':>26} {auc(-labd, y):>7.3f}")
    print(f"{'-position in the chain':>26} {auc(-slot, y):>7.3f}")
    print("\n  0.50 = knows nothing about which dragged members pop.")
    print("  All of these members are already same-kind and adjacent, so any")
    print("  separation is something the shipped rule does not have.")

    # Where the model would trim, and what that costs. A member below the bar
    # is one the chain would not have walked to.
    print(f"\n{'threshold':>10} {'kept':>7} {'kept clear%':>12} {'dropped clear%':>15}")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        keep = p >= t
        if keep.sum() and (~keep).sum():
            print(f"{t:>10.1f} {keep.mean():>7.1%} {y[keep].mean():>12.1%} "
                  f"{y[~keep].mean():>15.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
