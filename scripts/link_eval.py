"""Score linking rules the way a round scores them: tsums cleared per drag.

`scripts/group_eval.py` scores a rule by how it PARTITIONS a board, and it has
a flaw this fixes. It treats every tsum sharing the head's id as available to
the chain, which ignores reach entirely -- but `adjacency` will not join two
tsums 300px apart however well their colours agree, so a partition score
credits a rule with partners the chain could never walk to.

Here a candidate set is built the way the chain builder actually builds one:

* **today** -- the connected component of same-`kind` tsums reachable from the
  head through `adjacency` edges, at the shipped `link_px` and `block`. Chains
  hop tsum to tsum, so a partner five hops away is reachable and one 106px
  away in open space is not.
* **kinds N** -- the same, with `kind` replaced by an N-way grouping of face
  colour.
* **link model** -- every tsum the model gives at least `--threshold`
  probability of being linked, which is what it was trained to predict.

Scored against the game's own marks, in the units the round is scored in. A
chain draws up to `--cap` members from the candidate set, and the share of them
the game accepts is that set's own marked density -- so a set half full of
tsums the game will refuse fills half the chain with members that will not pop.
That charge is what stops the score rewarding a rule for simply proposing more.
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
from group_eval import kgroups  # noqa: E402
from link_net import blockers  # noqa: E402


def component(adj, kinds, head):
    """Everything reachable from `head` through same-kind edges."""
    seen, stack = {head}, [head]
    while stack:
        n = stack.pop()
        for m in adj[n]:
            if m not in seen and kinds[m] == kinds[head]:
                seen.add(m)
                stack.append(m)
    seen.discard(head)
    return seen


def reachable(adj, allowed, head):
    """The part of `allowed` a chain could actually walk to from `head`.

    Without this the link model is scored on a set it could never draw as a
    path, while `today` is scored on a connected component -- which would
    credit the model for partners on the far side of the board. The model
    answers head-to-candidate, so this is where walkability is put back.
    """
    seen, stack = {head}, [head]
    while stack:
        n = stack.pop()
        for m in adj[n]:
            if m not in seen and m in allowed:
                seen.add(m)
                stack.append(m)
    seen.discard(head)
    return seen


def expected_cleared(cand, marked, cap):
    """Tsums this candidate set would pop on one drag from the head.

    The chain takes up to `cap` of them, and the share the game accepts is the
    set's own marked density -- see the module docstring.
    """
    if not cand:
        return 1.0
    hit = len(cand & marked)
    drawn = min(cap - 1, len(cand))
    return 1.0 + drawn * hit / len(cand)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/link_colour.pt"))
    ap.add_argument("--cap", type=int, default=12, help="play.yaml max_chain")
    ap.add_argument("--link-px", type=float, default=105.0)
    ap.add_argument("--block", type=float, default=1.25)
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
    print(f"{args.model.name}: {blob['features']}, held-out AUC {blob['auc']:.4f}")

    # The SAME held-out sessions the model was trained against. Scoring it on
    # its own training rounds would measure memory, which is the mistake the
    # twenty-fourth round was made of.
    sessions = sorted(d.name for d in args.dir.glob("*/")
                      if (d / "samples.jsonl").exists())
    rng = np.random.RandomState(args.seed)
    order = list(sessions)
    rng.shuffle(order)
    test_s = set(order[:max(1, int(round(len(order) * args.holdout)))])
    print(f"held out: {len(test_s)} of {len(sessions)} sessions\n")

    thresholds = [0.3, 0.4, 0.5, 0.6]
    tally = {k: [] for k in ["today", "kinds4", "ORACLE"]}
    for t in thresholds:
        tally[f"link>={t}"] = []          # set as the model gives it
        tally[f"link+reach>={t}"] = []    # and the walkable part of it

    for d in sorted(args.dir.glob("*/")):
        if d.name not in test_s or not (d / "samples.jsonl").exists():
            continue
        for line in (d / "samples.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts, head, marked = r.get("tsums"), r.get("head"), r.get("marked")
            if not ts or head is None or head >= len(ts) or not marked:
                continue
            bgr = cv2.imread(str(d / f"{r['index']:04d}_before.jpg"))
            if bgr is None:
                continue
            radius = float(r.get("radius") or 25.0)
            objs = [T.Tsum(x=float(t["x"]), y=float(t["y"]),
                           r=float(t.get("r", radius)), kind=int(t["kind"]),
                           colour=(0, 0, 0)) for t in ts]
            mark = {int(i) for i in marked if int(i) != head}
            if not mark:
                continue
            lab = T._face_lab(bgr, objs, radius)
            adj = T.adjacency(objs, radius, link_px=args.link_px, block=args.block)

            saved = [t["kind"] for t in ts]
            tally["today"].append(expected_cleared(
                component(adj, saved, head), mark, args.cap))
            tally["kinds4"].append(expected_cleared(
                component(adj, kgroups(lab, 4).tolist(), head), mark, args.cap))
            # THE CEILING. A chain can only walk through tsums the game
            # accepts, so a perfect identity rule would offer exactly the
            # marked tsums reachable THROUGH marked tsums -- no more. Any rule
            # scoring near this is not being held back by identity, and no rule
            # can pass it. Worth knowing before optimising anything else.
            tally["ORACLE"].append(expected_cleared(
                reachable(adj, mark, head), mark, args.cap))

            feats = []
            for i, t in enumerate(ts):
                if i == head:
                    continue
                dx, dy = t["x"] - ts[head]["x"], t["y"] - ts[head]["y"]
                feats.append([np.hypot(dx, dy) / radius, abs(dx) / radius,
                              dy / radius, blockers(ts, head, i, radius),
                              len(ts) / 50.0,
                              float(np.linalg.norm(lab[i] - lab[head])) / 40.0,
                              1.0 if ts[i]["kind"] == ts[head]["kind"] else 0.0])
            idx = [i for i in range(len(ts)) if i != head]
            with torch.no_grad():
                x = (np.asarray(feats, np.float32)[:, :cols] - mu) / sd
                p = torch.sigmoid(net(torch.from_numpy(x).float())).numpy()
            for t in thresholds:
                allowed = {i for i, q in zip(idx, p) if q >= t}
                tally[f"link>={t}"].append(
                    expected_cleared(allowed, mark, args.cap))
                tally[f"link+reach>={t}"].append(
                    expected_cleared(reachable(adj, allowed, head), mark, args.cap))

    n = len(tally["today"])
    if not n:
        print("no held-out samples with marks")
        return 1
    base = np.asarray(tally["today"])
    print(f"{n} held-out drags, cap {args.cap}\n")
    print(f"{'rule':>12} {'cleared/drag':>13} {'vs today':>20}")
    for name, vals in tally.items():
        v = np.asarray(vals)
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        note = "--" if name == "today" else (
            f"{d.mean():+.3f} +/- {2 * se:.3f} ({d.mean() / base.mean():+.1%})"
            f"{'' if abs(d.mean()) > 2 * se else '  noise'}")
        print(f"{name:>12} {v.mean():>13.2f} {note:>20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
