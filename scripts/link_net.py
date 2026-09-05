"""Learn which tsums the game will actually link, instead of guessing.

`adjacency()` decides whether two tsums can chain with two hand-set numbers:
`link_px` (105) and `block` (1.25 radii of clearance from anything lying on
the line). Those were tuned by hand against a few boards, and every chain the
bot builds rests on them.

The game has answered the same question 80,417 times already. Holding a tsum
lights up everything it will link to, and every collected sample records it --
so "will the game link these two" is a supervised problem with a large free
label set, and the rule in use can be scored against it rather than assumed.

What this is NOT
----------------

It is not another identity model. That programme is closed: four independent
attempts and a data-scaling curve put crop-based identity at about 0.66 AUC,
and more rounds buy +0.0092 per doubling. This asks a different and more
useful question, because it is the one the bot actually acts on -- a mark
means same character AND reachable, which is exactly the condition for a chain
member to survive.

That also makes distance a legitimate input here, where in `embed_net.py` it
was a confound to be matched away. There the target was identity and distance
was a proxy that had to be removed; here the target includes reachability, so
distance is a cause rather than a leak.

The ablation is the point
-------------------------

Three feature sets, same pairs, same split:

* ``geom``    -- distance, offsets, blockers on the line, board density.
  This is what `adjacency` uses, learned rather than set by hand.
* ``colour``  -- geom plus the Lab distance between the two faces. This is
  everything the bot has today, `kind` included.
* ``full``    -- colour plus a small CNN over both crops, on the GPU.

If ``full`` does not beat ``colour``, then the pictures carry nothing the
median face colour does not already have, and that is a fifth independent
closing of the identity question -- in the units the round is scored in rather
than in AUC over crops.

And whatever wins is measured against the rule actually in use: `adjacency` at
the shipped `link_px` and `block` is scored on the same held-out pairs, so the
question is never "is the model good" but "is it better than what runs today".

    .venv/Scripts/python scripts/link_net.py --features geom
    .venv/Scripts/python scripts/link_net.py --features full --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CROP = 32          #: smaller than embed_net's 64: this model is mostly geometry
CACHE = Path("crops/link_cache.npz")


def auc(score, label):
    """Chance a linked pair outranks an unlinked one. Threshold-free."""
    order = np.argsort(score)
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = int((label == 1).sum()), int((label == 0).sum())
    if not npos or not nneg:
        return 0.5
    return float((ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cut(bgr, x, y, radius):
    """Face crop, padded at the frame edge rather than refused."""
    half = max(3, int(round(radius)))
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = int(x) - half, int(y) - half, int(x) + half + 1, int(y) + half + 1
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    patch = bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if patch.size == 0:
        return np.zeros((CROP, CROP, 3), np.uint8)
    if px0 or py0 or px1 or py1:
        patch = cv2.copyMakeBorder(patch, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
    return cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)


def blockers(ts, a, c, radius, block=1.25):
    """How many other tsums lie across the line from `a` to `c`.

    The same question `adjacency`'s `block` asks, kept as a feature so the
    learned rule is choosing how much to weigh it rather than being told.
    """
    ax, ay = ts[a]["x"], ts[a]["y"]
    cx, cy = ts[c]["x"], ts[c]["y"]
    vx, vy = cx - ax, cy - ay
    span = vx * vx + vy * vy
    if span <= 1e-6:
        return 0
    n = 0
    for i, t in enumerate(ts):
        if i in (a, c):
            continue
        s = ((t["x"] - ax) * vx + (t["y"] - ay) * vy) / span
        if not 0.0 < s < 1.0:
            continue
        px, py = ax + s * vx, ay + s * vy
        if np.hypot(t["x"] - px, t["y"] - py) < block * radius:
            n += 1
    return n


def build(root: Path, cache: Path, want_crops: bool):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"loaded {cache}: {len(z['y'])} pairs")
        return z["X"], z["y"], z["sess"], (z["ca"] if "ca" in z.files else None), \
            (z["cc"] if "cc" in z.files else None)

    from ttheart_sender.game import tsum as T

    X, y, sess, ca, cc = [], [], [], [], []
    boards = 0
    for d in sorted(root.glob("*/")):
        f = d / "samples.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
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
            lab = T._face_lab(bgr, objs, radius)
            mark = {int(i) for i in marked}
            ax, ay = ts[head]["x"], ts[head]["y"]
            crop_a = cut(bgr, ax, ay, radius) if want_crops else None

            for i, t in enumerate(ts):
                if i == head:
                    continue
                dx, dy = t["x"] - ax, t["y"] - ay
                dist = float(np.hypot(dx, dy))
                X.append([
                    dist / radius,
                    abs(dx) / radius,
                    dy / radius,                       # signed: the pile falls
                    blockers(ts, head, i, radius),
                    len(ts) / 50.0,
                    float(np.linalg.norm(lab[i] - lab[head])) / 40.0,
                    1.0 if ts[i]["kind"] == ts[head]["kind"] else 0.0,
                ])
                y.append(1 if i in mark else 0)
                sess.append(d.name)
                if want_crops:
                    ca.append(crop_a)
                    cc.append(cut(bgr, t["x"], t["y"], radius))
            boards += 1
        print(f"  {d.name}: {len(y)} pairs so far", end="\r", flush=True)

    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    sess = np.asarray(sess)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if want_crops:
        ca, cc = np.asarray(ca, np.uint8), np.asarray(cc, np.uint8)
        np.savez_compressed(cache, X=X, y=y, sess=sess, ca=ca, cc=cc)
    else:
        ca = cc = None
        np.savez_compressed(cache, X=X, y=y, sess=sess)
    print(f"\n{boards} boards -> {len(y)} pairs, {y.mean():.1%} linked")
    return X, y, sess, ca, cc


def adjacency_baseline(X, link_px=105.0, block_max=0):
    """The rule in use, scored on the same pairs.

    `adjacency` links when the centres are within `link_px` and nothing lies
    across the line. It is a hard yes/no, so as a ranking it has exactly two
    levels -- which is the point: a learned rule that cannot beat two levels
    is not worth shipping.
    """
    radius = 25.0
    dist_px = X[:, 0] * radius
    return ((dist_px <= link_px) & (X[:, 3] <= block_max)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--features", choices=["geom", "colour", "full"], default="colour")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.25, help="share of SESSIONS")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else "cpu" if args.device == "auto" else args.device)
    print(f"device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")

    X, y, sess, ca, cc = build(args.dir, args.cache, args.features == "full")
    if args.features == "full" and ca is None:
        print("cache has no crops -- delete it and re-run for --features full")
        return 1

    # Held out by SESSION: boards inside one round are near-duplicates, and a
    # pair-level split would let the net recognise the board.
    uniq = sorted(set(sess.tolist()))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq)
    test_s = set(uniq[:max(1, int(round(len(uniq) * args.holdout)))])
    te = np.array([s in test_s for s in sess])
    tr = ~te
    print(f"{len(uniq)} sessions -> {len(uniq) - len(test_s)} train / {len(test_s)} test")
    print(f"{int(tr.sum())} train pairs, {int(te.sum())} test pairs, "
          f"{y[te].mean():.1%} of the test pairs are linked\n")

    cols = {"geom": 5, "colour": 7, "full": 7}[args.features]
    Xt = X[:, :cols].copy()
    mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
    Xt = (Xt - mu) / sd

    base = adjacency_baseline(X[te])
    print(f"adjacency as it ships (link_px=105, block clear): "
          f"AUC {auc(base, y[te]):.3f}")

    xt = torch.from_numpy(Xt).float().to(dev)
    yt = torch.from_numpy(y).float().to(dev)

    if args.features == "full":
        class Net(nn.Module):
            """Geometry and colour, plus what a convolution can add to them."""

            def __init__(self, n):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(6, 24, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.head = nn.Sequential(
                    nn.Linear(n + 64, 128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, f, img):
                return self.head(torch.cat([f, self.cnn(img)], 1)).squeeze(1)
        net = Net(cols).to(dev)
        imgs = np.concatenate([ca, cc], axis=3)      # (n, 32, 32, 6)
    else:
        net = nn.Sequential(nn.Linear(cols, 96), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(96, 48), nn.ReLU(),
                            nn.Linear(48, 1), nn.Flatten(0)).to(dev)
        imgs = None

    pos = float(y[tr].mean())
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-6)], device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_i, te_i = np.where(tr)[0], np.where(te)[0]

    def run(idx, train):
        out = []
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            f = xt[b]
            if imgs is None:
                z = net(f)
            else:
                im = torch.from_numpy(imgs[b]).to(dev).float().permute(0, 3, 1, 2) / 255.0
                z = net(f, im)
            if train:
                loss = lossf(z, yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            out.append(z.detach().cpu().numpy())
        return np.concatenate(out)

    best, best_state, since = 0.0, None, 0
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        net.train()
        run(np.random.permutation(tr_i), True)
        net.eval()
        with torch.no_grad():
            a = auc(run(te_i, False), y[te])
        mark = ""
        if a > best:
            best, since = a, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            mark = "  <- best"
        else:
            since += 1
        print(f"  epoch {ep + 1:>2}/{args.epochs}  held-out AUC {a:.4f}{mark}",
              flush=True)
        if since >= 6:
            print("  stopping: 6 epochs with no gain")
            break

    print(f"\n{args.features}: best held-out AUC {best:.4f} "
          f"in {time.perf_counter() - t0:.0f}s")
    print(f"  adjacency as it ships:      {auc(base, y[te]):.4f}")
    if best_state is not None:
        Path("models").mkdir(exist_ok=True)
        # MANDATORY, and the reason is round 24: a model was scored +10% on
        # sessions it had trained on and -6.6% on ones it had not, because the
        # split lived in one script's head instead of in the artifact. The
        # sessions travel with the weights now, so an evaluation can be held to
        # them instead of remembering them.
        torch.save({"state": best_state, "mu": mu, "sd": sd, "cols": cols,
                    "features": args.features, "auc": best,
                    "holdout": args.holdout, "seed": args.seed,
                    "train_sessions": sorted(set(sess[tr].tolist())),
                    "test_sessions": sorted(test_s)},
                   f"models/link_{args.features}.pt")
        print(f"  weights -> models/link_{args.features}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
