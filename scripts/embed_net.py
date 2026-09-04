"""Learn an embedding that says *same character*, from the game's own marks.

This supersedes `scripts/identity_net.py`, which had the right idea and three
wrong pieces. It is worth naming them, because each one invalidated everything
downstream of it:

* **`WINDOW = 1.5` radii.** Detected tsums sit about 2.44 radii apart, so a
  1.5r crop reaches well into the neighbour. Every "face" in that training set
  was a face and a piece of the tsum beside it, and a net asked whether two of
  those match is partly being asked whether their neighbours match. 1.0r here,
  the same window `crops.py` and `CharacterModel` use.
* **Edge crops thrown away.** 31.5% of the faces on a board sit close enough
  to the board rect that a 1.0r window clips them. Dropping those trains on
  two-thirds of a board and leaves the other third unidentifiable at play
  time, which reintroduces exactly the split this is meant to close. Here the
  crop is padded by replication instead, and padded crops are in the training
  set, so inference on one is not a picture the net has never seen.
* **The `balanced` objective.** Agreement and split were averaged, which
  prices a missed partner and a false partner the same. The twenty-third round
  measured that trade and it is not symmetric: the game SKIPS a chain member
  it refuses and keeps linking, so a false partner costs one slot, while a
  missed partner is unreachable at any distance and costs the whole rest of
  the group. What this optimises for is therefore recall of the marked set at
  a bounded group count, and what scores it is `scripts/group_eval.py` in
  tsums cleared per drag.

Why an embedding and not a classifier
-------------------------------------

`models/character.onnx` names 15 characters at 95% on held-out sessions and is
a **-24%** regression in play. Two reasons, and an embedding answers both. It
knows only the characters somebody labelled, and the equipped tsum -- the most
numerous on any board -- is usually not one of them; 48% of its names were one
untrained character landing on its nearest neighbour. And it answers *which
character*, when the board only ever needs *which of these five piles*.

An embedding needs no class list, so an untrained character embeds consistently
with itself and clusters with itself. The labels cost nothing: holding a tsum
makes the game light up every tsum that is the same character and reachable, so
every sample already carries its own answer.

  * **positives** -- (head, marked) pairs beyond the glow, where a reaction
    means identity rather than proximity.
  * **negatives** -- (head, unmarked) tsums **at the same distance** as the
    positive each one is matched to. See `build`: whether a tsum is marked
    depends on distance as much as on identity, so unmatched negatives make
    this a distance net rather than an identity net. Measured -- unmatched, it
    scored 0.461 held-out; matched, 0.647 in the same single epoch.

    Noisy in ONE direction: a same-character tsum that was merely unreachable
    lands among the negatives and is trained against, so the negative half is
    pessimistic throughout.

**Held out by SESSION.** Boards inside one round are near-duplicates of each
other; a crop-level split leaves the same physical tsum on both sides and the
net scores itself on its own memory.

    .venv/Scripts/python scripts/embed_net.py --epochs 20
    .venv/Scripts/python scripts/group_eval.py        # the score that decides

Nothing ships unless it wins there. Export is ONNX read through `cv2.dnn`,
which is already bundled -- no new runtime dependency -- at a FIXED batch,
because OpenCV 5.0.0 sizes a net's buffers on the first batch it sees and a
larger one later kills the process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

CROP = 64          #: cached at 64px, exactly as `crops.py` writes them
WINDOW = 1.0       #: radii per crop. 1.5 reached into the neighbouring tsum.
BATCH_FIXED = 64   #: the exported net's batch, fixed -- see the module docstring


# ------------------------------------------------------------------ crop/data

def cut(bgr, x, y, radius):
    """One face crop at `WINDOW` radii, padded rather than refused at an edge.

    `CharacterModel._crop` returns None here, and that is right for a model
    whose training set excluded clipped crops -- a sliver stretched to a square
    is a picture it has never seen. The fix is not to keep refusing at play
    time but to put padded crops in the training set, which is what this does.
    """
    half = max(4, int(round(radius * WINDOW)))
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = int(x) - half, int(y) - half, int(x) + half + 1, int(y) + half + 1
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    patch = bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if patch.size == 0:
        return None
    if px0 or py0 or px1 or py1:
        patch = cv2.copyMakeBorder(patch, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
    return cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)


def build(root: Path, aura: float, tol: float, cache: Path, seed: int = 0):
    """Crops plus (anchor, other, same) pairs, from the marks already on disk.

    **Every negative is distance-matched to a positive**, and that is the most
    important line in this file.

    Whether the game marks a tsum depends on distance as well as on identity:
    measured over the corpus, 48.2% of tsums 1.5-2.5 radii from the pressed one
    are marked and only 11.7% of those beyond 10 radii. So a pair set that
    takes positives from one distance range and negatives from another is not a
    set of identity pairs at all -- it is a set of distance pairs, and a net
    trained on it learns whatever appearance happens to travel with distance:
    how buried the tsum is, how much of it is clipped, how the pile shades it.
    This project has already published two findings that turned out to be
    selection bias of exactly this shape.

    So for each positive (head, marked) pair at distance d, a negative is drawn
    from the unmarked tsums whose distance from the head is within `tol` of d.
    The two halves then have the same distance distribution by construction,
    the classes come out balanced, and appearance is the only thing left that
    separates them.
    """
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"loaded {cache}: {len(z['crops'])} crops, {len(z['pairs'])} pairs")
        return z["crops"], z["pairs"], z["sess"]

    rng = np.random.RandomState(seed)
    crops, pairs, sess = [], [], []
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
            img = d / f"{r['index']:04d}_before.jpg"
            bgr = cv2.imread(str(img))
            if bgr is None:
                continue
            radius = float(r.get("radius") or 25.0)
            hx, hy = ts[head]["x"], ts[head]["y"]
            mark = {int(i) for i in marked}

            local = {}

            def add(i):
                if i in local:
                    return local[i]
                c = cut(bgr, ts[i]["x"], ts[i]["y"], radius)
                if c is None:
                    return None
                local[i] = len(crops)
                crops.append(c)
                sess.append(d.name)
                return local[i]

            a = add(head)
            if a is None:
                continue
            # Distances in radii, so the floor means the same thing on any
            # board. Inside the glow a reaction means proximity, not identity.
            far = [(i, float(np.hypot(ts[i]["x"] - hx, ts[i]["y"] - hy)) / radius)
                   for i in range(len(ts)) if i != head]
            far = [(i, dd) for i, dd in far if dd >= aura]
            pos = [(i, dd) for i, dd in far if i in mark]
            neg = [(i, dd) for i, dd in far if i not in mark]
            if not pos or not neg:
                continue

            used = set()
            for i, dd in pos:
                near = [j for j, ee in neg
                        if j not in used and abs(ee - dd) <= tol * dd]
                if not near:
                    continue
                j = int(rng.choice(near))
                used.add(j)
                b, c = add(i), add(j)
                if b is not None:
                    pairs.append((a, b, 1))
                if c is not None:
                    pairs.append((a, c, 0))
            boards += 1

    crops = np.asarray(crops, np.uint8)
    pairs = np.asarray(pairs, np.int64)
    sess = np.asarray(sess)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, crops=crops, pairs=pairs, sess=sess)
    pos = int((pairs[:, 2] == 1).sum()) if len(pairs) else 0
    print(f"{boards} boards -> {len(crops)} crops, {len(pairs)} pairs "
          f"({pos} same, {len(pairs) - pos} different), cached to {cache}")
    return crops, pairs, sess


# ----------------------------------------------------------------------- net

def build_net(backbone: str, dim: int):
    """A pretrained trunk with an embedding head, L2-normalised at the end.

    The normalisation is part of the exported graph rather than something the
    caller remembers to do. `cv2.dnn` will happily return un-normalised vectors
    and clustering them still "works" -- badly, and silently, because vector
    length would then carry as much weight as direction.
    """
    import torch.nn as nn
    import torchvision.models as tv

    class Normalise(nn.Module):
        def forward(self, x):
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

    if backbone == "mobilenet_v3_small":
        net = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier = nn.Sequential(nn.Linear(576, 256), nn.Hardswish(),
                                       nn.Dropout(0.1), nn.Linear(256, dim))
    elif backbone == "resnet18":
        net = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(512, dim)
    else:
        raise SystemExit(f"unknown backbone {backbone}")
    return nn.Sequential(net, Normalise())


def augment(b, train: bool):
    """Flips and small jitter only.

    A tsum is upright and roughly centred by detection, so rotation and large
    translation would teach invariances the board never shows. Brightness
    matters: the same character reads differently at the top of the pile and
    under the FEVER flash, and this net exists to see through that.
    """
    import torch

    if not train:
        return b
    if torch.rand(1).item() < 0.5:
        b = torch.flip(b, dims=[3])
    b = b * (0.8 + 0.4 * torch.rand(len(b), 1, 1, 1))
    b = b + 0.08 * torch.randn(len(b), 1, 1, 1)
    return b.clamp(0.0, 1.0)


def as_batch(crops, idx, size, mean, std, train):
    import torch
    import torch.nn.functional as F

    b = torch.from_numpy(crops[idx]).float().permute(0, 3, 1, 2) / 255.0
    b = augment(b, train)
    if size != CROP:
        b = F.interpolate(b, size=(size, size), mode="bilinear", align_corners=False)
    return (b - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--cache", type=Path, default=Path("crops/embed_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("models/embed.onnx"))
    ap.add_argument("--backbone", default="mobilenet_v3_small")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.35,
                    help="cosine similarity a NEGATIVE pair may keep for free")
    ap.add_argument("--aura", type=float, default=2.5,
                    help="radii; the glow washes over everything nearer than "
                         "this, so a reaction there means proximity")
    ap.add_argument("--tol", type=float, default=0.2,
                    help="how close in distance a negative must be to the "
                         "positive it is matched against, as a fraction")
    ap.add_argument("--holdout", type=float, default=0.2, help="share of SESSIONS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many epochs with no held-out gain; "
                         "0 runs every epoch")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="2.5M backbone parameters against ~10k pairs from 74 "
                         "sessions is a shape that memorises rounds: the first "
                         "attempt scored +10%% on sessions it had trained on "
                         "and -6.6%% on ones it had not")
    ap.add_argument("--sessions", type=int, default=0,
                    help="train on only this many sessions, for a data-scaling "
                         "curve. 0 uses all of them")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    crops, pairs, sess = build(args.dir, args.aura, args.tol, args.cache,
                               args.seed)
    if len(pairs) < 200:
        print("not enough pairs -- collect more rounds with data collection on")
        return 1

    # Split by SESSION. A crop-level split leaves near-duplicates of the test
    # crop in training and the net scores its own memory.
    uniq = sorted(set(sess.tolist()))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * args.holdout)))
    test_s = set(uniq[:n_test])
    is_test = np.array([s in test_s for s in sess])
    if args.sessions:
        # Only the TRAINING half shrinks. The test sessions stay exactly as
        # they are, or the points on the curve would each be measured against
        # a different yardstick and could not be compared.
        pool = [u for u in uniq if u not in test_s][:args.sessions]
        keep = set(pool) | test_s
        drop = np.array([x not in keep for x in sess])
        print(f"data-scaling run: training on {len(pool)} sessions")
    else:
        drop = np.zeros(len(sess), bool)
    tr = (~is_test[pairs[:, 0]] & ~is_test[pairs[:, 1]]
          & ~drop[pairs[:, 0]] & ~drop[pairs[:, 1]])
    te = is_test[pairs[:, 0]] & is_test[pairs[:, 1]]
    print(f"{len(uniq)} sessions -> {len(uniq) - n_test} train / {n_test} test")
    print(f"{int(tr.sum())} train pairs, {int(te.sum())} test pairs")

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    net = build_net(args.backbone, args.dim)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    tr_pairs = pairs[tr]
    te_pairs = pairs[te]

    def embed(idx, train):
        return net(as_batch(crops, idx, args.size, mean, std, train))

    def evaluate():
        net.eval()
        sims, same = [], []
        with torch.no_grad():
            for i in range(0, len(te_pairs), 256):
                p = te_pairs[i:i + 256]
                a = embed(p[:, 0], False)
                b = embed(p[:, 1], False)
                sims.append((a * b).sum(1).numpy())
                same.append(p[:, 2])
        if not sims:
            return 0.0
        s = np.concatenate(sims)
        y = np.concatenate(same)
        # AUC: the chance a same-character pair scores above a different one.
        # Threshold-free, so it cannot be flattered by picking a cut later.
        order = np.argsort(s)
        ranks = np.empty(len(s), np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if not npos or not nneg:
            return 0.0
        return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    t0 = time.perf_counter()
    best, best_state, since = 0.0, None, 0
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(tr_pairs))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            p = tr_pairs[perm[i:i + args.batch]]
            a = embed(p[:, 0], True)
            b = embed(p[:, 1], True)
            sim = (a * b).sum(1)
            y = torch.from_numpy(p[:, 2]).float()
            # Positives pulled to 1; negatives pushed below `margin` and then
            # left alone, so the loss stops spending on pairs already apart.
            loss = (y * (1.0 - sim)
                    + (1.0 - y) * torch.clamp(sim - args.margin, min=0.0)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(p)
        a = evaluate()
        mark = ""
        if a > best:
            # KEEP THE BEST WEIGHTS, NOT THE LAST. This tracked `best` and
            # then exported whatever the final epoch happened to leave
            # behind, so the AUC reported and the model written were two
            # different models -- and on a run that overfits past its peak,
            # the one shipped is strictly worse than the one reported.
            best, since = a, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            mark = "  <- best"
        else:
            since += 1
        print(f"  epoch {ep + 1:>2}/{args.epochs}  loss {total / len(tr_pairs):.4f}  "
              f"held-out AUC {a:.3f}{mark}", flush=True)
        if args.patience and since >= args.patience:
            print(f"  stopping: {args.patience} epochs with no gain")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"\ntrained in {time.perf_counter() - t0:.0f}s, best held-out AUC {best:.3f}")
    print("  0.50 = knows nothing. The previous hand-made descriptors reached")
    print("  0.561 and the linear metric over them 0.510.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pt = args.out.with_suffix(".pt")
    torch.save(net.state_dict(), pt)          # BEFORE export: the exporter has
    print(f"weights -> {pt}")                 # killed a finished run before

    net.eval()
    dummy = torch.zeros(BATCH_FIXED, 3, args.size, args.size)
    torch.onnx.export(net, dummy, str(args.out), dynamo=False,
                      input_names=["x"], output_names=["e"], opset_version=17)
    args.out.with_suffix(".json").write_text(json.dumps({
        "dim": args.dim, "size": args.size, "batch": BATCH_FIXED,
        "window": WINDOW, "crop": CROP,
        "mean": mean.tolist(), "std": std.tolist(),
        "auc": round(float(best), 4),
        # The sessions this net never saw. Written here rather than left to be
        # recomputed, because any later evaluation that scores it on a training
        # session is measuring memory: `scripts/group_eval.py --held-out` reads
        # this list and refuses to score anything else.
        "test_sessions": sorted(test_s),
    }, indent=2), encoding="utf-8")
    print(f"model -> {args.out}  (fixed batch {BATCH_FIXED})")
    print("\nNow score it where it counts:")
    print("  .venv/Scripts/python scripts/group_eval.py --embed models/embed.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
