"""Stages 3-5: transfer-learn a character classifier, and score it honestly.

Trains on the folders `crops.py assign` produced:

    crops/labelled/Mickey/*.png
    crops/labelled/Pooh/*.png
    ...

and answers the question `Tsum.kind` cannot: **which character is this?**
`kind` is a per-frame k-means cluster id -- a partition of pixels that means
nothing between frames and agrees with the game about a confirmed partner only
37% of the time. A classifier gives a name that is stable across frames,
rounds and sessions, which is the thing every accumulated statistic in this
project has lacked a peg for.

Transfer learning, because the training set is small. A backbone pretrained on
ImageNet already knows edges, fur, eyes and roundness; only the last layer is
learned from scratch, and the rest is fine-tuned gently. MobileNetV3-Small by
default -- it is the smallest torchvision backbone that exports cleanly to
ONNX and runs 46 crops inside the play loop's budget.

Three things this does that the tutorials do not, and each is here because
this project has been burned without it:

* **Held out by SESSION, never by crop.** Boards inside one round are alike
  and the same physical tsum appears in consecutive frames, so a random split
  puts near-duplicates on both sides and reports a number that means nothing.
  The filenames carry their session; the split follows it.
* **A confusion matrix, not an accuracy.** Accuracy hides the failure that
  matters: two characters that share a dominant colour being swapped for each
  other. That is the whole hypothesis under test, and only the matrix shows it.
* **An explicit UNKNOWN.** A classifier forced to choose will name a
  half-buried sliver of a tsum with confidence, and `adjacency` will believe
  it. `--reject` sets the softmax floor below which a crop is called unknown,
  and the report prints coverage beside accuracy so the trade is visible.

    python scripts/classify.py --epochs 12
    python scripts/classify.py --epochs 12 --onnx models/character.onnx

Training needs torch and torchvision. Inference does not: the ONNX loads
through `cv2.dnn`, already bundled with the app.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("crops/labelled")
SIZE = 96          #: crops are 64px; the backbone wants more, this is the compromise

#: `<session>_<sample>_<index>_v<visible>.png`, written by `crops.py`. The
#: session prefix is what the split is made on.
NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2})_v(?P<vis>[\d.]+)$")


def load(root: Path, min_visible: float):
    """Paths, class ids, class names and session ids."""
    paths, ys, sessions = [], [], []
    names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not names:
        return [], np.zeros(0, int), [], np.zeros(0, int)
    sess_ids: dict[str, int] = {}
    for ci, name in enumerate(names):
        for p in sorted((root / name).glob("*.png")):
            m = NAME.match(p.stem)
            if m and float(m.group("vis")) < min_visible:
                continue
            sess = m.group("sess") if m else "unknown"
            paths.append(p)
            ys.append(ci)
            sessions.append(sess_ids.setdefault(sess, len(sess_ids)))
    return paths, np.asarray(ys), names, np.asarray(sessions)


def confusion(y, pred, names, reject=None):
    """Print the matrix, then the per-class numbers that the matrix implies."""
    k = len(names)
    m = np.zeros((k, k + (1 if reject is not None else 0)), int)
    for t, p in zip(y, pred):
        m[t, p if p >= 0 else k] += 1
    w = max(9, max(len(n) for n in names) + 1)
    head = "".join(f"{n[:7]:>8}" for n in names)
    corner = "true \\ pred"
    unk = f"{'UNK':>8}" if reject is not None else ""
    print(f"\n{corner:>{w}}{head}{unk}")
    for i, n in enumerate(names):
        row = "".join(f"{v:>8}" for v in m[i])
        print(f"{n:>{w}}{row}")

    print(f"\n{'class':>{w}}{'n':>7}{'recall':>9}{'precis':>9}")
    for i, n in enumerate(names):
        tp = m[i, i]
        support = m[i].sum()
        picked = m[:, i].sum()
        rec = tp / support if support else 0.0
        pre = tp / picked if picked else 0.0
        print(f"{n:>{w}}{support:7d}{rec:9.1%}{pre:9.1%}")

    named = np.asarray(pred) >= 0
    cov = float(named.mean())
    acc = float((np.asarray(pred)[named] == np.asarray(y)[named]).mean()) if named.any() else 0.0
    print(f"\n  accuracy on the crops it named: {acc:.1%}")
    if reject is not None:
        print(f"  coverage (crops it was willing to name): {cov:.1%} at reject {reject:.2f}")
    # The confusion that matters most: which pair gets swapped.
    off = [(m[i, j], names[i], names[j]) for i in range(k) for j in range(k) if i != j]
    off.sort(reverse=True)
    if off and off[0][0]:
        print("  worst confusions: " + ", ".join(f"{a}->{b} ({c})" for c, a, b in off[:3]))
    return acc, cov


def build_net(backbone: str, n_classes: int):
    """The backbone with its head replaced. Factored out so the label-cleaning
    pass trains exactly the same model this scores, rather than a copy that
    can drift away from it."""
    import torch.nn as nn
    from torchvision import models
    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
    else:
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, n_classes)
    return net


def augment(b):
    """The pile rotates, mirrors and shades its tsums; so does this."""
    import torch
    import torch.nn as nn
    n = b.shape[0]
    if torch.rand(1).item() < 0.5:
        b = torch.flip(b, dims=[3])
    ang = (torch.rand(n) * 2 - 1) * 0.35
    sc = 1.0 + (torch.rand(n) * 2 - 1) * 0.10
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.zeros(n, 2, 3)
    th[:, 0, 0] = cos; th[:, 0, 1] = -sin
    th[:, 1, 0] = sin; th[:, 1, 1] = cos
    g = nn.functional.affine_grid(th, b.shape, align_corners=False)
    b = nn.functional.grid_sample(b, g, align_corners=False, padding_mode="border")
    return b * (1.0 + (torch.rand(n, 1, 1, 1) * 2 - 1) * 0.2)


def as_tensor(paths):
    """Crops -> a normalised NCHW tensor, the same way for every caller."""
    import cv2
    import torch
    from torchvision import transforms
    X = np.zeros((len(paths), SIZE, SIZE, 3), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        X[i] = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
    return transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])(t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.3, help="share of SESSIONS")
    ap.add_argument("--min-visible", type=float, default=0.0,
                    help="skip crops showing less of a tsum than this")
    ap.add_argument("--min-class", type=int, default=100,
                    help="drop classes with fewer crops than this. A class of "
                         "20 cannot be trained OR scored -- it lands a handful "
                         "of crops in the held-out sessions and its recall is "
                         "then noise reported to two decimal places")
    ap.add_argument("--reject", type=float, default=0.0,
                    help="softmax floor below which a crop is UNKNOWN (0 = never)")
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.dir.exists():
        print(f"no {args.dir} -- label some clusters first:")
        print("  python scripts/crops.py extract --dir dataset")
        print("  python scripts/crops.py cluster --model models/embed.onnx")
        print("  python scripts/crops.py assign 03 Mickey")
        return 1

    paths, y, names, sess = load(args.dir, args.min_visible)
    if args.min_class > 1 and len(names):
        keep = {i for i, n in enumerate(names)
                if int((y == i).sum()) >= args.min_class}
        dropped = [names[i] for i in range(len(names)) if i not in keep]
        if dropped:
            print(f"dropping {len(dropped)} class(es) under {args.min_class} "
                  f"crops: {', '.join(sorted(dropped))}\n")
        sel = np.isin(y, list(keep))
        paths = [p for p, k in zip(paths, sel) if k]
        remap = {old: new for new, old in enumerate(sorted(keep))}
        y = np.array([remap[v] for v in y[sel]])
        sess = sess[sel]
        names = [names[i] for i in sorted(keep)]
    if len(names) < 2:
        print(f"need at least 2 labelled classes, found {len(names)}")
        return 1
    counts = Counter(y.tolist())
    print(f"{len(paths)} crops, {len(names)} classes, {len(set(sess.tolist()))} sessions")
    for i, n in enumerate(names):
        print(f"  {n:22} {counts.get(i, 0):6d}" +
              ("   <- thin" if counts.get(i, 0) < 100 else ""))
    if min(counts.values()) < 20:
        print("\nA class under 20 crops cannot be trained or scored. Label more.")
        return 1

    try:
        import torch, torch.nn as nn
        from torchvision import models, transforms
    except ImportError:
        print("\nneeds torch + torchvision (training only):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        return 1

    import cv2
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    uniq = sorted(set(sess.tolist()))
    cut = max(1, int(len(uniq) * (1 - args.holdout)))
    train_s = set(uniq[:cut])
    tr_i = np.array([i for i, s in enumerate(sess) if s in train_s])
    te_i = np.array([i for i, s in enumerate(sess) if s not in train_s])
    if not len(te_i):
        print("every crop is from one session -- cannot hold anything out.")
        return 1
    print(f"\nsessions: train {cut} / test {len(uniq) - cut}   "
          f"crops: train {len(tr_i)} / test {len(te_i)}")

    X = np.zeros((len(paths), SIZE, SIZE, 3), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        X[i] = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    Xt = norm(Xt)
    yt = torch.from_numpy(y).long()

    if args.backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, len(names))
    else:
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(names))

    # Class weights: an unbalanced set otherwise trains a model that predicts
    # the common character and scores well on accuracy while being useless.
    freq = np.array([counts.get(i, 1) for i in range(len(names))], np.float32)
    wts = torch.from_numpy((freq.sum() / (len(names) * freq)).astype(np.float32))
    lossf = nn.CrossEntropyLoss(weight=wts)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def aug(b):
        """The pile rotates, mirrors and shades its tsums; so does this."""
        n = b.shape[0]
        if torch.rand(1).item() < 0.5:
            b = torch.flip(b, dims=[3])
        ang = (torch.rand(n) * 2 - 1) * 0.35
        sc = 1.0 + (torch.rand(n) * 2 - 1) * 0.10
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        th = torch.zeros(n, 2, 3)
        th[:, 0, 0] = cos; th[:, 0, 1] = -sin
        th[:, 1, 0] = sin; th[:, 1, 1] = cos
        g = nn.functional.affine_grid(th, b.shape, align_corners=False)
        b = nn.functional.grid_sample(b, g, align_corners=False, padding_mode="border")
        return b * (1.0 + (torch.rand(n, 1, 1, 1) * 2 - 1) * 0.2)

    tr_t = torch.from_numpy(tr_i)
    print(f"\n{'epoch':>6}{'loss':>9}{'test acc':>10}")
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            out = net(aug(Xt[b]))
            loss = lossf(out, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            logits = torch.cat([net(Xt[torch.from_numpy(te_i[i:i + 256])])
                                for i in range(0, len(te_i), 256)])
        acc = float((logits.argmax(1).numpy() == y[te_i]).mean())
        print(f"{ep:6d}{tot / len(perm):9.4f}{acc:10.1%}", flush=True)
    print(f"\ntrained in {time.perf_counter() - t0:.0f}s")

    with torch.no_grad():
        logits = torch.cat([net(Xt[torch.from_numpy(te_i[i:i + 256])])
                            for i in range(0, len(te_i), 256)])
        prob = torch.softmax(logits, 1).numpy()
    pred = prob.argmax(1)
    if args.reject > 0:
        pred = np.where(prob.max(1) >= args.reject, pred, -1)
    print("\n== held-out confusion matrix (rows = truth) ==")
    confusion(y[te_i], pred, names, reject=args.reject if args.reject > 0 else None)

    print("\nRead the matrix, not the accuracy. Two characters swapped for each")
    print("other is the failure this whole line of work is about; a high accuracy")
    print("with one such pair still means the bot will chain them together.")

    if args.onnx:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        net.eval()
        torch.onnx.export(net, torch.zeros(1, 3, SIZE, SIZE), str(args.onnx),
                          input_names=["crop"], output_names=["logits"],
                          dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}},
                          dynamo=False)
        args.onnx.with_suffix(".json").write_text(
            json.dumps({"classes": names, "size": SIZE,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]}, indent=2), encoding="utf-8")
        print(f"\nexported {args.onnx} (+ .json with the class order)")
        print("Load it with cv2.dnn.readNetFromONNX -- no torch at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
