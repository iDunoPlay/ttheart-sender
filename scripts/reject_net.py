"""Is this detection a tsum at all?

A different question from "which character is this", and a more useful one.

`docs/IDENTITY.md` closed the identity programme: a classifier reads a whole
tsum at 97.5% and reads the 78% of a board that is buried at chance, so naming
characters cannot help. That conclusion was about OCCLUSION -- pictures that do
not contain the thing they are centred on.

This is the opposite failure and it is not occluded at all. When the player
labelled 5,669 crops they left 763 unnamed with a plain reason: *"shouldn't be
detected as it is background or unknown object, which is invalid for the
gameplay"*. Those crops are **more visible than the labelled ones** -- median
0.68 against 0.64 -- so they are not slivers of a buried tsum. They are empty
bowl, the board's own blue slot motif, off-board black, and effect flashes,
detected at full visibility and handed to the chain builder as tsums.

Why it costs a round
--------------------

A false detection carries a `kind` like any other, so `adjacency` will happily
put it in a chain. The stroke then passes over empty board, the game skips that
stop, and the chain spends a slot on nothing. It also inflates `dragged`, which
deflates the clear rate every rule in this project is priced against.

So this is a detection fix with a gameplay payout, and unlike identity it does
not need the picture to contain a whole tsum -- a picture of the bowl is a
complete and perfectly readable picture of the bowl.

    python scripts/reject_net.py --epochs 20 --onnx models/reject.onnx

Held out BY SESSION, like everything else here. The negatives come from 241
different sessions, so a random crop split would put near-identical bowl
squares on both sides and report a number that means nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import NAME, SIZE, build_net  # noqa: E402

#: The negative class, and it is ONE folder a person filled on purpose.
#:
#: Verified against the game before being trained on: of 705 `board` crops, the
#: game marked **0.0%** and none was ever a chain head. A patch of bowl cannot
#: be confirmed as a character, so that zero is what a clean negative looks
#: like.
#:
#: Two folders that look like they belong here and do NOT:
#:
#: * `unknown_lightball` is **86.8% marked**. It is the game's own LINK
#:   HIGHLIGHT -- the glow drawn over a tsum the game has just confirmed. The
#:   first version of this script listed it as junk and learned to reject the
#:   exact visual state that means "this one counts", scoring 0.999 AUC while
#:   throwing away game-confirmed tsums at 36.3% against 15.7% for the ones it
#:   kept. A lit tsum is a tsum.
#: * `coin` is **66.3% marked**, four times a real character's rate, so it is
#:   not the clean UI class it appears to be either.
#:
#: Hence one folder, not a guess at several. An absence of a label is not a
#: negative label; this is a label.
NOT_TSUM = ("board",)

#: The leftover pile, and it is NOT used by default -- see `--junk`.
#:
#: It was, once. "Left unlabelled" turned out to be a mixture of *this is
#: background* and *I cannot tell what this is*, and 21% of it was tsums the
#: game had drawn its highlight over. Cleaning the confirmed ones out did not
#: flip the sign. Inference from silence is not a label.
JUNK = Path("crops/all")


def confirmed(dataset: Path) -> set:
    """(session, frame, detection) for every tsum the GAME confirmed.

    A crop the game marked, or one the bot dragged and saw clear, is a real
    tsum whatever anyone labelled it. This is the only way to clean the
    negative pile without asking the player to look at it twice, and it is
    ground truth rather than a second opinion.

    It is needed because "left unlabelled" is not the same claim as "not a
    tsum". The click-UI names what is picked and leaves the rest, so the
    leftovers are a mixture: empty bowl and board graphics, which is what the
    player meant -- and tsums rendered in a state the player did not recognise,
    which the game is perfectly happy to confirm.
    """
    out = set()
    for d in sorted(dataset.iterdir()) if dataset.exists() else []:
        f = d / "samples.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for i in (row.get("marked") or []) + (row.get("cleared") or []):
                out.add((d.name, int(row["index"]), int(i)))
    return out


def linked_positives(dataset: Path, limit: int, seed: int = 0):
    """Crops of tsums the game MARKED, as extra positives.

    The one thing a crop cannot settle. The board is printed with empty
    tsum-shaped slots, and a tsum the game has linked into the chain is drawn
    as a flat silhouette -- same outline, same flat fill, same dark border. A
    person cannot tell them apart from one 64px square and neither can a net,
    which is why a model trained on a verified-clean `board` class still threw
    away game-confirmed tsums at 20.8% against 16.6%.

    The marks settle it for free: the game only lights up a real tsum of the
    head's character. So every marked detection is a positive, whatever it
    looks like, and the model gets to see the silhouette state labelled
    correctly instead of inferring it from an outline it shares with the board.
    """
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crops import WINDOW, _cut
    rng = np.random.default_rng(seed)
    out = []
    for d in sorted(dataset.iterdir()) if dataset.exists() else []:
        f = d / "samples.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            marked = row.get("marked") or []
            tsums = row.get("tsums") or []
            if not marked or row.get("fever"):
                continue
            img = None
            radius = float(row.get("radius") or 25.0)
            for i in marked:
                if i >= len(tsums) or rng.random() > 0.25:
                    continue
                if img is None:
                    img = cv2.imread(str(d / ("%04d_before.jpg" % row["index"])))
                    if img is None:
                        break
                t = tsums[i]
                c = _cut(img, t["x"], t["y"], radius, WINDOW)
                if c is not None:
                    out.append((c, d.name))
                    if len(out) >= limit:
                        return out
    return out


def load(labelled: Path, junk: Path, min_visible: float, keep=None):
    """Crops, y (1 = a chainable tsum), and the session each came from."""
    paths, ys, sess = [], [], []
    rescued = 0

    def add(p: Path, y: int):
        m = NAME.match(p.stem)
        if m and float(m.group("vis")) < min_visible:
            return
        paths.append(p)
        ys.append(y)
        sess.append(m.group("sess") if m else "unknown")

    for d in sorted(labelled.iterdir()) if labelled.exists() else []:
        if not d.is_dir():
            continue
        y = 0 if d.name in NOT_TSUM else 1
        for p in sorted(d.glob("*.png")):
            add(p, y)
    for p in sorted(junk.glob("*.png")) if junk and junk.exists() else []:
        m = NAME.match(p.stem)
        key = ((m.group("sess"), int(m.group("sample")), int(m.group("idx")))
               if m else None)
        if keep and key in keep:
            # The game confirmed this one. It is a tsum, and training against
            # it would teach the model to throw away the very detections the
            # marks are pointing at.
            rescued += 1
            add(p, 1)
            continue
        add(p, 0)
    if rescued:
        print(f"{rescued} of the unlabelled crops were CONFIRMED by the game "
              f"and are counted as tsums, not junk")
    return paths, np.asarray(ys), np.asarray(sess, dtype=object)


def auc(y, score) -> float:
    """Rank AUC, ties averaged. Threshold-free, which matters here: where the
    floor goes is a gameplay trade -- a rejected real tsum costs a chain
    member, an accepted fake costs a chain slot -- and it should be chosen
    from the curve rather than baked into the number that reports the model."""
    y = np.asarray(y)
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):                       # average the ranks of ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--junk", type=Path,
                    help="also train against the crops left unlabelled in this "
                         "directory. OFF by default: an absence of a label is "
                         "not a negative label, and doing it cost a round")
    ap.add_argument("--dataset", type=Path, default=Path("dataset"),
                    help="corpus to read the game's marks from, so a crop the "
                         "game confirmed is never trained against as junk")
    ap.add_argument("--linked", type=int, default=3000,
                    help="extra positives cut straight from the corpus: "
                         "detections the game MARKED. They are the only way to "
                         "teach the flat linked-silhouette state, which shares "
                         "its outline with the board's own empty slots and "
                         "cannot be told apart from one crop. 0 to skip")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.3, help="share of SESSIONS")
    ap.add_argument("--min-visible", type=float, default=0.0)
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.junk:
        print("WARNING: --junk trains against crops that were merely left "
              "unlabelled.")
        print("  21% of that pile was game-confirmed last time. See "
              "docs/DATASET-FINDINGS.md round 33.")
    extra = (linked_positives(args.dataset, args.linked, args.seed)
             if args.linked else [])
    if extra:
        print(f"{len(extra)} extra positives: detections the GAME marked, "
              f"whatever they look like")
    keep = confirmed(args.dataset) if args.dataset.exists() else set()
    if keep:
        print(f"{len(keep)} detections the game itself confirmed")
    paths, y, sess = load(args.dir, args.junk, args.min_visible, keep)
    # Appended BEFORE the split, carrying their own session names, so a
    # harvested positive lands on the same side of the cut as every other crop
    # from its round. Added after the split they would leak.
    if extra:
        y = np.concatenate([y, np.ones(len(extra), int)])
        sess = np.concatenate([sess, np.asarray([s for _, s in extra],
                                                dtype=object)])
    if len(set(y.tolist())) < 2:
        print(f"need both classes -- found {Counter(y.tolist())}.")
        print(f"Negatives come from {args.junk} (crops left unlabelled) plus "
              f"{', '.join(NOT_TSUM)}.")
        return 1
    print(f"{len(paths)} crops: {int((y == 1).sum())} tsums, "
          f"{int((y == 0).sum())} not")
    print("  negatives: " + ", ".join(
        f"{n} {len(list((args.dir / n).glob('*.png')))}"
        for n in NOT_TSUM if (args.dir / n).exists())
        + (f" + {len(list(args.junk.glob('*.png')))} unlabelled"
           if args.junk and args.junk.exists() else ""))

    import cv2
    import torch
    import torch.nn as nn
    from torchvision import transforms
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       or args.device == "cuda" else "cpu")
    print(f"device: {dev}")

    uniq = sorted(set(sess.tolist()))
    rng = np.random.default_rng(args.seed)
    uniq = [uniq[i] for i in rng.permutation(len(uniq))]
    cut = max(1, int(len(uniq) * (1 - args.holdout)))
    train_s, test_s = set(uniq[:cut]), set(uniq[cut:])
    tr = np.array([i for i, s in enumerate(sess) if s in train_s])
    te = np.array([i for i, s in enumerate(sess) if s not in train_s])
    if not len(te) or not len(tr):
        print("cannot split by session -- too few sessions")
        return 1
    print(f"sessions: train {len(train_s)} / test {len(test_s)}   "
          f"crops: train {len(tr)} / test {len(te)} "
          f"({int((y[te] == 0).sum())} of the test crops are not tsums)")

    X = np.zeros((len(paths) + len(extra), SIZE, SIZE, 3), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        X[i] = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    for j, (c, _) in enumerate(extra):
        X[len(paths) + j] = cv2.resize(c, (SIZE, SIZE),
                                       interpolation=cv2.INTER_LINEAR)
    Xt = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)).to(dev)
    yt = torch.from_numpy(y).long().to(dev)

    net = build_net(args.backbone, 2).to(dev)
    freq = np.array([max(1, int((y == c).sum())) for c in (0, 1)], np.float32)
    wts = torch.from_numpy((freq.sum() / (2 * freq)).astype(np.float32)).to(dev)
    lossf = nn.CrossEntropyLoss(weight=wts)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def aug(b):
        n = b.shape[0]
        if torch.rand(1).item() < 0.5:
            b = torch.flip(b, dims=[3])
        ang = (torch.rand(n, device=b.device) * 2 - 1) * 0.35
        sc = 1.0 + (torch.rand(n, device=b.device) * 2 - 1) * 0.10
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        th = torch.zeros(n, 2, 3, device=b.device)
        th[:, 0, 0] = cos; th[:, 0, 1] = -sin
        th[:, 1, 0] = sin; th[:, 1, 1] = cos
        g = nn.functional.affine_grid(th, b.shape, align_corners=False)
        b = nn.functional.grid_sample(b, g, align_corners=False,
                                      padding_mode="border")
        return b * (1.0 + (torch.rand(n, 1, 1, 1, device=b.device) * 2 - 1) * 0.2)

    def scores(idx):
        with torch.no_grad():
            out = torch.cat([net(Xt[torch.from_numpy(idx[i:i + 256]).to(dev)]).cpu()
                             for i in range(0, len(idx), 256)])
        return torch.softmax(out, 1)[:, 1].numpy()

    tr_t = torch.from_numpy(tr).to(dev)
    best, best_state, best_ep = -1.0, None, 0
    print(f"\n{'epoch':>6}{'loss':>9}{'held-out AUC':>14}")
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = lossf(net(aug(Xt[b])), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        net.eval()
        a = auc(y[te], scores(te))
        if a > best:
            best, best_ep = a, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        print(f"{ep:6d}{tot / len(perm):9.4f}{a:14.4f}"
              + ("   <- best" if ep == best_ep else ""), flush=True)
    if best_state is not None:
        net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    print(f"\nbest held-out AUC {best:.4f} (epoch {best_ep}) "
          f"in {time.perf_counter() - t0:.0f}s")

    net.eval()
    s = scores(te)
    print("\nWhere to put the floor is a gameplay trade, not a modelling one:")
    print("a rejected real tsum costs a chain MEMBER, an accepted fake costs a")
    print("chain SLOT and a stroke over empty board.\n")
    print(f"{'floor':>7}{'fakes rejected':>17}{'real tsums lost':>18}"
          f"{'kept per 40-tsum board':>25}")
    for f in (0.10, 0.25, 0.50, 0.75, 0.90):
        keep = s >= f
        fake_rej = float((~keep)[y[te] == 0].mean()) if (y[te] == 0).any() else 0.0
        real_lost = float((~keep)[y[te] == 1].mean()) if (y[te] == 1).any() else 0.0
        print(f"{f:7.2f}{fake_rej:17.1%}{real_lost:18.1%}"
              f"{40 * (1 - real_lost):25.1f}")

    if args.onnx:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        net.cpu().eval()
        import torch as _t
        _t.onnx.export(net, _t.zeros(1, 3, SIZE, SIZE), str(args.onnx),
                       input_names=["crop"], output_names=["logits"],
                       dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}},
                       dynamo=False)
        _t.save({"state": net.state_dict(), "backbone": args.backbone},
                str(args.onnx.with_suffix(".pt")))
        args.onnx.with_suffix(".json").write_text(json.dumps({
            "classes": ["not_a_tsum", "tsum"], "size": SIZE,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "backbone": args.backbone, "held_out_auc": round(best, 4),
            "best_epoch": best_ep, "seed": args.seed,
            "not_tsum_classes": list(NOT_TSUM),
            "train_sessions": sorted(train_s),
            "test_sessions": sorted(test_s)}, indent=2), encoding="utf-8")
        print(f"\nexported {args.onnx} (+ .json, + .pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
