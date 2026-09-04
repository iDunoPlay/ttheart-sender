"""Find the crops whose label is wrong, by asking a model that never saw them.

Labels made by clustering are wrong in bulk, not at random. One folder of this
project's first labelled set was ~60% a different character -- the detections
were centred off their tsum, so the crops were pictures of the neighbour, and
the person naming them was right about the tsum and wrong about the picture.
That single folder produced almost every error in the confusion matrix and
dragged an innocent class down with it (Randall: 73% recall with it, 90% and
100% precision without).

Finding those by eye means opening every folder. A model finds them in one
pass -- but only if it never saw the crop it is judging, and that is the whole
design of this script:

**k-fold by session.** The sessions are split into `--folds` groups; for each
group a model is trained on the others and predicts it. Every crop therefore
gets an out-of-fold prediction from a model with no memory of it. Scoring a
crop with a model that trained on it is asking whether the model memorised the
label, which it did, and that is worth nothing.

A disagreement is flagged when the out-of-fold model is CONFIDENT and wrong:

    predicted != label  AND  softmax >= --confidence

Both halves matter. Low-confidence disagreements are the model being unsure,
which is not evidence about the label; confident ones are the model insisting,
on evidence it learned from other sessions, that this picture is a different
character.

It never edits anything by itself. It writes one contact sheet per
(label -> predicted) pair into `crops/review/`, and `--apply` moves the flagged
crops only after a person has looked:

    python scripts/clean_labels.py                     # look
    python scripts/clean_labels.py --apply             # then move them
    python scripts/clean_labels.py --apply --only MissBianca

Flagged crops move to `crops/labelled/<predicted>/`, except with
`--to-rejected`, which parks them in `crops/rejected/` instead -- the right
choice when the crop is not a character at all (a coin, an offset crop, a
fragment) and the model's second opinion is just as wrong as the first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import ROOT, as_tensor, augment, build_net, load  # noqa: E402

REVIEW = Path("crops/review")


def out_of_fold(paths, y, names, sess, folds, epochs, batch, lr, backbone, seed):
    """A prediction for every crop, from a model trained without its session."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = as_tensor(paths)
    yt = torch.from_numpy(y).long()

    uniq = sorted(set(sess.tolist()))
    # Sessions, not crops: the same physical tsum appears in consecutive
    # frames of one round, so a crop-level fold leaves near-duplicates of the
    # test crop in training and the model recognises the board, not the
    # character.
    assign = {s: i % folds for i, s in enumerate(uniq)}
    fold_of = np.array([assign[s] for s in sess])

    pred = np.full(len(paths), -1, np.int64)
    conf = np.zeros(len(paths), np.float32)

    for f in range(folds):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        if not len(te) or not len(tr):
            continue
        # A class absent from this fold's training half cannot be predicted,
        # and every one of its crops would be flagged as wrong. Skip scoring
        # those rather than reporting them as errors.
        present = set(y[tr].tolist())
        net = build_net(backbone, len(names))
        freq = np.array([max(1, int((y[tr] == i).sum())) for i in range(len(names))],
                        np.float32)
        w = torch.from_numpy((freq.sum() / (len(names) * freq)).astype(np.float32))
        lossf = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        tr_t = torch.from_numpy(tr)
        net.train()
        for ep in range(epochs):
            perm = tr_t[torch.randperm(len(tr_t))]
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                loss = lossf(net(augment(X[b])), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logits = torch.cat([net(X[torch.from_numpy(te[i:i + 256])])
                                for i in range(0, len(te), 256)])
            prob = torch.softmax(logits, 1).numpy()
        for j, idx in enumerate(te):
            if y[idx] in present:
                pred[idx] = int(prob[j].argmax())
                conf[idx] = float(prob[j].max())
        print(f"  fold {f + 1}/{folds}: scored {len(te)} crops", flush=True)
    return pred, conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default="mobilenet_v3_small")
    ap.add_argument("--min-class", type=int, default=60)
    ap.add_argument("--confidence", type=float, default=0.90,
                    help="only flag a disagreement the model is this sure of")
    ap.add_argument("--apply", action="store_true", help="move the flagged crops")
    ap.add_argument("--only", help="restrict to one label")
    ap.add_argument("--to-rejected", action="store_true",
                    help="park flagged crops in crops/rejected/ instead of "
                         "moving them to the predicted class -- for crops that "
                         "are not a character at all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths, y, names, sess = load(args.dir, 0.0)
    if not paths:
        print(f"nothing labelled under {args.dir}")
        return 1
    keep = {i for i in range(len(names)) if int((y == i).sum()) >= args.min_class}
    if len(keep) < 2:
        print(f"need 2+ classes with {args.min_class}+ crops")
        return 1
    sel = np.isin(y, list(keep))
    paths = [p for p, k in zip(paths, sel) if k]
    remap = {o: n for n, o in enumerate(sorted(keep))}
    y = np.array([remap[v] for v in y[sel]])
    sess = sess[sel]
    names = [names[i] for i in sorted(keep)]
    print(f"{len(paths)} crops, {len(names)} classes, "
          f"{len(set(sess.tolist()))} sessions, {args.folds} folds\n")

    t0 = time.perf_counter()
    pred, conf = out_of_fold(paths, y, names, sess, args.folds, args.epochs,
                             args.batch, args.lr, args.backbone, args.seed)
    print(f"\nout-of-fold in {time.perf_counter() - t0:.0f}s")

    scored = pred >= 0
    agree = (pred == y) & scored
    print(f"the model agrees with {agree.sum()}/{scored.sum()} labels "
          f"({agree.sum() / max(scored.sum(), 1):.1%})")

    flag = scored & (pred != y) & (conf >= args.confidence)
    if args.only:
        flag &= np.array([names[v] == args.only for v in y])
    print(f"confidently disagrees on {int(flag.sum())} "
          f"({flag.sum() / max(scored.sum(), 1):.1%}) at confidence "
          f"{args.confidence:.2f}\n")
    if not flag.any():
        print("nothing to review -- the labels look clean at this threshold")
        return 0

    pairs = Counter((names[y[i]], names[pred[i]]) for i in np.where(flag)[0])
    print(f"{'label':>18} -> {'model says':<18}{'crops':>7}")
    for (a, b), n in pairs.most_common():
        print(f"{a:>18} -> {b:<18}{n:>7}")

    if not args.apply:
        from crops import montage
        if REVIEW.exists():
            shutil.rmtree(REVIEW)
        REVIEW.mkdir(parents=True, exist_ok=True)
        for (a, b) in pairs:
            grp = [paths[i] for i in np.where(flag)[0]
                   if names[y[i]] == a and names[pred[i]] == b]
            montage(grp, REVIEW / f"{a}_MODEL_SAYS_{b}.png", cols=12, rows=6)
        print(f"\nsheets written to {REVIEW}. Look at them, then:")
        print("  python scripts/clean_labels.py --apply            # move them all")
        print("  python scripts/clean_labels.py --apply --only X   # just one label")
        print("  ... --apply --to-rejected                          # not a character")
        print("\nA sheet where the model is RIGHT is a label to fix. A sheet where")
        print("it is wrong is a class that needs more crops, not a correction.")
        return 0

    moved = 0
    for i in np.where(flag)[0]:
        src = Path(paths[i])
        dest = (Path("crops/rejected") / names[y[i]] if args.to_rejected
                else args.dir / names[pred[i]])
        dest.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dest / src.name))
            moved += 1
    where = "crops/rejected" if args.to_rejected else "the class the model named"
    print(f"\nmoved {moved} crops to {where}")
    print("Retrain to see what it bought:  python scripts/classify.py --epochs 16 --reject 0.6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
