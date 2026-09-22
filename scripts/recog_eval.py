"""Score the SHIPPED character model on the fixed held-out sessions.

`classify.py` trains and prints a confusion matrix, but it prints it for the
model it has just trained, in memory, in torch. This scores the artifact the
bot actually loads -- `models/character.onnx` through `cv2.dnn` -- on the
sessions recorded inside `models/character.json` as held out when it was
trained. Nothing here trains anything, so the same command run twice gives the
same numbers, and the test set cannot drift.

    python scripts/recog_eval.py
    python scripts/recog_eval.py --reject 0.85        # the shipped floor

**Read the visibility warning it prints before quoting any number from it.**
Every labelled crop in this project is at least 0.55 visible -- measured, not
assumed: 5,642 of 5,642. The median tsum on a real board shows 0.41. So this
scores recognition on the readable fifth of a board and says nothing about the
rest, and `docs/IDENTITY.md` section 4 measures the rest at close to chance. A
high number here is not evidence that the bot knows what is on the board.

The metrics are the ones a single accuracy figure hides:

* **macro F1**, which weights a 20-crop class the same as a 566-crop one;
* **top-3**, which separates "wrong" from "had no idea";
* **confidence split by correct/wrong**, which is how you find out whether a
  reject floor can help -- in this project's history it could not;
* **latency per crop**, because this runs inside a ~100ms decision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game.tsum import _forward, _warm_up  # noqa: E402

ROOT = Path("crops/labelled")
NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2,})_v(?P<vis>[\d.]+)$")

#: Folders under `crops/labelled` that are not characters. `crops.py` writes
#: them so the labeller can put non-tsums somewhere, and scoring them as
#: identities would report a number for a question nobody asked.
NOT_CHARACTERS = {"board", "score", "junk"}


def train_support(root, trained):
    """How many crops of each class the model was actually SHOWN.

    A class whose every session fell in the held-out half scores 0% by
    arithmetic and drags the headline down with it, which reads as the model
    failing at something it was never asked. `classify.py` prints both numbers
    for exactly this reason; scoring the shipped artifact has to do the same or
    it reports a worse model than the one that exists. Cleo is the live case:
    41 held-out crops from one session, **0 training crops**.
    """
    out = Counter()
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in NOT_CHARACTERS:
            continue
        for p in d.glob("*.png"):
            m = NAME.match(p.stem)
            if m and m.group("sess") in trained:
                out[d.name] += 1
    return out


def load(root, sessions):
    """Crops, their class names, their sessions and their visibilities."""
    paths, labels, sess, vis = [], [], [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in NOT_CHARACTERS:
            continue
        for p in sorted(d.glob("*.png")):
            m = NAME.match(p.stem)
            if not m:
                continue
            if sessions is not None and m.group("sess") not in sessions:
                continue
            paths.append(p)
            labels.append(d.name)
            sess.append(m.group("sess"))
            vis.append(float(m.group("vis")))
    return paths, labels, sess, np.asarray(vis, np.float32)


def probabilities(net, paths, size, mean, std):
    """Softmax per crop, plus milliseconds per crop, through `cv2.dnn`."""
    X = np.zeros((len(paths), size, size, 3), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        X[i] = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    blob = X.astype(np.float32) / 255.0
    blob = (blob - mean) / std
    blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)
    t0 = time.perf_counter()
    logits = _forward(net, blob, "the character model")
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(paths))
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1, keepdims=True)
    return prob, ms


def per_class(y, pred, names):
    """Precision, recall and F1 per class, and the macro mean of the F1s."""
    rows, f1s = [], []
    for ci, n in enumerate(names):
        support = int((y == ci).sum())
        picked = int((pred == ci).sum())
        tp = int(((y == ci) & (pred == ci)).sum())
        rec = tp / support if support else float("nan")
        pre = tp / picked if picked else float("nan")
        f1 = (2 * pre * rec / (pre + rec)
              if support and picked and (pre + rec) > 0 else
              (0.0 if support else float("nan")))
        rows.append((n, support, pre, rec, f1))
        # A class with no held-out crop is not a failure, it is an absence.
        # Averaging a 0 in for it would report the model failing at something
        # it was never asked -- the same arithmetic `classify.py` guards.
        if support:
            f1s.append(f1)
    return rows, float(np.mean(f1s)) if f1s else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--dir", type=Path, default=ROOT)
    ap.add_argument("--golden", type=Path, default=None,
                    help="score on the FROZEN golden sessions instead of this "
                         "model's own held-out split. The only way two "
                         "candidates can be compared: models split differently "
                         "are scored on different boards, and this project "
                         "once read 97.9%% against 94.9%% for two models that "
                         "were exactly as good")
    ap.add_argument("--all-sessions", action="store_true",
                    help="score every session, INCLUDING the ones the model "
                         "trained on. Reports memory, not skill; here only "
                         "because the gap between the two is worth seeing")
    args = ap.parse_args()

    meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    names = list(meta["classes"])
    size = int(meta.get("size", 96))
    mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
    held = set(meta.get("test_sessions", []))
    trained = set(meta.get("train_sessions", []))
    source = "held out at training"
    if args.golden:
        held = set(json.loads(args.golden.read_text(encoding="utf-8"))["sessions"])
        source = "the FROZEN golden set"
        leaked = sorted(held & trained)
        if leaked:
            # REFUSED, not warned. A model trained on part of the golden set
            # scores well on it for the wrong reason, and every comparison
            # against every other candidate silently becomes meaningless.
            print("REFUSED: this model trained on %d golden session(s): %s"
                  % (len(leaked), ", ".join(leaked[:5])))
            return 2
    if not held and not args.all_sessions:
        print("This model records no test_sessions. Retrain with classify.py, "
              "or pass --all-sessions and do not call the result held out.")
        return 2

    paths, labels, sess, vis = load(args.dir, None if args.all_sessions else held)
    if not paths:
        print("no crops matched")
        return 2

    # Classes the model has no output for cannot be named at any threshold.
    # Counted and set aside, never scored as errors: a folder added since the
    # last training run is a retraining job, not a recognition failure.
    index = {n: i for i, n in enumerate(names)}
    missing = Counter(n for n in labels if n not in index)
    keep = [i for i, n in enumerate(labels) if n in index]
    paths = [paths[i] for i in keep]
    y = np.asarray([index[labels[i]] for i in keep])
    vis = vis[keep]
    sess = [sess[i] for i in keep]

    net = cv2.dnn.readNetFromONNX(str(args.model))
    _warm_up(net, size)
    prob, ms = probabilities(net, paths, size, mean, std)
    order = np.argsort(-prob, axis=1)
    pred, conf = order[:, 0], prob.max(1)
    top3 = np.array([y[i] in order[i, :3] for i in range(len(y))])
    right = pred == y

    print("\nmodel     %s  (%d classes, %s, best epoch %s)"
          % (args.model, len(names), meta.get("backbone", "?"),
             meta.get("best_epoch", "?")))
    print("scored on %d crops from %d sessions%s"
          % (len(paths), len(set(sess)),
             "  -- ALL SESSIONS, TRAINING INCLUDED" if args.all_sessions
             else " from %s (%d trained on)" % (source, len(trained))))
    print("          visibility %.2f-%.2f, median %.2f"
          % (vis.min(), vis.max(), float(np.median(vis))))
    print("\n  !! Every labelled crop is at least 0.55 visible. The median")
    print("     tsum on a real board is 0.41. This scores the readable fifth")
    print("     of a board only -- see docs/IDENTITY.md section 4.")
    if missing:
        print("\n  %d crop(s) in %d class(es) the model has no output for, set "
              "aside rather than scored: %s"
              % (sum(missing.values()), len(missing),
                 ", ".join("%s (%d)" % (n, c) for n, c in missing.most_common())))

    shown = train_support(args.dir, trained)
    untaught = {n for n in names if not shown.get(n)}

    rows, macro = per_class(y, pred, names)
    scored = [r for r in rows if r[1]]
    print("\n%18s%7s%9s%9s%9s" % ("class", "n", "precis", "recall", "F1"))
    for n, support, pre, rec, f1 in sorted(scored, key=lambda r: r[4]):
        print("%18s%7d%9.1f%%%8.1f%%%8.1f%%%s"
              % (n, support, pre * 100, rec * 100, f1 * 100,
                 "   NEVER TRAINED" if n in untaught else ""))

    # Both headlines, so neither can be quoted alone. The first is what the
    # model does on characters it was taught; the second includes the ones it
    # was not, and the gap between them is a dataset gap, not a model one.
    taught = np.array([names[t] not in untaught for t in y])
    absent = [r[0] for r in rows if not r[1]]
    # Recomputed on the taught subset, not filtered out of the table above.
    # An untaught class does not only score 0 itself -- its crops land on some
    # other class and wreck that class's PRECISION. Beast is the live case:
    # 25 of the 31 crops called Beast are Cleo, which the model was never
    # shown, so Beast reads as the worst taught class while its recall is fine.
    trows, tmacro = per_class(y[taught], pred[taught], names)
    tscored = [r for r in trows if r[1]]
    print("\n  -- over the %d class(es) the model was TAUGHT --"
          % sum(1 for n in names if shown.get(n)))
    print("  top-1 accuracy      %8.1f%%   on %d crops"
          % (right[taught].mean() * 100, int(taught.sum())))
    print("  top-3 accuracy      %8.1f%%" % (top3[taught].mean() * 100))
    print("  macro F1            %8.1f%%   over %d scored class(es)"
          % (tmacro * 100, len(tscored)))
    worst = min(tscored, key=lambda r: r[4])
    print("  worst class F1      %8.1f%%   (%s)" % (worst[4] * 100, worst[0]))

    print("\n  -- including class(es) with NO training crop --")
    print("  top-1 accuracy      %8.1f%%   on all %d crops"
          % (right.mean() * 100, len(y)))
    print("  top-3 accuracy      %8.1f%%" % (top3.mean() * 100))
    print("  macro F1            %8.1f%%   over the %d class(es) with "
          "held-out crops" % (macro * 100, len(scored)))
    held_untaught = sorted(n for n in untaught if any(r[0] == n and r[1] for r in rows))
    if held_untaught:
        print("  scored at 0%% by arithmetic, never shown one example: %s"
              % ", ".join(held_untaught))
    if absent:
        print("  %d class(es) have no held-out crop and are not averaged in: %s"
              % (len(absent), ", ".join(absent)))
    print("  latency          %11.2f ms/crop  (%.1f ms for a 46-tsum board)"
          % (ms, ms * 46))

    off = Counter()
    for t, p in zip(y, pred):
        if t != p:
            off[(names[t], names[p])] += 1
    if off:
        print("\n  most confused pairs (true -> predicted):")
        for (a, b), c in off.most_common(5):
            print("    %s -> %s   %d" % (a, b, c))

    print("\n  confidence   median %.3f   correct %.3f   %s"
          % (float(np.median(conf)), float(np.median(conf[right])),
             ("wrong %.3f" % float(np.median(conf[~right]))
              if (~right).any() else "wrong --")))
    print("\n  A reject floor only helps if wrong crops are less confident\n"
          "  than right ones. Coverage is the share it is willing to name:")
    print("\n  and accuracy WITHIN each confidence band, which the coverage\n"
          "  curve below hides -- a band no better than the one under it means\n"
          "  the score is not a probability of being right:")
    print("\n%14s%9s%11s" % ("confidence", "n", "accuracy"))
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9),
                   (0.9, 0.99), (0.99, 1.01)):
        m = (conf >= lo) & (conf < hi)
        print("%14s%9d%10s" % ("%.2f-%.2f" % (lo, hi), int(m.sum()),
                               "%.1f%%" % (100 * right[m].mean()) if m.any()
                               else "--"))

    print("\n%9s%11s%11s" % ("floor", "coverage", "accuracy"))
    for floor in (0.0, 0.50, 0.70, 0.85, 0.90, 0.95, 0.99):
        take = conf >= floor
        cov = float(take.mean())
        acc = float(right[take].mean()) if take.any() else float("nan")
        print("%9.2f%10.1f%%%10.1f%%%s"
              % (floor, cov * 100, acc * 100,
                 "   <- shipped" if abs(floor - 0.85) < 1e-9 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
