"""Is the character model recognising the tsum, or its colour?

`classify.py` reports 97.9% on held-out SESSIONS, and that number is honest --
the split is by session, so it is not near-duplicates leaking across the cut.
But a high score says nothing about WHICH evidence earned it, and this project
has one specific reason to doubt: `classify.py:augment` perturbs geometry
(flip, +/-20 deg, +/-10% scale) and a single global brightness gain, and
NOTHING ELSE. Hue and saturation are never touched, so across the whole
training set a character's colour is a perfectly stable cue -- the cheapest
one available, and the one a network will therefore take.

This measures the size of that dependence WITHOUT retraining, by serving the
SHIPPED model its own held-out crops under perturbations that leave the object
intact and move the colour:

    grey        saturation -> 0, luminance preserved
    desat       saturation halved
    hue+N       hue rotated, saturation and value untouched
    bright/dark value scaled (the one thing training already does)
    contrast    contrast scaled about mid-grey
    rot/shift/scale  geometry, as a control

A model that reads the object survives the colour rows. A model that reads the
colour does not. The geometry rows are the control: they must stay high, or
the harness is broken rather than the model.

Read with `--per-class` to see which characters carry the dependence; the
plan's hypothesis is that same-silhouette, different-colour pairs fall first.

    .venv/Scripts/python scripts/colour_probe.py
    .venv/Scripts/python scripts/colour_probe.py --per-class
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import NAME, NOT_CHARACTERS  # noqa: E402


def load_split(root: Path, test_sessions, classes):
    """The shipped model's OWN held-out crops, by the sessions its json names."""
    keep = {c: i for i, c in enumerate(classes)}
    paths, ys = [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in NOT_CHARACTERS or d.name not in keep:
            continue
        for p in sorted(d.glob("*.png")):
            m = NAME.match(p.stem)
            sess = m.group("sess") if m else "unknown"
            if sess in test_sessions:
                paths.append(p)
                ys.append(keep[d.name])
    return paths, np.asarray(ys, int)


# ---- perturbations. Each takes and returns uint8 BGR, and must not move
# ---- pixels except where that is the point.

def _hsv(img, dh=0.0, ds=1.0, dv=1.0):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32))
    h = (h + dh / 2.0) % 180.0           # OpenCV hue is 0..179 for 0..359 deg
    s = np.clip(s * ds, 0, 255)
    v = np.clip(v * dv, 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _grey(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _contrast(img, k):
    return np.clip((img.astype(np.float32) - 128.0) * k + 128.0,
                   0, 255).astype(np.uint8)


def _affine(img, ang=0.0, dx=0.0, dy=0.0, sc=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
    M[0, 2] += dx * w
    M[1, 2] += dy * h
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _disc(img, keep_inside: bool, frac: float = 0.80):
    """Blank everything outside (or inside) the centred tsum disc.

    The crop rule puts the tsum at the centre at 1.0 detected radii, so the
    inscribed disc IS the tsum and the corners are its neighbours and the
    board behind it. Two questions, one mask each:

        keep_inside=True    only the tsum survives -- can the model still
                            name it with every contextual cue gone?
        keep_inside=False   only the surround survives -- can the model name
                            it from the neighbours ALONE? A score here much
                            above chance is background leakage, and it means
                            some of the headline accuracy is being earned by
                            what happens to be lying next to a character
                            rather than by the character.

    Filled with the crop's own edge mean rather than black, because a flat
    black field is itself a strong learnable cue and would answer a different
    question than the one asked.
    """
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r = frac * min(h, w) / 2.0
    inside = (yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2 <= r * r
    mask = inside if keep_inside else ~inside
    fill = np.array([img[0].mean(0), img[-1].mean(0)]).mean(0)
    out = np.empty_like(img)
    out[:] = fill.astype(img.dtype)
    out[mask] = img[mask]
    return out


VARIANTS = [
    ("normal",        lambda im: im,                          "baseline"),
    ("tsum only",     lambda im: _disc(im, True),             "context"),
    ("surround only", lambda im: _disc(im, False),            "context"),
    ("grey",          _grey,                                  "colour"),
    ("desat 50%",     lambda im: _hsv(im, ds=0.5),            "colour"),
    ("hue +30",       lambda im: _hsv(im, dh=30),             "colour"),
    ("hue +60",       lambda im: _hsv(im, dh=60),             "colour"),
    ("hue +120",      lambda im: _hsv(im, dh=120),            "colour"),
    ("hue +180",      lambda im: _hsv(im, dh=180),            "colour"),
    ("bright x1.3",   lambda im: _hsv(im, dv=1.3),            "light"),
    ("dark x0.7",     lambda im: _hsv(im, dv=0.7),            "light"),
    ("contrast x1.4", lambda im: _contrast(im, 1.4),          "light"),
    ("contrast x0.6", lambda im: _contrast(im, 0.6),          "light"),
    ("rot 8deg",      lambda im: _affine(im, ang=8),          "geometry"),
    ("shift 8%",      lambda im: _affine(im, dx=.08, dy=.08), "geometry"),
    ("scale 1.15",    lambda im: _affine(im, sc=1.15),        "geometry"),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--dir", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("scratchpad/recog/colour_probe.json"))
    args = ap.parse_args()

    meta = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    classes = meta["classes"]
    size = int(meta.get("size", 96))
    mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
    test_sessions = set(meta.get("test_sessions") or [])
    if not test_sessions:
        print("this model's json names no test_sessions -- the probe must not "
              "score a model on crops it trained on.")
        return 1

    paths, y = load_split(args.dir, test_sessions, classes)
    if not len(paths):
        print("no held-out crops found for this model's sessions")
        return 1
    print("%s: %d classes, %d held-out crops from %d sessions\n"
          % (args.model.name, len(classes), len(paths), len(test_sessions)))

    net = cv2.dnn.readNetFromONNX(str(args.model))
    raw = [cv2.imread(str(p)) for p in paths]

    def run(fn):
        pred = np.zeros(len(raw), int)
        for i in range(0, len(raw), 256):
            chunk = raw[i:i + 256]
            batch = np.stack([cv2.resize(fn(im), (size, size),
                                         interpolation=cv2.INTER_LINEAR)
                              for im in chunk]).astype(np.float32) / 255.0
            batch = (batch - mean) / std
            net.setInput(np.ascontiguousarray(batch.transpose(0, 3, 1, 2),
                                              dtype=np.float32))
            pred[i:i + len(chunk)] = net.forward().argmax(axis=1)
        return pred

    base_pred, base_acc, rows = None, 0.0, []
    print("%14s%10s%10s%9s%8s" % ("variant", "kind", "accuracy", "vs base",
                                  "agree"))
    for name, fn, kind in VARIANTS:
        pred = run(fn)
        acc = float((pred == y).mean())
        if base_pred is None:
            base_pred, base_acc = pred, acc
        agree = float((pred == base_pred).mean())
        rows.append({"variant": name, "kind": kind, "accuracy": round(acc, 4),
                     "delta": round(acc - base_acc, 4),
                     "agree": round(agree, 4), "pred": pred.tolist()})
        print("%14s%10s%9.1f%%%+9.1f%%%8.1f%%"
              % (name, kind, 100 * acc, 100 * (acc - base_acc), 100 * agree))

    if args.per_class:
        colour = [r for r in rows if r["kind"] == "colour"]
        grey = next(r for r in rows if r["variant"] == "grey")
        print("\nper-class recall, baseline vs worst colour variant\n")
        print("%18s%6s%9s%9s%9s  %s"
              % ("class", "n", "normal", "grey", "worst", "variant"))
        out = []
        for ci, cname in enumerate(classes):
            sel = y == ci
            n = int(sel.sum())
            if not n:
                continue
            b = float((np.asarray(rows[0]["pred"])[sel] == ci).mean())
            g = float((np.asarray(grey["pred"])[sel] == ci).mean())
            worst, wv = 1.1, ""
            for r in colour:
                v = float((np.asarray(r["pred"])[sel] == ci).mean())
                if v < worst:
                    worst, wv = v, r["variant"]
            out.append((b - worst, cname, n, b, g, worst, wv))
        for _, cname, n, b, g, worst, wv in sorted(out, reverse=True):
            print("%18s%6d%9.0f%%%9.0f%%%9.0f%%  %s"
                  % (cname, n, 100 * b, 100 * g, 100 * worst, wv))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": str(args.model), "n": len(paths), "classes": classes,
         "y": y.tolist(), "rows": rows}, indent=1), encoding="utf-8")
    print("\n-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
