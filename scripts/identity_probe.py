"""Is there a learnable identity signal in the face crops, beyond colour?

Every round from the tenth on has circled one number: rebuild a chain from the
tsums the game itself marked and it comes back **29% shorter** using the bot's
own `kind` ids, **11% longer** using the game's word on identity. Same tsums,
same graph, same frame -- the whole spread is who is believed about colour.
`docs/DATASET-FINDINGS.md` then priced colour as an identity signal and found
it exhausted: the same character spreads 45.6 Lab *within one frame*, and a
game-confirmed same-character set is only 1.18x more colour-coherent than a
random group off the same board.

That closes colour. It does not open anything. This script asks the next
question, and deliberately in the cheapest possible form:

    over pairs the GAME has labelled, does any descriptor computable from the
    face crop separate same-character from different-character better than the
    colour the pipeline already uses -- and does a metric LEARNED from those
    pairs beat the same descriptor unlearned?

Nothing here plays, ships, or changes a round. It reads a collected corpus and
prints AUCs. If the learned metric does not clear the colour baseline on
held-out SESSIONS, the classifier idea is dead for the cost of an afternoon
rather than the cost of a capture mode, a labelling UI and a training loop --
which is the order `docs/IMPROVEMENT-LOOP.md` step 7 asks for.

The labels are free, and they are not the bot's own opinion
-----------------------------------------------------------
Holding a tsum makes the game light up every tsum that is the same character
*and* reachable from it. So per sample, `head` and everything in `marked` are
the same character, stated by the game. That is what makes this corpus
trainable without anyone labelling a single crop -- and what keeps it clear of
the trap of training a model on its own predictions.

Two label-hygiene rules, both measured elsewhere in this project:

* **positives skip the glow.** `hold_aura` (90px) is the radius the highlight
  washes over, and 18% of marks sit inside it, where a reaction means
  proximity rather than identity. Those are dropped rather than trusted.
* **negatives are noisy, and that is stated rather than hidden.** An unmarked
  tsum can be the same character and merely unreachable, so "unmarked" is not
  a clean negative. Every AUC here is therefore a LOWER BOUND. It stays a fair
  comparison, because every descriptor is scored on the identical pairs.

Run:

    python scripts/identity_probe.py --dir dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------
# descriptors -- cv2 + numpy only, so anything that wins here can ship without
# a new runtime dependency or a change to the frozen build.
# --------------------------------------------------------------------------
def _crops(bgr: np.ndarray, tsums: list, radius: float, box: int = 24,
           window: float = 1.5, mask: bool = False):
    """One fixed-size Lab crop and one grayscale crop per tsum.

    Sampled over `window` radii rather than the 0.45r inner disk `_face_lab`
    uses. That window is deliberately tight -- it exists to read ONE colour off
    a face while dodging the outline -- and a window that tight cannot carry
    shape, which is the whole point of asking this question. 1.5r takes the
    face, the outline and the ears, and the ears are most of what separates two
    characters that share a face colour.

    The cost of the wider window is the reason for `mask`. Tsums sit in a pile
    at ~2.44r centre to centre, so a 1.5r box is part tsum and part whatever
    happens to be lying next to it *on this frame* -- and the neighbours change
    every frame while the character does not. `mask` zeroes everything outside
    the tsum's own disk, which is the difference between describing a character
    and describing a character's current company.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    half = max(4, int(round(radius * window)))

    disk = None
    if mask:
        yy, xx = np.ogrid[:box, :box]
        c = (box - 1) / 2.0
        # The disk in crop coordinates: the face fills `radius` of a `half`
        # half-width, so its share of the box is radius/half.
        disk = ((yy - c) ** 2 + (xx - c) ** 2) <= (box / 2.0 * (radius / half)) ** 2

    labs = np.zeros((len(tsums), box, box, 3), np.float32)
    grays = np.zeros((len(tsums), box, box), np.float32)
    for i, t in enumerate(tsums):
        cx, cy = int(round(t["x"])), int(round(t["y"]))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, cx + half + 1), min(h, cy + half + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        labs[i] = cv2.resize(lab[y0:y1, x0:x1], (box, box), interpolation=cv2.INTER_AREA)
        grays[i] = cv2.resize(gray[y0:y1, x0:x1], (box, box), interpolation=cv2.INTER_AREA)
        if disk is not None:
            labs[i][~disk] = 0.0
            grays[i][~disk] = 0.0
    return labs, grays


def _feat_lab3(labs: np.ndarray) -> np.ndarray:
    """Today's signal: one median Lab colour per face, inner disk only.

    Reproduced here rather than imported so the baseline is scored on exactly
    the same crops as everything else -- a candidate that finds a signal on a
    different window has not beaten anything.
    """
    box = labs.shape[1]
    yy, xx = np.ogrid[:box, :box]
    c = (box - 1) / 2.0
    disk = ((yy - c) ** 2 + (xx - c) ** 2) <= (box * 0.30) ** 2
    return np.median(labs[:, disk, :], axis=1)


def _feat_patch(labs: np.ndarray, cells: int = 8) -> np.ndarray:
    """The face as a small picture rather than as one colour.

    Downsampled to cells x cells x 3: the colour baseline plus layout. Worth
    trying first because it is nearly free.
    """
    n = labs.shape[0]
    out = np.stack([cv2.resize(labs[i], (cells, cells), interpolation=cv2.INTER_AREA)
                    for i in range(n)])
    return out.reshape(n, -1)


def _feat_shape(grays: np.ndarray, cells: int = 4, bins: int = 8) -> np.ndarray:
    """Gradient-orientation histograms -- a HOG, by hand.

    Colour-free on purpose. This is the signal `docs/COMMANDS.md` names when it
    says separating characters "needs a signal colour does not carry -- shape,
    or a small learned classifier over the face crop", and the point of the
    probe is whether that sentence is true or merely plausible.

    Written out rather than called: OpenCV 5.0 dropped `cv2.HOGDescriptor`, so
    the classic route is gone, and this is nine lines of numpy anyway.
    """
    n, box, _ = grays.shape
    gx = np.gradient(grays, axis=2)
    gy = np.gradient(grays, axis=1)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (bins / (2 * np.pi))
    idx = np.clip(ang.astype(int), 0, bins - 1)

    step = box // cells
    out = np.zeros((n, cells, cells, bins), np.float32)
    for cy in range(cells):
        for cx in range(cells):
            m = mag[:, cy * step:(cy + 1) * step, cx * step:(cx + 1) * step]
            b = idx[:, cy * step:(cy + 1) * step, cx * step:(cx + 1) * step]
            for k in range(bins):
                out[:, cy, cx, k] = (m * (b == k)).sum(axis=(1, 2))
    # Per-cell L2, so a bright tsum and a shadowed one of the same character
    # describe the same shape. Without it the descriptor is mostly contrast.
    out /= (np.linalg.norm(out, axis=3, keepdims=True) + 1e-6)
    return out.reshape(n, -1)


# --------------------------------------------------------------------------
# pairs
# --------------------------------------------------------------------------
def _pairs(row: dict, aura: float, neg_max: float, min_vis: float = 0.0):
    """(positive, negative) index lists for one sample, per the hygiene rules."""
    tsums = row["tsums"]
    head = row["head"]
    if head >= len(tsums):
        return [], []
    marked = set(row.get("marked") or [])
    if not marked:
        return [], []
    rad = float(row.get("radius") or 1.0)
    if min_vis > 0 and tsums[head]["r"] / rad < min_vis:
        return [], []
    hx, hy = tsums[head]["x"], tsums[head]["y"]

    pos, neg = [], []
    for i, t in enumerate(tsums):
        if i == head:
            continue
        if min_vis > 0 and t["r"] / rad < min_vis:
            continue
        d = float(np.hypot(t["x"] - hx, t["y"] - hy))
        if d <= aura:
            continue          # inside the glow: a reaction means proximity
        if i in marked:
            pos.append(i)
        elif d <= neg_max:
            neg.append(i)     # near enough that reach is unlikely to be why
    return pos, neg


def _auc(same: np.ndarray, diff: np.ndarray) -> float:
    """P(a same-character pair scores closer than a different-character pair).

    Rank-based, so it needs no threshold and cannot be flattered by one -- the
    mistake `purity` made, where a default of 35 sat above the largest distance
    in the corpus and therefore dropped nothing.
    """
    if not len(same) or not len(diff):
        return float("nan")
    allv = np.concatenate([same, diff])
    order = allv.argsort()
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]      # ties share the average rank
    n1, n2 = len(same), len(diff)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return 1.0 - u / (n1 * n2)        # smaller distance = same, so invert


# --------------------------------------------------------------------------
# the learned part
# --------------------------------------------------------------------------
def _learn_metric(dp: np.ndarray, dn: np.ndarray, keep: int, reg: float = 1e-2):
    """A linear metric fitted from labelled PAIRS, closed-form.

    Difference vectors of same-character pairs should be small and those of
    different-character pairs large, so the projection wanted is the one
    maximising the ratio of their scatters -- a generalised eigenproblem, the
    same shape as LDA, solved once with no training loop, no learning rate and
    no framework. Live it is a matmul over ~46 crops, which is why it is the
    first rung: if this works there is nothing to bundle into the .exe.

    Regularised because the positive scatter is near-singular in the directions
    where a character genuinely never varies.
    """
    sp = dp.T @ dp / max(1, len(dp))
    sn = dn.T @ dn / max(1, len(dn))
    sp = sp + reg * np.trace(sp) / sp.shape[0] * np.eye(sp.shape[0])
    w, v = np.linalg.eigh(sp)
    w = np.maximum(w, 1e-9)
    white = v @ np.diag(w ** -0.5) @ v.T
    ew, ev = np.linalg.eigh(white @ sn @ white)
    order = ew.argsort()[::-1][:keep]
    return white @ ev[:, order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dataset")
    ap.add_argument("--aura", type=float, default=90.0,
                    help="glow radius; marks inside it are not used as positives")
    ap.add_argument("--neg-max", type=float, default=260.0,
                    help="unmarked tsums beyond this are not used as negatives")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="share of SESSIONS held out; sessions are the unit")
    ap.add_argument("--keep", type=int, default=24, help="learned metric dimensions")
    ap.add_argument("--limit", type=int, default=0, help="stop after N samples")
    ap.add_argument("--window", type=float, default=1.5,
                    help="crop half-width in radii")
    ap.add_argument("--no-fever", action="store_true",
                    help="skip FEVER frames -- the board is repainted neon and "
                         "animated, which is label noise on 39% of drags")
    ap.add_argument("--min-vis", type=float, default=0.0,
                    help="only use tsums whose detected blob is at least this "
                         "fraction of the full radius, i.e. the least buried. "
                         "The decisive test: if a signal exists but occlusion "
                         "is burying it, it shows up here and nowhere else")
    ap.add_argument("--mask", action="store_true",
                    help="zero everything outside the tsum's own disk, so the "
                         "descriptor cannot describe its neighbours instead")
    args = ap.parse_args()

    root = Path(args.dir)
    sessions = sorted(p for p in root.glob("*/samples.jsonl"))
    if not sessions:
        print(f"no sessions under {root}")
        return 1

    t0 = time.perf_counter()
    feats = {"lab3": [], "patch": [], "shape": []}
    pairs = []
    offset = 0
    n_samples = kinds_agree = kinds_total = skipped = 0
    negs_agree = negs_total = 0

    for si, sfile in enumerate(sessions):
        folder = sfile.parent
        for line in sfile.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.no_fever and row.get("fever"):
                skipped += 1
                continue
            pos, neg = _pairs(row, args.aura, args.neg_max, args.min_vis)
            if not pos or not neg:
                skipped += 1
                continue
            img = cv2.imread(str(folder / ("%04d_before.jpg" % row["index"])))
            if img is None:
                skipped += 1
                continue

            tsums = row["tsums"]
            labs, grays = _crops(img, tsums, float(row["radius"]),
                                 window=args.window, mask=args.mask)
            feats["lab3"].append(_feat_lab3(labs))
            feats["patch"].append(_feat_patch(labs))
            feats["shape"].append(_feat_shape(grays))

            head = row["head"]
            # The baseline the project already runs on: how often does the
            # bot's per-frame cluster id agree with the game about a partner
            # the game just named?
            for i in pos:
                kinds_total += 1
                kinds_agree += int(tsums[i]["kind"] == tsums[head]["kind"])
                pairs.append((si, offset, head, i, 1))
            for i in neg:
                negs_total += 1
                negs_agree += int(tsums[i]["kind"] == tsums[head]["kind"])
                pairs.append((si, offset, head, i, 0))

            offset += len(tsums)
            n_samples += 1
            if args.limit and n_samples >= args.limit:
                break
        if args.limit and n_samples >= args.limit:
            break

    if not pairs:
        print("no usable pairs -- is this a schema 1 corpus?")
        return 1

    for k in list(feats):
        feats[k] = np.concatenate(feats[k]).astype(np.float32)
    pairs_arr = np.array(pairs, np.int64)
    cut = max(1, int(len(sessions) * (1 - args.holdout)))
    train = pairs_arr[pairs_arr[:, 0] < cut]
    test = pairs_arr[pairs_arr[:, 0] >= cut]

    print("corpus      %d samples over %d sessions (%d unusable), read in %.0fs"
          % (n_samples, len(sessions), skipped, time.perf_counter() - t0))
    print("pairs       %d total: %d same-character (the game's word), %d different"
          % (len(pairs_arr), int(pairs_arr[:, 4].sum()),
             int((1 - pairs_arr[:, 4]).sum())))
    print("holdout     train %d sessions / test %d sessions (%d / %d pairs)"
          % (cut, len(sessions) - cut, len(train), len(test)))
    print()
    # Today's identity, scored as a classifier on exactly these pairs: what
    # share of real partners does `kind` link, and what share of non-partners
    # does it link anyway. A recall this low with a false rate this high is the
    # -29% rebuild in `docs/DATASET-FINDINGS.md` seen one drag at a time.
    recall = 100 * kinds_agree / max(1, kinds_total)
    false = 100 * negs_agree / max(1, negs_total)
    print("BASELINE    today's `kind` links %.1f%% of the game's confirmed "
          "partners (%d/%d)" % (recall, kinds_agree, kinds_total))
    print("            ...and links %.1f%% of non-partners anyway (%d/%d)"
          % (false, negs_agree, negs_total))
    print()
    print("AUC = P(a same-character pair scores closer than a different one). "
          "0.50 knows nothing.")
    print("%-28s%6s%9s%9s" % ("descriptor", "dim", "train", "TEST"))

    def dist(f, idx):
        a = f[idx[:, 1] + idx[:, 2]]
        b = f[idx[:, 1] + idx[:, 3]]
        return np.linalg.norm(a - b, axis=1)

    tp, tn = train[train[:, 4] == 1], train[train[:, 4] == 0]
    sp, sn = test[test[:, 4] == 1], test[test[:, 4] == 0]

    results = {}
    combo = np.hstack([feats["patch"], feats["shape"] * 40.0])
    for name, f in (("lab3", feats["lab3"]), ("patch", feats["patch"]),
                    ("shape", feats["shape"]), ("patch+shape", combo)):
        tr = _auc(dist(f, tp), dist(f, tn))
        te = _auc(dist(f, sp), dist(f, sn))
        results[name] = te
        print("%-28s%6d%9.3f%9.3f" % (name + " (raw)", f.shape[1], tr, te))

    print()
    for name, f in (("patch", feats["patch"]), ("shape", feats["shape"]),
                    ("patch+shape", combo)):
        dp = f[tp[:, 1] + tp[:, 2]] - f[tp[:, 1] + tp[:, 3]]
        dn = f[tn[:, 1] + tn[:, 2]] - f[tn[:, 1] + tn[:, 3]]
        mu, sd = f.mean(0), f.std(0) + 1e-6
        proj = _learn_metric(dp / sd, dn / sd, args.keep)
        g = ((f - mu) / sd) @ proj
        tr = _auc(dist(g, tp), dist(g, tn))
        te = _auc(dist(g, sp), dist(g, sn))
        print("%-28s%6d%9.3f%9.3f   %+.3f vs raw"
              % (name + " (LEARNED)", g.shape[1], tr, te, te - results[name]))

    print()
    print("Read it the way step 6 of docs/IMPROVEMENT-LOOP.md asks: the TEST "
          "column is the only one that means anything, negatives are noisy so "
          "every number here is a floor, and a learned metric that does not "
          "clear its own raw descriptor on held-out sessions has learned the "
          "corpus rather than the characters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
