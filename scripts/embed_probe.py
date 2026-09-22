"""Sort the board into five piles using the LEARNED face, not the raw colour.

THE PLAYER'S ARGUMENT, and it is the right one:

> "It merges the wrong tsums. This is why recognition must be in place. I want
> the app to know only 5 kinds of tsum are in the game and recognise all of
> them correctly, not guessing, so it won't link the wrong tsum. Each tsum I
> labelled has its face pattern with colour -- just train the app to remember
> those."

Three findings sit behind it, and they agree:

* per-frame k-means makes 7 or 8 groups on a board holding 5 characters, so
  the average character is split across about one and a half groups
  (`group_probe.py`);
* forcing k-means to 4 or 5 groups does not fix it -- it merges the WRONG
  tsums, because the feature is one median Lab colour per face and two
  characters can share a colour (`group_probe.py`);
* the classifier is accurate where it is asked and is asked about a fifth of
  the board, so naming cannot carry the grouping either (`naming_gain.py`).

WHAT THIS TRIES. Not classification -- clustering, in the space the classifier
LEARNED. `models/character.onnx` is MobileNetV3-Small, and its penultimate
layer `/classifier/classifier.1/HardSwish` is a 1024-d description of a tsum
face trained to separate characters. That is exactly "the face pattern with
colour", and using it as the k-means feature instead of median Lab asks the
question the player is asking.

WHY IT COULD WORK WHERE NAMING DOES NOT. Naming needs a confident answer about
one crop against sixty classes, and refuses below 0.85 -- which is why it
reaches a fifth of the board. Clustering needs only that two crops of the same
character sit closer together than two crops of different ones. No floor, no
vocabulary, no per-character labels, and it works on a character nobody has
ever labelled and on a base tsum the classifier has never seen.

    python scripts/embed_probe.py
    python scripts/embed_probe.py --features lab,prob,embed --groups 4,5,6

SCORED THE WAY `group_probe.py` SCORES: the chain the round could actually drag
from the pressed tsum, against the game's own marks, and reported as the RUN of
correct members before the first wrong one -- because the game refuses 86% of
what follows a refusal, so a chain is worth its leading run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ttheart_sender.game import tsum as T  # noqa: E402

_EMBED = None

#: MobileNetV3-Small's classifier is Linear(576, 1024) -> HardSwish ->
#: Dropout -> Linear(1024, classes). The HardSwish output is the description
#: of the face; the layer after it is already an opinion about which of sixty
#: characters it is, which is the thing that does not generalise.
#:
#: Read with TORCH, not cv2.dnn: OpenCV 5's engine wants a tensor name in
#: `forward()` and `getLayerNames()` returns layer names, so there is no
#: supported way to pull an intermediate blob out of the shipped ONNX. The same
#: weights are in the .pt beside it. If this earns a round, the ONNX export
#: gains a second output -- the player does not gain a torch dependency.
EMBED_CHECKPOINT = "models/character_candidate.pt"


class _Embedder:
    """The trained net, cut off before it commits to a character."""

    def __init__(self, path):
        import torch
        from torchvision import models as tvm

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        net = tvm.mobilenet_v3_small(weights=None)
        n = len(ck["classes"])
        net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, n)
        net.load_state_dict(ck["state"])
        net.eval()
        # Everything up to and including the HardSwish, and nothing after.
        self.body = net
        self.classes = ck["classes"]

    def __call__(self, blob):
        """blob is NCHW, already normalised. Returns (N, 1024)."""
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(blob)
            x = self.body.features(x)
            x = self.body.avgpool(x)
            x = t.flatten(x, 1)
            x = self.body.classifier[0](x)
            x = self.body.classifier[1](x)      # HardSwish
            return x.numpy()


def prototypes(embed, crops: Path, model, limit_per_class: int = 60):
    """One remembered face per labelled character.

    The MEAN embedding of that character's crops, L2-normalised. A mean is the
    right summary here for the same reason a median colour is in `_face_lab`:
    the crops differ mostly by how much of the tsum is showing, and that
    averages out, while what makes the character itself does not.
    """
    out, names = [], []
    for d in sorted(crops.iterdir()) if crops.is_dir() else []:
        if not d.is_dir() or d.name in ("board", "score", "junk"):
            continue
        rows = []
        for p in sorted(d.glob("*.png"))[:limit_per_class]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (model.size, model.size),
                             interpolation=cv2.INTER_AREA)
            rows.append(img)
        if len(rows) < 5:
            continue
        blob = np.stack(rows).astype(np.float32) / 255.0
        blob = (blob - model.mean) / model.std
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), np.float32)
        vec = embed(blob)
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6)
        mean = vec.mean(axis=0)
        out.append(mean / max(np.linalg.norm(mean), 1e-6))
        names.append(d.name)
    return np.asarray(out, np.float32), names


def assign(feats, ok, protos, allowed, before, face_lab=None):
    """Give every tsum the nearest remembered face, and group by that.

    `allowed` restricts the choice to the round's own cast. A tsum whose crop
    could not be cut is placed by face colour against the piles the others
    built -- the same fallback `cluster` uses, and for the same reason: leaving
    it in its old group is what turned five piles into eleven.
    """
    idx = np.flatnonzero(ok)
    out = [0] * len(before)
    if not len(idx):
        return list(before)
    sub_p = protos[allowed]
    # Cosine, both sides unit length, so this is a dot product.
    pick = np.argmax(feats[idx] @ sub_p.T, axis=1)
    for i, p_ in zip(idx, pick):
        out[i] = int(p_)
    missing = np.flatnonzero(~ok)
    if not len(missing) or face_lab is None:
        for i in missing:
            out[i] = len(allowed)
        return out
    centres = np.zeros((len(allowed), face_lab.shape[1]), np.float32)
    for g in range(len(allowed)):
        members = idx[pick == g]
        if len(members):
            centres[g] = np.median(face_lab[members], axis=0)
    for i in missing:
        out[i] = int(np.argmin(((centres - face_lab[i]) ** 2).sum(axis=1)))
    return out


def _probe():
    spec = importlib.util.spec_from_file_location(
        "group_probe", HERE / "group_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boards(root: Path, want: int, seed: int):
    rows = []
    for jl in sorted(root.glob("*/samples.jsonl")):
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            o = d.get("options") or {}
            if o.get("character") or o.get("recolour") or o.get("kinds"):
                continue
            head = d.get("head")
            if (not d.get("tsums") or not d.get("marked")
                    or head is None or not (0 <= head < len(d["tsums"]))):
                continue
            img = jl.parent / ("%04d_before.jpg" % d.get("index", -1))
            if img.exists():
                rows.append((d, img))
    random.Random(seed).shuffle(rows)
    return rows[:want] if want else rows


def features(model, bgr, tsums, radius, kind: str):
    """One row per tsum, and a mask saying which rows are real.

    EVERY tsum gets a row. A crop the model cannot cut -- at the frame edge --
    has no description, and rather than drop it (which would renumber the board
    and break the marks) it is marked unusable and left in the group it already
    had. That is the same hybrid rule naming follows.
    """
    n = len(tsums)
    if kind == "lab":
        return T._face_lab(bgr, tsums, radius), np.ones(n, bool)

    crops = [model._crop(bgr, t, radius) for t in tsums]
    ok = np.array([c is not None for c in crops], bool)
    if not ok.any():
        return np.zeros((n, 1), np.float32), ok
    blob = np.stack([c for c in crops if c is not None]).astype(np.float32) / 255.0
    blob = (blob - model.mean) / model.std
    blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)
    if kind == "embed":
        out = _EMBED(blob)
    else:
        model.net.setInput(blob)
        out = model.net.forward()
    out = np.asarray(out, np.float32).reshape(len(blob), -1)
    if kind == "prob":
        out = out - out.max(axis=1, keepdims=True)
        out = np.exp(out)
        out /= out.sum(axis=1, keepdims=True)
    else:
        # L2, because k-means minimises squared distance and an unnormalised
        # embedding lets overall brightness dominate the direction that
        # actually carries identity.
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norm, 1e-6)
    feats = np.zeros((n, out.shape[1]), np.float32)
    feats[ok] = out
    return feats, ok


def cluster(feats, ok, groups: int, before, face_lab=None):
    """Sort the board into exactly `groups` piles, and leave nothing out.

    EVERY tsum gets one of the `groups` ids. The first version of this left a
    tsum whose crop could not be cut in the group it already had -- the hybrid
    rule naming uses -- and here that is exactly wrong: about 40% of a board is
    refused at the frame edge, so 40% kept up to eight old ids while five new
    ones were added and the board came out in ELEVEN groups. More fragmented
    than the rule being replaced, which is the failure this is meant to fix.

    A refused tsum is assigned by FACE COLOUR to the nearest pile the embedding
    built. Colour is a weak identity signal alone -- that is the finding that
    started this -- but "which of these five piles is this nearest" is a much
    easier question than "which of sixty characters is this", and it keeps the
    count where the game says it should be.
    """
    idx = np.flatnonzero(ok)
    out = [0] * len(before)
    if len(idx) <= groups:
        return list(before)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(np.ascontiguousarray(feats[idx], np.float32),
                              groups, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    for i, lab in zip(idx, labels):
        out[i] = int(lab)

    missing = np.flatnonzero(~ok)
    if not len(missing):
        return out
    if face_lab is None:
        # Nothing to place them by. Their own bucket, once, rather than one
        # each -- still better than eight leftovers from the old grouping.
        for i in missing:
            out[i] = groups
        return out
    centres = np.zeros((groups, face_lab.shape[1]), np.float32)
    for g in range(groups):
        members = idx[labels == g]
        if len(members):
            centres[g] = np.median(face_lab[members], axis=0)
    for i in missing:
        out[i] = int(np.argmin(((centres - face_lab[i]) ** 2).sum(axis=1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="4,5,6")
    ap.add_argument("--checkpoint", default=EMBED_CHECKPOINT,
                    help="the torch weights the ONNX was exported from")
    ap.add_argument("--features", default="lab,prob,embed",
                    help="lab = one median colour per face, what ships. "
                         "prob = the classifier's opinion. embed = the "
                         "penultimate 1024-d description of the face")
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--min-visible", type=float, default=0.0,
                    help="0 asks the model about the WHOLE board. Clustering "
                         "needs no confident answer, so the floor that naming "
                         "needs does not apply -- but it is here to measure")
    ap.add_argument("--link-px", type=float, default=105.0)
    ap.add_argument("--block", type=float, default=1.25)
    ap.add_argument("--max-chain", type=int, default=12)
    args = ap.parse_args()

    gp = _probe()
    model = T.load_character_model(args.model, 0.85, args.min_visible)
    global _EMBED
    protos, proto_names = None, []
    if any(f in args.features for f in ("embed", "proto")):
        _EMBED = _Embedder(args.checkpoint)
    if "proto" in args.features:
        protos, proto_names = prototypes(_EMBED, args.crops, model)
        print("remembered %d character face(s) from %s"
              % (len(proto_names), args.crops))
    kinds_want = [int(g) for g in args.groups.split(",") if g.strip()]
    feats_want = [f.strip() for f in args.features.split(",") if f.strip()]

    rows = boards(args.dir, args.samples, args.seed)
    if not rows:
        print("no drags with marks were collected under the shipped rule")
        return 2

    arms = [("k-means", None, None)] + [(f, f, g) for f in feats_want
                                        for g in kinds_want]
    # The prototype arms do not take a group count -- the count is however many
    # characters the round turns out to use.
    proto_arms = [f for f in feats_want if f.startswith("proto")]
    feats_want = [f for f in feats_want if not f.startswith("proto")]
    run = {a[0] if a[1] is None else (a[1], a[2]): [] for a in arms
           if a[1] is None or not a[1].startswith("proto")}
    for f in proto_arms:
        run[(f, 0)] = []
    rec = {k: [] for k in run}
    prec = {k: [] for k in run}
    off = {k: [] for k in run}
    n = 0

    for d, img_path in rows:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        radius = float(d["radius"])
        head = int(d["head"])
        raw = d["tsums"]
        truth = {int(i) for i in d["marked"]
                 if 0 <= int(i) < len(raw) and int(i) != head}
        if not truth:
            continue
        n += 1
        before = [int(t["kind"]) for t in raw]
        _ts = [T.Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                      kind=0, colour=(0, 0, 0)) for t in raw]
        lab_for_fallback = T._face_lab(frame, _ts, radius)
        made = {}
        for f in feats_want:
            ts = [T.Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                         kind=0, colour=(0, 0, 0)) for t in raw]
            made[f] = features(model, frame, ts, radius, f)

        arms_now = [("k-means", before)]
        arms_now += [((f, g), cluster(made[f][0], made[f][1], g,
                                      before, lab_for_fallback))
                     for f in feats_want for g in kinds_want]
        for f in proto_arms:
            fe, okm = features(model, frame, _ts, radius, "embed")
            if f == "proto":
                allowed = np.arange(len(proto_names))
            else:
                # The round's own cast: the five prototypes this board's tsums
                # most often sit nearest to. Voted on the board rather than
                # given, so nothing here knows anything a round would not.
                votes = np.argmax(fe[okm] @ protos.T, axis=1) if okm.any() else []
                top = [c for c, _ in Counter(votes.tolist()).most_common(5)]
                allowed = np.array(top or [0])
            arms_now.append(((f, 0),
                             assign(fe, okm, protos, allowed, before,
                                    lab_for_fallback)))

        for key, kinds in arms_now:
            ts = [T.Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                         kind=0, colour=(0, 0, 0)) for t in raw]
            order = gp.reachable(ts, kinds, head, radius, args.link_px,
                                 args.block, args.max_chain)
            same = set(order)
            hit = len(same & truth)
            rec[key].append(hit / len(truth))
            prec[key].append(hit / len(same) if same else 1.0)
            off[key].append(len(same))
            good = 0
            for i in order:
                if i not in truth:
                    break
                good += 1
            run[key].append(good)

    if not n:
        print("no scorable drags")
        return 2

    print("%d drags with marks. Grouping feature vs the game's own marks," % n)
    print("scored on the chain the round could drag, capped at %d."
          % args.max_chain)
    print("model asked about tsums at or above %.2f visible."
          % args.min_visible)
    print()
    print("  feature        groups   recall   precision   offered   RUN")
    base = st.mean(run["k-means"])
    for key in run:
        if not run[key]:
            continue
        label = ("median face colour (ships)" if key == "k-means"
                 else {"lab": "median face colour",
                       "prob": "classifier opinion",
                       "embed": "learned face (1024-d)",
                       "proto": "nearest remembered face",
                       "proto5": "nearest of the round's 5"}[key[0]])
        groups = "" if key == "k-means" or key[1] == 0 else str(key[1])
        print("  %-26s %4s   %5.1f%%      %5.1f%%   %6.2f   %5.2f%s"
              % (label, groups, 100 * st.mean(rec[key]),
                 100 * st.mean(prec[key]), st.mean(off[key]),
                 st.mean(run[key]),
                 "" if key == "k-means"
                 else "  %+.0f%%" % (100 * (st.mean(run[key]) / base - 1))))
    print()
    for key in run:
        if key == "k-means" or not run[key]:
            continue
        d_r = [a - b for a, b in zip(run[key], run["k-means"])]
        better = sum(1 for v in d_r if v > 1e-9)
        worse = sum(1 for v in d_r if v < -1e-9)
        print("  %-8s %d groups: better on %d, worse on %d, unchanged on %d"
              % (key[0], key[1], better, worse, len(d_r) - better - worse))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
