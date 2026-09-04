"""Score ways of deciding tsum identity against the game's own answer.

The bot has never had a way to check its identity guesses. It has one now, and
it has had it all along without anyone using it: every saved sample carries
``marked``, the tsums the game lit up when the bot pressed one. That is the
game saying *these are the same character as the one you are holding*. So any
proposal about identity can be scored offline, on 1,200 real boards, before it
is allowed anywhere near a round.

What is scored, on identical footing
------------------------------------

Only sessions played with the character model **off** are used, because there
the saved ``kind`` is still a pure k-means cluster. In a session where the
model ran, ``kind`` has already been overwritten for the crops it named and
the colour baseline cannot be recovered. Every method below is then computed
from the same saved frame, so the comparison is between methods and not
between rounds.

* ``kmeans``     -- the saved cluster id. What the bot does today.
* ``named``      -- the current model: rename where confident, otherwise keep
                    the cluster. Simulated here for rounds that never ran it.
* ``colour5``    -- face colour, forced into 5 groups. The control that
                    matters: if this alone closes the gap, the model is not
                    what was missing, the group *count* was.
* ``model5``     -- the model's 15-way output used as a fingerprint rather
                    than a name, forced into 5 groups.
* ``both5``      -- fingerprint and colour together, forced into 5 groups.

Why 5: a Tsum Tsum board holds at most five characters, four with an item.
That is a hard rule of the game, and it makes identity a closed problem --
sort the board into five piles -- rather than the open one of naming 40-odd
characters, most of which nobody has labelled. It also means a *wrong* name is
harmless as long as it is consistent. If every Beast is confidently called
"Grim" and no real Grim is on the board, every Beast still chains with every
other Beast. What costs a round is one character carrying two ids.

How it is scored
----------------

``agreement`` is the share of marked tsums given the same id as the pressed
one. On its own it is worthless -- putting every tsum in one group scores
100% -- so it is always reported beside ``base``, the share that would agree
by chance given the sizes of the groups that method actually made. ``lift`` is
the ratio, and it is the number to read: it is what the method knows that its
own group sizes do not already give it for free.

``split`` counts the ids handed to a single marked group. The game says those
tsums are one character; 1.0 means we agreed, and anything higher is a
character we have cut in half and will never chain across.

Run:  .venv/Scripts/python scripts/group_eval.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttheart_sender.game import tsum as T  # noqa: E402
from embed_net import cut as embed_cut  # noqa: E402

GROUPS = 5          # the game's own limit on characters per board
UNKNOWN = -1        # a crop the model could not read; never chains


# ------------------------------------------------------------------ grouping

def kgroups(feat: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Force `feat` into at most `k` groups. One id per row, always.

    Every row gets a group. That is the point of the exercise: the hybrid the
    bot runs today leaves half the board on colour ids and renames the other
    half, so one character routinely ends up as a name *and* two clusters. A
    method that refuses to answer for some rows reintroduces exactly that.
    """
    n = len(feat)
    if n == 0:
        return np.zeros(0, np.int32)
    if n <= k:
        return np.arange(n, dtype=np.int32)
    data = np.ascontiguousarray(feat, np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(data, k, None, crit, 3,
                              cv2.KMEANS_PP_CENTERS)
    return labels.ravel().astype(np.int32)


def grid_lab(bgr, tsums, radius, cells=3):
    """A `cells` x `cells` grid of Lab means over each tsum's face.

    `_face_lab` returns one median colour per tsum, which is the right summary
    for asking *what colour is this* and the wrong one for asking *which
    character is this*: many Tsum Tsum characters share a body colour and
    differ in where the other colours sit -- a muzzle, an ear, a patch. A grid
    keeps that layout at a cost of nothing, and unlike a learned descriptor it
    needs no labels, so it works on the characters nobody has trained.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    lab = lab.astype(np.float32)
    h, w = lab.shape[:2]
    half = max(2, int(round(radius * 0.72)))
    out = np.zeros((len(tsums), cells * cells * 3), np.float32)
    for i, t in enumerate(tsums):
        x0, y0 = max(0, int(t.x) - half), max(0, int(t.y) - half)
        x1, y1 = min(w, int(t.x) + half + 1), min(h, int(t.y) + half + 1)
        patch = lab[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        small = cv2.resize(patch, (cells, cells), interpolation=cv2.INTER_AREA)
        out[i] = small.reshape(-1)
    return out


class Embedder:
    """`models/embed.onnx` through `cv2.dnn`, one vector per tsum.

    Every tsum gets one: the crop is padded at the frame edge rather than
    refused, which is why the net was trained on padded crops. A method that
    leaves some of the board unidentified reintroduces the split that made
    `--character` a regression.
    """

    def __init__(self, path: Path):
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size = int(meta["size"])
        self.batch = int(meta["batch"])
        self.mean = np.asarray(meta["mean"], np.float32)
        self.std = np.asarray(meta["std"], np.float32)

    def embed(self, bgr, tsums, radius):
        crops = [embed_cut(bgr, t.x, t.y, radius) for t in tsums]
        keep = [i for i, c in enumerate(crops) if c is not None]
        if not keep:
            return None, []
        blob = np.stack([crops[i] for i in keep]).astype(np.float32) / 255.0
        blob = np.stack([cv2.resize(b, (self.size, self.size),
                                    interpolation=cv2.INTER_LINEAR) for b in blob])
        blob = (blob - self.mean) / self.std
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), np.float32)
        outs = []
        for i in range(0, len(blob), self.batch):
            chunk = blob[i:i + self.batch]
            n = len(chunk)
            if n < self.batch:      # FIXED batch -- cv2 5.0.0 sizes on the first
                pad = np.zeros((self.batch,) + chunk.shape[1:], np.float32)
                pad[:n] = chunk
                chunk = pad
            self.net.setInput(np.ascontiguousarray(chunk))
            outs.append(np.asarray(self.net.forward(), np.float32)[:n].copy())
        return np.concatenate(outs), keep


def methods(bgr, tsums, radius, saved_kind, model, embedder=None):
    """Every candidate's id-per-tsum for one board."""
    n = len(tsums)
    out = {"kmeans": np.asarray(saved_kind, np.int64)}

    lab = T._face_lab(bgr, tsums, radius)
    out["colour5"] = kgroups(lab, GROUPS)
    for k in (2, 3, 4, 6, 7):
        out[f"colour{k}"] = kgroups(lab, k)

    grid = grid_lab(bgr, tsums, radius)
    out["grid5"] = kgroups(grid, GROUPS)
    out["grid6"] = kgroups(grid, 6)

    idx, prob = model.probabilities(bgr, tsums, radius)
    # `named`: the production rule -- rename where sure, else keep the cluster.
    named = np.asarray(saved_kind, np.int64).copy()
    for i, p in zip(idx, prob):
        b = int(p.argmax())
        if float(p[b]) >= model.confidence:
            named[i] = T.CHARACTER_KIND + b
    out["named"] = named

    # Fingerprint methods. sqrt of the softmax: the raw vector is spiky enough
    # that two crops of one character agreeing on the top class but differing
    # in the tail sit far apart under a plain L2 distance.
    fp = np.full((n, prob.shape[1] if len(prob) else 1), np.nan, np.float32)
    for i, p in zip(idx, prob):
        fp[i] = np.sqrt(p)
    ok = ~np.isnan(fp[:, 0])
    for name, feat in (("model5", fp),
                       ("both5", np.hstack([fp, lab / 40.0]) if len(prob) else fp)):
        ids = np.full(n, UNKNOWN, np.int64)
        if ok.any():
            ids[ok] = kgroups(feat[ok], GROUPS)
        out[name] = ids

    if embedder is not None:
        emb, keep = embedder.embed(bgr, tsums, radius)
        if emb is not None:
            for k in (3, 4, 5):
                ids = np.full(n, UNKNOWN, np.int64)
                ids[keep] = kgroups(emb, k)
                out[f"embed{k}"] = ids
    return out, int((~ok).sum())


# ------------------------------------------------------------------- scoring

class Score:
    def __init__(self, cap=12):
        self.agree = self.pairs = 0
        self.base_num = self.base_den = 0.0
        self.ids_per_group = []
        self.ids_per_board = []
        self.cleared = []
        self.cap = cap

    def board(self, ids, head, marked, keep=None):
        n = len(ids)
        if keep is not None:
            # Score every method over the SAME tsums. A crop the frame edge
            # clipped is unreadable to the model and perfectly readable to
            # colour, so leaving those in compares coverage, not judgement.
            self.ids_per_board.append(len({int(v) for v, k in zip(ids, keep) if k}))
            if head is None or not keep[head]:
                return
            marked = [i for i in marked if i < n and keep[i]]
        else:
            self.ids_per_board.append(len(set(ids.tolist())))
        if head is None or head >= n or not marked:
            return
        hk = ids[head]
        peers = [i for i in marked if i != head and i < n]
        if not peers:
            return
        self.agree += sum(1 for i in peers if ids[i] == hk)
        self.pairs += len(peers)
        # What agreement would come free from this method's group sizes: the
        # chance a random other tsum on THIS board shares the head's id. Over
        # the same population the pairs came from, or the control is not one.
        pool = [int(v) for v, k in zip(ids, keep)
                if k] if keep is not None else [int(v) for v in ids]
        same = sum(1 for v in pool if v == hk) - 1
        self.base_num += len(peers) * (same / max(len(pool) - 1, 1))
        self.base_den += len(peers)
        self.ids_per_group.append(len({int(ids[i]) for i in peers + [head]}))

        # The number the round is actually scored on, and the only one that
        # says which way to go on the group count.
        #
        # A chain takes up to `cap` tsums sharing the head's id. The hole
        # analysis showed the game SKIPS a member it refuses rather than
        # ending the chain there, so a wrong member costs its own slot and
        # nothing more -- which is why over-merging looks free on `agreement`
        # and is not. It is the SLOT that is scarce. A group half full of the
        # wrong character fills half the chain with tsums that will not pop.
        #
        # So: the chain draws `cap` members from the group, and the share of
        # them the game will accept is the group's own marked density.
        group = [i for i in range(n) if ids[i] == hk and i != head
                 and (keep is None or keep[i])]
        if not group:
            self.cleared.append(1.0)
            return
        hit = sum(1 for i in group if i in set(peers))
        drawn = min(self.cap - 1, len(group))
        self.cleared.append(1.0 + drawn * hit / len(group))

    def row(self, name):
        a = self.agree / self.pairs if self.pairs else 0.0
        b = self.base_num / self.base_den if self.base_den else 0.0
        lift = a / b if b else 0.0
        return (f"{name:>9}  {a:>8.1%}  {b:>7.1%}  {lift:>5.2f}x  "
                f"{np.mean(self.ids_per_group):>6.2f}  "
                f"{np.mean(self.ids_per_board):>7.2f}  "
                f"{np.mean(self.cleared):>8.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--confidence", type=float, default=0.85)
    ap.add_argument("--min-tsums", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="stop after N boards")
    ap.add_argument("--cap", type=int, default=12,
                    help="longest chain the bot builds (play.yaml max_chain)")
    ap.add_argument("--embed", type=Path,
                    help="also score models/embed.onnx used as a fingerprint")
    ap.add_argument("--held-out", action="store_true",
                    help="score ONLY the sessions --embed never trained on. A "
                         "learned method scored on its own training rounds is "
                         "reporting memory: boards inside one round are near "
                         "duplicates, so a session it saw is a session it can "
                         "recognise. The colour baselines learn nothing and are "
                         "unaffected, which is what makes the comparison fair.")
    ap.add_argument("--readable-only", action="store_true",
                    help="score every method over only the crops the model "
                         "can read, separating what it judges badly from what "
                         "it cannot see at all")
    args = ap.parse_args()

    model = T.CharacterModel(args.model, args.confidence)
    embedder = Embedder(args.embed) if args.embed else None
    only = None
    if args.held_out:
        if not args.embed:
            print("--held-out needs --embed: the split lives in the model")
            return 1
        meta = json.loads(args.embed.with_suffix(".json").read_text(encoding="utf-8"))
        only = set(meta.get("test_sessions") or [])
        if not only:
            print("that model records no test_sessions -- retrain to record them")
            return 1
        print(f"held-out only: {len(only)} sessions the embedding never saw")
    scores, boards, unread, total = {}, 0, 0, 0
    t0 = time.perf_counter()

    for d in sorted(args.dir.glob("*/")):
        f = d / "samples.jsonl"
        if not f.exists() or (only is not None and d.name not in only):
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = r.get("tsums") or []
            if len(raw) < args.min_tsums:
                continue
            # Pure-k-means rounds only: elsewhere `kind` is already hybrid and
            # there is no colour baseline left to compare against.
            if any(t.get("kind", 0) >= T.CHARACTER_KIND for t in raw):
                continue
            img = d / f"{r['index']:04d}_before.jpg"
            bgr = cv2.imread(str(img))
            if bgr is None:
                continue
            radius = float(r.get("radius") or 25.0)
            tsums = [T.Tsum(x=float(t["x"]), y=float(t["y"]),
                            r=float(t.get("r", radius)), kind=int(t["kind"]),
                            colour=(0, 0, 0)) for t in raw]
            saved = [int(t["kind"]) for t in raw]

            ids, miss = methods(bgr, tsums, radius, saved, model, embedder)
            unread += miss
            total += len(tsums)
            keep = None
            if args.readable_only:
                keep = [v != UNKNOWN for v in ids["model5"]]
                if sum(keep) < 3:
                    continue
            for name, v in ids.items():
                scores.setdefault(name, Score(args.cap)).board(
                    v, r.get("head"), r.get("marked") or [], keep)
            boards += 1
            if args.limit and boards >= args.limit:
                break
        if args.limit and boards >= args.limit:
            break

    if not boards:
        print("no usable boards -- are there sessions played with the model off?")
        return 1

    print(f"{boards} boards, {total} tsums, {time.perf_counter() - t0:.0f}s")
    print(f"{unread} crops ({unread / total:.1%}) the frame edge clipped -- "
          f"unreadable, left ungrouped\n")
    print("   method  agreement     base    lift  ids/grp  ids/board   cleared")
    print("   " + "-" * 66)
    for name in ("kmeans", "named", "colour2", "colour3", "colour4", "colour5",
                 "colour6", "colour7", "grid5", "grid6", "model5", "both5",
                 "embed3", "embed4", "embed5"):
        if name in scores:
            print("  " + scores[name].row(name))
    # Paired, because the methods are scored on the SAME drags: the spread
    # between boards is far larger than the spread between methods, and an
    # unpaired comparison drowns a real difference in it.
    base = np.asarray(scores["kmeans"].cleared, np.float64)
    print(f"\n  vs kmeans, same {len(base)} drags "
          f"(paired mean difference +/- 2 s.e.):")
    for name in ("named", "colour3", "colour4", "colour5", "grid5", "model5",
                 "embed3", "embed4", "embed5"):
        if name not in scores:
            continue
        d = np.asarray(scores[name].cleared, np.float64) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = "REAL" if abs(d.mean()) > 2 * se else "noise"
        print(f"    {name:>8}  {d.mean():+6.3f} +/- {2 * se:.3f} tsums/drag  "
              f"({d.mean() / base.mean():+6.1%})  {verdict}")

    print("\n  agreement = marked tsums we gave the head's id (higher better)")
    print("  base      = what this method's own group sizes give for free")
    print("  lift      = agreement / base; 1.00x means it knows nothing")
    print("  ids/grp   = ids handed to one marked group; 1.00 is perfect")
    print("  ids/board = distinct ids on a board; the game allows at most 5")
    print(f"  cleared   = tsums this grouping would pop per drag, cap {args.cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
