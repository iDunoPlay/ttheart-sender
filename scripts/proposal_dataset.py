"""One row per PROPOSED CHAIN MEMBER, with the game's own verdict on it.

WHY THIS, AND WHY NOW
---------------------

The clean A/B of 2026-09-05 moved the target. The board filter deleted
detections, and the two arms turned out to agree about the board to within 0.2
detections per frame (40.19 against 40.40, p=0.775) while disagreeing about
what the game accepted (3.261 against 3.435, p=0.008). Detection is not the
bottleneck. **The bot proposes chains and the game refuses about a quarter of
what it proposes**, and nothing in this repository has ever looked at that
refusal one member at a time.

Every other dataset here is per BOARD or per ROUND. This one is per member of
a proposed chain, which is the unit the decision is actually made in.

THE LABEL IS FREE, AND IT IS THE GAME'S
---------------------------------------

`kept` in `samples.jsonl` is not the bot's opinion. It is
`marked_by_game(...)`: the chain is held, the game lights up the members it
will link, and the read is what gets recorded
([tsum.py:3697](../ttheart_sender/game/tsum.py#L3697)). So for a proposed
member, `accepted = member in kept` is the game's own answer, at every chain
position, on every round ever collected.

Three labels come out of one sample and they are not the same question:

* ``accepted``  -- the game lit it while the chain was held. The proposal was
  right about this member.
* ``dragged``   -- the stroke actually went through it.
* ``cleared``   -- it left the board afterwards.

`accepted` is the one this is built for. The others are recorded because the
gap between them is itself a finding nobody has measured.

WHAT IS DELIBERATELY NOT A FEATURE
----------------------------------

``marked``. It is the same reading the label comes from -- a superset of
`kept` over the whole board -- so a model given it would score near 1.0 and
have learned nothing. It is written to the file as ``leak_marked`` so the
column is auditable rather than quietly absent, and `--eval` refuses to use
any column whose name starts with ``leak_``.

    .venv/Scripts/python scripts/proposal_dataset.py
    .venv/Scripts/python scripts/proposal_dataset.py --eval
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

#: The columns, in order. Named here so the file, the printout and `--eval`
#: cannot drift: a feature added below without a name here fails loudly.
FEATURES = [
    "position",          # 1 = first member after the head
    "dist_prev_r",       # to the previous member, in radii
    "dist_head_r",       # to the head, in radii
    "abs_dx_r", "dy_r",  # signed dy: the pile falls, so up and down differ
    "turn_cos",          # cos of the angle at the previous member
    "blockers_prev",     # tsums lying across the segment from the previous
    "blockers_head",     # ... and across the segment from the head
    "density",           # neighbours within 1.5 radii of this member
    "visible",           # r / radius
    "board_n",           # detections on the board / 50
    "lab_prev",          # Lab distance to the previous member's face / 40
    "lab_head",          # ... and to the head's
    "same_kind_head",    # the bot's own grouping said these match
    "chain_len",         # how long the proposal was when it was chosen
    "path_so_far_r",     # cumulative stroke length up to here, in radii
    "prev_accepted",     # did the game take the member before this one
    "fever",
    "is_base",           # the head is the equipped character's group
]

#: Written, never trained on. See the module docstring.
LEAKS = ["leak_marked"]


def _lab_faces(bgr, tsums, radius: float):
    """The median Lab of each tsum's face, the way `link_net.py` cuts it."""
    out = np.zeros((len(tsums), 3), np.float32)
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    half = max(2, int(radius * 0.5))
    for i, t in enumerate(tsums):
        x, y = int(t["x"]), int(t["y"])
        patch = lab[max(0, y - half):min(h, y + half + 1),
                    max(0, x - half):min(w, x + half + 1)]
        out[i] = patch.reshape(-1, 3).mean(0) if patch.size else 0
    return out


def _blockers(ts, a, c, radius, block=1.25):
    """How many other tsums lie across the line from `a` to `c`.

    The same question `adjacency`'s `block` asks, and the same implementation
    as `link_net.blockers`, so a feature here means what it means there.
    """
    ax, ay, cx, cy = ts[a]["x"], ts[a]["y"], ts[c]["x"], ts[c]["y"]
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
        if math.hypot(t["x"] - (ax + s * vx), t["y"] - (ay + s * vy)) < block * radius:
            n += 1
    return n


def build(root: Path, version: str, arm: str):
    rows, labels, sess, meta = [], [], [], []
    boards = skipped = 0
    for rj in sorted(root.glob("*/round.json")):
        try:
            j = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if version and j.get("version") != version:
            continue
        if arm and j.get("ab_arm") != arm:
            continue
        d = rj.parent
        sp = d / "samples.jsonl"
        if not sp.exists():
            continue
        for line in sp.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts, radius = o.get("tsums") or [], o.get("radius") or 0
            prop, kept = o.get("proposed") or [], o.get("kept")
            if not radius or len(prop) < 2 or kept is None:
                continue
            img = d / f"{o['index']:04d}_before.jpg"
            bgr = cv2.imread(str(img))
            if bgr is None:
                skipped += 1
                continue
            lab = _lab_faces(bgr, ts, radius)
            pts = np.array([[t["x"], t["y"]] for t in ts], np.float64)
            keptset = set(kept)
            dragset = set(o.get("dragged") or [])
            clearset = set(o.get("cleared") or [])
            markset = set(o.get("marked") or [])
            base_kind = (o.get("base") or {}).get("kind")
            head = prop[0]
            path = 0.0
            prev_ok = 1.0
            boards += 1
            for k in range(1, len(prop)):
                i, prev = prop[k], prop[k - 1]
                if i >= len(ts) or prev >= len(ts) or head >= len(ts):
                    continue
                v_prev = pts[i] - pts[prev]
                d_prev = float(np.hypot(*v_prev))
                d_head = float(np.hypot(*(pts[i] - pts[head])))
                path += d_prev
                turn = 1.0
                if k >= 2 and prop[k - 2] < len(ts):
                    u = pts[prev] - pts[prop[k - 2]]
                    nu, nv = np.linalg.norm(u), np.linalg.norm(v_prev)
                    if nu > 1e-6 and nv > 1e-6:
                        turn = float(np.clip((u @ v_prev) / (nu * nv), -1, 1))
                near = int(np.count_nonzero(
                    np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
                    < 1.5 * radius) - 1)
                rows.append([
                    float(k),
                    d_prev / radius, d_head / radius,
                    abs(v_prev[0]) / radius, v_prev[1] / radius,
                    turn,
                    float(_blockers(ts, prev, i, radius)),
                    float(_blockers(ts, head, i, radius)),
                    near / 10.0,
                    ts[i]["r"] / radius,
                    len(ts) / 50.0,
                    float(np.linalg.norm(lab[i] - lab[prev])) / 40.0,
                    float(np.linalg.norm(lab[i] - lab[head])) / 40.0,
                    1.0 if ts[i]["kind"] == ts[head]["kind"] else 0.0,
                    float(len(prop)),
                    path / radius,
                    prev_ok,
                    1.0 if o.get("fever") else 0.0,
                    1.0 if (base_kind is not None
                            and ts[head]["kind"] == base_kind) else 0.0,
                ])
                labels.append([
                    1 if i in keptset else 0,
                    1 if i in dragset else 0,
                    1 if i in clearset else 0,
                    1 if i in markset else 0,          # the leak column
                ])
                sess.append(d.name)
                meta.append((o["index"], k, i))
                prev_ok = 1.0 if i in keptset else 0.0
    return (np.asarray(rows, np.float32), np.asarray(labels, np.int64),
            np.asarray(sess), boards, skipped)


def roc_auc(score, label):
    """Mann-Whitney AUC, with TIES SHARING A RANK.

    The tie handling is not a detail here. The shipped `adjacency` rule scores
    every proposed member identically -- it has no opinion to rank them by --
    and a plain `argsort` breaks those ties in array order, which invented an
    AUC of 0.576 for a constant. Averaged ranks give a constant predictor
    exactly 0.5, which is the true and much more interesting answer.
    """
    score, label = np.asarray(score, float), np.asarray(label)
    if label.min() == label.max():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average the ranks within each run of equal scores.
    srt = score[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos, neg = label.sum(), (1 - label).sum()
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_auc(score, label):
    """Average precision, which is what a rare-positive score is read on."""
    order = np.argsort(-np.asarray(score))
    lab = np.asarray(label)[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    total = lab.sum()
    return float((prec * lab).sum() / total) if total else float("nan")


def calibration(prob, label, bins=5):
    out = []
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob <= hi)
        if m.sum() < 5:
            continue
        out.append((float(prob[m].mean()), float(label[m].mean()), int(m.sum())))
    return out


def evaluate(X, y, sess):
    import torch
    accepted = y[:, 0]
    print(f"\n{'model':16s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'Brier':>7s} "
          f"{'base rate':>10s}")
    base = accepted.mean()
    results = {}

    # The rule in use, scored on the same rows. `adjacency` links within
    # `link_px` and refuses when anything lies across the line, so its
    # "probability" is that test, and it is a constant for every member it
    # proposed -- which is the point: it has no opinion to rank them by.
    rule = np.full(len(X), base, np.float32)
    print(f"{'adjacency':16s} {roc_auc(rule, accepted):8.4f} "
          f"{pr_auc(rule, accepted):7.4f} "
          f"{np.mean((rule - accepted) ** 2):7.4f} {base:10.1%}"
          f"   <- constant: it has no opinion to rank by")

    for name in ("link_geom", "link_colour"):
        p = Path(f"models/{name}.pt")
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cols, mu, sd = ck["cols"], np.asarray(ck["mu"]), np.asarray(ck["sd"])
        # The link model's own 5/7 columns, in ITS order, rebuilt from ours:
        # dist/r, |dx|/r, dy/r, blockers-from-head, n/50, lab-to-head, same-kind.
        idx = [FEATURES.index(n) for n in
               ("dist_head_r", "abs_dx_r", "dy_r", "blockers_head", "board_n",
                "lab_head", "same_kind_head")][:cols]
        Z = (X[:, idx] - mu) / sd
        net = torch.nn.Sequential(
            torch.nn.Linear(cols, 96), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(96, 48), torch.nn.ReLU(),
            torch.nn.Linear(48, 1))
        try:
            net.load_state_dict(ck["state"])
        except RuntimeError as exc:
            print(f"{name:16s} architecture moved on: {exc}")
            continue
        net.eval()
        with torch.no_grad():
            prob = torch.sigmoid(
                net(torch.from_numpy(Z.astype(np.float32))).squeeze(1)).numpy()
        results[name] = prob
        print(f"{name:16s} {roc_auc(prob, accepted):8.4f} "
              f"{pr_auc(prob, accepted):7.4f} "
              f"{np.mean((prob - accepted) ** 2):7.4f} {base:10.1%}")

    for name, prob in results.items():
        rows = calibration(prob, accepted)
        if not rows:
            continue
        print(f"\n  {name} calibration (predicted vs actual):")
        for pm, am, n in rows:
            print(f"    predicted {pm:.3f}   actual {am:.3f}   n={n}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--version", default="1.11.6c",
                    help="build to read; the clean A/B baseline by default")
    ap.add_argument("--arm", default="",
                    help="restrict to one A/B arm ('off' is the baseline)")
    ap.add_argument("--out", type=Path, default=Path("dataset/proposals.npz"))
    ap.add_argument("--eval", action="store_true",
                    help="score the existing link models and the shipped rule")
    args = ap.parse_args()

    X, y, sess, boards, skipped = build(args.dir, args.version, args.arm)
    if not len(X):
        print(f"no proposal rows for version={args.version!r} arm={args.arm!r}")
        return 1

    np.savez_compressed(args.out, X=X, y=y, sess=sess,
                        features=np.asarray(FEATURES), labels=np.asarray(
                            ["accepted", "dragged", "cleared"] + LEAKS))
    print(f"{len(X):,} proposed members from {boards:,} presses over "
          f"{len(set(sess.tolist())):,} rounds"
          + (f" ({skipped} frames unreadable)" if skipped else ""))
    print(f"  -> {args.out}")

    acc, drag, clr, mark = (y[:, i] for i in range(4))
    print(f"\nlabels")
    print(f"  accepted by the game   {acc.mean():6.1%}")
    print(f"  actually dragged       {drag.mean():6.1%}")
    print(f"  cleared the board      {clr.mean():6.1%}")
    print(f"  (leak) marked anywhere {mark.mean():6.1%}")

    print(f"\nacceptance by chain position -- the thing no round-level number "
          f"could show")
    pos = X[:, FEATURES.index("position")]
    print(f"  {'position':>8s} {'n':>7s} {'accepted':>9s} {'cleared':>8s}")
    for k in range(1, int(pos.max()) + 1):
        m = pos == k
        if m.sum() < 20:
            continue
        print(f"  {k:8d} {int(m.sum()):7,d} {acc[m].mean():9.1%} "
              f"{clr[m].mean():8.1%}")

    # A refusal is not the end of the chain: the twenty-sixth round measured
    # that the game skips a refused member and keeps linking. Checked here on
    # the members themselves rather than on drag totals.
    prev = X[:, FEATURES.index("prev_accepted")]
    for v, label in ((1.0, "after an ACCEPTED member"),
                     (0.0, "after a REFUSED member")):
        m = prev == v
        if m.sum() > 20:
            print(f"  {label:28s} n={int(m.sum()):6,d}  "
                  f"accepted {acc[m].mean():.1%}")

    if args.eval:
        evaluate(X, y, sess)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
