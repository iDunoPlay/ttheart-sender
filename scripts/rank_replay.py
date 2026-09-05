"""Would ranking chains by EXPECTED ACCEPTED members beat picking the longest?

Answered offline, on presses already recorded. Nothing here touches play.

THE SHAPE OF THE QUESTION, AND THE PART THAT IS ACTUALLY VERIFIABLE
-------------------------------------------------------------------

The bot sorts candidate chains by ``(is_base, len)`` -- longest wins. The
proposal dataset says the late members of a long chain are mostly refused:
97.9% of first members are accepted and 19.1% of sevenths. So a shorter chain
the game takes whole may be worth more than a long one it takes a third of.

Testing that runs straight into the counterfactual wall. For a chain the bot
did NOT play, the game never said what it would have accepted, so any
"expected accepted" for it is the model talking to itself.

**One case escapes that, and it is the case that matters.** When the new rule
picks a PREFIX of the chain that was actually played, every member of it was
held in front of the game and the game answered. Truncation is therefore
measurable with real labels, and this script reports it separately and
loudly. Re-ranking onto a different chain is reported as prediction only, and
labelled as such.

WHAT TRUNCATION ACTUALLY TRADES
-------------------------------

Cutting a chain at its first predicted refusal buys stroke time and loses the
members past the cut that the game would have taken anyway -- 13.8% of them.
Time was the reason to want it, and the 140-round baseline says presses per
round correlate **-0.17** with score, so time bought is not obviously worth
having. Both sides are measured below rather than assumed.

    .venv/Scripts/python scripts/rank_replay.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game.tsum import Tsum, find_chains  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposal_dataset import FEATURES, _blockers, _lab_faces  # noqa: E402

import cv2  # noqa: E402

#: Never an input. `prev_accepted` is the previous member's LABEL and the bot
#: proposes the whole chain before the game has answered anything, so a model
#: given it is scoring a question that is never asked at decision time.
LEAKY = {"prev_accepted"}


def chain_features(ts, pts, lab, radius, nodes, chain_len, fever, is_base):
    """The per-member rows for ANY chain, real or hypothetical.

    Every column `proposal_dataset` defines except the leak, computed from the
    board and the chain alone -- which is what makes a candidate the bot never
    played scoreable at all.
    """
    head = nodes[0]
    rows, path = [], 0.0
    for k in range(1, len(nodes)):
        i, prev = nodes[k], nodes[k - 1]
        v = pts[i] - pts[prev]
        d_prev = float(np.hypot(*v))
        path += d_prev
        turn = 1.0
        if k >= 2:
            u = pts[prev] - pts[nodes[k - 2]]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 1e-6 and nv > 1e-6:
                turn = float(np.clip((u @ v) / (nu * nv), -1, 1))
        near = int(np.count_nonzero(
            np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            < 1.5 * radius) - 1)
        rows.append({
            "position": float(k),
            "dist_prev_r": d_prev / radius,
            "dist_head_r": float(np.hypot(*(pts[i] - pts[head]))) / radius,
            "abs_dx_r": abs(v[0]) / radius, "dy_r": v[1] / radius,
            "turn_cos": turn,
            "blockers_prev": float(_blockers(ts, prev, i, radius)),
            "blockers_head": float(_blockers(ts, head, i, radius)),
            "density": near / 10.0,
            "visible": ts[i]["r"] / radius,
            "board_n": len(ts) / 50.0,
            "lab_prev": float(np.linalg.norm(lab[i] - lab[prev])) / 40.0,
            "lab_head": float(np.linalg.norm(lab[i] - lab[head])) / 40.0,
            "same_kind_head": 1.0 if ts[i]["kind"] == ts[head]["kind"] else 0.0,
            "chain_len": float(chain_len),
            "path_so_far_r": path / radius,
            "prev_accepted": 0.0,          # never used; see LEAKY
            "fever": 1.0 if fever else 0.0,
            "is_base": 1.0 if is_base else 0.0,
        })
    return np.array([[r[f] for f in FEATURES] for r in rows], np.float32)


def train(X, y, sess, device, seed=0):
    """The acceptance model, held out by ROUND, leak columns removed."""
    import torch
    import torch.nn as nn
    cols = [i for i, f in enumerate(FEATURES) if f not in LEAKY]
    rounds = sorted(set(sess.tolist()))
    rng = np.random.default_rng(seed)
    rounds = [rounds[i] for i in rng.permutation(len(rounds))]
    test = set(rounds[int(len(rounds) * 0.7):])
    tr = np.array([i for i, s in enumerate(sess) if s not in test])
    te = np.array([i for i, s in enumerate(sess) if s in test])
    Xc = X[:, cols]
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0) + 1e-6
    dev = torch.device(device)
    Xt = torch.tensor((Xc[tr] - mu) / sd, device=dev)
    yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    Xv = torch.tensor((Xc[te] - mu) / sd, device=dev)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(len(cols), 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = 0.0, None
    for _ in range(80):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(Xt[b]).squeeze(1), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xv).squeeze(1)).cpu().numpy()
        a = _auc(p, y[te])
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, cols, mu, sd, best, test, dev


def _auc(score, label):
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score), float)
    r[order] = np.arange(1, len(score) + 1)
    p, n = label.sum(), (1 - label).sum()
    return float((r[label == 1].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--data", type=Path, default=Path("dataset/proposals.npz"))
    ap.add_argument("--version", default="1.11.6c")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--export", type=Path,
                    help="write the acceptance model as ONNX (+ .json) so the "
                         "play loop can load it the way it loads the others")
    ap.add_argument("--length-bonus", type=float, default=0.0,
                    help="added per member, so a small preference for length "
                         "can be mixed back in")
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    d = np.load(args.data, allow_pickle=True)
    net, cols, mu, sd, auc, test_rounds, dev = train(
        d["X"], d["y"][:, 0], d["sess"], args.device)
    print(f"acceptance model: held-out AUC {auc:.4f} on {len(test_rounds)} "
          f"rounds it never saw, device {dev}")
    print(f"  {len(cols)} features; `prev_accepted` excluded as a leak")

    # A SECOND model, trained on the same rounds with a different seed and a
    # different split, used ONLY to score the chain the first one picked.
    #
    # Without this the headline number is circular. Choosing the argmax of a
    # noisy estimate and then reporting that estimate is the winner's curse:
    # over ~40 candidates the winner is whichever one the model most
    # over-estimates, so the gain is guaranteed positive and measures nothing.
    # Selecting with A and scoring with B removes exactly that, because B's
    # errors are independent of A's choice.
    net_b, cols_b, mu_b, sd_b, auc_b, _tb, _db = train(
        d["X"], d["y"][:, 0], d["sess"], args.device, seed=7)
    print(f"  second model for honest scoring: AUC {auc_b:.4f}\n")

    if args.export:
        # Exported with a DYNAMIC batch axis and the normalisation baked into
        # the sidecar, so the runtime does the same arithmetic this script did
        # -- a model whose mu/sd drifted from its training set would score a
        # different question at play time and nothing would say so.
        args.export.parent.mkdir(parents=True, exist_ok=True)
        cpu = net.cpu().eval()
        torch.onnx.export(cpu, torch.zeros(1, len(cols)), str(args.export),
                          input_names=["members"], output_names=["logit"],
                          dynamic_axes={"members": {0: "n"}, "logit": {0: "n"}},
                          dynamo=False)
        args.export.with_suffix(".json").write_text(json.dumps({
            "features": [FEATURES[i] for i in cols],
            "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
            "held_out_auc": round(float(auc), 4),
            "excluded_as_leak": sorted(LEAKY),
            "trained_on": "1.11.6c proposal members",
        }, indent=2), encoding="utf-8")
        net.to(dev)
        print(f"  exported -> {args.export} (+ .json)")

    def predict_with(n_, c_, m_, s_, F):
        with torch.no_grad():
            z = torch.tensor((F[:, c_] - m_) / s_, dtype=torch.float32, device=dev)
            return torch.sigmoid(n_(z).squeeze(1)).cpu().numpy()

    def predict_b(F):
        return predict_with(net_b, cols_b, mu_b, sd_b, F)

    def predict(F):
        with torch.no_grad():
            z = torch.tensor((F[:, cols] - mu) / sd, dtype=torch.float32, device=dev)
            return torch.sigmoid(net(z).squeeze(1)).cpu().numpy()

    # ---- replay, on the ROUNDS THE MODEL NEVER TRAINED ON -----------------
    changed = same = 0
    dlen, dexp, dhon = [], [], []
    trunc_actual, calib = [], []          # (played accepted, truncated accepted, cut)
    examples = []
    presses = regen_ok = 0
    for rj in sorted(args.dir.glob("*/round.json")):
        j = json.loads(rj.read_text(encoding="utf-8"))
        if j.get("version") != args.version or rj.parent.name not in test_rounds:
            continue
        for line in (rj.parent / "samples.jsonl").read_text(
                encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts, radius = o.get("tsums") or [], o.get("radius") or 0
            prop, kept = o.get("proposed") or [], o.get("kept")
            if not radius or len(prop) < 2 or kept is None:
                continue
            img = rj.parent / f"{o['index']:04d}_before.jpg"
            bgr = cv2.imread(str(img))
            if bgr is None:
                continue
            opts = o.get("options") or {}
            objs = [Tsum(t["x"], t["y"], t["r"], t["kind"], (0, 0, 0)) for t in ts]
            base_kind = (o.get("base") or {}).get("kind")
            cands = find_chains(
                objs, radius,
                link_px=float(opts.get("link_px") or 105.0),
                block=float(opts.get("block") or 1.25),
                base_kind=base_kind if opts.get("use_base", True) else None,
                max_chain=int(opts.get("max_chain") or 12))
            cands = [c for c in cands if len(c) >= int(opts.get("min_chain") or 3)]
            if not cands:
                continue
            presses += 1
            played = list(prop)
            if any(list(c.nodes) == played for c in cands):
                regen_ok += 1
            lab = _lab_faces(bgr, ts, radius)
            pts = np.array([[t["x"], t["y"]] for t in ts], np.float64)

            scored = []
            for c in cands[:40]:            # `find_chains` returns best-first
                nodes = list(c.nodes)
                if len(nodes) < 2:
                    continue
                F = chain_features(ts, pts, lab, radius, nodes, len(nodes),
                                   o.get("fever"), c.is_base)
                p = predict(F)
                scored.append((nodes, c.is_base,
                               float(p.sum()) + args.length_bonus * len(nodes),
                               float(predict_b(F).sum())))
            if not scored:
                continue
            old = max(scored, key=lambda s: (s[1], len(s[0])))
            new = max(scored, key=lambda s: (s[1], s[2]))
            if list(new[0]) == list(old[0]):
                same += 1
            else:
                changed += 1
                dlen.append(len(new[0]) - len(old[0]))
                dexp.append(new[2] - old[2])
                dhon.append(new[3] - old[3])
                if len(examples) < 6:
                    examples.append((len(old[0]), old[2], len(new[0]), new[2]))

            # Is `expected accepted` worth anything as a NUMBER? On the
            # played chain, predicted total against what the game actually
            # took. If those disagree, no comparison between candidates that
            # rests on the total means anything.
            Fp = chain_features(ts, pts, lab, radius, played, len(played),
                                o.get("fever"), False)
            pp = predict(Fp)
            calib.append((float(pp.sum()),
                          sum(1 for m in played[1:] if m in set(kept))))

            # The verifiable slice: cutting the PLAYED chain at the first
            # member the model would refuse. Every label here is the game's.
            F = chain_features(ts, pts, lab, radius, played, len(played),
                               o.get("fever"), False)
            p = predict(F)
            cut = next((k for k, pk in enumerate(p, start=1) if pk < 0.5),
                       len(played))
            keptset = set(kept)
            acc_full = sum(1 for m in played[1:] if m in keptset)
            acc_cut = sum(1 for m in played[1:cut] if m in keptset)
            trunc_actual.append((acc_full, acc_cut, len(played), cut))

    print(f"{presses:,} presses replayed on held-out rounds; "
          f"{regen_ok / max(presses, 1):.0%} regenerated the exact chain the "
          f"bot played (a check that the candidates are the bot's own)\n")

    total = changed + same
    print("=" * 62)
    print("RE-RANKING (prediction only -- the game never saw these chains)")
    print("=" * 62)
    print(f"  the ranking changes the pick on {changed:,}/{total:,} presses "
          f"({changed / max(total, 1):.1%})")
    if dlen:
        print(f"  when it changes: {np.mean(dlen):+.2f} members "
              f"({np.mean([d < 0 for d in dlen]):.0%} of the time it is SHORTER)")
        print(f"  predicted accepted, SAME model that chose:  "
              f"{np.mean(dexp):+.3f}  <- circular, ignore it")
        print(f"  predicted accepted, INDEPENDENT model:      "
              f"{np.mean(dhon):+.3f}  <- the honest one")
        over_all = np.mean(dhon) * changed / max(total, 1)
        print(f"  averaged over every press (changed or not): "
              f"{over_all:+.3f} accepted members")
        print("\n  examples (old -> new):")
        for ol, oe, nl, ne in examples:
            print(f"    {ol}-member chain, predicted accepted {oe:.2f}   ->   "
                  f"{nl}-member, predicted accepted {ne:.2f}")

    print("\n" + "=" * 62)
    print("IS THE PREDICTED TOTAL WORTH ANYTHING? (played chains, real labels)")
    print("=" * 62)
    if calib:
        pr = np.array([c[0] for c in calib], float)
        ac = np.array([c[1] for c in calib], float)
        print(f"  mean predicted accepted {pr.mean():.3f}   "
              f"mean ACTUAL accepted {ac.mean():.3f}   "
              f"bias {pr.mean() - ac.mean():+.3f}")
        print(f"  correlation predicted vs actual, per press: "
              f"{np.corrcoef(pr, ac)[0, 1]:+.3f}")
        print("\n  by predicted band:")
        edges = np.quantile(pr, [0, .2, .4, .6, .8, 1.0])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pr >= lo) & (pr <= hi)
            if m.sum() < 5:
                continue
            print(f"    predicted {pr[m].mean():5.2f}   actual {ac[m].mean():5.2f}"
                  f"   n={int(m.sum())}")

    print("\n" + "=" * 62)
    print("TRUNCATION (REAL labels -- the game answered every one of these)")
    print("=" * 62)
    if trunc_actual:
        full = np.array([t[0] for t in trunc_actual], float)
        cutd = np.array([t[1] for t in trunc_actual], float)
        plen = np.array([t[2] for t in trunc_actual], float)
        clen = np.array([t[3] for t in trunc_actual], float)
        moved = clen < plen
        print(f"  presses where the cut would fire: {moved.mean():.1%}")
        print(f"  members ACTUALLY accepted, chain as played: {full.mean():.3f}")
        print(f"  members ACTUALLY accepted, chain truncated: {cutd.mean():.3f}")
        print(f"  accepted members LOST to the cut:           "
              f"{(full - cutd).mean():.3f} per press")
        print(f"  members not dragged (stroke saved):         "
              f"{(plen - clen).mean():.2f} per press")
        print(f"\n  acceptance ratio  as played {full.sum() / (plen - 1).sum():.1%}"
              f"   truncated {cutd.sum() / np.maximum(clen - 1, 1).sum():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
