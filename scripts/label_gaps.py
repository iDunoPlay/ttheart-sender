"""Find the characters the classifier has no name for, and only those.

The player's report: rounds go badly when a tsum appears that was never
labelled. That is a checkable claim, and this is the check -- but the point of
this script is not the measurement, it is the cost of the fix.

Labelling every crop in the corpus is ~40,000 decisions and will not happen.
Labelling every crop in the *unlabelled sessions* is ~26,000 and will not
happen either. But the classifier already names most of them correctly, and
the crops it CANNOT name are the ones carrying information it does not have.
So:

  1. run the trained classifier over the sessions no crop was ever labelled in
  2. keep only the crops it refuses to name (softmax below --reject)
  3. group those, and write one contact sheet per group

which turns "label a corpus" into "name a dozen sheets". The sheets go through
the click-UI that already exists, unchanged:

    python scripts/label_gaps.py
    python scripts/crops.py label 00          # click, type the name
    python scripts/classify.py --epochs 30 --split random --onnx models/character.onnx

WHY THE UNNAMED PILE IS THE RIGHT PILE
--------------------------------------

A board holds at most 5 characters (4 with an item), so a session in which the
classifier refuses a large share of crops is not a session it found difficult
-- it is a session holding a character that has no class at all. The
per-session table is sorted by that share for exactly this reason: the top
rows are the rounds worth naming, and a round at the bottom would teach the
model nothing it does not already know.

A REFUSAL IS NOT ALWAYS A NEW CHARACTER
---------------------------------------

Some of the pile will be fragments, bowl, and half-buried tsums. Those are
fine to leave unlabelled -- the click-UI names what you pick and leaves the
rest -- and they are why this refuses to auto-assign anything at all. The
model groups; the person decides.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crops import CROP, ROOT, WINDOW, _cut, montage  # noqa: E402
from classify import SIZE, build_net  # noqa: E402

#: A crop is "mostly bowl" when this share of its pixels sits within
#: BOWL_TOL of the board's own Lab colour. Both numbers are `crops.py`'s,
#: kept the same so a crop dropped there is dropped here.
BOWL_TOL, BOWL_MOST = 40.0, 0.55


def labelled_sessions(root: Path) -> set[str]:
    """Sessions that already have at least one crop with a name on it.

    Read off the file names, which carry `<session>_<sample>_<index>_v<vis>`.
    A session with even one labelled crop was looked at by a person, so its
    characters are as named as they are going to get without being asked
    again -- and asking again is the cost this script exists to avoid.
    """
    out = set()
    for p in root.rglob("*.png"):
        parts = p.stem.rsplit("_", 3)
        if len(parts) == 4:
            out.add(parts[0])
    return out


def load_model(onnx: Path):
    """The trained classifier, as torch on the GPU.

    Deliberately NOT the cv2.dnn path the play loop would use: this is an
    offline sweep over tens of thousands of crops, and the runtime path would
    spend an hour on what the card does in a minute. Nothing here ships.
    """
    import torch
    meta = json.loads(onnx.with_suffix(".json").read_text(encoding="utf-8"))
    ckpt = torch.load(str(onnx.with_suffix(".pt")), map_location="cpu",
                      weights_only=False)
    net = build_net(meta.get("backbone", "mobilenet_v3_small"),
                    len(meta["classes"]))
    net.load_state_dict(ckpt["state"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.eval().to(dev), meta["classes"], meta, dev


def predict(net, dev, crops, batch: int = 512):
    """(best class, its probability) for each BGR crop."""
    import torch
    from torchvision import transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    best, conf = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            X = np.stack([cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
                          for c in crops[i:i + batch]])
            t = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)
            p = torch.softmax(net(norm(t).to(dev)), 1).cpu().numpy()
            best.append(p.argmax(1))
            conf.append(p.max(1))
    if not best:
        return np.zeros(0, int), np.zeros(0, np.float32)
    return np.concatenate(best), np.concatenate(conf)


def signature(crop):
    """What groups two crops of the same UNKNOWN character together.

    Deliberately not the classifier's own features. For a character it has no
    class for, its features are whatever the wrong classes happen to excite,
    and two crops of one unknown tsum are alike there only by accident. A 3x3
    grid of Lab means is crude, but it is a property of the picture rather
    than of a model already shown not to know what it is looking at -- and
    grouping is all that is being asked of it.
    """
    lab = cv2.cvtColor(cv2.resize(crop, (24, 24)), cv2.COLOR_BGR2LAB)
    cells = [lab[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8].reshape(-1, 3).mean(0)
             for r in range(3) for c in range(3)]
    return np.concatenate(cells).astype(np.float32)


def bowl_colour(crops, step: int = 37):
    """The bowl's Lab colour, taken from the outer rings of the crops.

    The border of a crop is board far more often than it is the tsum's own
    face, so the median border colour over a few hundred crops is the bowl.
    """
    rings = [cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[np.r_[0:6, CROP - 6:CROP]]
             .reshape(-1, 3) for c in crops[::step]]
    return (np.median(np.concatenate(rings), axis=0) if rings
            else np.zeros(3, np.float32))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--reject", type=float, default=0.60,
                    help="softmax below which a crop counts as unnamed")
    ap.add_argument("--per-session", type=int, default=6,
                    help="frames sampled per session; boards inside one round "
                         "barely change, so every extra frame buys very little")
    ap.add_argument("--min-visible", type=float, default=0.55,
                    help="skip crops showing less of a tsum than this. A mostly "
                         "buried tsum gets refused for being buried, not for "
                         "being a character with no class, and it would fill "
                         "the sheets with slivers")
    ap.add_argument("--k", type=int, default=14, help="contact sheets to write")
    ap.add_argument("--all-sessions", action="store_true",
                    help="include sessions that already have labelled crops")
    ap.add_argument("--keep-marked", action="store_true",
                    help="include detections the game had MARKED in that frame. "
                         "Off by default: the game draws its link highlight "
                         "over a marked tsum -- a white glow, or a flat "
                         "silhouette -- so the crop is a picture of the mark "
                         "rather than of the character. 167 of them reached a "
                         "sheet once and were labelled `unknown_lightball`, an "
                         "honest reading of an unreadable picture, and they are "
                         "86.8% marked against ~15% for a real character class")
    ap.add_argument("--base-sheets", type=int, default=4,
                    help="extra sheets of the EQUIPPED BASE TSUM, whatever the "
                         "model thinks of it. Every sample row records which "
                         "colour cluster the skill icon matched, and about an "
                         "eighth of a board is that character -- so this is a "
                         "large free supply of crops of the one tsum the round "
                         "is actually played for, which is otherwise the least "
                         "labelled thing in the set. 0 to skip")
    args = ap.parse_args()

    for suffix in (".json", ".pt"):
        if not args.model.with_suffix(suffix).exists():
            print(f"no {args.model.with_suffix(suffix)} -- train one first:")
            print("  python scripts/classify.py --epochs 30 --split random "
                  "--onnx models/character.onnx")
            return 1

    known = labelled_sessions(ROOT / "labelled")
    sessions = sorted(d for d in args.dir.iterdir()
                      if d.is_dir() and (d / "samples.jsonl").exists()
                      and (args.all_sessions or d.name not in known))
    if not sessions:
        print("every session already has labelled crops -- nothing to triage")
        return 0

    net, classes, meta, dev = load_model(args.model)
    print(f"{len(classes)} classes, {len(sessions)} unlabelled session(s) of "
          f"{len(known) + len(sessions)}, device {dev}")
    if meta.get("held_out_accuracy"):
        print(f"model scored {meta['held_out_accuracy']:.1%} on "
              f"{len(meta.get('test_sessions', []))} held-out sessions")

    crops, origin = [], []
    base_crops, base_origin = [], []
    for d in sessions:
        seen = 0
        for line in (d / "samples.jsonl").open(encoding="utf-8"):
            if seen >= args.per_session:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # FEVER repaints the board dark and lays COMBO text over it. Those
            # crops get refused for being dark, not for being unknown, and a
            # sheet of them teaches nobody a character's name.
            if row.get("fever"):
                continue
            img = cv2.imread(str(d / ("%04d_before.jpg" % row["index"])))
            if img is None:
                continue
            seen += 1
            radius = float(row.get("radius", 25.0))
            # Which colour cluster the equipped tsum's skill icon matched.
            # A k-means id, so it is only as good as k-means -- which is why
            # these crops go to a person to confirm rather than straight into
            # a class. What it does give is a shortlist that is mostly one
            # character, and the character is the one that matters most.
            base = row.get("base") or {}
            base_kind = base.get("kind") if isinstance(base, dict) else None
            # A marked tsum is drawn with the game's link highlight over it, so
            # its crop shows the mark and not the character. Nobody can label
            # that, and asking is how `unknown_lightball` happened.
            hidden = (set() if args.keep_marked
                      else set(row.get("marked") or []) | {row.get("head")})
            for i, t in enumerate(row["tsums"]):
                vis = t["r"] / radius
                if vis < args.min_visible or i in hidden:
                    continue
                c = _cut(img, t["x"], t["y"], radius, WINDOW)
                if c is None:
                    continue
                if base_kind is not None and t.get("kind") == base_kind:
                    base_crops.append(c)
                    base_origin.append((d.name, row["index"], i, vis))
                else:
                    crops.append(c)
                    origin.append((d.name, row["index"], i, vis))
    if not crops:
        print("no crops -- are the *_before.jpg frames present?")
        return 1

    # Drop the ones whose centre is bowl rather than tsum, BEFORE anything is
    # counted. About one detection in eight is one, and every one of them
    # would be refused by the classifier and land in the pile looking exactly
    # like a character it has never been taught.
    board = bowl_colour(crops)
    keep = [i for i, c in enumerate(crops)
            if float((np.linalg.norm(
                cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
                .reshape(-1, 3) - board, axis=1) < BOWL_TOL).mean()) <= BOWL_MOST]
    dropped = len(crops) - len(keep)
    crops = [crops[i] for i in keep]
    origin = [origin[i] for i in keep]
    print(f"\n{len(crops) + dropped} crops, dropped {dropped} "
          f"({dropped / max(len(crops) + dropped, 1):.0%}) whose centre was "
          f"bowl rather than a tsum -> {len(crops)} triaged\n")

    pred, conf = predict(net, dev, crops)
    unnamed = conf < args.reject

    per = {}
    for (sess, _, _, _), u in zip(origin, unnamed):
        a, b = per.get(sess, (0, 0))
        per[sess] = (a + int(u), b + 1)
    rows = sorted(per.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))
    print(f"{'session':>24}{'crops':>7}{'unnamed':>9}{'share':>7}   "
          f"what it did name")
    for sess, (u, n) in rows[:15]:
        top = Counter(classes[p] for (s, _, _, _), p, ok
                      in zip(origin, pred, ~unnamed) if s == sess and ok)
        print(f"{sess:>24}{n:7d}{u:9d}{u / max(n, 1):7.0%}   "
              + ", ".join(c for c, _ in top.most_common(4)))
    if len(rows) > 15:
        print(f"{'... and ' + str(len(rows) - 15) + ' more':>24}")

    print(f"\n{int(unnamed.sum())} of {len(crops)} crops ({unnamed.mean():.0%}) "
          f"have no name at reject {args.reject:.2f}.")
    print("A board holds at most 5 characters, so a session near the top of "
          "that table is\nnot a hard session -- it is a session holding a "
          "character with no class.")

    idx = np.flatnonzero(unnamed)
    out_dir, sheets = ROOT / "all", ROOT / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets.mkdir(parents=True, exist_ok=True)
    manifest, at = {}, 0

    def write_one(pool, where, tag):
        """One sheet from one pile, in the order given -- no clustering."""
        nonlocal at
        paths = []
        for j in pool:
            sess, sample, i, vis = (base_origin if tag == "base" else origin)[j]
            q = out_dir / f"{sess}_{sample:04d}_{i:02d}_v{vis:.2f}.png"
            cv2.imwrite(str(q), where[j])
            paths.append(str(q))
        key = f"{at:02d}"
        manifest[key] = paths
        montage(paths, sheets / f"{key}.png")
        at += 1
        return key, len(paths)

    def write_sheets(pool, where, n_sheets, tag):
        """Cluster one pile and write it as contact sheets, continuing the
        numbering. One flat 00.. sequence rather than two prefixes, because
        `crops.py label` takes a two-digit cluster id and nothing about the
        click-UI needs to know which pile a sheet came from."""
        nonlocal at
        if not len(pool) or n_sheets < 1:
            return []
        sig = np.stack([signature(where[i]) for i in pool])
        sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
        k = max(1, min(n_sheets, max(1, len(pool) // 40)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        cv2.setRNGSeed(0)
        if k > 1:
            _, lbl, _ = cv2.kmeans(np.ascontiguousarray(sig, np.float32), k,
                                   None, crit, 5, cv2.KMEANS_PP_CENTERS)
            lbl = lbl.ravel()
        else:
            lbl = np.zeros(len(pool), np.int32)
        made = []
        for c in range(k):
            paths = []
            for j in pool[lbl == c]:
                sess, sample, i, vis = (base_origin if tag == "base"
                                        else origin)[j]
                p = out_dir / f"{sess}_{sample:04d}_{i:02d}_v{vis:.2f}.png"
                cv2.imwrite(str(p), where[j])
                paths.append(str(p))
            key = f"{at:02d}"
            manifest[key] = paths
            montage(paths, sheets / f"{key}.png")
            made.append((key, len(paths)))
            at += 1
        return made

    unnamed_sheets = write_sheets(idx, crops, args.k, "unnamed")

    # The equipped tsum, whatever the model made of it. Kept as its own pile
    # rather than mixed in: these crops are not here because the classifier
    # failed on them, they are here because this is the character every round
    # is played FOR and it is among the least labelled things in the set.
    #
    # Grouped by SESSION, never clustered by colour. Measured: the base-kind
    # shortlist is 100% one character within a session -- the skill icon match
    # is doing its job -- but the equipped tsum CHANGES between sessions, so
    # colour-clustering the pile across sessions shuffles three characters
    # back together and hands back sheets that are as mixed as the ones this
    # was meant to improve on. A sheet that is one session is one character.
    base_sheets = []
    if args.base_sheets and base_crops:
        keep = [i for i, c in enumerate(base_crops)
                if float((np.linalg.norm(
                    cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
                    .reshape(-1, 3) - board, axis=1) < BOWL_TOL).mean()) <= BOWL_MOST]
        keep.sort(key=lambda i: base_origin[i][0])
        per = max(1, len(keep) // max(1, args.base_sheets))
        page, last = [], None
        for i in keep:
            sess = base_origin[i][0]
            # Break on a session boundary once the page is full, so a sheet
            # never straddles two equipped tsums.
            if page and len(page) >= per and sess != last:
                base_sheets.append(write_one(page, base_crops, "base"))
                page = []
            page.append(i)
            last = sess
        if page:
            base_sheets.append(write_one(page, base_crops, "base"))

    (ROOT / "clusters.json").write_text(json.dumps(manifest), encoding="utf-8")
    print("")
    if unnamed_sheets:
        print(f"sheets {unnamed_sheets[0][0]}-{unnamed_sheets[-1][0]}: "
              f"{sum(n for _, n in unnamed_sheets)} crops the model could not "
              f"name")
    if base_sheets:
        print(f"sheets {base_sheets[0][0]}-{base_sheets[-1][0]}: "
              f"{sum(n for _, n in base_sheets)} crops of the EQUIPPED BASE "
              f"TSUM, from the skill icon's own colour match")
    print(f"-> {sheets}")
    print("Look at the sheets, then name what you recognise:")
    print("    python scripts/crops.py label 00")
    print("Leave fragments and half-buried tsums unpicked. Unlabelled is a")
    print("perfectly good answer, and a wrong name costs more than a missing one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
