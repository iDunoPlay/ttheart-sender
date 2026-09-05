"""Score the character model against the game's own marks -- no labels needed.

When the bot holds a tsum, the game lights up every tsum that is the SAME
CHARACTER and reachable. `marked` in `samples.jsonl` is therefore ground truth
about identity that nobody wrote down, on sessions nobody labelled, at every
visibility -- including the 78% of a board that no labelled crop has ever come
from. It is the only measurement here that reaches the crops the training set
cannot describe.

The measurement
---------------

Name the head, but only where the model is on solid ground: well visible, and
confident. Then ask what it calls each marked partner. **They must agree** --
the game already said they are the same character. Every disagreement is a
real error, at a visibility read straight off the detection.

    python scripts/mark_probe.py
    python scripts/mark_probe.py --model models/character.onnx --floor-sweep

What it does not measure
------------------------

A mistaken head name mislabels a whole group, so this is a floor on accuracy
rather than an estimate of it -- but a systematically wrong head would drag
every visibility band down together, and the bands do not move together.

And a disagreement has two causes this cannot separate: the model reading a
picture wrongly, or the picture not containing the tsum at all -- a crop
centred on a tsum that is 30% visible is mostly whatever is lying on top of
it. Both are real failures and both cost the same chain, but only the first
can be fixed by labelling more. `harvest` exists to tell them apart: train on
mark-labelled buried crops and re-run this. If the low bands stay flat, the
picture was never there to read.

Harvesting
----------

    python scripts/mark_probe.py harvest --out crops/marked

writes the marked partners into `crops/marked/<character>/`, named by the head
the game confirmed them against. Those are MACHINE labels and are kept apart
from `crops/labelled/` for that reason -- so that training with and without
them is one flag, and so a person's label is never quietly overwritten by a
model's guess.
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
from crops import ROOT, WINDOW, _cut  # noqa: E402
import re  # noqa: E402
from label_gaps import labelled_sessions, load_model, predict  # noqa: E402

#: A head's name is only trusted this far. Both are deliberately strict: the
#: head's name is the only model output the measurement depends on, and every
#: partner in the group inherits it, so a loose head is a poisoned group.
HEAD_VISIBLE, HEAD_CONFIDENCE = 0.60, 0.90

#: The visibility bands the agreement is reported in. Narrow around 0.55
#: because that is where the training set's own cut-off sits and the question
#: is what happens on either side of it.
BANDS = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 1.01]


#: `<session>_<sample>_<index>_v<visible>.png` -- the name `crops.py` writes,
#: which is a primary key back into `samples.jsonl`: session, frame, and which
#: detection in that frame. That is what makes a human label re-findable.
CROP_NAME = re.compile(r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2})_v")


def human_names(root: Path) -> dict:
    """(session, sample, detection) -> the name a person gave that crop.

    The crop file names are a primary key back into the corpus, so a labelled
    crop can be found again as the exact detection it was cut from. Where that
    detection was the HEAD of a marked group, the group's character is known
    to a person's standard rather than a model's -- and every partner in it
    inherits a human label without anyone opening another sheet.
    """
    out = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            m = CROP_NAME.match(p.stem)
            if m:
                out[(m.group("sess"), int(m.group("sample")),
                     int(m.group("idx")))] = d.name
    return out


def groups(sessions, per_session: int = 0):
    """Every (head, partners) the game confirmed, as crops with visibilities.

    Yields one dict per marked group. `partners` excludes the head itself, and
    a group with no readable partner is not yielded at all.
    """
    for d in sessions:
        seen = 0
        for line in (d / "samples.jsonl").open(encoding="utf-8"):
            if per_session and seen >= per_session:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # FEVER repaints the board and lays COMBO text over it. The marks
            # are still honest there, but the pictures are not the pictures
            # the model was trained on, and mixing the two would answer a
            # different question than the one asked.
            if row.get("fever"):
                continue
            head, marked = row.get("head"), row.get("marked") or []
            tsums = row.get("tsums") or []
            if head is None or not marked or head >= len(tsums):
                continue
            img = cv2.imread(str(d / ("%04d_before.jpg" % row["index"])))
            if img is None:
                continue
            radius = float(row.get("radius") or 25.0)
            h = tsums[head]
            head_crop = _cut(img, h["x"], h["y"], radius, WINDOW)
            if head_crop is None:
                continue
            seen += 1
            partners = []
            for j in marked:
                if j == head or j >= len(tsums):
                    continue
                t = tsums[j]
                c = _cut(img, t["x"], t["y"], radius, WINDOW)
                if c is not None:
                    partners.append((c, t["r"] / radius, j))
            if partners:
                yield {"session": d.name, "index": row["index"],
                       "head": (head_crop, h["r"] / radius, head),
                       "partners": partners}


def collect(sessions, net, dev, per_session: int = 0):
    """Groups, with the model's answer for the head and for every partner."""
    gs = list(groups(sessions, per_session))
    crops = []
    for g in gs:
        crops.append(g["head"][0])
        crops.extend(c for c, _, _ in g["partners"])
    if not crops:
        return gs, np.zeros(0, int), np.zeros(0, np.float32)
    pred, conf = predict(net, dev, crops)
    return gs, pred, conf


def _walk(gs, pred, conf):
    """Re-associate the flat predictions with the groups they came from."""
    at = 0
    for g in gs:
        head_p, head_c = int(pred[at]), float(conf[at])
        at += 1
        rows = []
        for _, vis, j in g["partners"]:
            rows.append((int(pred[at]), float(conf[at]), vis, j))
            at += 1
        yield g, head_p, head_c, rows


def report(gs, pred, conf, classes, heads_from: str = "model",
           human: dict = None, head_pred=None, head_conf=None,
           head_classes=None) -> None:
    """Agreement with the game's marks, banded by how visible the partner is.

    `heads_from="labels"` names each group from the crop a PERSON labelled at
    that exact detection. It matters when two models are being compared: with
    model-named heads each model picks its own set of groups to be judged on,
    so the two tables are computed over different rounds and the difference
    between them is partly a difference of population. Human heads pin the
    same groups for every model, and take the model out of its own numerator.
    """
    human = human or {}
    trusted = 0
    vis, agree, pc = [], [], []
    heads = []
    # When a second model names the heads, the group set and the head names
    # are identical for every model being compared. Without it each model
    # picks the groups it happens to feel sure about, and two tables are then
    # computed over different rounds -- so part of any difference between them
    # is a difference of population rather than of skill.
    named_by = (list(_walk(gs, head_pred, head_conf))
                if head_pred is not None else None)
    for k, (g, head_p, head_c, rows) in enumerate(_walk(gs, pred, conf)):
        if named_by is not None:
            # Carried across as a NAME, not an index. Two models trained on
            # different crop sets can have different class lists, and an index
            # from one of them read against the other's list is a different
            # character -- silently, and in a way the table cannot show.
            _, hp, head_c, _ = named_by[k]
            name = (head_classes or classes)[hp]
            if name not in classes:
                continue
            head_p = classes.index(name)
        if heads_from == "labels":
            name = human.get((g["session"], g["index"], g["head"][2]))
            if name is None or name not in classes:
                continue
            head_p = classes.index(name)
        elif g["head"][1] < HEAD_VISIBLE or head_c < HEAD_CONFIDENCE:
            continue
        trusted += 1
        heads.append(head_p)
        for p, c, v, _ in rows:
            vis.append(v); agree.append(p == head_p); pc.append(c)
    if not vis:
        print("no group had a head the model was sure enough about")
        return
    vis, agree, pc = np.array(vis), np.array(agree), np.array(pc)
    print(f"{len(gs)} marked groups, {trusted} with a usable head "
          + ("named by a person" if heads_from == "labels" else
             f"the model is sure of (visible >= {HEAD_VISIBLE}, "
             f"confidence >= {HEAD_CONFIDENCE})"))
    print(f"{len(vis)} confirmed partners to check against it\n")

    print(f"{'partner visible':>17}{'n':>7}{'agrees':>10}{'mean conf':>11}")
    for a, b in zip(BANDS, BANDS[1:]):
        m = (vis >= a) & (vis < b)
        if m.sum() < 20:
            continue
        print(f"{a:6.2f}-{b:<10.2f}{int(m.sum()):7d}{agree[m].mean():10.1%}"
              f"{pc[m].mean():11.2f}")
    print(f"{'ALL':>17}{len(vis):7d}{agree.mean():10.1%}{pc.mean():11.2f}")

    # What agreement would be if the model named every partner with whichever
    # class its heads are most often. Without this the table has no zero.
    base = Counter(heads).most_common(1)[0][1] / max(len(heads), 1)
    print(f"\nchance rate (always the commonest head class): {base:.1%}")
    print(f"{'reject floor':>14}{'agrees':>9}{'on this share of partners':>27}")
    for floor in (0.0, 0.60, 0.80, 0.90, 0.95):
        m = pc >= floor
        if m.sum():
            print(f"{floor:14.2f}{agree[m].mean():9.1%}{m.mean():27.1%}")
    print("\nA reject floor that does not buy accuracy is a floor the model's "
          "confidence\nis not tracking its competence through -- read the two "
          "columns together.")


def harvest(gs, pred, conf, classes, out: Path, min_visible: float,
            heads: str, human: dict) -> None:
    """Write the confirmed partners as crops named by the head's character.

    The GROUPING is always the game's, and that is what makes this work: the
    relation is ground truth even at a visibility where the model is useless,
    so a buried partner gets a name that could never have been read off it.

    Only the head's NAME has to come from somewhere, and there are two
    sources, in this order of preference:

    * **A person.** A labelled crop's file name is a primary key back into the
      corpus, so where the head is a crop somebody already named, the group
      inherits a human label and no model is involved at any point.
    * **The model**, on heads it is well placed to read -- visible, and
      confident. Weaker, and reported separately for that reason: a mistaken
      head silently mislabels every partner behind it.
    """
    out.mkdir(parents=True, exist_ok=True)
    kept, source = Counter(), Counter()
    for g, head_p, head_c, rows in _walk(gs, pred, conf):
        key = (g["session"], g["index"], g["head"][2])
        name = human.get(key) if heads in ("labels", "both") else None
        by = "person"
        if name is None:
            if heads == "labels":
                continue
            if g["head"][1] < HEAD_VISIBLE or head_c < HEAD_CONFIDENCE:
                continue
            name, by = classes[head_p], "model"
        for crop, vis, j in g["partners"]:
            if vis < min_visible:
                continue
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{g['session']}_{g['index']:04d}_"
                              f"{j:02d}_v{vis:.2f}.png"), crop)
            kept[name] += 1
            source[by] += 1
    print("")
    print(f"{sum(kept.values())} crops -> {out}   "
          + ", ".join(f"{n} via a {b}-named head"
                      for b, n in source.most_common()))
    for name, n in kept.most_common():
        print(f"  {name:22} {n:6d}")
    print("")
    print("The GAME grouped these; only the head's name came from elsewhere.")
    print("They stay apart from crops/labelled/ so that training with and")
    print("without them is one flag, and so no person's label is ever")
    print("overwritten by a model's guess.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="probe",
                    choices=["probe", "harvest"])
    ap.add_argument("--dir", type=Path, default=Path("dataset"))
    ap.add_argument("--model", type=Path, default=Path("models/character.onnx"))
    ap.add_argument("--head-model", type=Path,
                    help="name the heads with THIS model and score the "
                         "partners with --model. Pass the same --head-model to "
                         "every model being compared and they are all judged "
                         "on one fixed set of groups, so the difference "
                         "between their tables is skill and nothing else")
    ap.add_argument("--out", type=Path, default=Path("crops/marked"))
    ap.add_argument("--per-session", type=int, default=0,
                    help="frames per session (0 = all)")
    ap.add_argument("--min-visible", type=float, default=0.0,
                    help="harvest: skip partners below this. 0 keeps the "
                         "buried ones, which is the entire point")
    ap.add_argument("--sessions", default="unlabelled",
                    choices=["unlabelled", "all"],
                    help="unlabelled uses only sessions no crop was ever "
                         "labelled from, so the model has not seen one pixel "
                         "of them. `all` measures on its own training data and "
                         "is worth nothing as a score")
    ap.add_argument("--heads", default="both",
                    choices=["labels", "model", "both"],
                    help="where a group's character name comes from. `labels` "
                         "uses only heads a person already named, which makes "
                         "the harvest entirely model-free")
    ap.add_argument("--half", default="all", choices=["all", "first", "second"],
                    help="halve the session list, deterministically. Harvest "
                         "from `first` and probe on `second` -- otherwise the "
                         "model is retrained on the very groups it is then "
                         "scored against, and the gain it reports is its own "
                         "memory of them")
    args = ap.parse_args()

    if not args.model.with_suffix(".pt").exists():
        print(f"no {args.model.with_suffix('.pt')} -- train one first:")
        print("  python scripts/classify.py --epochs 30 --split random "
              "--onnx models/character.onnx")
        return 1

    known = labelled_sessions(ROOT / "labelled")
    every = sorted(d for d in args.dir.iterdir()
                   if d.is_dir() and (d / "samples.jsonl").exists())
    # The halves are cut from the WHOLE corpus, before `--sessions` narrows
    # it. Cutting after would make "the first half of all sessions" and "the
    # second half of the unlabelled ones" two overlapping sets, and a harvest
    # and a probe taken that way would share rounds without either saying so.
    if args.half != "all":
        cut = len(every) // 2
        every = every[:cut] if args.half == "first" else every[cut:]
    sessions = [d for d in every
                if args.sessions == "all" or d.name not in known]
    if not sessions:
        print("no sessions to score")
        return 1

    net, classes, meta, dev = load_model(args.model)
    print(f"{len(classes)} classes over {len(sessions)} {args.sessions} "
          f"session(s)" + (f", {args.half} half" if args.half != "all" else "")
          + f", device {dev}")
    print("")
    gs, pred, conf = collect(sessions, net, dev, args.per_session)
    if not gs:
        print("no marked groups found -- were these rounds played with marks "
              "recorded? (`marked` in samples.jsonl)")
        return 1

    if args.mode == "harvest":
        human = human_names(ROOT / "labelled")
        print(f"{len(human)} human-labelled detections can name a head")
        harvest(gs, pred, conf, classes, args.out, args.min_visible,
                args.heads, human)
    else:
        hp = hc = head_classes = None
        if args.head_model and args.head_model != args.model:
            hnet, hclasses, _, hdev = load_model(args.head_model)
            _, hp, hc = collect(sessions, hnet, hdev, args.per_session)
            head_classes = hclasses
        report(gs, pred, conf, classes, args.heads,
               human_names(ROOT / "labelled"), hp, hc,
               head_classes if args.head_model else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
