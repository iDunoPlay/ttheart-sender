"""Stage 1-2 of the classifier pipeline: get crops out, and get them named cheaply.

A classifier needs "this crop is Mickey". Nothing in this project has ever
recorded that. The game's marks -- the label every other measurement here
leans on -- only say *same character as the one being held*, never who either
one is, and `tsum label`'s group mode is the same relation drawn by hand. So
the names have to come from a person, once, and the only question worth
engineering is how many decisions that costs.

Naively it is one per crop: ~2,000 crops to reach a few hundred per class,
which is an evening of clicking and the reason this has never been done.

This makes it one per CLUSTER instead. The pairwise net from
`embed_net.py` already maps a crop to an embedding where same-character
crops sit close together -- it scores 61% balanced against k-means' 53%, which
is far from good enough to *decide* identity but is easily good enough to
*group* candidates for a person to name. Cluster the embeddings, emit one
montage sheet per cluster, and a human names 20 sheets instead of 2,000 crops.

The clusters will be impure, and that is expected and handled: the labelling
step writes a per-crop file, so a wrong crop in a sheet is fixed by moving one
file, not by redoing the sheet. The net groups; the person decides.

    python scripts/crops.py extract --dir dataset          # corpus -> crops/
    python scripts/crops.py extract --live --frames 40     # LDPlayer -> crops/
    python scripts/crops.py cluster --model models/embed.onnx --k 24
    #   ... look at crops/sheets/*.png, then:
    python scripts/crops.py assign 03 Mickey               # cluster 3 is Mickey
    python scripts/crops.py status

`assign` moves a cluster's crops into `crops/labelled/<name>/`, which is the
layout `classify.py` trains from and the layout every torchvision tutorial
expects. Unassigned clusters stay put and cost nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CROP = 64          #: saved at 64px; the classifier downsamples, a person cannot

#: Half-width of a crop, in detected radii. **1.0, not 1.5, and the difference
#: is the whole quality of the training set.**
#:
#: Two touching tsums sit ~2.44r apart centre to centre (measured, see
#: `detect`'s de-duplication). A crop of half-width 1.5r therefore reaches
#: 1.5r from its own centre while the neighbour's body starts at 2.44-1.0 =
#: 1.44r -- so every single crop contained a slice of its neighbours, and the
#: player reviewing the first sheets said so plainly: "some crops contain
#: multiple tsums".
#:
#: That is not a labelling annoyance, it is a poisoned training set: the
#: pairwise net was asked to say whether two pictures are the same character
#: while each picture held two or three. 1.0 keeps the face and its outline
#: and stops at the gap.
WINDOW = 1.0
ROOT = Path("crops")


# --------------------------------------------------------------------------
# stage 1: extract
# --------------------------------------------------------------------------
def _cut(img, x, y, radius, window=WINDOW):
    """One square crop centred on a tsum, or None if it cannot be one.

    REFUSES a crop the image edge would clip. Clamping to the bounds and
    resizing anyway turns a thin strip at the top of the board rect into a
    square, which is why a whole cluster came back looking "too zoomed in" --
    it was a sliver of a tsum stretched to fill 64x64. A partial crop is not a
    smaller picture of a character, it is a different picture, and it teaches
    a classifier nothing except what the board edge looks like.
    """
    half = max(4, int(round(radius * window)))
    h, w = img.shape[:2]
    x0, y0 = int(x) - half, int(y) - half
    x1, y1 = int(x) + half + 1, int(y) + half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = img[y0:y1, x0:x1]
    if patch.size == 0 or min(patch.shape[:2]) < 4:
        return None
    return cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)


def extract_corpus(root: Path, out: Path, limit: int, window=WINDOW,
                   skip_fever=True) -> int:
    """Every detection in the collected corpus becomes one crop on disk.

    Uses the positions the collector already recorded rather than re-detecting,
    so a crop here is exactly a tsum the live pipeline saw -- which is the
    population the classifier has to work on.
    """
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(root.glob("*/samples.jsonl")):
        folder = f.parent
        for line in f.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # FEVER repaints the board dark and lays COMBO text over it, and
            # detections from those frames are what filled the junk clusters:
            # dim fragments and pieces of the overlay. They are real frames the
            # bot plays on, but they are not pictures of a character.
            if skip_fever and row.get("fever"):
                continue
            img = cv2.imread(str(folder / ("%04d_before.jpg" % row["index"])))
            if img is None:
                continue
            radius = float(row.get("radius", 25.0))
            for i, t in enumerate(row["tsums"]):
                c = _cut(img, t["x"], t["y"], radius, window)
                if c is None:
                    continue
                # The name carries provenance: session, sample, index, and how
                # much of the tsum was visible. `visible` matters because a
                # crop of a 26%-visible tsum is mostly its neighbours, and a
                # training set should be able to exclude those on demand.
                vis = t["r"] / radius
                cv2.imwrite(str(out / f"{folder.name}_{row['index']:04d}_"
                                      f"{i:02d}_v{vis:.2f}.png"), c)
                n += 1
                if limit and n >= limit:
                    return n
    return n


def extract_live(out: Path, frames: int, interval: float) -> int:
    """Crops straight off LDPlayer, for characters the corpus never saw.

    The corpus is whatever was played while collecting. Equipping a different
    tsum, or playing a different event board, puts characters on screen that
    no sample holds -- and a classifier silently has no class for them. This
    is how you add one without playing a measured round.
    """
    from ttheart_sender.app import Application
    from ttheart_sender.game import tsum as T

    out.mkdir(parents=True, exist_ok=True)
    app = Application.create(config_path="config.yaml")
    app.attach_window(prepare=False)
    rect = app.content_rect()
    n = 0
    for k in range(frames):
        frame = app.capture.grab(rect)
        bx, by, bw, bh = T._board_rect(frame.shape, None)
        crop = frame[by:by + bh, bx:bx + bw]
        radius = T._layout_radius(frame.shape) or 25.0
        tsums, radius, _ = T.detect(crop, k=12, radius=radius,
                                    include_dark=True, fit_effort=3)
        stamp = time.strftime("%H%M%S")
        for i, t in enumerate(tsums):
            c = _cut(crop, t.x, t.y, radius)
            if c is None:
                continue
            cv2.imwrite(str(out / f"live{stamp}_{k:03d}_{i:02d}_"
                                  f"v{min(t.r / radius, 9.99):.2f}.png"), c)
            n += 1
        print(f"  frame {k + 1}/{frames}: {len(tsums)} tsums", flush=True)
        time.sleep(interval)
    return n


# --------------------------------------------------------------------------
# stage 2: cluster, so a person names groups instead of crops
# --------------------------------------------------------------------------
def embed(paths, model: Path) -> np.ndarray:
    """Embed every crop with the pairwise net, through cv2.dnn.

    Deliberately not torch: this is the same code path the play loop would use
    if the classifier ever ships, so if it runs here it runs there.
    """
    net = cv2.dnn.readNetFromONNX(str(model))
    out = []
    for i in range(0, len(paths), 256):
        batch = []
        for p in paths[i:i + 256]:
            im = cv2.imread(str(p))
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
            batch.append(im)
        blob = np.stack(batch).astype(np.float32) / 255.0
        blob = blob.transpose(0, 3, 1, 2)
        net.setInput(blob)
        out.append(net.forward().copy())
    return np.concatenate(out)


def kmeans(X, k, iters=40, seed=0):
    """Plain k-means on the embeddings. cv2.kmeans wants float32 and 2-D."""
    X = np.ascontiguousarray(X, np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 0.01)
    cv2.setRNGSeed(seed)
    _, labels, centres = cv2.kmeans(X, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centres


def board_colour(paths, step=37):
    """The bowl's own Lab colour, taken from the crops themselves.

    The outer ring of a crop is board or a neighbour far more often than it is
    the tsum's own face, so the modal ring colour over thousands of crops is
    the bowl. Derived here rather than imported so this measure is INDEPENDENT
    of the pipeline's own `board_colours` -- the point is to cross-check
    detection, and a check that reuses the thing it is checking proves
    nothing.
    """
    ring = []
    for p in paths[::step]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        m = np.ones(lab.shape[:2], bool)
        m[6:-6, 6:-6] = False
        ring.append(np.median(lab[m], axis=0))
    return np.median(np.stack(ring), axis=0) if ring else np.zeros(3)


def bowl_share(paths, board, tol=40.0, most=0.55):
    """Share of crops that are mostly bowl -- i.e. are not tsums at all.

    Measures the WHOLE crop, not just its centre. The centre-only version let
    through a class of crop the player picked out immediately: four fifths
    empty board with one small fragment sitting in the middle, which passes a
    centre test and is plainly not a character. A real tsum at this window
    fills its crop; anything more than `most` bowl is board with litter on it.
    """
    hits = n = 0
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.linalg.norm(lab.reshape(-1, 3) - board, axis=1)
        n += 1
        hits += float((d < tol).mean()) > most
    return hits / max(n, 1)


def montage(paths, out: Path, cols=12, rows=8):
    """One contact sheet, so a person can name a cluster at a glance."""
    cells = []
    for p in paths[:cols * rows]:
        im = cv2.imread(str(p))
        cells.append(im if im is not None else np.zeros((CROP, CROP, 3), np.uint8))
    if not cells:
        return False
    while len(cells) < cols * rows:
        cells.append(np.full((CROP, CROP, 3), 30, np.uint8))
    grid = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(out), grid)
    return True


def label_ui(cluster: str | None, cols: int, rows: int) -> int:
    """Click the crops that are one character, name them, repeat.

    `assign` names a whole cluster, and that only works when a cluster IS one
    character. It is usually not: the clusters group by colour more than by
    identity, so a sheet routinely holds three characters a player can name on
    sight. Whole-sheet labelling cannot express that, and asking someone to
    sort 4,000 files by hand is the cost this pipeline exists to avoid.

    So: a page of crops, click the ones that belong together, type the name
    once. A sheet with three characters costs three passes instead of 96
    decisions, and the crops nobody claims are simply left alone -- unlabelled
    is a perfectly good answer for a fragment or a character you do not know.
    """
    man_path = ROOT / "clusters.json"
    if not man_path.exists():
        print("no clusters.json -- run `cluster` first")
        return 1
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if not cluster:
        print("clusters: " + ", ".join(sorted(man)))
        print("open one with:  python scripts/crops.py label 00")
        return 0
    key = cluster.zfill(2)
    if key not in man:
        print(f"no cluster {key} -- have {', '.join(sorted(man))}")
        return 1

    paths = [Path(p) for p in man[key] if Path(p).exists()]
    if not paths:
        print(f"cluster {key} has no files left (already labelled?)")
        return 0

    per = cols * rows
    page = 0
    picked: set[int] = set()
    win = f"cluster {key} -- click to pick, then S to name them"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def draw():
        start = page * per
        cells = []
        for j in range(per):
            i = start + j
            if i < len(paths):
                im = cv2.imread(str(paths[i]))
                im = im if im is not None else np.zeros((CROP, CROP, 3), np.uint8)
                im = cv2.resize(im, (CROP, CROP))
                if i in picked:
                    # A thick green frame, drawn INSIDE the cell so picking
                    # never changes the grid geometry the clicks are mapped on.
                    cv2.rectangle(im, (1, 1), (CROP - 2, CROP - 2), (0, 255, 0), 3)
            else:
                im = np.full((CROP, CROP, 3), 30, np.uint8)
            cells.append(im)
        grid = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols])
                          for r in range(rows)])
        bar = np.full((28, grid.shape[1], 3), 20, np.uint8)
        pages = (len(paths) + per - 1) // per
        cv2.putText(bar, f"page {page + 1}/{pages}   picked {len(picked)}"
                         f"   [click] pick  [a] all  [d] none  [n/p] page"
                         f"  [s] name them  [q] quit",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.imshow(win, np.vstack([grid, bar]))

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        c, r = x // CROP, y // CROP
        if c >= cols or r >= rows:
            return
        i = page * per + r * cols + c
        if i < len(paths):
            picked.symmetric_difference_update({i})
            draw()

    cv2.setMouseCallback(win, on_mouse)
    draw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("a"):
            picked.update(range(page * per, min((page + 1) * per, len(paths))))
            draw()
        elif k == ord("d"):
            picked.clear(); draw()
        elif k == ord("n"):
            page = min(page + 1, (len(paths) - 1) // per); draw()
        elif k == ord("p"):
            page = max(page - 1, 0); draw()
        elif k == ord("s") and picked:
            cv2.destroyWindow(win)
            name = input(f"name for {len(picked)} crops (blank cancels): ").strip()
            if name:
                dest = ROOT / "labelled" / name
                dest.mkdir(parents=True, exist_ok=True)
                moved = 0
                for i in sorted(picked):
                    src = paths[i]
                    if src.exists():
                        shutil.move(str(src), str(dest / src.name)); moved += 1
                print(f"  {moved} crops -> {dest}")
                # Moved, not copied: a labelled crop must not reappear on the
                # next page and be labelled twice as two characters.
                paths = [q for j, q in enumerate(paths) if j not in picked]
                man[key] = [str(q) for q in paths]
                man_path.write_text(json.dumps(man), encoding="utf-8")
            picked.clear()
            page = min(page, max(0, (len(paths) - 1) // per))
            if not paths:
                print("cluster finished")
                break
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(win, on_mouse)
            draw()
    cv2.destroyAllWindows()
    print(f"{len(paths)} crops left unlabelled in cluster {key} -- that is fine")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="corpus or live screen -> crops/all/")
    e.add_argument("--dir", default="dataset", type=Path)
    e.add_argument("--live", action="store_true", help="capture LDPlayer instead")
    e.add_argument("--frames", type=int, default=30)
    e.add_argument("--interval", type=float, default=1.0)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--window", type=float, default=WINDOW,
                   help="crop half-width in detected radii; past ~1.0 the crop "
                        "starts containing the neighbouring tsums")
    e.add_argument("--keep-fever", action="store_true",
                   help="keep FEVER frames (dark, and overlaid with COMBO text)")
    e.add_argument("--keep-board", action="store_true",
                   help="keep crops whose centre is the bowl rather than a tsum")

    c = sub.add_parser("cluster", help="group crops so they can be named in bulk")
    c.add_argument("--model", type=Path, default=Path("models/embed.onnx"))
    c.add_argument("--k", type=int, default=24)
    c.add_argument("--sample", type=int, default=6000,
                   help="crops to cluster; the rest are assigned to the nearest")
    c.add_argument("--min-visible", type=float, default=0.0,
                   help="skip crops showing less than this fraction of a tsum")

    lb = sub.add_parser("label", help="name INDIVIDUAL crops in a cluster (click them)")
    lb.add_argument("cluster", nargs="?", help="cluster to open; omit to list them")
    lb.add_argument("--cols", type=int, default=12)
    lb.add_argument("--rows", type=int, default=8)

    a = sub.add_parser("assign", help="name a WHOLE cluster -> crops/labelled/<name>/")
    a.add_argument("cluster")
    a.add_argument("name")
    a.add_argument("--move", action="store_true",
                   help="move instead of copy (frees disk, loses the sheet)")

    sub.add_parser("status", help="what is labelled so far")
    args = ap.parse_args()

    all_dir = ROOT / "all"
    if args.cmd == "extract":
        if all_dir.exists():
            # Cleared, not added to: crops from an earlier `--window` are a
            # different picture of the same tsum, and mixing the two makes a
            # training set nobody can reason about.
            shutil.rmtree(all_dir)
        n = (extract_live(all_dir, args.frames, args.interval) if args.live
             else extract_corpus(args.dir, all_dir, args.limit,
                                 args.window, not args.keep_fever))
        print(f"\n{n} crops -> {all_dir}")
        if not args.keep_board:
            # Measured over the whole corpus: about one detection in eight has
            # the bowl at its centre rather than a tsum, and being clearly
            # visible does not protect against it. They are not characters and
            # they should not reach a labelling sheet, let alone a class.
            paths = sorted(all_dir.glob("*.png"))
            board = board_colour(paths)
            gone = 0
            for q in paths:
                if bowl_share([q], board) > 0.5:
                    q.unlink()
                    gone += 1
            print(f"dropped {gone} ({gone / max(len(paths), 1):.1%}) whose centre "
                  f"was the bowl rather than a tsum -> {len(paths) - gone} left")
        print("next: python scripts/crops.py cluster --model models/embed.onnx")
        return 0

    if args.cmd == "cluster":
        if not args.model.exists():
            print(f"no model at {args.model} -- train one first:")
            print("  python scripts/embed_net.py --dir dataset --onnx models/embed.onnx")
            return 1
        paths = sorted(all_dir.glob("*.png"))
        if args.min_visible > 0:
            paths = [p for p in paths
                     if float(p.stem.rsplit("_v", 1)[-1]) >= args.min_visible]
        if not paths:
            print(f"no crops in {all_dir} -- run `extract` first")
            return 1
        idx = np.arange(len(paths))
        if args.sample and len(paths) > args.sample:
            idx = np.random.RandomState(0).choice(len(paths), args.sample, False)
        pick = [paths[i] for i in idx]
        print(f"embedding {len(pick)} of {len(paths)} crops...")
        X = embed(pick, args.model)
        labels, centres = kmeans(X, args.k)
        sheets = ROOT / "sheets"
        # Cleared, not merged. Re-clustering at a different `k` leaves the old
        # run's higher-numbered sheets on disk, and `assign 47 Mickey` would
        # then name a cluster from a manifest that no longer exists -- or
        # worse, one that does exist and means something else.
        if sheets.exists():
            shutil.rmtree(sheets)
        sheets.mkdir(parents=True, exist_ok=True)
        groups = {}
        for p, l in zip(pick, labels):
            groups.setdefault(int(l), []).append(p)
        board = board_colour(pick)
        manifest = {}
        bowlish = {}
        for l, ps in sorted(groups.items()):
            montage(ps, sheets / f"cluster_{l:02d}.png")
            manifest[f"{l:02d}"] = [str(p) for p in ps]
            bowlish[l] = bowl_share(ps, board)
        (ROOT / "clusters.json").write_text(json.dumps(manifest), encoding="utf-8")
        print(f"\n{len(groups)} clusters, sheets in {sheets}")
        worth = [l for l in sorted(groups) if bowlish[l] < 0.5]
        for l, ps in sorted(groups.items()):
            flag = ""
            if bowlish[l] >= 0.5:
                flag = "   <- MOSTLY EMPTY BOARD, skip it"
            elif bowlish[l] >= 0.25:
                flag = "   <- part board"
            print(f"  cluster {l:02d}: {len(ps):5d} crops   board {bowlish[l]:4.0%}{flag}")
        print(f"\n{len(worth)} of {len(groups)} clusters look like actual tsums: "
              + ", ".join(f"{l:02d}" for l in worth))
        print("\nLook at THOSE sheets, then name the ones you recognise:")
        print("  python scripts/crops.py assign 03 Mickey")
        print("A sheet that is clearly two characters: skip it, or assign it and")
        print("delete the wrong files from crops/labelled/<name>/ afterwards.")
        return 0

    if args.cmd == "label":
        return label_ui(args.cluster, args.cols, args.rows)

    if args.cmd == "assign":
        man = json.loads((ROOT / "clusters.json").read_text(encoding="utf-8"))
        key = args.cluster.zfill(2)
        if key not in man:
            print(f"no cluster {key} -- have {', '.join(sorted(man))}")
            return 1
        dest = ROOT / "labelled" / args.name
        dest.mkdir(parents=True, exist_ok=True)
        op = shutil.move if args.move else shutil.copy2
        n = 0
        for p in man[key]:
            src = Path(p)
            if src.exists():
                op(str(src), str(dest / src.name)); n += 1
        print(f"cluster {key}: {n} crops -> {dest}")
        return 0

    if args.cmd == "status":
        lab = ROOT / "labelled"
        total = len(list((ROOT / "all").glob("*.png"))) if (ROOT / "all").exists() else 0
        print(f"crops extracted : {total}")
        if not lab.exists():
            print("nothing labelled yet")
            return 0
        counts = Counter()
        for d in sorted(lab.iterdir()):
            if d.is_dir():
                counts[d.name] = len(list(d.glob("*.png")))
        print(f"classes labelled: {len(counts)}")
        for name, n in counts.most_common():
            bar = "#" * min(40, n // 25)
            print(f"  {name:22} {n:6d}  {bar}")
        print(f"\ntotal labelled  : {sum(counts.values())}")
        thin = [n for n, c in counts.items() if c < 100]
        if thin:
            print(f"under 100 crops : {', '.join(thin)}  <- too thin to train on")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
