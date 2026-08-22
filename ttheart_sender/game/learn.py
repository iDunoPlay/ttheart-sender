"""Turn collected samples into a palette the play loop can actually use.

The collector in :mod:`ttheart_sender.game.dataset` writes labels and nothing
reads them. This is the other half: it reads every session, fits one colour
palette over the whole corpus, scores it against the labels, and saves it. The
play loop then hands that palette to :func:`~ttheart_sender.game.tsum.detect`
instead of letting k-means re-derive one per frame.

Why a palette is the thing worth learning
-----------------------------------------

``Tsum.kind`` is documented as "colour-cluster id; stable within one frame
only", and that sentence is the ceiling on everything above it. Identity is
re-invented from scratch on every fit, so:

* cluster #3 is Pooh on one frame and the bowl on the next, and nothing can be
  remembered about #3 between them;
* the fit runs on one 525x456 crop, so whatever that crop happens to show
  decides the colours -- which is why FEVER, a dim animated wash, produces a
  palette that reads the board at the wrong scale;
* every one of the places `play_loop` discards the palette pays for a refit
  against a board that may be mid-collapse.

A palette fitted over thousands of frames from many rounds addresses all three
at once, and it does it through a seam that already exists:
``detect(palette=...)`` has always accepted centres from an earlier frame.
Passing learned centres instead of last-frame centres changes no detection code
at all -- cluster #3 simply means the same thing on every frame of every round.

That is also what makes anything further possible. A persistent id is the peg
that per-character statistics, learned link rules and any future model hang on;
without one there is nothing to accumulate against.

What the label proves
---------------------

Holding a tsum makes the game light up every tsum that is the same character
*and* reachable. So for one sample, ``head`` and everything in ``marked`` are
the same character -- stated by the game, not inferred. That gives the one
measurement that matters here:

    over held-out samples, how often does the palette give the head and its
    marked partners the same id?

The recorded ``kind`` of each detection is the answer per-frame k-means gave
live, so the same question scored against it is a free baseline on identical
frames. Two numbers, same samples, one difference. `tsum dataset` already
reports the baseline side of it (26-40% on the collections so far), so the
figures are directly comparable to what is in ``docs/DATASET-FINDINGS.md``.

Nothing here runs during a round. Fitting is offline, the artifact is a file,
and the play loop only reads it when it is pointed at one.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Bumped when the saved file's meaning changes, not merely its contents. A
#: palette from an older schema is refused rather than half-read: the whole
#: point is that an id means one thing, so silently loading centres fitted
#: under different rules would break exactly the guarantee being sold.
SCHEMA = 1

#: Samples below this many detections were not a board -- the Home screen and
#: the results screen both score a handful of phantoms, and their colours are
#: menu furniture. The play loop's own floor is 20; this is deliberately
#: looser because a FEVER frame is a legitimate board at ~20 and its colours
#: are worth having.
MIN_TSUMS = 12

#: Marks this close to the pressed tsum are not evidence and are dropped from
#: every score below. The glow is about 90px across and washes over whatever
#: is under it, so a tsum in there clears the bar for being *near the press*
#: rather than for being the same character -- `marked_by_game` says as much,
#: and `tsum dataset` has always excluded them from its own appearance test.
#: Measured over this corpus, 18% of recorded marks sit inside it.
#:
#: Same default as ``--hold-aura``. Scoring a palette against glow-washed
#: marks would credit it for agreeing with the wrong answer.
AURA = 90.0


@dataclass
class Palette:
    """Learned colour centres, plus what the labels said about them.

    `centres` is exactly what :func:`~ttheart_sender.game.tsum._quantise`
    returns and accepts -- (k, 3) float32 in Lab. Everything else is
    provenance: which corpus produced it, how it scored, and which ids turned
    out to be faces rather than bowl or ink.
    """

    centres: np.ndarray
    #: Ids the labels landed on, and how often. A centre no marked tsum ever
    #: matched is board, outline, or a colour from a screen that was not a
    #: board -- useful to see, and never used to filter, because a character
    #: that is rare in the corpus must not be deleted from the palette.
    face_counts: dict = field(default_factory=dict)
    #: How the fit scored on samples it never saw. See :func:`agreement`.
    metrics: dict = field(default_factory=dict)
    #: Corpus provenance -- sessions, samples, and when.
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.centres.shape[0])

    def faces(self) -> list:
        """Ids the game's own marks confirmed as characters, commonest first."""
        return sorted(self.face_counts, key=lambda i: -self.face_counts[i])

    # -- persistence -----------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": SCHEMA,
            "k": self.k,
            # Rounded to two places: Lab centres are distances in a space
            # where 1.0 is near the limit of what a colour difference means,
            # so full float precision is noise that only makes diffs unreadable.
            "centres": [[round(float(v), 2) for v in c] for c in self.centres],
            "face_counts": {str(i): int(n) for i, n in sorted(self.face_counts.items())},
            "metrics": self.metrics,
            "meta": self.meta,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Palette":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        got = int(data.get("schema", 0))
        if got != SCHEMA:
            raise ValueError(
                f"{path} is schema {got}, this build reads schema {SCHEMA}. "
                f"Re-fit it with `tsum learn` rather than using it as-is -- an "
                f"id from another schema does not mean what this one expects.")
        centres = np.asarray(data["centres"], np.float32)
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError(f"{path}: centres must be (k, 3) Lab, got {centres.shape}")
        return cls(
            centres=centres,
            face_counts={int(i): int(n) for i, n in (data.get("face_counts") or {}).items()},
            metrics=data.get("metrics") or {},
            meta=data.get("meta") or {},
        )


# --------------------------------------------------------------------------
# reading a collection
# --------------------------------------------------------------------------
def iter_rows(root) -> Iterator:
    """Every sample in every session folder under `root`, oldest first."""
    root = Path(root)
    if not root.is_dir():
        return
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        jl = folder / "samples.jsonl"
        if not jl.exists():
            continue
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                yield folder, json.loads(line)
            except ValueError:
                # A truncated last line is what a killed round leaves behind.
                # One unreadable row is not a reason to drop a session.
                continue


def usable(row: dict) -> bool:
    """Is this row one the palette should be fitted on or scored against?

    Only structural checks. The collector already refused samples taken while
    the board was moving or while half the screen lit up, so re-litigating
    quality here would be second-guessing a gate that had information this
    does not.
    """
    tsums = row.get("tsums") or []
    head = int(row.get("head", -1))
    return len(tsums) >= MIN_TSUMS and 0 <= head < len(tsums)


def crop_of(folder, row: dict) -> Optional[np.ndarray]:
    """The board as detection saw it, or None if the JPEG is missing."""
    path = Path(folder) / f"{int(row['index']):04d}_before.jpg"
    if not path.exists():
        return None
    # np.fromfile rather than cv2.imread: paths under the user's profile can
    # go through a non-ASCII codepage on Windows and imread returns None on
    # them with no error of its own.
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def _outside_glow(tsums: Sequence, head: int, aura: float = AURA) -> np.ndarray:
    """Boolean mask of detections far enough from the press to be evidence."""
    hx, hy = float(tsums[head]["x"]), float(tsums[head]["y"])
    return np.array([(float(t["x"]) - hx) ** 2 + (float(t["y"]) - hy) ** 2 > aura * aura
                     for t in tsums], bool)


def _as_tsums(row: dict) -> list:
    """Recorded detections as objects `_face_lab` will accept."""
    from .tsum import Tsum
    return [Tsum(x=float(t["x"]), y=float(t["y"]), r=float(t["r"]),
                 kind=int(t["kind"]), colour=(0, 0, 0))
            for t in row["tsums"]]


def face_colours(folder, row: dict) -> Optional[np.ndarray]:
    """Median face Lab of every detection in one sample, (n, 3).

    Sampled with the same window detection itself uses, so a colour measured
    here is the colour the pipeline would have measured live.
    """
    from .tsum import _face_lab
    crop = crop_of(folder, row)
    if crop is None:
        return None
    return _face_lab(crop, _as_tsums(row), float(row.get("radius", 25.0)))


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit(rows: Sequence, *, k: int = 12, px_per_frame: int = 4000,
        seed: int = 0, progress=None) -> np.ndarray:
    """One k-means over pixels pooled from many frames. Returns (k, 3) Lab.

    `px_per_frame` rather than every pixel: the fit only needs the colour
    *distribution*, and a few thousand pixels per frame pin it as well as a
    quarter of a million while keeping a thousand-frame corpus in memory. The
    per-frame count is what matters, not the total -- pooling all pixels from
    a handful of frames would hand those frames' boards the whole palette.

    The blur and the Lab conversion are the ones `_quantise` applies, because
    the centres have to live in the space the live labels will be assigned in.
    """
    pool = []
    used = 0
    rng = np.random.default_rng(seed)
    for folder, row in rows:
        crop = crop_of(folder, row)
        if crop is None:
            continue
        lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0),
                           cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        take = min(px_per_frame, lab.shape[0])
        pool.append(lab[rng.choice(lab.shape[0], take, replace=False)])
        used += 1
        if progress and used % 250 == 0:
            progress(used)
    if not pool:
        raise ValueError("no readable frames to fit on")

    flat = np.concatenate(pool)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    cv2.setRNGSeed(seed)
    # More attempts and a tighter epsilon than the per-frame fit can afford:
    # this runs once, offline, and every frame of every round afterwards
    # inherits the result. Ten seconds here is free.
    _, _, centres = cv2.kmeans(flat, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    return np.asarray(centres, np.float32)


def assign(colours: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest centre for each row of `colours`. Same rule as `_quantise`."""
    colours = np.asarray(colours, np.float32)
    d = ((centres * centres).sum(axis=1)[None, :] - 2.0 * (colours @ centres.T))
    return d.argmin(axis=1)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def agreement(rows: Sequence, centres: np.ndarray) -> dict:
    """Score a palette on samples it was not fitted on.

    For each sample the game named a set of tsums as one character: `head`
    plus everything in `marked`. A palette is right about identity to the
    extent it gives them all the same id.

    Reported alongside is the same count over the `kind` each detection
    carried live, which is per-frame k-means' answer on the identical frames.
    That is the number to beat, and beating it is the entire claim.

    `split` is the other half: the share of weak negatives given a *different*
    id. An unmarked tsum is only a weak negative, because the game marks
    same-character AND reachable, so a same-character tsum on the far side of
    the board is unmarked and counting it as a different character would be
    wrong -- which is why a correct palette scores well under 100% here and
    this must never be maximised on its own.

    **Neither number decides anything alone, and the reason is measured.**
    Across k on the first real corpus, agreement rose exactly as split fell:
    k=6 scored 37.6%/72.4% and k=24 scored 26.6%/86.6%. Reading agreement by
    itself would have called k=6 a 3.9-point win over per-frame clustering,
    when all it had done was merge characters -- the failure `split` exists to
    catch, arriving too gradually for a threshold to see. So the verdict is
    `balanced`, the mean of the two, computed the same way for the learned
    palette and for the per-frame `kind` that ran live. On that measure no k
    beat per-frame clustering on that corpus, which is the honest result.
    """
    pairs = same = base_same = 0
    neg = neg_split = base_split = 0
    scored = 0
    for folder, row in rows:
        head = int(row["head"])
        tsums = row["tsums"]
        far = _outside_glow(tsums, head)
        marked = [int(i) for i in (row.get("marked") or [])]
        # `far` is what makes these marks evidence rather than proximity --
        # see :data:`AURA`. Applied to the negatives too, so both sides of the
        # score are drawn from the same population.
        marked = [i for i in marked if 0 <= i < len(tsums) and i != head and far[i]]
        if not marked:
            continue
        colours = face_colours(folder, row)
        if colours is None:
            continue
        ids = assign(colours, centres)
        scored += 1
        for i in marked:
            pairs += 1
            same += int(ids[i] == ids[head])
            base_same += int(int(tsums[i]["kind"]) == int(tsums[head]["kind"]))

        # Weak negatives: everything the game left dark. Sampled down to the
        # number of positives rather than taken whole, so a board of sixty
        # does not drown the ~5 pairs that carry the real label. Seeded off
        # the sample so the figure is reproducible run to run.
        # Anything the game lit is excluded whether or not it survived the
        # `far` filter: a glow-washed mark is not evidence of sameness, but it
        # is not evidence of difference either, and counting it as a negative
        # would be the same mistake in the other direction.
        lit = {int(i) for i in (row.get("marked") or [])} | {head}
        dark = [i for i in range(len(tsums)) if i not in lit and far[i]]
        for i in random.Random(head).sample(dark, min(len(marked), len(dark))):
            neg += 1
            neg_split += int(ids[i] != ids[head])
            base_split += int(int(tsums[i]["kind"]) != int(tsums[head]["kind"]))

    if not pairs or not neg:
        return {"pairs": pairs, "negatives": neg}

    agree, split = same / pairs, neg_split / neg
    base_agree, base_sp = base_same / pairs, base_split / neg
    return {
        "samples": scored,
        "pairs": pairs,
        "negatives": neg,
        # Share of game-confirmed same-character pairs the palette agrees on.
        "agreement": round(agree, 4),
        # Share of weak negatives it keeps apart.
        "split": round(split, 4),
        # Both of the above for the per-frame clustering that ran live on the
        # identical frames. `baseline` keeps its name -- it is what the
        # verdict used to be read off, and the CLI still prints it.
        "baseline": round(base_agree, 4),
        "baseline_split": round(base_sp, 4),
        # The verdict. Reading agreement alone rewards a palette for merging
        # characters, because merging lifts agreement and costs only split.
        "balanced": round((agree + split) / 2, 4),
        "baseline_balanced": round((base_agree + base_sp) / 2, 4),
    }


def split_rows(rows: Sequence, holdout: float, seed: int = 0) -> tuple:
    """Divide a corpus into fit and score halves BY SESSION, not by sample.

    Samples inside one session are the same board minutes apart under the same
    equipped tsum, so a random per-sample split puts near-duplicates on both
    sides and scores a palette against frames it effectively saw. Holding out
    whole sessions is the only split that answers the question actually being
    asked, which is whether this palette works on the NEXT round.
    """
    sessions = sorted({folder for folder, _ in rows})
    if len(sessions) < 2 or holdout <= 0:
        return list(rows), []
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    n_out = max(1, int(round(len(shuffled) * holdout)))
    held = set(shuffled[:n_out])
    return ([r for r in rows if r[0] not in held],
            [r for r in rows if r[0] in held])


def build(root, *, k: int = 12, holdout: float = 0.25, px_per_frame: int = 4000,
          seed: int = 0, limit: int = 0, progress=None) -> Palette:
    """Read a collection, fit a palette on part of it, score it on the rest."""
    rows = [(f, r) for f, r in iter_rows(root) if usable(r)]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no usable samples under {root}")

    train, test = split_rows(rows, holdout, seed)
    centres = fit(train, k=k, px_per_frame=px_per_frame, seed=seed, progress=progress)

    # Scored on held-out sessions when there are any, and on the fit set when
    # there is only one session -- flagged either way, because a number
    # measured on the frames that produced it is not evidence and must never
    # read as if it were.
    scored_on = test or train
    metrics = agreement(scored_on, centres)
    metrics["held_out"] = bool(test)

    face_counts: dict = {}
    for folder, row in rows:
        marked = [int(i) for i in (row.get("marked") or [])
                  if 0 <= int(i) < len(row["tsums"])]
        if not marked:
            continue
        colours = face_colours(folder, row)
        if colours is None:
            continue
        ids = assign(colours, centres)
        for i in marked:
            key = int(ids[i])
            face_counts[key] = face_counts.get(key, 0) + 1

    return Palette(
        centres=centres, face_counts=face_counts, metrics=metrics,
        meta={
            "fitted": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root": str(Path(root).resolve()),
            "sessions": len({f for f, _ in rows}),
            "samples": len(rows),
            "fit_samples": len(train),
            "score_samples": len(scored_on),
            "px_per_frame": px_per_frame,
            "seed": seed,
        })
