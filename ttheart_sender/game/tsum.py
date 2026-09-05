"""Tsum Tsum board reader: find the tsums, build the adjacency graph, pick a chain.

Two ways in. Flows use the ``play_tsum`` action (see
:mod:`ttheart_sender.automation.tsum_actions`), which calls :func:`play_loop`.
The module is also a CLI, kept because tuning this needs to be done against
saved frames rather than a live game::

    python -m ttheart_sender.game.tsum analyze board.png -o out.png
    python -m ttheart_sender.game.tsum synth -o board.png   # fake board to test on
    python -m ttheart_sender.game.tsum live -n 20           # grab LDPlayer, time the loop

`analyze` writes an overlay PNG (every detected tsum + the best chain drawn on
top) and a JSON sidecar with the drag waypoints in image coordinates. `live`
reports the real capture+detect+path cost per frame, which is the number that
decides how fast a bot built on this could actually react.

Detection deliberately avoids template matching: tsums overlap, tilt, and get
half-buried, which is exactly the case template matching handles worst. Instead
we quantise colour (k-means in Lab), then split each colour's blob into
individual tsums with a distance transform -- the standard "count the
overlapping coins" trick. Nothing here is tsum-specific, so a new character
costs nothing, and it is blind to orientation -- which matters, because tsums
settle at whatever angle physics drops them at.

Chains are ranked around the *base tsum*: the skill icon bottom-left shows the
tsum you equipped, and clearing that type is what charges the skill, so a
3-chain of the base beats a longer chain of anything else.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from ..exceptions import StopRequested, TTHeartError

log = logging.getLogger(__name__)

# Play area as a fraction of the full game screen (540x960 portrait). Measured
# off a 540x960 grab: the bowl starts under the score bar and ends above the
# FEVER strip. Override with --board if your capture is cropped differently.
DEFAULT_BOARD = (0.015, 0.245, 0.970, 0.565)  # x, y, w, h

# Layouts we have actually measured, keyed by captured frame size (h, w).
# Fractions alone can't cover both cases: a live LDPlayer grab carries the
# emulator's title bar and side toolbar, so the game sits at a different offset
# than in a saved screenshot of the game surface. Matching on frame size picks
# the right one without the caller having to know or pass anything.
LAYOUTS: dict[tuple[int, int], dict] = {
    (994, 578): {                       # live `main.py shot` of LDPlayer
        # Two rects, and which one is used depends on FEVER -- see
        # :class:`FeverWatch`.
        #
        # `board` is tight on the middle of the pile. Offline metrics prefer
        # the wider rect below (median radius 22.9 vs 16.2 over 151 captured
        # in-play frames, and a radius collapse below 12px on 13.9% of frames
        # against 26.5%), because a rect that slices tsums at its edge makes
        # them read smaller than they are and drags the radius estimate down.
        # It is kept anyway: it plays better, which is the measure that counts,
        # and those proxies say nothing about whether the chains found are ones
        # worth dragging.
        #
        # `fever_board` is the rect FEVER borrows for its ~10s. It was
        # re-measured with `main.py region` on 2026-08-19: it sits higher than
        # `board` (the pile rides up under FEVER) and is inset at the sides.
        # The metrics above were taken against the older, wider FEVER rect it
        # replaces (8,265,522,535), so they no longer describe this one.
        "board": (10, 314, 525, 456),
        "fever_board": (22, 291, 502, 451),
        "base": "83,858,26",
        # EVERY TSUM IS THE SAME SIZE -- the sprite does not change with the
        # character, so this is a property of the capture geometry and not
        # something to work out per round. Measured two ways that agree: a
        # ruler over one face reads ~27px, and over the captured frames whose
        # detection looks healthy (a realistic 38-48 detections covering a
        # third of the rect) the estimator reads 24-27.
        #
        # Given here, `detect` is told the radius instead of inferring it, and
        # the whole `--radius-lock` warm-up is skipped. That matters because
        # the estimator is unreliable: over 151 captured frames it ranges 8-38px
        # on this same board, and a round that locks low reads the board at
        # half scale all the way through.
        #
        # To re-measure on a new emulator size: `python main.py region`, mark
        # the two sides of one tsum's face, halve the width.
        "radius": 25.0,
    },
    (956, 542): {                       # saved screenshot, no emulator chrome
        # Deliberately left at the full play area: it is a different capture
        # geometry, and mapping the crop above onto it would be arithmetic on
        # an unmeasured offset rather than a measurement. Re-run `region` on a
        # saved screenshot if the labelled boards need to match.
        "board": (8, 258, 524, 505),
        "base": "76,830,26",
    },
}


def _layout(shape) -> dict:
    return LAYOUTS.get((shape[0], shape[1]), {})


def _layout_radius(shape) -> Optional[float]:
    """The measured tsum radius for this capture size, if there is one."""
    value = _layout(shape).get("radius")
    return float(value) if value else None


#: How long FEVER runs once the meter maxes out. The template only marks the
#: moment it fills, so the state has to be held open for its duration.
FEVER_SECONDS = 10.0

#: How well the FEVER BONUS banner's glyph shape has to match. Measured over
#: 151 captured frames, hand-checked: every frame showing the banner scores
#: 0.438 or better, every frame without it 0.275 or worse, and the gap between
#: those two is the largest in the whole sorted list. 0.35 sits in the middle
#: of it. The few frames that score 0.23-0.25 are the banner fading in or out,
#: real FEVER with the text half-transparent -- :data:`FEVER_HOLD` covers those
#: rather than a lower threshold, which would start letting menus through.
FEVER_BANNER_CONFIDENCE = 0.35

#: How long one banner sighting keeps FEVER open.
#:
#: This has to outlast the longest stretch the banner can go unread, and there
#: are two of those. The banner fades in and out at each end of a run, and
#: those frames score 0.20-0.28 against a 0.35 threshold. Bigger: **a skill
#: firing pauses FEVER while its animation plays**, and the animation covers
#: the screen. 3.0s is sized for the animation, which is the longer of the two
#: and was not known about when this was first set to 1.0 -- at 1.0 a played
#: round logged one eleven-second FEVER as three separate ones.
#:
#: The cost of it being too long is noticing the END of FEVER late, which is
#: mild: a few seconds on the FEVER board rect and the FEVER count floor. The
#: cost of too short is a flip, and a flip discards the palette and re-reads
#: the base tsum -- the identity of the character being played for.
FEVER_HOLD = 3.0

#: `max_fever` is the meter at full, gold, an instant before FEVER starts.
#: Measured over 151 captured frames it scores 0.79-0.82 on the three frames
#: that really are the trigger and 0.66 on the next best, so the shipped 0.85
#: default misses it and anything in 0.70-0.78 separates it cleanly.
FEVER_CONFIDENCE = 0.75

MIN_CHAIN = 3  # game rule: a chain of 3+ same tsums clears


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
@dataclass
class Tsum:
    x: float
    y: float
    r: float
    kind: int          # colour-cluster id; stable within one frame only
    colour: tuple      # BGR of that cluster, for drawing


@dataclass
class Chain:
    kind: int
    colour: tuple
    nodes: list = field(default_factory=list)   # indices into the tsum list
    is_base: bool = False                       # matches the equipped tsum

    def __len__(self) -> int:
        return len(self.nodes)


#: How hard the per-frame fit tries, as (attempts, iterations, epsilon).
#:
#: The fit is k-means with a seed, so the same frame can be read twice and
#: come back different -- and how different is measured. Over 60 collected
#: boards, re-fitting one frame at three seeds and comparing which tsums each
#: read found:
#:
#: ===== ================== ========== ============= ==========
#: level criteria           two reads  count spread  cost / fit
#: ===== ================== ========== ============= ==========
#: 1     4, 20, 1.0         73.1%      9.1 tsums     51ms
#: 2     8, 40, 0.5         83.3%      6.2 tsums     138ms
#: 3     16, 60, 0.25       91.4%      3.0 tsums     349ms
#: ===== ================== ========== ============= ==========
#:
#: At level 1 a quarter of the board is decided by the seed, and the count
#: swings by nine tsums between two reads of one frame. Sampling more pixels
#: does not help -- 160k instead of 40k reads 82.3%, no better than level 2 at
#: four times the cost -- so it is the fit's own restarts that matter, not how
#: much of the board it looks at.
#:
#: Level 1 stays the *code* default, so a bare `detect()` call and every
#: offline measurement keep reading the way the rounds in
#: `docs/DATASET-FINDINGS.md` were taken. What ships to a played round is
#: level 3: the thirteenth round put 4,306 drags through it and gave up
#: nothing against the level 1 baseline -- stability 74% -> 93% re-measured on
#: the new corpus, detections per board, `balanced` and refusals all
#: unchanged -- so `flows/*.yaml` set `fit_effort: 3`, and the panel box that
#: used to switch it went away with the experiment. Raising it costs nothing
#: per *frame*: the loop
#: caches its centres and only refits on a fever transition, a shuffle or a
#: recalibration -- about five times in a round, so level 3 costs ~1.5s of a
#: round that runs for minutes.
#:
#: Two defaults rather than one is deliberate: moving the code default would
#: silently re-score every offline number this project has taken.
FIT_EFFORT = {1: (4, 20, 1.0), 2: (8, 40, 0.5), 3: (16, 60, 0.25)}


def _quantise(bgr: np.ndarray, k: int, palette: Optional[np.ndarray] = None,
              seed: int = 0, effort: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """k-means the image in Lab space. Returns (label image, cluster centres).

    Pass `palette` to reuse centres from an earlier frame. Worth doing in a live
    loop: the fit is most of the cost here and the colours on the board don't
    change mid-run, so refitting every frame buys nothing.

    `effort` picks how many restarts the fit gets when there is no palette to
    reuse -- see :data:`FIT_EFFORT`, which measures what each level buys.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    flat = lab.reshape(-1, 3).astype(np.float32)

    if palette is None:
        # Fitting on every pixel is pointless -- 40k samples pins the centres
        # just as well, and measured against 160k it is not the sampling that
        # limits the fit.
        step = max(1, flat.shape[0] // 40_000)
        # A level that is not one of the three plays at today's, rather than
        # raising: this value can arrive from a flow's `options:` mapping, and
        # a typo there must not end a round in the middle of a board.
        try:
            level = int(effort)
        except (TypeError, ValueError):
            level = 1
        attempts, iters, eps = FIT_EFFORT.get(level, FIT_EFFORT[1])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, eps)
        cv2.setRNGSeed(seed)
        _, _, palette = cv2.kmeans(flat[::step], k, None, criteria, attempts,
                                   cv2.KMEANS_PP_CENTERS)

    # ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2, and ||x||^2 is constant per pixel so
    # it drops out of the argmin. Leaves one N*k GEMM instead of materialising an
    # N*k*3 broadcast -- same answer, a fraction of the memory traffic.
    dist = (palette * palette).sum(axis=1)[None, :] - 2.0 * (flat @ palette.T)
    labels = dist.argmin(axis=1).reshape(lab.shape[:2]).astype(np.int32)
    return labels, palette


def _background_clusters(labels: np.ndarray, centres: np.ndarray) -> set[int]:
    """Clusters that make up the bowl behind the tsums.

    The border ring of the crop is always board, never tsum, so whatever
    dominates it is background -- plus anything sitting near it in Lab, which
    catches the lighter rim gradient as one background rather than a "colour".
    """
    ring = np.ones(labels.shape, bool)
    ring[8:-8, 8:-8] = False
    modal = int(np.bincount(labels[ring].ravel(), minlength=len(centres)).argmax())

    bg = {modal}
    for i, c in enumerate(centres):
        if np.linalg.norm(c - centres[modal]) < 18:
            bg.add(i)
    return bg


def board_colours(bgr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """The Lab colours of the bowl behind the tsums, for a frame already fitted.

    :func:`_background_clusters` answers the same question but wants the whole
    label image, which only :func:`detect` has. This reads the crop's border
    ring -- always board, never tsum -- assigns just those few thousand pixels
    to the palette, and applies the identical rule: whatever dominates the ring
    is background, plus anything within 18 Lab of it, which catches the lighter
    rim gradient as one background rather than as a colour of its own.

    Verified to pick the same clusters as `_background_clusters` on all ten
    labelled boards, for ~2.5ms against a re-quantisation of the full crop.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    mask = np.ones(bgr.shape[:2], bool)
    mask[8:-8, 8:-8] = False
    px = lab[mask].reshape(-1, 3).astype(np.float32)
    if px.size == 0:
        return palette[:0]

    dist = (palette * palette).sum(axis=1)[None, :] - 2.0 * (px @ palette.T)
    modal = int(np.bincount(dist.argmin(axis=1), minlength=len(palette)).argmax())
    bg = [i for i, c in enumerate(palette)
          if i == modal or np.linalg.norm(c - palette[modal]) < 18]
    return palette[bg]


def _lightness(lab_centre: np.ndarray) -> float:
    """Lightness of a cluster centre, 0-255.

    Trivial, and that is the point: it names the assumption that a centre's
    channel 0 is lightness. That is true of Lab and of nothing else. Feed this
    pipeline BGR centres and channel 0 is *blue*, so every warm tsum -- Pooh,
    Tigger, Pluto -- reads as dark and gets discarded before it is ever a
    candidate. Measured: swapping the quantiser to raw BGR without touching
    this test drops detection f1 from 0.759 to 0.504, and nothing in the
    failure points at the colourspace.

    So the rule lives behind a name. Changing :func:`_quantise` to work in a
    different space now means either converting here or seeing this function
    and knowing it has to change -- rather than silently losing half the board.
    """
    return float(lab_centre[0])


def _lab_to_bgr(lab_centre: np.ndarray) -> tuple:
    px = np.uint8([[lab_centre]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0][0]
    return int(b), int(g), int(r)


def _fill_holes(mask: np.ndarray, max_area: Optional[float] = None) -> np.ndarray:
    """Close regions fully enclosed by the mask.

    Critical, not cosmetic: eyes, mouths and outlines are dark, so they land in
    a different colour cluster and punch holes straight through the face. The
    distance transform then measures the gap to the nearest *eye* instead of to
    the tsum's edge, and every radius comes out ~5x too small.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = ((flooded == 0) & (padded == 0)).astype(np.uint8)[1:-1, 1:-1]

    if max_area is None:
        return mask | holes

    # Once the radius is known, only fill things eye-sized or smaller -- a real
    # pocket of board enclosed by same-coloured tsums is not part of a tsum and
    # filling it would invent a peak there.
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] <= max_area
    return mask | keep[lbl].astype(np.uint8)


def _merge_enclosed(labels: np.ndarray, centres: np.ndarray, exclude: set[int],
                    *, min_frac: float = 0.6) -> tuple[np.ndarray, int]:
    """Fold colour patches that sit *inside* another colour into their host.

    One tsum is frequently two colours -- Pluto's gold face carries a white
    muzzle, Piglet's pink face a pale snout -- and k-means has no way to know
    they belong together. Split like that, neither half is a disc, so neither
    produces a distance-transform peak and the whole tsum goes undetected.

    Colour distance can't fix it: Mickey and Donald sit 30.6 apart in Lab,
    closer than the genuinely-split pairs, so any merge threshold that unites
    Pluto's two halves also fuses Mickey into Donald. Geometry can, because the
    muzzle is always *surrounded* by gold and Donald's face never is.

    Crucially this works per connected component, not per cluster: Donald's face
    and Pluto's muzzle are the *same* white cluster, so a cluster-level rule
    would have to treat them identically. Component-level, the muzzle is ringed
    by gold and gets absorbed while Donald's face is ringed by board and stays.
    """
    out = labels.copy()
    h, w = labels.shape
    ring_kernel = np.ones((3, 3), np.uint8)
    merged = 0

    for c in range(len(centres)):
        if c in exclude:
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == c).astype(np.uint8), 8)
        for i in range(1, n):
            x, y, bw, bh, _area = stats[i]
            # Work in the component's own bounding box -- a few hundred pixels
            # instead of the whole board, per component.
            x0, y0 = max(0, x - 4), max(0, y - 4)
            x1, y1 = min(w, x + bw + 4), min(h, y + bh + 4)
            sub = comp[y0:y1, x0:x1] == i
            ring = cv2.dilate(sub.astype(np.uint8), ring_kernel, iterations=2).astype(bool) & ~sub

            neigh = labels[y0:y1, x0:x1][ring]
            if neigh.size == 0:
                continue
            counts = np.bincount(neigh, minlength=len(centres))
            counts[c] = 0
            for e in exclude:
                counts[e] = 0  # board and outlines host nothing
            host = int(counts.argmax())
            if counts[host] and counts[host] / neigh.size >= min_frac:
                out[y0:y1, x0:x1][sub] = host
                merged += 1

    return out, merged


def _peaks(dt: np.ndarray, radius: float, floor: float,
           ceiling: float = 1.45) -> list[tuple[float, float, float]]:
    """Local maxima of a distance transform -- one per tsum in the blob.

    `ceiling` (in radii) throws out peaks that are too *deep* to be a tsum. A
    tsum can't be much rounder than its own radius, so anything well past that
    is open board: the dark bowl shadow behind the pile is a colour cluster like
    any other, and without this it reports itself as a handful of giant tsums.
    """
    ksize = max(3, int(round(radius * 1.1)) | 1)
    local_max = cv2.dilate(dt, np.ones((ksize, ksize), np.uint8))
    mask = ((dt >= local_max - 1e-4) & (dt >= floor) & (dt <= radius * ceiling)).astype(np.uint8)

    n, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        cx, cy = centroids[i]
        out.append((float(cx), float(cy), float(dt[int(round(cy)), int(round(cx))])))
    return out


def _recolour(bgr: np.ndarray, tsums: list[Tsum], radius: float,
              thresh: float) -> list[Tsum]:
    """Re-decide which tsums are the same character, one sample per tsum.

    Pixel-level k-means segments the board well but classifies it badly: a
    single character lit differently across the board can land in two clusters,
    and then two adjacent tsums that obviously match get treated as unrelated.
    Measured against hand-labelled links, that cost ~9% of them.

    Colour distance between *cluster centres* can't fix it -- one genuine split
    pair sat 43.8 apart in Lab while unrelated characters sat at 31, so the
    distributions overlap and no threshold separates them. Sampling per tsum
    instead gives cleaner evidence: one median colour per face, taken from the
    middle of the face where lighting is most consistent, then merged
    agglomeratively.

    OFF BY DEFAULT, and the reason is a lesson about the labels themselves.
    Tuning `thresh` against labelled links looked like a win -- 80% -> 89% of
    drawn links got the same kind. But labels only record pairs that DO belong
    together, and a positive-only score always improves by merging more, right
    up to "everything is one character". Checked end to end it was a clear
    regression: chains of 27 and 31 tsums on real boards, and links the model
    would actually chain fell 85% -> 63%.

    Calibrating this needs negative examples -- pairs a human marks as
    look-alike but genuinely different -- which the current labelling flow
    does not collect. Left in, off, for when it does.
    """
    if len(tsums) < 2:
        return tsums

    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    sample = max(2, int(radius * 0.45))
    feats = []
    for t in tsums:
        patch = np.zeros(bgr.shape[:2], np.uint8)
        cv2.circle(patch, (int(t.x), int(t.y)), sample, 1, -1)
        px = lab[patch.astype(bool)]
        feats.append(np.median(px, axis=0) if px.size else np.zeros(3, np.float32))
    feats = np.asarray(feats, np.float32)

    dist = np.linalg.norm(feats[:, None] - feats[None, :], axis=2)
    parent = list(range(len(tsums)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = sorted(((dist[i, j], i, j)
                    for i in range(len(tsums)) for j in range(i + 1, len(tsums))),
                   key=lambda p: p[0])
    for d, i, j in order:
        if d > thresh:
            break
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    groups: dict[int, list[int]] = {}
    for i in range(len(tsums)):
        groups.setdefault(find(i), []).append(i)

    for new_kind, (_root, members) in enumerate(sorted(groups.items())):
        colour = _lab_to_bgr(np.median(feats[members], axis=0))
        for i in members:
            tsums[i].kind = new_kind
            tsums[i].colour = colour
    return tsums


def _regroup(bgr: np.ndarray, tsums: list["Tsum"], radius: float,
             groups: int) -> list["Tsum"]:
    """Sort the board into exactly `groups` identities, by face colour.

    The same evidence as :func:`_recolour` -- one median colour per face --
    decided a different way, and the difference is the whole point.

    `_recolour` merges everything closer than a *distance*, and its docstring
    records why that could never be calibrated here: the labels are
    positive-only, so the score always improves by merging more, and the
    threshold that looked best produced chains of 27 and 31 tsums. There is no
    natural place to stop.

    The game supplies one. **A board holds at most five characters, four when
    an item is used.** That is a hard rule, and it turns identity from an open
    problem -- name a character out of forty-odd, most of them never labelled
    -- into a closed one: sort this board into five piles. A count cannot run
    away the way a threshold can, because the group sizes are bounded by the
    board, and it needs no labels at all, so it works on the characters nobody
    has trained and on a base tsum the classifier has never seen.

    Measured over 1,117 saved boards against the game's own marks, by
    `scripts/group_eval.py`, in expected tsums cleared per drag:

        pixel k-means (today)   3.26
        groups=3                3.31  (+1.4%, inside the noise)
        groups=4                3.41  (+4.6%, paired 2 s.e. 0.047)
        groups=5                3.34  (+2.4%)
        groups=6                3.23  (-1.0%)

    An interior optimum, which is the shape that says the number means
    something: too many groups splits a character and its partners become
    unreachable; too few fills the chain with tsums the game will refuse. The
    win is real and it is small. OFF BY DEFAULT, and what settles it is a
    played round, not this table.
    """
    if len(tsums) <= max(1, groups):
        return tsums

    feats = _face_lab(bgr, tsums, radius)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    # Three attempts: k-means on 40-odd points is cheap, and a bad start here
    # costs a whole frame's identity rather than a slightly worse centroid.
    _, labels, _ = cv2.kmeans(np.ascontiguousarray(feats, np.float32), groups,
                              None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    for new_kind in range(groups):
        members = [i for i in range(len(tsums)) if labels[i] == new_kind]
        if not members:
            continue
        colour = _lab_to_bgr(np.median(feats[members], axis=0))
        for i in members:
            tsums[i].kind = new_kind
            tsums[i].colour = colour
    return tsums


def _face_lab(bgr: np.ndarray, tsums: Sequence["Tsum"], radius: float,
              *, fill: float = 0.0) -> np.ndarray:
    """Median Lab colour of each tsum's inner face, one row per tsum.

    The middle 0.45r only. Eyes, outline and the rim gradient all live further
    out, and a sample that reaches them is a blend no character actually has --
    the same window :func:`_recolour` and :func:`purity_filter` sample.

    Sampled in a box around each tsum rather than against a full-frame mask:
    same medians, but the cost stops scaling with board size, which matters
    because `--bowl-reject` runs this on every frame.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    sample = max(2, int(radius * 0.45))
    out = np.full((len(tsums), 3), fill, np.float32)

    for i, t in enumerate(tsums):
        cx, cy = int(t.x), int(t.y)
        x0, y0 = max(0, cx - sample), max(0, cy - sample)
        x1, y1 = min(w, cx + sample + 1), min(h, cy + sample + 1)
        patch = lab[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= sample * sample
        px = patch[disc]
        if px.size:
            out[i] = np.median(px, axis=0)
    return out


#: Least of a tsum that must be showing before the character model is asked.
#:
#: **The training set's own rule, which the runtime never knew.** `crops.py`
#: wrote its crops at `v >= 0.55` and every one of the 3,791 labelled crops is
#: above it, while the median detection on a real board shows 0.41 -- so 78% of
#: the board is a picture the model has not seen a single example of.
#:
#: Asked anyway, it does not hedge. Measured against the game's own marks over
#: 94 sessions no labelled crop came from, it agrees with the character the
#: game confirmed 74% of the time above 0.55 and 0-28% below it, against a
#: 12.5% chance rate -- at a mean confidence of 0.75-0.88 the whole way down.
#: Confidence does not fall where competence does, so `--character-confidence`
#: alone cannot catch this and never could.
#:
#: Below the floor a tsum keeps its k-means `kind`, which agrees with the game
#: 37% of the time. That is not good, but it is more than twice what the model
#: manages there, and it does not arrive dressed as certainty.
CHARACTER_MIN_VISIBLE = 0.55

#: Where a character id starts, so it cannot collide with a k-means `kind`.
#: The two live in the same field on purpose -- `adjacency` and `find_chains`
#: ask only whether two `kind`s are equal, so a character id drops in without
#: either of them learning a new concept. The offset is what keeps "character
#: 3" from meaning the same thing as "colour cluster 3".
CHARACTER_KIND = 1000


#: Every forward pass is padded to exactly this many crops.
#:
#: **This is a workaround for a crash in OpenCV, not a tuning knob.** cv2 5.0.0
#: sizes a `dnn` net's internal buffers on the first batch it sees and does not
#: grow them; a later, larger batch writes past the end and Windows kills the
#: process with 0xc0000409 (STACK_BUFFER_OVERRUN) inside cv2.pyd -- no Python
#: traceback, no error box, the tray icon simply vanishes. Reproduced exactly:
#: a batch of 1 followed by a batch of 46 dies on the 46, and 1,2,3,...  dies
#: at 13.
#:
#: Live that is guaranteed to happen: a round's first frame is the board still
#: filling and reads ~11 tsums, and a settled board reads ~45. The net gets
#: sized for 11 and dies on the next real frame.
#:
#: So the batch is CAPPED here and the buffers are sized here, once, by
#: :func:`_warm_up` at load time -- a single forward on 64 rows of zeros,
#: before any board is read. After that every real board is sent at its own
#: size, because the buffers can only ever SHRINK from a maximum they were
#: already given, and shrinking is not what kills it.
#:
#: That the shrink is safe is measured, not assumed. On cv2 5.0.0, warmed at
#: 64 and then fed 41, 33, 47, 12, 58, 1, 64, 40, 5 and 44 in that order, the
#: net survives every one and its answers are **bit-identical** to the same
#: crops padded out to 64 -- so this is a speed change with no decision in it.
#:
#: It was worth doing because the padding was not the "about 10ms" the first
#: estimate here guessed. A board is ~41 detections and a forward costs 10.4ms
#: at 41 against 18.3ms at 64, so **43% of every forward was zero rows**, on a
#: filter measured at 11% of the frame rate in play. See `updates.md`,
#: 2026-09-05.
#:
#: A board larger than this still goes round again rather than growing the
#: batch, which is the one thing that must never happen.
CHARACTER_BATCH = 64


def _warm_up(net, size: int) -> None:
    """Size a freshly loaded net's buffers at the largest batch it will see.

    Called once per net, at load, so the first real board cannot be the thing
    that sizes them. Without it a round whose opening frame reads 11 tsums
    sizes the net for 11 and is killed by the next settled board at 45 --
    which is the crash :data:`CHARACTER_BATCH` was written for. It costs one
    forward on zeros, once per process, against a class of failure that takes
    the whole app down mid-round.
    """
    net.setInput(np.zeros((CHARACTER_BATCH, 3, int(size), int(size)),
                          np.float32))
    net.forward()


def _forward(net, blob, name: str) -> np.ndarray:
    """Every crop's logits, in order, in batches that never exceed the warm-up.

    Shared by both models so a change to the batching rule cannot reach one
    of them and not the other -- the same reason :func:`_character_crop` is
    module level.
    """
    outs = []
    for i in range(0, len(blob), CHARACTER_BATCH):
        chunk = np.ascontiguousarray(blob[i:i + CHARACTER_BATCH])
        net.setInput(chunk)
        out = np.asarray(net.forward(), np.float32)
        if out.shape[:1] != chunk.shape[:1]:
            raise ValueError(f"{name} returned {out.shape} for "
                             f"{len(chunk)} crops")
        outs.append(out.copy())
    if not outs:
        return np.zeros((0, 1), np.float32)
    return np.concatenate(outs)


#: One loaded net per (path, confidence), for the life of the process.
#:
#: `play_loop` runs once per ROUND, and the tray plays rounds back to back for
#: hours. Constructing a `cv2.dnn` net per round means a fresh native parse and
#: a fresh set of layer buffers each time, with the old ones released only if
#: and when Python gets around to it -- and the reported failure was exactly
#: that shape: two rounds fine, the third kills the process with no traceback
#: and no error box.
#:
#: The model is immutable and identical every round, so there is nothing to
#: reload. Cached here rather than on the caller because the caller is a new
#: `play_loop` frame every time.
_CHARACTER_CACHE: dict = {}


def load_character_model(path, confidence: float,
                         min_visible: float = CHARACTER_MIN_VISIBLE
                         ) -> "CharacterModel":
    """The shared model for this path, loaded at most once per process."""
    key = (str(path), round(float(confidence), 4), round(float(min_visible), 4))
    model = _CHARACTER_CACHE.get(key)
    if model is None or model.net is None:
        # `net is None` means it switched itself off mid-round. Rebuilt rather
        # than handed back dead, because a cached corpse clears `failed` below
        # and then names nothing for the rest of the process with nothing said.
        model = CharacterModel(path, confidence, min_visible)
        _CHARACTER_CACHE[key] = model
    # Counters are per round, the net is not.
    model.named = model.seen = model.buried = 0
    model.failed = ""
    return model


def load_chain_model(path, bonus: float) -> "ChainModel":
    """The shared chain ranker for this path, loaded at most once per process.

    Same cache and the same reason as the other two loaders: `play_loop` runs
    once per ROUND and a fresh `cv2.dnn` net per round is how this project
    last killed the tray.
    """
    key = ("chain", str(path), round(float(bonus), 4))
    model = _CHARACTER_CACHE.get(key)
    if model is None or model.net is None:
        model = ChainModel(path, bonus)
        _CHARACTER_CACHE[key] = model
    model.seen = model.moved = 0
    model.failed = ""
    return model


def load_reject_model(path, floor: float) -> "RejectModel":
    """The shared board filter for this path, loaded at most once per process.

    Same cache and the same reason as :func:`load_character_model`: the tray
    plays rounds back to back for hours, and building a `cv2.dnn` net per round
    means a fresh native parse and a fresh set of layer buffers each time, with
    the old ones released only if and when Python gets around to it. The
    reported failure of that shape was two rounds fine and the third killing
    the process with no traceback.

    The filter was constructed per round from the day it shipped and had not
    been noticed because it had never played a long session -- it first ran on
    2026-09-05, for 14 rounds. Cached before it gets the chance.
    """
    key = (str(path), round(float(floor), 4))
    model = _CHARACTER_CACHE.get(key)
    if model is None or model.net is None:
        model = RejectModel(path, floor)
        _CHARACTER_CACHE[key] = model
    # Counters are per round, the net is not.
    model.seen = model.dropped = 0
    model.failed = ""
    return model


def _character_crop(bgr, t, radius: float, size: int):
    """The picture a crop-trained model was shown, or None.

    Must match `scripts/crops.py` exactly -- window 1.0 radii, saved at 64px,
    and REFUSED when the frame edge would clip it. A crop the edge clips is a
    sliver stretched to a square, which the training set excluded; feeding one
    in at play time asks the model about a picture it has never seen one
    example of.

    Module level, and shared by every model that reads a crop, so a change to
    the rule cannot reach one of them and not the other.
    """
    half = max(4, int(round(radius)))
    h, w = bgr.shape[:2]
    x0, y0 = int(t.x) - half, int(t.y) - half
    x1, y1 = int(t.x) + half + 1, int(t.y) + half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0 or min(patch.shape[:2]) < 4:
        return None
    # 64 then `size`, not straight to `size`: the training crops were written
    # to disk at 64 and upsampled from there, and a single resize would hand
    # the model a sharper image than it ever trained on.
    small = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)


class RejectModel:
    """Drops detections that are the board rather than a tsum.

    The player labelled 705 crops `board` on purpose, and the label is clean:
    across the whole corpus the game marked **0.0%** of them and none was ever
    a chain head. A patch of bowl cannot be confirmed as a character, so that
    zero is what a real negative looks like. Trained on it, held out by
    session, the net reads 0.998 AUC.

    **What it cannot settle, and neither can anyone.** The board is printed
    with empty tsum-shaped SLOTS, and a tsum the game has linked into a chain
    is drawn as a flat SILHOUETTE -- same outline, same flat fill, same dark
    border. From one 64px square they are the same picture. Trained on `board`
    alone the net threw away game-confirmed tsums at 20.8% against 16.6% for
    the ones it kept; the marks are what fixed it, harvested as positives so
    the linked state is labelled rather than inferred from an outline it
    shares with the board.

    After that, its rejections are confirmed by the game at 15.4% against a
    17.2% base rate -- no longer enriched for real tsums, but only just below
    chance. **That test can rule a filter out and cannot certify one**, which
    is why this ships off and a played round decides.
    """

    #: Below this the detection is dropped. Chosen from the measured trade
    #: rather than rounded to a half: at 0.10 the model rejects 2.2% of a board
    #: and those are the least likely to be real; raising it rejects more and
    #: the ones it adds are progressively more often confirmed tsums.
    FLOOR = 0.10

    def __init__(self, path, floor: float = FLOOR):
        meta = json.loads(Path(str(path)).with_suffix(".json")
                          .read_text(encoding="utf-8"))
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size = int(meta.get("size", 96))
        # Before the first board, not on it: see `_warm_up`.
        _warm_up(self.net, self.size)
        self.mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
        self.floor = float(floor)
        self.seen = self.dropped = 0
        self.failed = ""

    def keep(self, bgr: np.ndarray, tsums: Sequence["Tsum"],
             radius: float) -> list["Tsum"]:
        """The detections worth playing, in their original order.

        A crop the frame edge clips is KEPT, not dropped. The model has never
        seen one, and refusing a tsum at the board's rim because the picture
        of it is awkward would quietly shrink the playable board -- the same
        mistake `CharacterModel` made by being asked about crops the training
        set had excluded.
        """
        if self.net is None or not tsums:
            return list(tsums)
        self.seen += len(tsums)
        idx, batch = [], []
        for i, t in enumerate(tsums):
            c = _character_crop(bgr, t, radius, self.size)
            if c is not None:
                idx.append(i)
                batch.append(c)
        if not batch:
            return list(tsums)
        blob = np.stack(batch).astype(np.float32) / 255.0
        blob = (blob - self.mean) / self.std
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)
        try:
            logits = _forward(self.net, blob, "the board filter")
        except (cv2.error, ValueError) as exc:
            # Off for the rest of the round rather than taking the round with
            # it, and said once: a filter that quietly stops filtering looks
            # exactly like one that was never on.
            self.failed = str(exc)
            self.net = None
            return list(tsums)
        if logits.ndim != 2 or logits.shape[0] != len(batch):
            self.failed = f"model returned {logits.shape} for {len(batch)} crops"
            self.net = None
            return list(tsums)
        logits = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        drop = {i for i, p in zip(idx, prob[:, 1]) if float(p) < self.floor}
        self.dropped += len(drop)
        return [t for i, t in enumerate(tsums) if i not in drop]

    def summary(self) -> str:
        if not self.seen:
            return ""
        out = (f"board filter dropped {self.dropped}/{self.seen} detections "
               f"({self.dropped / self.seen:.0%}) below {self.floor:.2f}")
        if self.failed:
            out += f" -- SWITCHED OFF mid-round: {self.failed}"
        return out


class ChainModel:
    """Ranks candidate chains by how many members the GAME will accept.

    The bot has always sorted chains by ``(is_base, len)`` -- longest wins.
    The proposal corpus says that is the wrong end of the board to optimise:
    of 7,352 proposed members, the game accepts 97.9% of first members and
    19.1% of sevenths, and once it refuses one it refuses 86% of what follows.
    A shorter chain taken whole can be worth more than a long one taken a
    third of.

    So each candidate is scored as **expected accepted members** -- the sum of
    a per-member acceptance probability -- and the best of those is played.

    WHAT IS MEASURED, AND WHAT IS NOT
    ---------------------------------

    * The per-member model reads **0.876 AUC** on rounds it never trained on,
      against **0.500** for the `adjacency` rule it replaces -- which is not a
      failure to measure `adjacency` but the whole point: every member it
      proposes passed its own test, so it scores them all alike and has
      nothing to rank by.
    * The expected total is **calibrated**: over 597 held-out presses it
      predicted 2.274 accepted against 2.310 actually accepted, a bias of
      -0.036, tracking across every band.
    * Re-ranking changes the pick on 23.8% of presses and is worth
      **+0.109 accepted members per press** averaged over all of them. That
      survives scoring with a SECOND, independently trained model
      (+0.457 against the choosing model's +0.467), so it is not the winner's
      curse of taking an argmax over a noisy estimate.

    **It has never been played.** +0.109 on 2.31 is +4.7%, and the corpus's own
    spread needs ~136 rounds an arm to resolve that on accepted-per-press and
    ~204 on `cleared`. Ships off, one line to revert, exactly like everything
    else here -- and unlike the board filter, which had 0.998 AUC offline and
    lost 143 rounds of play.
    """

    def __init__(self, path, bonus: float = 0.0):
        meta = json.loads(Path(str(path)).with_suffix(".json")
                          .read_text(encoding="utf-8"))
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.features = list(meta["features"])
        self.mu = np.asarray(meta["mu"], np.float32)
        self.sd = np.asarray(meta["sd"], np.float32)
        #: Added per member, so a preference for length can be mixed back in.
        self.bonus = float(bonus)
        self.seen = self.moved = 0
        self.failed = ""
        _warm_up_rows(self.net, len(self.features))

    def rank(self, chains, tsums, radius: float, fever: bool):
        """`chains` best-first by expected accepted members.

        Returns the list unchanged on any failure. A ranker that quietly
        stops ranking looks exactly like one that was never on, so a failure
        switches it off for the round and says so once.
        """
        if self.net is None or len(chains) < 2:
            return chains
        pts = np.array([[t.x, t.y] for t in tsums], np.float64)
        lab = _cluster_lab(tsums)
        rows, spans = [], []
        for c in chains:
            f = _chain_rows(tsums, pts, lab, radius, list(c.nodes),
                            fever, c.is_base, self.features)
            spans.append(len(f))
            if len(f):
                rows.append(f)
        if not rows:
            return chains
        blob = (np.concatenate(rows) - self.mu) / self.sd
        try:
            logit = _forward(self.net, np.ascontiguousarray(
                blob, dtype=np.float32), "the chain ranker")
        except (cv2.error, ValueError) as exc:
            self.failed = str(exc)
            self.net = None
            return chains
        prob = 1.0 / (1.0 + np.exp(-logit.reshape(-1)))
        out, at = [], 0
        for c, n in zip(chains, spans):
            total = float(prob[at:at + n].sum()) + self.bonus * len(c) if n else 0.0
            at += n
            out.append((c, total))
        # `is_base` still leads: clearing the equipped character is what
        # charges the skill, and that is a rule about the ROUND rather than
        # about this press. Within it, expected accepted decides.
        ranked = [c for c, _ in sorted(out, key=lambda kv: (kv[0].is_base, kv[1]),
                                       reverse=True)]
        self.seen += 1
        if ranked and chains and list(ranked[0].nodes) != list(chains[0].nodes):
            self.moved += 1
        return ranked

    def summary(self) -> str:
        if not self.seen:
            return ""
        out = (f"chain ranker re-picked {self.moved}/{self.seen} presses "
               f"({self.moved / self.seen:.0%})")
        if self.failed:
            out += f" -- SWITCHED OFF mid-round: {self.failed}"
        return out


def _warm_up_rows(net, width: int) -> None:
    """Size a row-shaped net's buffers at the cap, before any board.

    Same rule and same reason as :func:`_warm_up`, which shapes its zeros like
    a batch of crops; this one shapes them like a batch of feature rows.
    """
    net.setInput(np.zeros((CHARACTER_BATCH, int(width)), np.float32))
    net.forward()


def _cluster_lab(tsums):
    """Each tsum's cluster colour in Lab, for the ranker's colour features.

    NOT `_face_lab`, which already exists and reads patches out of a frame.
    Naming this one the same shadowed it and broke six unrelated tests -- the
    second shadowing bug in two days, after a dict-valued `experiments`
    property hid a set-valued one and made every experiment read as armed.

    Read off `Tsum.colour`, which is the cluster's own BGR, rather than
    re-cutting the frame: the ranker runs on every candidate of every frame
    and a per-tsum patch read would cost more than the model does.
    """
    bgr = np.asarray([[t.colour for t in tsums]], np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0].astype(np.float32)


def _chain_rows(tsums, pts, lab, radius, nodes, fever, is_base, want):
    """One feature row per member after the head, in `want`'s order.

    `want` comes from the model's own sidecar, so a model trained on a
    different set of columns cannot be fed this one's -- the order is read
    from the file rather than assumed to match.
    """
    if len(nodes) < 2:
        return np.zeros((0, len(want)), np.float32)
    head = nodes[0]
    rows, path = [], 0.0
    for k in range(1, len(nodes)):
        i, prev = nodes[k], nodes[k - 1]
        vx, vy = pts[i][0] - pts[prev][0], pts[i][1] - pts[prev][1]
        d_prev = float(math.hypot(vx, vy))
        path += d_prev
        turn = 1.0
        if k >= 2:
            ux, uy = pts[prev][0] - pts[nodes[k - 2]][0], pts[prev][1] - pts[nodes[k - 2]][1]
            nu, nv = math.hypot(ux, uy), d_prev
            if nu > 1e-6 and nv > 1e-6:
                turn = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
        near = int(np.count_nonzero(
            np.hypot(pts[:, 0] - pts[i][0], pts[:, 1] - pts[i][1])
            < 1.5 * radius) - 1)
        row = {
            "position": float(k),
            "dist_prev_r": d_prev / radius,
            "dist_head_r": float(math.hypot(pts[i][0] - pts[head][0],
                                            pts[i][1] - pts[head][1])) / radius,
            "abs_dx_r": abs(vx) / radius, "dy_r": vy / radius,
            "turn_cos": turn,
            "blockers_prev": float(_on_segment(tsums, prev, i, radius)),
            "blockers_head": float(_on_segment(tsums, head, i, radius)),
            "density": near / 10.0,
            "visible": tsums[i].r / radius,
            "board_n": len(tsums) / 50.0,
            "lab_prev": float(np.linalg.norm(lab[i] - lab[prev])) / 40.0,
            "lab_head": float(np.linalg.norm(lab[i] - lab[head])) / 40.0,
            "same_kind_head": 1.0 if tsums[i].kind == tsums[head].kind else 0.0,
            "chain_len": float(len(nodes)),
            "path_so_far_r": path / radius,
            "fever": 1.0 if fever else 0.0,
            "is_base": 1.0 if is_base else 0.0,
        }
        rows.append([row.get(name, 0.0) for name in want])
    return np.asarray(rows, np.float32)


def _on_segment(tsums, a, c, radius, block=1.25):
    """How many other tsums lie across the line from `a` to `c`.

    The same question `adjacency`'s `block` asks and the same arithmetic as
    `scripts/proposal_dataset.blockers`, because the model was trained on that
    one -- a different definition here would feed it a feature it has never
    seen under a name it has.
    """
    ax, ay, cx, cy = tsums[a].x, tsums[a].y, tsums[c].x, tsums[c].y
    vx, vy = cx - ax, cy - ay
    span = vx * vx + vy * vy
    if span <= 1e-6:
        return 0
    n = 0
    for i, t in enumerate(tsums):
        if i in (a, c):
            continue
        s = ((t.x - ax) * vx + (t.y - ay) * vy) / span
        if not 0.0 < s < 1.0:
            continue
        if math.hypot(t.x - (ax + s * vx), t.y - (ay + s * vy)) < block * radius:
            n += 1
    return n


class CharacterModel:
    """Names the character in each face crop, where it is sure enough.

    `kind` is a per-frame k-means cluster: it means nothing between frames and
    agrees with the game about a confirmed partner 37% of the time, because
    face colour does not identify a Tsum Tsum character -- many of them share
    one. A classifier trained on crops the game's own marks helped label reads
    ~95% on held-out sessions.

    **Hybrid, not a replacement, and that is the whole safety of it.** The
    model knows the characters somebody labelled and no others, and a real
    board is full of the rest. A crop it is unsure of keeps the k-means `kind`
    it already had, so an unknown character plays exactly as it does today and
    a known one plays better. Replacing `kind` outright would make every
    unlabelled character unchainable and stop the bot playing half the board.

    Costs ~11ms for a 46-tsum board through `cv2.dnn`, against a ~100ms
    decision. No new dependency: OpenCV is already bundled.
    """

    #: Least of a tsum that must be showing before the model is asked at all.
    #: See :data:`CHARACTER_MIN_VISIBLE`.
    #:
    #: **This is the training set's own rule, which the runtime never knew.**
    #: `crops.py` wrote its crops at `v >= 0.55` and every one of the 3,791
    #: labelled crops is above it, while the median detection on a real board
    #: shows 0.41 -- so 78% of the board is a picture the model has not seen a
    #: single example of. Asked anyway, it does not hedge: measured against the
    #: game's own marks on 94 sessions no labelled crop came from, it agrees
    #: with the character the game confirmed 74% of the time above 0.55 and
    #: 0-28% below it, against a 12.5% chance rate -- at a mean confidence of
    #: 0.75-0.88 the whole way down. Confidence does not fall where competence
    #: does, so `confidence` alone cannot catch this and never could.
    #:
    #: Below the floor the tsum keeps its k-means `kind`, which agrees with the
    #: game 37% of the time. That is not good, but it is more than twice what
    #: the model manages there, and it does not come dressed as certainty.
    def __init__(self, path, confidence: float = 0.85,
                 min_visible: float = CHARACTER_MIN_VISIBLE):
        meta_path = Path(str(path)).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.classes = list(meta["classes"])
        self.size = int(meta.get("size", 96))
        # Before the first board, not on it: see `_warm_up`.
        _warm_up(self.net, self.size)
        self.mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
        self.confidence = float(confidence)
        self.min_visible = float(min_visible)
        #: Counted so the end-of-round line can say how much of the board was
        #: never offered to the model, rather than lumping it in with crops it
        #: saw and declined.
        self.buried = 0
        #: Counted per round and reported, because a model that names nothing
        #: is indistinguishable from one that is switched off.
        self.named = 0
        self.seen = 0
        #: Set when the net is switched off mid-round, and reported at the end.
        self.failed = ""

    def _crop(self, bgr, t, radius):
        """The picture the training set held, or None.

        Delegates to :func:`_character_crop`, which `RejectModel` also uses:
        both are trained on crops written by the same extractor, so a change to
        the window, the 64px round trip or the edge refusal must reach both or
        neither.
        """
        return _character_crop(bgr, t, radius, self.size)

    def probabilities(self, bgr: np.ndarray, tsums: Sequence["Tsum"],
                      radius: float) -> tuple[list[int], np.ndarray]:
        """Which crops the model could read, and what it thinks each one is.

        Split out of :meth:`apply` so that offline analysis scores the same
        pictures play does. The crop rules here are fussy and easy to
        re-implement subtly differently -- window, the 64px round trip, the
        refusal at the frame edge, the fixed batch -- and an evaluation that
        got any of them wrong would be measuring a model the bot never runs.
        """
        if self.net is None:
            return [], np.zeros((0, 0), np.float32)
        # The visibility floor first, so a buried tsum is never cropped, never
        # batched and never scored. Refusing it here rather than discarding its
        # answer later is what keeps the two counters meaning different things.
        showing = [(i, t) for i, t in enumerate(tsums)
                   if radius <= 0 or t.r / radius >= self.min_visible]
        self.buried += len(tsums) - len(showing)
        usable = [(i, self._crop(bgr, t, radius)) for i, t in showing]
        usable = [(i, c) for i, c in usable if c is not None]
        self.seen += len(tsums)
        if not usable:
            return [], np.zeros((0, 0), np.float32)
        blob = np.stack([c for _, c in usable]).astype(np.float32) / 255.0
        blob = (blob - self.mean) / self.std
        # CONTIGUOUS, and not as a tidiness point: `transpose` returns a view
        # with permuted strides, and handing one to `cv2.dnn` reaches native
        # code that assumes packed memory. When that goes wrong it does not
        # raise -- the process dies with no Python traceback, which is exactly
        # the shape of a crash this rule was reported to cause on its first
        # round. Unreproduced offline, so this is a hardening rather than a
        # fix, but the copy costs ~0.5ms on a 46-crop board and a silent
        # process death costs a round.
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)
        try:
            # CAPPED batch, never a GROWING one -- see CHARACTER_BATCH. The
            # buffers were sized at the cap by `_warm_up` when the net loaded,
            # so a real board only ever shrinks them.
            logits = _forward(self.net, blob, "the character model")
        except (cv2.error, ValueError) as exc:
            # Switch the model off for the rest of the round rather than
            # taking the round down with it. Said once, loudly: a model that
            # quietly stops naming looks identical to one that was never on.
            self.failed = str(exc)
            self.net = None
            return [], np.zeros((0, 0), np.float32)
        if logits.ndim != 2 or logits.shape[0] != len(usable):
            self.failed = f"model returned {logits.shape} for {len(usable)} crops"
            self.net = None
            return [], np.zeros((0, 0), np.float32)
        logits = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        return [i for i, _ in usable], prob

    def apply(self, bgr: np.ndarray, tsums: Sequence["Tsum"], radius: float) -> int:
        """Rewrite `kind` to a character id where the model is confident."""
        usable, prob = self.probabilities(bgr, tsums, radius)
        named = 0
        for i, p in zip(usable, prob):
            best = int(p.argmax())
            if float(p[best]) >= self.confidence:
                tsums[i].kind = CHARACTER_KIND + best
                named += 1
        self.named += named
        return named

    def summary(self) -> str:
        if not self.seen:
            return ""
        out = (f"character model named {self.named}/{self.seen} detections "
               f"({self.named / self.seen:.0%}); the rest kept their colour "
               f"cluster")
        if self.buried:
            out += (f", {self.buried} of them never asked "
                    f"({self.buried / self.seen:.0%} under "
                    f"{self.min_visible:.2f} visible)")
        if self.failed:
            out += f" -- SWITCHED OFF mid-round: {self.failed}"
        return out


def _base_from_faces(bgr: np.ndarray, tsums: Sequence["Tsum"], radius: float,
                     icon_lab) -> Optional[int]:
    """Which `kind` is the equipped tsum, when the cluster ids are not k-means'.

    :func:`read_base_kind` matches the skill icon against the palette centres
    and returns one of their indices. That is the right answer while `kind` IS
    a palette index; it is meaningless once :func:`_recolour` has regrouped the
    board under its own numbering.

    The icon's Lab colour survives both, and is the closest thing this project
    has to naming the character -- read off raw pixels of a fixed sprite in a
    fixed place. So the group whose faces sit nearest it is the equipped one.
    Medians rather than means: one member landing on a neighbour's face should
    not drag a whole group's colour.
    """
    if icon_lab is None or not len(tsums):
        return None
    faces = _face_lab(bgr, tsums, radius)
    icon = np.asarray(icon_lab, np.float32)
    best, best_d = None, float("inf")
    for kind in {t.kind for t in tsums}:
        sel = faces[[i for i, t in enumerate(tsums) if t.kind == kind]]
        if not len(sel):
            continue
        d = float(np.linalg.norm(np.median(sel, axis=0) - icon))
        if d < best_d:
            best, best_d = kind, d
    return best


def purity_filter(bgr: np.ndarray, tsums: Sequence[Tsum], nodes: Sequence[int],
                  radius: float, tol: float) -> list[int]:
    """Drop chain members that don't actually look like the rest of the chain.

    A chain is only worth dragging if every stop really is the same character.
    One misclassified member turns a 3-chain into a 2-link the game ignores --
    the drag runs, nothing clears, and the board is unchanged. Cheap to check
    directly: sample each member's colour off the image and discard anything
    far from the chain's median.
    """
    if len(nodes) < 2:
        return list(nodes)

    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    sample = max(2, int(radius * 0.45))
    feats = []
    for i in nodes:
        m = np.zeros(bgr.shape[:2], np.uint8)
        cv2.circle(m, (int(tsums[i].x), int(tsums[i].y)), sample, 1, -1)
        px = lab[m.astype(bool)]
        feats.append(np.median(px, axis=0) if px.size else np.full(3, 1e6, np.float32))

    feats = np.asarray(feats, np.float32)
    centre = np.median(feats, axis=0)
    keep = [n for n, f in zip(nodes, feats) if float(np.linalg.norm(f - centre)) <= tol]
    return keep


def _face_coverage(tsums: Sequence[Tsum], radius: float, shape) -> float:
    """What fraction of the board rect the detected faces add up to.

    The check that a radius reading is self-consistent, and the one thing the
    detection count cannot tell you on its own. Count and radius are not
    independent: a rect only holds so much. ~40 faces at 26px cover about a
    third of it; the same rect read at half scale reports ~60 faces at 12px,
    which covers a twelfth. So the two failures look identical by count -- both
    are "a plausible number of tsums" -- and completely different by area.

    Measured over the frames a round would sample, split by whether the radius
    came out near the truth:

        good reads       coverage p10 0.22, p50 0.30, p90 0.38
        collapsed reads  coverage p10 0.08, p50 0.18, p90 0.23

    At 0.25 that keeps 39 of 48 good frames and rejects 48 of 51 collapsed
    ones. It is a floor, not a band: over-reading the radius is what
    `open_ratio` guards, and a frame cannot cover more than the rect anyway.

    Note the dependence on `--bowl-reject`, which removes detections and so
    lowers coverage. The figures above are at 40, the shipped setting; turning
    it off raises coverage and makes this floor looser, never tighter.
    """
    if not tsums or radius <= 0:
        return 0.0
    area = float(shape[0] * shape[1])
    return (len(tsums) * math.pi * radius * radius / area) if area else 0.0


def _snap_to_mask(solid_dt: np.ndarray, cx: float, cy: float,
                  reach: int) -> Optional[tuple[float, float, float]]:
    """Move a peak to the deepest nearby point that is genuinely on the mask.

    Returns (x, y, depth), or None when there is no mask at all within `reach`
    -- which means the peak was an artefact of mask healing, not a tsum.
    """
    h, w = solid_dt.shape
    x, y = int(round(cx)), int(round(cy))
    y0, y1 = max(0, y - reach), min(h, y + reach + 1)
    x0, x1 = max(0, x - reach), min(w, x + reach + 1)
    win = solid_dt[y0:y1, x0:x1]
    if win.size == 0 or win.max() <= 0:
        return None
    dy, dx = np.unravel_index(int(win.argmax()), win.shape)
    return float(x0 + dx), float(y0 + dy), float(win.max())


def detect(
    bgr: np.ndarray,
    *,
    k: int = 12,
    radius: Optional[float] = None,
    include_dark: bool = False,
    dark_l: float = 60.0,
    merge: bool = False,
    heal_frac: float = 0.9,
    open_ratio: float = 2.2,
    recolour: float = 0.0,
    kinds: int = 0,
    bowl_reject: float = 0.0,
    floor_frac: float = 0.42,
    hole_frac: float = 0.8,
    palette: Optional[np.ndarray] = None,
    scale: float = 1.0,
    fit_effort: int = 1,
    debug_dir: Optional[Path] = None,
) -> tuple[list[Tsum], float, np.ndarray]:
    """Locate every tsum. Coordinates and radius come back in `bgr` pixels.

    `scale` < 1 runs the whole pipeline on a downscaled copy: cost is roughly
    quadratic in resolution, and tsums are big enough that half-size still
    separates them. Measured on a 523x542 board: 49ms at 1.0 -> 13ms at 0.5,
    same chain, a couple of deeply buried tsums lost.

    `fit_effort` only matters when there is no `palette` to reuse, and decides
    how repeatable that fit is -- see :data:`FIT_EFFORT`.
    """
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if radius is not None:
            radius *= scale

    labels, centres = _quantise(bgr, k, palette, effort=fit_effort)
    skip = _background_clusters(labels, centres)
    # Dark pixels are two different things at once: every tsum's outline, ears
    # and shadow, AND the face of a black tsum like classic Mickey. Treating
    # them as one cluster fuses the whole board; discarding them loses Mickey
    # entirely. So they get split off and processed under their own rules below.
    dark = {i for i, c in enumerate(centres) if _lightness(c) < dark_l} - skip
    if not include_dark:
        skip |= dark
        dark = set()

    if merge:
        # Dark is excluded on both sides: an outline must never adopt a face,
        # and a black tsum must never be dissolved into its neighbour.
        labels, _ = _merge_enclosed(labels, centres, skip | dark)

    bright = [i for i in range(len(centres)) if i not in skip and i not in dark]

    kernel = np.ones((3, 3), np.uint8)

    def _mask(kind: int, open_px: int = 0) -> np.ndarray:
        m = (labels == kind).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        if open_px:
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ek)
        return m

    def _dt(mask: np.ndarray, max_hole: Optional[float]) -> np.ndarray:
        return cv2.distanceTransform(_fill_holes(mask, max_hole), cv2.DIST_L2, 5)

    def _estimate(dts: Iterable[np.ndarray]) -> float:
        # For a disc of radius r the area with dt > t is ((r-t)/r)^2, so the
        # 99th percentile of dt across the board sits at ~0.9r. Fully visible
        # tsums dominate that tail; buried ones only drag it down slightly.
        pool = [d[d > 0].ravel() for d in dts]
        pool = np.concatenate(pool) if pool else np.zeros(0)
        return float(np.percentile(pool, 99.0) / 0.9) if pool.size else 30.0

    bright_masks = {k: _mask(k) for k in bright}

    # Board that isn't caught by the border-ring test -- the bowl often splits
    # into a lit rim and a darker centre, and only one of them touches the
    # crop edge. The survivor then poses as a tsum colour, and because it is a
    # wide open region its distance transform runs far deeper than any face,
    # dragging the radius estimate up with it. An inflated radius then raises
    # every threshold downstream and quietly suppresses detection board-wide.
    #
    # Depth is the giveaway: a tsum cluster can only be about as deep as a tsum
    # is round, so a cluster several times deeper than the typical one is open
    # board, whatever colour it happens to be.
    probe = {k: _dt(m, None) for k, m in bright_masks.items()}
    depth = {k: float(np.percentile(d[d > 0], 99)) for k, d in probe.items() if (d > 0).any()}
    if depth:
        typical = float(np.median(list(depth.values())))
        open_board = {k for k, v in depth.items() if v > open_ratio * typical}
        # Never let this empty the board: if everything looks like an outlier,
        # the median is the unreliable part, so change nothing.
        if open_board and len(open_board) < len(depth):
            skip |= open_board
            for k in open_board:
                bright_masks.pop(k, None)
                probe.pop(k, None)
            bright = [k for k in bright if k not in open_board]

    if radius is None:
        # Chicken and egg: sizing a hole needs the radius, and the radius comes
        # from a hole-free mask. Fill everything for a first estimate, then redo
        # with a sane cap. Bright clusters only -- the dark mask is every tsum's
        # outline fused together and would poison the estimate.
        radius = _estimate(probe.values())

    # A face is often not a solid disc: Pluto's muzzle and Piglet's snout are
    # big pale patches that cut the face colour down to a crescent, and a
    # crescent has no deep distance-transform peak -- those tsums went missing
    # entirely. Hole-filling can't help when the patch runs off the face edge
    # and so isn't enclosed, but a closing this wide bridges it either way.
    # Safe against merging two same-colour neighbours: closing fuses the masks
    # but leaves both peaks, which is exactly what peak-finding expects.
    heal = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(round(radius * heal_frac)) | 1),) * 2)

    max_hole = math.pi * radius * radius * hole_frac
    work: list[tuple[int, np.ndarray, float]] = [
        (k, _dt(cv2.morphologyEx(m, cv2.MORPH_CLOSE, heal), max_hole), radius * floor_frac)
        for k, m in bright_masks.items()
    ]

    # The healed mask is good for *counting* tsums but not for aiming at them:
    # the closing above invents up to ~heal_frac*r of mask that no tsum
    # occupies, and a peak can settle in that invented area -- a coordinate
    # floating in the gap between tsums rather than on one. Measured on a real
    # board, 19 of 55 centres landed on a colour that wasn't their own, which
    # is a drag the game would simply not register.
    #
    # So peaks get snapped back onto solid ground: the deepest point of the
    # *un-healed* mask near the peak. Same tsum, but a coordinate guaranteed to
    # sit on real pixels of that colour.
    # Note: raw mask, NOT hole-filled. Filling would let a centre land on an
    # eye or a muzzle -- still on the tsum, but not on a pixel of the colour
    # we think we are clicking. Snapping to raw colour makes the coordinate
    # literally verifiable: the pixel under the cursor is that tsum.
    solid = {k: cv2.distanceTransform(m, cv2.DIST_L2, 5) for k, m in bright_masks.items()}

    # Dark clusters (off by default -- see `include_dark`). Two filters try to
    # find a black-faced tsum without the outline mesh dragging the whole board
    # in with it: an opening wide enough to erase strokes but not faces, and a
    # higher peak floor -- an ear is ~0.4r across, so anything that deep in the
    # dark mask is a face, not trim. Kept as a best-effort pass, not a solved
    # one: on a sparsely-filled board the gaps between tsums can be wide enough
    # to clear both filters, reading as phantom tsums sitting in open board --
    # measured as high as 13 phantoms on one real frame. Real black-faced
    # detections have never outnumbered false ones in testing.
    for kind in dark:
        opened = _mask(kind, open_px=max(1, int(round(radius * 0.18))))
        work.append((kind, _dt(opened, max_hole), radius * 0.62))

    # Peak depth is carried beside each Tsum rather than read back off `t.r`,
    # because `t.r` is clipped at the radius: every fully visible tsum ties
    # there, and the sort below then falls back to insertion order -- i.e. to
    # whichever cluster id k-means happened to number first.
    found: list[tuple[float, Tsum]] = []
    reach = max(2, int(round(radius)))
    for kind, dt, floor in work:
        colour = _lab_to_bgr(centres[kind])
        ground = solid.get(kind)
        for cx, cy, score in _peaks(dt, radius, floor=floor):
            if ground is not None:
                snapped = _snap_to_mask(ground, cx, cy, reach)
                if snapped is None:
                    # Nothing of this colour within a radius: the peak lives
                    # entirely in healed-in area, so there is no tsum here.
                    continue
                cx, cy, score = snapped
            found.append((score, Tsum(cx, cy, min(score, radius), kind, colour)))

    # A blob can peak twice on one tsum (a highlight splits the mask, or a
    # loose peak floor finds two maxima on one face). Keep only the deepest
    # peak in a neighbourhood. 1.6r is set by geometry, not taste: two tsums
    # that genuinely touch sit ~2.44r apart, so anything closer than 1.6r is
    # two readings of one tsum. Measured over 14 real boards, 1.6 removes every
    # impossible overlap (223 -> 0) while keeping 730 detections.
    #
    # Sorting on unclipped depth rather than on `t.r` scores identically over
    # the ten labelled boards (f1 0.762 either way); it is here to make the
    # winner of a collision follow from the evidence instead of from cluster
    # numbering, which is not stable between fits.
    #
    # DO NOT normalise depth per kind before this sort. An external HSV-window
    # reader found it had to -- its hand-authored windows nest, so a window
    # that is a superset of another scores higher on the narrow character's own
    # sprites and wins every collision, erasing that character board-wide. That
    # failure cannot happen here: k-means centres partition colour space, so no
    # cluster is a superset of another and depth is already comparable. Ported
    # over anyway and measured, it is a straight loss -- f1 0.762 -> 0.748 as a
    # global sort key, and 0.762 -> 0.749 restricted to colliding pairs of
    # different kinds. It lifts every peak of a shallow cluster, and the
    # shallow clusters are mostly the spurious ones, so phantoms take slots off
    # real tsums (boards 4 and 5 lost 4 real detections each).
    found.sort(key=lambda p: -p[0])
    kept: list[Tsum] = []
    for _, t in found:
        if all((t.x - o.x) ** 2 + (t.y - o.y) ** 2 > (radius * 1.6) ** 2 for o in kept):
            kept.append(t)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = np.zeros((*labels.shape, 3), np.uint8)
        for i, c in enumerate(centres):
            vis[labels == i] = (40, 40, 40) if i in skip else _lab_to_bgr(c)
        for i in dark:
            vis[labels == i] = (0, 0, 255)  # what the dark pass has to work with
        cv2.imwrite(str(debug_dir / "clusters.png"), vis)

    if bowl_reject > 0 and kept:
        # A detection that landed on the bowl rather than on a tsum carries the
        # bowl's colour. Measured over the ten labelled boards, real tsums sit a
        # median 173 Lab from the board's own colour and phantoms 76, with the
        # real p10 at 74 -- separated enough to act on, overlapping enough that
        # the cutoff buys phantoms with real tsums rather than for free.
        #
        #   off   precision 0.675  recall 0.874  f1 0.762
        #   40    precision 0.734  recall 0.844  f1 0.785
        #   60    precision 0.773  recall 0.810  f1 0.791
        #   80    precision 0.813  recall 0.724  f1 0.766
        #
        # 40-60 is a plateau, not a tuned edge. OFF BY DEFAULT: the reading
        # above is offline f1, and what matters live is whether the drags the
        # bot loses were ones the game would have cleared. That is the A/B.
        bowl = board_colours(bgr, centres)
        if len(bowl):
            faces = _face_lab(bgr, kept, radius)
            far = np.linalg.norm(faces[:, None] - bowl[None], axis=2).min(axis=1)
            kept = [t for t, d in zip(kept, far) if d >= bowl_reject]

    if recolour > 0:
        kept = _recolour(bgr, kept, radius, recolour)
    if kinds > 0:
        kept = _regroup(bgr, kept, radius, kinds)

    if scale != 1.0:
        radius /= scale
        for t in kept:
            t.x, t.y, t.r = t.x / scale, t.y / scale, t.r / scale

    return kept, radius, centres


# --------------------------------------------------------------------------
# base tsum (the skill icon bottom-left = the type you're actually playing)
# --------------------------------------------------------------------------
# Centre and radius of the skill icon's face, as fractions of a 540x960 screen.
# Sampled from the middle of the glyph, well inside the button plate.
BASE_ICON = (0.213, 0.852, 0.052)  # cx, cy, r


def read_base_kind(
    full_bgr: np.ndarray,
    centres: np.ndarray,
    *,
    spec: Optional[str] = None,
    debug_dir: Optional[Path] = None,
    out: Optional[dict] = None,
) -> tuple[Optional[int], float]:
    """Identify which colour cluster the bottom-left skill icon shows.

    That icon is the tsum you equipped, so it's the one whose chains actually
    charge the skill gauge -- which makes it the chain worth finding, not
    whichever type happens to have the longest run on the board.

    Returns (cluster index, Lab distance). A large distance means no cluster
    really matched and the caller should not trust it.

    `out` receives the icon's own median Lab colour, which is the closest
    thing this project has to the *name* of the equipped tsum. The cluster
    index cannot serve: it is a per-frame k-means id and means nothing between
    frames, let alone between sessions. The icon colour is read off raw pixels
    of a fixed sprite in a fixed place, so two collections of the same
    character land within a few Lab units of each other and two different
    characters do not. The collector writes it down for exactly that reason --
    three rounds running, which tsum was equipped was knowable only by asking
    the player.
    """
    h, w = full_bgr.shape[:2]
    if not spec:
        spec = _layout(full_bgr.shape).get("base")
    if spec:
        cx, cy, r = (float(v) for v in spec.split(","))
    else:
        cx, cy, r = BASE_ICON[0] * w, BASE_ICON[1] * h, BASE_ICON[2] * w

    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(w, cx + r)), int(min(h, cy + r))
    patch = full_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None, float("inf")

    lab = cv2.cvtColor(cv2.GaussianBlur(patch, (5, 5), 0), cv2.COLOR_BGR2LAB)
    disc = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(disc, (patch.shape[1] // 2, patch.shape[0] // 2),
               max(2, int(min(patch.shape[:2]) * 0.35)), 1, -1)

    # Median, not mean: the glyph has eyes and highlights, and a mean would
    # blend them into a colour no cluster has.
    sample = lab[disc.astype(bool)].reshape(-1, 3).astype(np.float32)
    if sample.size == 0:
        return None, float("inf")
    icon_lab = np.median(sample, axis=0)
    if out is not None:
        out["icon_lab"] = [round(float(v), 1) for v in icon_lab]

    d = np.linalg.norm(centres - icon_lab[None, :], axis=1)
    best = int(d.argmin())

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(debug_dir / "base_icon.png"), vis)

    return best, float(d[best])


# --------------------------------------------------------------------------
# graph + chain search
# --------------------------------------------------------------------------
def adjacency(tsums: Sequence[Tsum], radius: float, link: float = 1.5,
              block: float = 0.75, link_px: Optional[float] = None) -> list[set[int]]:
    """Same-kind tsums count as linked when you could actually drag between them.

    Distance alone is not that test. Two same-kind tsums a comfortable gap apart
    usually have a *different* tsum sitting between them, and the game won't
    join those -- a pure radius threshold invents chains you cannot play. So a
    candidate pair is also rejected when any other tsum's centre lies within
    `block` radii of the segment joining them.

    `link` is in *detected* diameters, which are not true diameters: the
    distance transform measures the visible inscribed radius, and a half-buried
    tsum reads smaller than it is. Measured across ten real boards, the median
    gap to the nearest tsum of any kind is 1.22 detected diameters -- and in a
    pile that dense, your nearest neighbour is one you are touching. So 1.22 is
    what contact actually looks like, and the default of 1.5 is contact plus
    about 23% slack. Push it to 2.0 and you are allowing two thirds of a tsum
    of clear space between "linked" tsums, which chains ones that visibly are
    not connected.
    """
    n = len(tsums)
    adj: list[set[int]] = [set() for _ in tsums]
    if n < 2:
        return adj

    pts = np.array([[t.x, t.y] for t in tsums], np.float64)
    # Absolute pixels when given, and that is the preferred path. A threshold
    # in "detected diameters" inherits every wobble in the radius estimate:
    # the same board measured at r=24 vs r=30 moves the cutoff by 27%, which
    # was enough to flip real links from accepted to rejected between runs.
    # Measured straight off hand-labelled links, the spacing of tsums a human
    # chains is tight in pixels (median 74, p90 95) and only looks noisy once
    # divided by an estimate. The emulator runs at a fixed resolution, so
    # pixels are the stable unit here.
    reach = (link_px ** 2) if link_px else ((link * 2 * radius) ** 2)
    block_d2 = (block * radius) ** 2

    for i in range(n):
        for j in range(i + 1, n):
            if tsums[i].kind != tsums[j].kind:
                continue
            ab = pts[j] - pts[i]
            seg2 = float(ab @ ab)
            if seg2 > reach:
                continue

            # Perpendicular distance from every other centre to segment i-j,
            # but only counting the ones that actually sit between the two --
            # a tsum off the end of the segment blocks nothing.
            t = ((pts - pts[i]) @ ab) / seg2
            interior = (t > 0.2) & (t < 0.8)
            interior[i] = interior[j] = False
            if interior.any():
                proj = pts[i] + t[interior, None] * ab
                if (((pts[interior] - proj) ** 2).sum(axis=1) < block_d2).any():
                    continue

            adj[i].add(j)
            adj[j].add(i)
    return adj


def blob_adjacency(labels: np.ndarray, tsums: Sequence[Tsum], radius: float, *,
                   grow: float = 0.9, reach: float = 2.2) -> list[set[int]]:
    """Link two tsums when their colour blobs actually join up, not when their
    centres are close enough.

    :func:`adjacency` models contact as "near, with nothing on the segment
    between". That needs the blocking test precisely because distance alone
    misreads two cases: sprites are not circles, so two tsums can overlap at
    the ears while the centre line runs through the gap between them; and a
    third tsum sitting between two others bridges the line, so a real gap reads
    as contact. Testing the mask directly answers the question the segment test
    approximates -- is there a continuous run of this colour joining these two
    specifically -- whatever shape they are and whatever sits nearby.

    `grow` is the part that is not optional. Clusters here are *face* colour,
    and two touching sprites still have an outline and a body between their
    faces: measured over the labelled boards, the median gap between the face
    blobs of two tsums a human chained is 23.4px against a detected radius of
    ~24, and only 2.2% of those pairs share a connected component at all. So
    the mask is dilated outward to approximate the sprite's full extent before
    connectivity is tested. Below ~0.6r almost nothing links; the score
    saturates at 0.9r and is flat to 1.2r, so 0.9 is the middle of a plateau
    rather than a tuned edge.

    Replayed through the 97 hand-drawn links in `score`, against the same
    detections: 86.6% accepted, versus 76.3% for adjacency() at --link-px 105
    and an 84.5% ceiling for adjacency() at any distance. It does that with a
    sparser graph than the loose setting it beats -- 291 edges against 387 at
    --link-px 150 -- so it is accepting more of the links a human drew while
    inventing fewer of its own.

    Costs ~60ms on a 62-tsum board against ~1.3ms for adjacency(), which is why
    this is opt-in: it roughly doubles the per-frame think time. Worth it when
    reaction speed is not the binding constraint, not when it is.

    NOTE ON THE SCORE: the labels record only links a human *did* draw, so
    acceptance is a positive-only measure and rises for free as a rule accepts
    more -- the same trap documented on :func:`_recolour`. The edge count above
    is the control, and this rule wins on both at once, which is real evidence
    but not the end-to-end A/B. Treat 86.6% as promising, not settled.
    """
    n = len(tsums)
    adj: list[set[int]] = [set() for _ in tsums]
    if n < 2:
        return adj

    h, w = labels.shape
    span = radius * reach
    g = max(1, int(round(radius * grow)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2)
    grown: dict[int, np.ndarray] = {}   # dilate each kind once, not once per pair

    for i in range(n):
        for j in range(i + 1, n):
            a, b = tsums[i], tsums[j]
            if a.kind != b.kind:
                continue
            if (a.x - b.x) ** 2 + (a.y - b.y) ** 2 > (2 * span) ** 2:
                continue
            if a.kind not in grown:
                grown[a.kind] = cv2.dilate((labels == a.kind).astype(np.uint8), kernel)

            # The question is local -- these two tsums and the corridor between
            # them -- so it gets answered in a box around the pair. Running it
            # over the whole board instead costs ~6x for the same answer.
            x0, x1 = max(0, int(min(a.x, b.x) - span)), min(w, int(max(a.x, b.x) + span) + 1)
            y0, y1 = max(0, int(min(a.y, b.y) - span)), min(h, int(max(a.y, b.y) + span) + 1)
            sub_mask = grown[a.kind][y0:y1, x0:x1]
            if sub_mask.size == 0:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            disks = ((((xx - a.x) ** 2 + (yy - a.y) ** 2) < span ** 2)
                     | (((xx - b.x) ** 2 + (yy - b.y) ** 2) < span ** 2))
            _, comp = cv2.connectedComponents(sub_mask * disks.astype(np.uint8))

            def blobs_at(t: Tsum) -> set:
                cy, cx = int(t.y) - y0, int(t.x) - x0
                win = comp[max(0, cy - 6):cy + 7, max(0, cx - 6):cx + 7]
                return set(np.unique(win)) - {0}

            if blobs_at(a) & blobs_at(b):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _components(nodes: Iterable[int], adj: list[set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    out = []
    for start in nodes:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            n = stack.pop()
            group.append(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        out.append(group)
    return out


def longest_path(component: Sequence[int], adj: list[set[int]], budget: float = 0.4,
                 tsums: Optional[Sequence[Tsum]] = None) -> list[int]:
    """Longest simple path through a component -- the longest drag you can make.

    You can only chain tsums you can drag *through* in one stroke, so component
    size is an upper bound, not the answer: a star-shaped clump of 7 still only
    scores 3. Exhaustive DFS, pruned by "even taking everything still reachable
    won't beat the best so far", with a wall-clock budget because the search is
    exponential in the worst case. Components are small (rarely past 20) so the
    budget almost never binds; when it does we return the best found so far.

    Pass `tsums` to break ties on total drag length. Node count still decides;
    this only chooses between paths that tie on it, and there are usually
    several. Without it the winner is whichever the search happened to reach
    first, which is how a chain ends up zig-zagging across a clump it could
    have walked in order -- every leg legal, but each one a long jump that the
    emulator's touch sampling is more likely to drop.
    """
    inside = set(component)
    deadline = time.perf_counter() + budget
    best: list[int] = []
    best_cost = 0.0

    def reachable(start: int, visited: set[int]) -> int:
        stack, seen = [start], {start}
        while stack:
            n = stack.pop()
            for m in adj[n]:
                if m in inside and m not in visited and m not in seen:
                    seen.add(m)
                    stack.append(m)
        return len(seen)

    def dfs(node: int, visited: set[int], path: list[int]) -> None:
        nonlocal best, best_cost
        cost = _tour_length(path, tsums) if tsums else 0.0
        if len(path) > len(best) or (tsums and len(path) == len(best) and cost < best_cost):
            best, best_cost = list(path), cost
        if time.perf_counter() > deadline:
            return
        # Ties are pruned, deliberately. Exploring them to find the shortest
        # path of maximal length is exponential, and this runs inside a loop
        # that wants ten decisions a second. Nearest-first ordering below gets
        # most of the same effect for nothing: the first maximal path the
        # search reaches is already one that took short hops.
        if len(path) + reachable(node, visited) - 1 <= len(best):
            return
        for nxt in sorted(adj[node] & inside - visited,
                          key=lambda n: (_leg(tsums, node, n) if tsums else len(adj[n]))):
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt, visited, path)
            path.pop()
            visited.remove(nxt)

    # Endpoints of a longest path tend to be low-degree, so start there: the
    # first DFS then finds a long path early and prunes everything after it.
    for start in sorted(component, key=lambda n: len(adj[n])):
        if time.perf_counter() > deadline and best:
            break
        dfs(start, {start}, [start])
        if len(best) == len(component):
            break  # can't do better than the whole component
    return best


def _leg(tsums: Sequence[Tsum], a: int, b: int) -> float:
    return math.hypot(tsums[a].x - tsums[b].x, tsums[a].y - tsums[b].y)


def orient_chain(path: Sequence[int], tsums: Sequence[Tsum],
                 first_leg_px: float = 0.0) -> list[int]:
    """Point the drag at the end whose opening hop is shortest, and trim it.

    The first leg is not like the others. The tsum you press first decides the
    character for the whole stroke, so if that opening hop is too long to
    register, nothing connects and the entire drag is wasted -- where a bad leg
    later on costs one skipped tsum and the stroke carries on. A path a-b-c-d
    is the same set of tsums dragged either way, so the direction is free: take
    whichever end opens shorter.

    With `first_leg_px` set, a chain that still opens too wide gives up its
    leading tsum rather than the drag, down to the game's three-tsum minimum.
    """
    path = list(path)
    if len(path) < 2:
        return path
    if _leg(tsums, path[-1], path[-2]) < _leg(tsums, path[0], path[1]):
        path.reverse()
    while (first_leg_px > 0 and len(path) > MIN_CHAIN
           and _leg(tsums, path[0], path[1]) > first_leg_px):
        path.pop(0)
        if _leg(tsums, path[-1], path[-2]) < _leg(tsums, path[0], path[1]):
            path.reverse()
    return path


def _nearest_neighbor_tour(members: Sequence[int], tsums: Sequence[Tsum]) -> list[int]:
    """Visit every point in `members` via greedy nearest-neighbour.

    Used for "reach" mode, where every same-kind tsum is linkable regardless of
    distance: order barely matters for legality (the drag just passes harmlessly
    over whatever's between two stops), but a short route still means a faster,
    more reliable drag than a random one -- fewer long legs for the emulator's
    touch sampling to drop, and less total time per chain.
    """
    remaining = set(members)
    start = min(remaining, key=lambda i: (tsums[i].y, tsums[i].x))
    order = [start]
    remaining.remove(start)
    while remaining:
        last = tsums[order[-1]]
        nxt = min(remaining, key=lambda i: (tsums[i].x - last.x) ** 2 + (tsums[i].y - last.y) ** 2)
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _tour_length(order: Sequence[int], tsums: Sequence[Tsum]) -> float:
    return sum(math.hypot(tsums[a].x - tsums[b].x, tsums[a].y - tsums[b].y)
               for a, b in zip(order, order[1:]))


def _compact_chain(members: Sequence[int], tsums: Sequence[Tsum], cap: int) -> list[int]:
    """Pick at most `cap` same-kind tsums forming the shortest drag.

    Truncating a whole-board tour would leave a chain strung across the entire
    board -- the same length limit but with all the long legs kept, which is
    the opposite of the point. So instead every tsum gets tried as a seed, each
    seed takes its `cap` nearest same-kind neighbours, and the tightest cluster
    wins. Cheap: a colour rarely has more than ~15 on screen.
    """
    if cap <= 0 or len(members) <= cap:
        return _nearest_neighbor_tour(members, tsums)

    best, best_len = None, float("inf")
    for seed in members:
        s = tsums[seed]
        near = sorted(members,
                      key=lambda i: (tsums[i].x - s.x) ** 2 + (tsums[i].y - s.y) ** 2)[:cap]
        order = _nearest_neighbor_tour(near, tsums)
        total = _tour_length(order, tsums)
        if total < best_len:
            best, best_len = order, total
    return best


def find_chains(
    tsums: Sequence[Tsum],
    radius: float,
    link: float = 1.75,
    *,
    block: float = 0.75,
    link_px: Optional[float] = None,
    base_kind: Optional[int] = None,
    base_only: bool = False,
    mode: str = "touch",
    max_chain: int = 0,
    first_leg_px: float = 0.0,
    labels: Optional[np.ndarray] = None,
) -> list[Chain]:
    """Every playable chain, best first.

    With `base_kind` set, chains of that type sort ahead of all others however
    long the others are: clearing your equipped tsum is what charges the skill,
    so a 3-chain of the base beats a 7-chain of something else.

    `mode`:
      "touch" (the default) -- the conservative model: only tsums whose circles
        are within `link_px` pixels (or `link` diameters) and not blocked by a
        third tsum on the segment between them count as linked. See
        :func:`adjacency`.
      "reach" -- the drag can pass harmlessly over off-type tsums, so any two
        same-kind tsums are linkable regardless of distance; the only real
        constraint is visiting them all in one continuous stroke. Verified
        against a hand-marked board: touching-only chains topped out around 3-4
        tsums where the actual game reaches 6+ by weaving between obstacles, so
        this is arguably the mode that matches how the game is actually played.
        It is not the default because it has not been A/B'd over a full round
        against "touch" -- when it is, this comment should say which won.
      "blob" -- like "touch", but contact is read off the mask instead of
        inferred from distance; needs `labels` (the cluster map that produced
        these tsums). Accepts more of the links a human actually drew and
        invents fewer of its own, at roughly 45x the cost. See
        :func:`blob_adjacency`. Falls back to "touch" if `labels` is missing.
    """
    chains: list[Chain] = []
    by_kind: dict[int, list[int]] = {}
    for i, t in enumerate(tsums):
        by_kind.setdefault(t.kind, []).append(i)

    if mode == "reach":
        for kind, members in by_kind.items():
            if base_only and kind != base_kind:
                continue
            if len(members) < MIN_CHAIN:
                continue
            order = orient_chain(_compact_chain(members, tsums, max_chain),
                                 tsums, first_leg_px)
            if len(order) < MIN_CHAIN:
                continue
            chains.append(Chain(kind, tsums[order[0]].colour, order, is_base=kind == base_kind))
        chains.sort(key=lambda c: (c.is_base, len(c)), reverse=True)
        return chains

    if mode == "blob" and labels is not None:
        adj = blob_adjacency(labels, tsums, radius)
    else:
        adj = adjacency(tsums, radius, link, block, link_px)
    for kind, members in by_kind.items():
        if base_only and kind != base_kind:
            continue
        for comp in _components(members, adj):
            if len(comp) < MIN_CHAIN:
                continue
            path = longest_path(comp, adj, tsums=tsums)
            # Orient before truncating, not after: the trim keeps a prefix, so
            # deciding which end is the front has to happen while both ends are
            # still there.
            path = orient_chain(path, tsums, first_leg_px)
            if max_chain > 0:
                path = path[:max_chain]
            if len(path) >= MIN_CHAIN:
                chains.append(Chain(kind, tsums[comp[0]].colour, path, is_base=kind == base_kind))

    chains.sort(key=lambda c: (c.is_base, len(c)), reverse=True)
    return chains


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def draw(bgr: np.ndarray, tsums: Sequence[Tsum], chains: Sequence[Chain], radius: float) -> np.ndarray:
    out = cv2.addWeighted(bgr, 0.45, np.zeros_like(bgr), 0, 30)

    for t in tsums:
        cv2.circle(out, (int(t.x), int(t.y)), int(radius * 0.92), t.colour, 2, cv2.LINE_AA)

    # Runner-up chains stay faint so the eye goes to the one we'd actually play.
    for chain in chains[1:]:
        pts = np.array([[int(tsums[i].x), int(tsums[i].y)] for i in chain.nodes], np.int32)
        cv2.polylines(out, [pts], False, (90, 90, 90), 3, cv2.LINE_AA)

    if chains:
        best = chains[0]
        pts = [(int(tsums[i].x), int(tsums[i].y)) for i in best.nodes]
        for t in (tsums[i] for i in best.nodes):
            overlay = out.copy()
            cv2.circle(overlay, (int(t.x), int(t.y)), int(radius), t.colour, -1, cv2.LINE_AA)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            cv2.circle(out, (int(t.x), int(t.y)), int(radius), (255, 255, 255), 2, cv2.LINE_AA)

        arr = np.array(pts, np.int32)
        cv2.polylines(out, [arr], False, (0, 0, 0), 9, cv2.LINE_AA)      # halo
        cv2.polylines(out, [arr], False, (60, 255, 255), 4, cv2.LINE_AA)  # path
        cv2.arrowedLine(out, pts[-2], pts[-1], (60, 255, 255), 4, cv2.LINE_AA, tipLength=0.45)

        for n, (px, py) in enumerate(pts, 1):
            cv2.circle(out, (px, py), 12, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(out, str(n), (px - 6 * len(str(n)), py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 255), 1, cv2.LINE_AA)

        label = f"{'BASE' if best.is_base else 'chain'}: {len(best)} tsums"
    else:
        label = "no chain of 3+ found"

    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, f"{label}   |   {len(tsums)} detected   r={radius:.0f}px",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# synthetic board (for testing without a real capture)
# --------------------------------------------------------------------------
SYNTH_PALETTE = [  # BGR, eyeballed off a real screenshot
    ((196, 176, 245), "piglet"),
    ((248, 245, 245), "donald"),
    ((60, 90, 139), "pumbaa"),
    ((200, 160, 168), "eeyore"),
    ((168, 199, 232), "chip"),
    ((30, 30, 32), "mickey"),   # near-black on purpose: the hard case
]
BASE_INDEX = 5                   # which palette entry the skill icon shows


def _draw_tsum(img: np.ndarray, x: float, y: float, r: float, colour, angle: float = 0.0) -> None:
    """One tsum, tilted by `angle` radians.

    Tsums land at whatever angle physics gives them, so the test board tilts
    them too -- the detector is supposed to be orientation-blind and this is
    what proves it rather than assuming it.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def at(dx: float, dy: float) -> tuple[int, int]:
        return (int(x + dx * cos_a - dy * sin_a), int(y + dx * sin_a + dy * cos_a))

    for sx in (-1, 1):
        cv2.circle(img, at(sx * r * 0.72, -r * 0.72), int(r * 0.42), (24, 24, 24), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), int(r), tuple(int(c) for c in colour), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), int(r), (30, 30, 30), 1, cv2.LINE_AA)
    for sx in (-1, 1):  # eyes, so the mask has holes to survive like the real thing
        cv2.circle(img, at(sx * r * 0.3, -r * 0.1), max(2, int(r * 0.11)),
                   (20, 20, 20), -1, cv2.LINE_AA)


def synth(width: int = 540, height: int = 960, count: int = 46, seed: int = 7,
          tilt: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (110, 60, 30), np.uint8)

    bx, by, bw, bh = (int(DEFAULT_BOARD[0] * width), int(DEFAULT_BOARD[1] * height),
                      int(DEFAULT_BOARD[2] * width), int(DEFAULT_BOARD[3] * height))
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (128, 74, 38), -1)

    r = 31
    placed: list[tuple[int, int, tuple]] = []
    cx, cy = bx + bw / 2, by + bh / 2
    for _ in range(4000):
        if len(placed) >= count:
            break
        x = rng.uniform(bx + r, bx + bw - r)
        y = rng.uniform(by + r, by + bh - r)
        # Keep it in a bowl shape like the real pile, not a rectangle.
        if ((x - cx) / (bw / 2 - r)) ** 2 + ((y - cy) / (bh / 2 - r)) ** 2 > 1.0:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (r * 1.55) ** 2 for px, py, *_ in placed):
            continue
        placed.append((x, y, SYNTH_PALETTE[rng.integers(len(SYNTH_PALETTE))][0],
                       rng.uniform(-math.pi, math.pi) * tilt))

    for x, y, colour, angle in placed:
        _draw_tsum(img, x, y, r, colour, angle)

    # Skill icon bottom-left, so read_base_kind has something to read.
    cx, cy, ir = BASE_ICON[0] * width, BASE_ICON[1] * height, BASE_ICON[2] * width
    cv2.circle(img, (int(cx), int(cy)), int(ir * 2.0), (150, 95, 45), -1, cv2.LINE_AA)
    _draw_tsum(img, cx, cy, ir * 1.15, SYNTH_PALETTE[BASE_INDEX][0])
    return img


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
#: `label`'s marking modes, in the order their number keys select them.
_LABEL_MODES = ("path", "missed", "false", "group")


def _board_rect(shape, spec: Optional[str], *, fever: bool = False):
    """The play area for this frame. `fever` picks the wider FEVER rect.

    An explicit `spec` always wins, FEVER or not: someone who passed `--board`
    asked for that rect and swapping it underneath them would be surprising.
    """
    h, w = shape[:2]
    if spec is None:
        layout = _layout(shape)
        measured = (layout.get("fever_board") if fever else None) or layout.get("board")
        if measured:
            return measured
        f = DEFAULT_BOARD
        return int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h)
    if spec == "full":
        return 0, 0, w, h
    x, y, bw, bh = (int(v) for v in spec.split(","))
    return x, y, bw, bh


@dataclass
class PlayReport:
    """What one :func:`play_loop` run did.

    `played` alone cannot compare two settings: a run that plays fifteen
    3-chains and one that plays eight 6-chains are not close, and the second
    scores far better. `cleared` and `stalled` are what an A/B actually turns
    on -- total tsums removed, and how many drags the game ignored.
    """

    played: int = 0
    #: Time spent waiting for the board to stop moving, and how often that wait
    #: hit its cap. Measured because it is the biggest number in a round and
    #: nobody had looked: over 26 logged rounds the gap between chains averages
    #: 839ms, of which only 144ms is thinking. The other 695ms is the stroke
    #: and this wait, and until they are separated neither can be tuned.
    settle_s: float = 0.0
    settles: int = 0
    settle_timeouts: int = 0
    #: Frames read while FEVER was up, and frames read at all.
    #:
    #: THE NUMBER THE SCORE ACTUALLY TURNS ON. Over 15 rounds carrying both a
    #: score and telemetry, score tracks the share of a round spent in FEVER at
    #: r=+0.81, and mean chain length at r=-0.01. The two catastrophic rounds
    #: in that set -- 9,841 and 25,065 against a normal 600-900k -- are the two
    #: that never entered FEVER at all. Nothing in the loop optimised for it and
    #: nothing reported it, so a round could collapse by two orders of magnitude
    #: with every other number looking ordinary.
    fever_frames: int = 0
    frames: int = 0
    #: Total tsums dragged through. Not a result: with the clear check off it
    #: is the length of the chains proposed, and the game rejects some of them
    #: outright -- a 3-chain it only marks two of pops nothing at all.
    dragged: int = 0
    #: Tsums seen to leave the board, and only ever set when `--verify-clears`
    #: measured it. The throughput number, when there is one.
    cleared: int = 0
    #: Was `cleared` measured at all? Without this a quiet 0 reads as a round
    #: that cleared nothing rather than one that never looked.
    checked: bool = False
    #: Drags the emulator registered but the game refused to clear -- the
    #: chain was not really one character. `--verify-clears` only.
    rejected: int = 0
    #: Which side of an `--ab` run this round was played on: "on", "off", or
    #: "" when no A/B was running. Recorded on the ROUND rather than left to be
    #: read out of the samples, because a round that wrote no sample -- the
    #: dataset off, or its per-round limit already spent -- still has to say
    #: which arm it was, or the arms silently stop being balanced.
    ab_arm: str = ""
    #: The option `--ab` alternated, for a corpus that mixes experiments.
    ab: str = ""
    #: Drags that ran but changed nothing -- the cost of an over-permissive
    #: link rule, and the half a positive-only label set cannot measure.
    stalled: int = 0
    #: --verify-hold only: chain members the game declined to mark, dropped
    #: before the stroke walked to them.
    trimmed: int = 0
    #: --verify-hold only: presses released without dragging, because what the
    #: game marked was already below min_chain.
    abandoned: int = 0
    #: Drags that paid for a check before moving. With `--verify-reach` this is
    #: the price actually paid, and it is the number the setting is tuned on:
    #: too high and the check is being bought on chains that did not need it,
    #: zero and the threshold never triggered.
    verified: int = 0
    #: Checks whose frame showed no mark anywhere on the board. The chain is
    #: dragged as proposed rather than trimmed on a reading that failed -- a
    #: blank frame is the game saying nothing, not the game saying no. Worth
    #: printing: it is the share of the `verified` cost that bought nothing,
    #: and a run where it is most of them is a capture problem, not a chain
    #: problem.
    unreadable: int = 0
    #: --verify-extend only: checks whose marks rebuilt a longer chain than the
    #: trim would have dragged, and the members those rebuilds added. This is
    #: the benefit side of the rule, and `verified` above is its cost -- the
    #: two are only worth reading together.
    extended: int = 0
    extended_by: int = 0
    #: Why the loop ended -- shown by the CLI and returned to the flow.
    reason: str = ""
    #: True when the stop key ended it rather than a normal exit condition.
    stopped: bool = False

    def describe(self) -> str:
        chains = f"played {self.played} chains, dragged {self.dragged} tsums"
        if self.played:
            chains += f" (mean {self.dragged / self.played:.1f}/chain)"
        if self.checked:
            # The two numbers are worth seeing together: the gap between them
            # is the waste, and it is the only figure here worth tuning for.
            share = f" ({100 * self.cleared / self.dragged:.0f}%)" if self.dragged else ""
            chains += f", cleared {self.cleared}{share}"
        if self.frames:
            chains += (f", FEVER on {100 * self.fever_frames / self.frames:.0f}% "
                       f"of frames")
        if self.settles:
            cap = (f", {100 * self.settle_timeouts / self.settles:.0f}% hit the cap"
                   if self.settle_timeouts else "")
            chains += (f", waited {self.settle_s:.0f}s for the board to settle "
                       f"({1000 * self.settle_s / self.settles:.0f}ms a drag{cap})")
        out = f"{chains}, {self.stalled} drag(s) did not register"
        if self.rejected:
            out += f", {self.rejected} the game would not accept"
        if self.trimmed or self.abandoned:
            out += (f"; the game trimmed {self.trimmed} member(s) and rejected "
                    f"{self.abandoned} chain(s) outright")
        if self.verified:
            # The cost side of --verify-reach, next to the trimmed/abandoned
            # benefit above: the two together are the whole trade.
            share = f" ({100 * self.verified / self.played:.0f}% of drags)" if self.played else ""
            out += f"; checked {self.verified} chain(s) before dragging{share}"
            if self.unreadable:
                out += (f", {self.unreadable} of which read nothing and were "
                        f"dragged as proposed")
            if self.extended:
                # Cost and benefit in one line, which is what step 7 of
                # docs/IMPROVEMENT-LOOP.md asks every opt-in rule for: how
                # many of the checks paid for above came back with a longer
                # chain than the trim, and how many tsums that added.
                share = (f" ({100 * self.extended / self.verified:.0f}% of "
                         f"checks)" if self.verified else "")
                out += (f"; the marks rebuilt {self.extended} chain(s){share}, "
                        f"adding {self.extended_by} member(s)")
        return out


class FeverWatch:
    """Is FEVER running right now?

    FEVER repaints the board and animates it for about ten seconds, and it is
    the state the detector reads worst -- so it is worth knowing about, and
    worth reading from the game rather than inferred from detection going
    wrong. There are two ways to know, and they are not equally good.

    `max_fever` is the meter at full, gold, an instant before FEVER starts: a
    *trigger*, visible for a moment, not a state. One match has to be turned
    into a ten-second assumption, so a missed trigger means the whole of FEVER
    is played on the wrong rect and the wrong floor, and a stretched or
    shortened FEVER is not noticed at all.

    `fever_bonus` is the banner the game shows for as long as FEVER is running.
    That can be *asked* every frame instead of assumed, which is strictly
    better, so it leads when it is available. It still opens a short window
    (:data:`FEVER_HOLD`) rather than being read raw: the banner fades in and
    out at each end, and those few frames score too low to match. The trigger
    is kept alongside it, because `max_fever` fires an instant *before* the
    banner is drawn and covers the first frame or two.

    Degrades in two steps rather than one: banner + trigger, trigger only with
    its ten-second assumption, then "never FEVER" if `templates/` predates
    both.
    """

    def __init__(self, matcher, templates, *, seconds: float = FEVER_SECONDS,
                 confidence: float = FEVER_CONFIDENCE,
                 name: str = "max_fever", banner: str = "fever_bonus",
                 banner_confidence: float = FEVER_BANNER_CONFIDENCE,
                 hold: float = FEVER_HOLD, use_banner: bool = True,
                 board: Optional[str] = None, clock=time.monotonic) -> None:
        self.matcher = matcher
        self.seconds = seconds
        self.confidence = confidence
        self.banner_confidence = banner_confidence
        self.hold = hold
        #: The board rect spec, only so the banner search can be anchored to
        #: the top of the board. Always the NORMAL rect: which rect is in force
        #: depends on the answer this class is about to give.
        self.board = board
        self.clock = clock
        self._until = 0.0
        self._was = False
        #: When the banner was last seen, to tell a new run from a top-up.
        self._banner_seen = -1e9
        self.template = self._load(templates, name)
        self.banner = self._load(templates, banner) if use_banner else None
        if self.banner is not None:
            # Reduced once here, not per frame: it is the same glyph mask every
            # time and the frame side is the only half that changes.
            self.banner_ink = _glyph_ink(self.banner.image)
        else:
            self.banner_ink = None

    @staticmethod
    def _load(templates, name):
        try:
            return templates.get(name)
        except Exception:  # noqa: BLE001 - a missing template is not an error here
            log.debug("no %s template", name)
            return None

    @property
    def enabled(self) -> bool:
        return self.template is not None or self.banner is not None

    @property
    def reads_state(self) -> bool:
        """True when FEVER is being asked about rather than assumed."""
        return self.banner_ink is not None

    def update(self, frame) -> bool:
        """Look at one frame; return whether FEVER is running as of now.

        Re-arms on every match rather than only on the first. The meter stays
        full for several frames as it fills, and each of those pushes the
        deadline out -- which is right, because FEVER has not started counting
        down until the bar actually starts draining.

Every banner sighting opens the same :data:`FEVER_HOLD` window. Two
        earlier designs are worth recording because both were wrong in ways
        that are not obvious:

        * **A 1.0s window.** Sized for the banner's fade and nothing else. It
          makes the state only as reliable as the least readable frame, and a
          played round showed it at once -- one real eleven-second FEVER logged
          as three (FEVER :37, NORMAL :40, FEVER :40, NORMAL :43, FEVER :43,
          NORMAL :48). Every flip discards the palette and re-reads the base
          tsum, and the base came back a different cluster each time, so a
          missed frame cost the identity of the character being played for.

        * **A full ten seconds on the first sighting, topped up after.** The
          idea was that FEVER's length is a game rule and so covers the middle
          of a run for free. It is not a rule: **a skill firing pauses FEVER
          until its animation finishes**, and some skills start FEVER outright,
          so a run has no fixed length to lean on. Worse, it made a single
          spurious read cost ten seconds of wrong state instead of one.

        So: no assumption about how long FEVER lasts, one window length, and it
        is sized to outlast a skill animation because that is the longest the
        banner can go unread mid-run.
        """
        now = self.clock()
        if self.banner_ink is not None:
            board_top = _board_rect(frame.shape, self.board)[1]
            score = fever_banner_score(frame, self.banner_ink, board_top)
            if score >= self.banner_confidence:
                self._banner_seen = now
                self._until = max(self._until, now + self.hold)
        if self.template is not None and self.matcher is not None:
            if self.matcher.find(frame, self.template, confidence=self.confidence):
                # Ten seconds only when the trigger is all there is. With the
                # banner reading the state, the trigger's job shrinks to
                # covering the frame or two between the meter filling and the
                # banner being drawn -- and a ten-second window off a single
                # match is exactly the assumption the banner exists to remove.
                # Measured over 151 frames, the two never overlap: the trigger
                # matches on 3 frames, all immediately before FEVER, and the
                # banner on the 54 frames of FEVER itself.
                span = self.hold if self.banner_ink is not None else self.seconds
                self._until = max(self._until, now + span)
        return self.active

    @property
    def active(self) -> bool:
        return self.clock() < self._until

    def took_effect(self) -> Optional[str]:
        """"fever"/"normal" the first time the state flips, else None.

        Lets a caller log the transition and drop cached state without keeping
        its own copy of the previous value.
        """
        now = self.active
        if now == self._was:
            return None
        self._was = now
        return "fever" if now else "normal"


def _glyph_ink(bgr: np.ndarray) -> np.ndarray:
    """The gold-and-white pixels a FEVER banner is drawn with.

    Bold yellow letters with a white rim, and that pairing is what the game
    uses for this banner and not for the board behind it.
    """
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    gold = (r > 150) & (g > 120) & (b < 150) & (r - b > 60)
    white = (r > 200) & (g > 200) & (b > 200)
    return ((gold | white).astype(np.uint8)) * 255


def fever_banner_score(frame: np.ndarray, template: np.ndarray,
                       board_top: int) -> float:
    """How much the FEVER BONUS banner is on screen, 0-1.

    Not :class:`TemplateMatcher`, and the reason is worth recording, because
    the obvious two attempts both fail on this particular template.

    The banner is *text over the live board*: the board shows through between
    and behind the letters and is different on every frame, so most of the
    template's box is noise. Straight `TM_CCOEFF_NORMED` reads ~0.61 whether
    the banner is on screen or not -- it locks onto the right place and cannot
    tell whether the text is there. Masking the template to the glyphs is the
    natural repair, but a mask makes `TemplateMatcher` switch to
    `TM_CCORR_NORMED` (see its `_match`), which has no mean subtraction and
    reads ~0.966 on everything.

    What is invariant is the *shape*: identical glyphs every time, whatever is
    behind them. So both sides are reduced to "is this pixel banner-coloured"
    and those two binary images are correlated. Mean-subtracted correlation of
    two binary images is a shape match, and it does not care what colour is
    showing through. Measured over 151 frames: banner present 0.438-0.561,
    banner absent 0.000-0.275.

    Searched in a band around the top of the board rather than over the whole
    frame -- same scores to three decimals, 0.7ms instead of 15.8ms.
    """
    h = template.shape[0]
    top = max(0, board_top - 3 * h)
    bottom = min(frame.shape[0], board_top + 2 * h)
    band = frame[top:bottom]
    if band.shape[0] < h or band.shape[1] < template.shape[1]:
        return 0.0
    scores = cv2.matchTemplate(_glyph_ink(band), template, cv2.TM_CCOEFF_NORMED)
    return float(scores.max()) if scores.size else 0.0


@dataclass
class Driver:
    """Everything the play loop needs from the outside world.

    The loop is written against this instead of against :class:`Application` or
    a flow's :class:`RunContext` so one implementation serves both entry
    points. They already own the same component objects -- the same capture,
    the same matcher, the same template library -- so each constructor below is
    a handful of lines and no play logic is duplicated.

    ``rect`` is the emulator's content area in screen coordinates. Everything
    the loop computes is content-relative; this is the only thing that knows
    where that content area actually sits on the desktop.
    """

    capture: Any
    matcher: Any
    templates: Any
    rect: Any
    #: Raises StopRequested when the user hits the stop key.
    check_stop: Callable[[], None] = lambda: None
    say: Callable[[Any], None] = print

    @classmethod
    def from_app(cls, app, rect, *, watcher=None, say=print) -> "Driver":
        """Drive from the CLI's :class:`Application`."""
        stop = watcher.check if watcher is not None else (lambda: None)
        return cls(app.capture, app.matcher, app.templates, rect,
                   check_stop=stop, say=say)

    @classmethod
    def from_context(cls, ctx, *, say=None) -> "Driver":
        """Drive from a flow's run context (the ``play_tsum`` action)."""
        return cls(ctx.capture, ctx.matcher, ctx.templates, ctx.require_content_rect(),
                   check_stop=ctx.stop.check, say=say or log.info)

    def grab(self):
        return self.capture.grab(self.rect)

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Content-relative point -> absolute screen coordinates."""
        return int(self.rect.left + x), int(self.rect.top + y)


def drag_chain(points: Sequence[tuple[int, int]], *, step_px: float = 8.0,
               hold: float = 0.05, per_step: float = 0.004,
               after_press: Optional[Callable[[], Sequence[tuple[int, int]]]] = None) -> None:
    """Drag through every point in order, as one continuous stroke.

    The emulator turns mouse movement into touch movement, and it only sees the
    positions it is actually given. Jumping corner to corner would teleport the
    cursor straight past the tsums in between, so each leg is walked in ~8px
    steps -- the chain is built from what the finger passes *over*, not from
    where it stops.

    `after_press` runs once the first point is held and before anything moves,
    and returns the points to actually walk. That window is the only moment the
    game will tell you which tsums it accepts from here -- see
    :func:`marked_by_game` -- and it costs nothing to use, because pressing the
    first tsum is the start of the drag either way. Returning a single point
    walks nowhere and releases, which is how a chain the game rejects gets
    abandoned before it wastes a stroke.
    """
    import pyautogui

    pyautogui.PAUSE = 0.0
    pyautogui.moveTo(*points[0])
    time.sleep(hold)
    pyautogui.mouseDown()
    try:
        if after_press is not None:
            points = list(after_press()) or list(points[:1])
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / step_px))
            for s in range(1, steps + 1):
                pyautogui.moveTo(x0 + (x1 - x0) * s / steps, y0 + (y1 - y0) * s / steps)
                time.sleep(per_step)
        time.sleep(hold)
    finally:
        pyautogui.mouseUp()


def cleared_by_drag(before_crop: np.ndarray, after_crop: np.ndarray,
                    tsums: Sequence[Tsum], nodes: Sequence[int], *,
                    tol: float) -> tuple[list[int], list[float], float]:
    """Which of the dragged tsums are actually gone?

    The whole-crop diff `--verify` uses answers a different question -- did
    the emulator see the stroke at all -- and cannot answer this one, because
    a live board animates. The score counter ticks, the timer runs, the FEVER
    meter fills and idle tsums jiggle, so the mean across the crop clears
    `--change-tol` whether or not anything popped.

    Looking only where the chain was is what separates the two. A tsum that
    cleared is replaced by whatever falls into its place, which is a large
    change inside its own disk; a tsum that merely jiggled is still the same
    face in nearly the same spot.

    Returns the members that changed, their per-disk means, and the median
    over the tsums that were *not* dragged -- the board's own idle noise, so a
    reading always carries the yardstick it should be judged against.
    """
    diff = cv2.absdiff(after_crop, before_crop).max(axis=2)
    radius = max(2, int(min(t.r for t in tsums) * 0.55))

    def _disk(t: Tsum) -> float:
        mask = np.zeros(diff.shape, np.uint8)
        cv2.circle(mask, (int(t.x), int(t.y)), radius, 1, -1)
        return float(diff[mask.astype(bool)].mean())

    chain = set(nodes)
    values = [_disk(tsums[n]) for n in nodes]
    idle = [_disk(t) for i, t in enumerate(tsums) if i not in chain]
    baseline = float(np.median(idle)) if idle else 0.0
    gone = [n for n, value in zip(nodes, values) if value > tol]
    return gone, values, baseline


def marked_by_game(drv, before_crop: np.ndarray, board: tuple, tsums: Sequence[Tsum],
                   nodes: Sequence[int], *, delay: float, threshold: float,
                   aura: float, out: Optional[dict] = None,
                   frames: int = 1, gap: float = 0.0,
                   floor_mult: float = 0.0) -> list[int]:
    """Of a chain's members, which ones did the game light up?

    Holding a tsum makes the game mark everything you can link to it -- both
    the same character *and* actually reachable, which are the two judgements
    this module makes worst. Measured on a real board, pressing one Piglet
    marked five more that colour clustering had filed under three different
    kinds. So this asks the game rather than trusting the clusters.

    Members inside `aura` of the pressed tsum are kept regardless. The glow is
    about 90px across and washes over whatever is under it, so a reaction there
    means nothing either way -- and a tsum that close was probably linkable in
    any case. Only distant members are ever dropped, which makes this a
    conservative trim: it removes the long reaches that the guesswork gets
    wrong, and never second-guesses the near ones.

    Two things decide whether the answer is worth anything, and both default
    to what ``--verify-hold`` can afford rather than to what reads best:

    * `delay` under ~0.15s photographs the board before the game has drawn the
      highlight. ``--verify-hold`` runs at 0.10 on purpose -- it pays this on
      every drag -- and the sample collector, which pays on one drag in four,
      does not and should not.
    * `frames` of 1 diffs against a board that is still moving. Reading
      several and keeping only what changed in *all* of them is the same
      defence :func:`marks_on_board` takes, for the same reason: a mark is
      present in every frame of the hold, a settling tsum is somewhere else by
      the next one.

    `out` receives the reading as well as the frame -- `values` per tsum,
    the `baseline` of idle motion they should be judged against, and the
    `marked` indices over the whole board. A caller that only wants to trim a
    chain can ignore it; the collector writes it down, because a sample whose
    baseline swamps its marks is one that has to be thrown away rather than
    trained on.
    """
    bx, by, bw, bh = board
    time.sleep(delay)
    grabs = [drv.grab()[by:by + bh, bx:bx + bw]]
    for _ in range(max(0, frames - 1)):
        if gap:
            time.sleep(gap)
        grabs.append(drv.grab()[by:by + bh, bx:bx + bw])
    crop = grabs[0]
    diff = np.minimum.reduce(
        [cv2.absdiff(g, before_crop) for g in grabs]).max(axis=2)

    head = tsums[nodes[0]]
    r = max(2, int(min(t.r for t in tsums) * 0.55))

    def _disk(t: Tsum) -> float:
        m = np.zeros(diff.shape, np.uint8)
        cv2.circle(m, (int(t.x), int(t.y)), r, 1, -1)
        return float(diff[m.astype(bool)].mean())

    values = [_disk(t) for t in tsums]
    # The board's own idle jiggle, measured off the tsums this stroke is not
    # touching -- the same yardstick `cleared_by_drag` reports, and the one
    # thing that says whether a reading over `threshold` means anything.
    chain = set(nodes)
    idle = [v for i, v in enumerate(values) if i not in chain]
    baseline = float(np.median(idle)) if idle else 0.0

    # A mark has to clear the board's own floor, not just a fixed number.
    # Measured with `hold` on four boards, the game's marks land 8x to 25x
    # above the median untouched tsum -- 61-65 against a floor of 7.5, 95
    # against 3.8, 167 against 6.8. The fixed 8.0 sits *inside* that floor's
    # noise on a live board and admitted 23 to 35 tsums per press, which is
    # why every reading looked like half the board. Confirmed at scale on
    # 11,537 samples: below 8x the floor the tsums that react are no more
    # this character than the board average is (lift 1.00, same-kind 25%
    # against a 25% base rate), and above it the reading turns into a label.
    #
    # This is the bar the *label* is read at. The trim below deliberately
    # keeps the fixed threshold: `--verify-hold` is the only caller that
    # trims, its A/B was measured under 8.0, and re-scoring it here would
    # invalidate that measurement -- silently, and only in the combination
    # where collection happened to be on at the same time.
    bar = max(threshold, floor_mult * baseline) if floor_mult else threshold

    keep = [nodes[0]]
    for n in nodes[1:]:
        t = tsums[n]
        if math.hypot(t.x - head.x, t.y - head.y) <= aura:
            keep.append(n)
            continue
        if values[n] > threshold:
            keep.append(n)

    # The marked frame is the only record of the game's answer, and it is gone
    # a moment later. `out` hands it to a caller that wants to keep it (the
    # sample collector) without this function knowing what a dataset is.
    if out is not None:
        out["marked_frame"] = crop
        out["values"] = values
        out["baseline"] = baseline
        out["bar"] = bar
        out["marked"] = [i for i, v in enumerate(values)
                         if v > bar and i != nodes[0]]
    return keep


def chain_from_marks(tsums: Sequence[Tsum], nodes: Sequence[int],
                     keep: Sequence[int], marked: Sequence[int], radius: float,
                     *, link_px: float, block: float, max_chain: int,
                     budget: float = 0.02) -> list[int]:
    """The chain the game just described, not the one it was asked about.

    :func:`marked_by_game` reads the press and the trim throws away every
    proposed member the game did not light. That answer contains more than a
    trim can use: the marks name partners the chain never contained. Measured
    over 4,306 collected drags the game marks a mean 6.1 tsums per press and
    **4.0 of them were never proposed** -- the recall gap, and the biggest
    single number in the collection.

    Why the graph cannot reach them by itself, which is the finding that makes
    this worth doing: :func:`adjacency` refuses to link two tsums whose `kind`
    differs, and `kind` is a per-frame k-means id. Rebuilt from the marks with
    the bot's own ids the chain gets *shorter* (-29% cleared over the drags a
    260px check fires on); rebuilt with the game's word on identity it gets
    longer. **The recall gap is a colour problem wearing a graph's clothes**,
    and a press that has already been paid for is the one moment the colour
    question has an authoritative answer. Priced end to end over the whole
    corpus by `scripts/replay_decisions.py`, against the trim at an identical
    reading cost: +5.7%, +6.0% and +6.3% clears/s across the three cost
    columns, with the 6+ chain share rising 6.7% -> 9.4%.

    So the marked tsums are treated as the pressed character -- the game said
    so -- while every other tsum stays as detected, because they are still
    needed as *blockers*: drop them and the segment test invents links through
    tsums that are really in the way.

    Returns a node list starting at `nodes[0]`, never shorter than `keep`.
    """
    head = int(nodes[0])
    lit = {head} | {int(i) for i in keep} | {
        int(i) for i in marked if 0 <= i < len(tsums)}
    if len(lit) <= len(keep):
        return list(keep)

    kind = tsums[head].kind
    recoloured = [replace(t, kind=kind) if i in lit else t
                  for i, t in enumerate(tsums)]
    adj = adjacency(recoloured, radius, block=block, link_px=link_px)

    cap = max_chain if max_chain > 0 else len(tsums)
    best = [head]
    deadline = time.monotonic() + budget

    def walk(path: list[int], seen: set[int]) -> None:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        # Same shape as `longest_path`: exhaustive, but on a wall clock. The
        # candidate set is the marks plus the trim -- single figures, so the
        # budget almost never binds; when it does, the best found so far is
        # still a chain the game named.
        if len(path) >= cap or time.monotonic() > deadline:
            return
        for nxt in adj[path[-1]]:
            if nxt in lit and nxt not in seen:
                path.append(nxt)
                seen.add(nxt)
                walk(path, seen)
                path.pop()
                seen.discard(nxt)

    walk([head], {head})
    return best if len(best) > len(keep) else list(keep)


# --------------------------------------------------------------------------
# assist: the user presses, we read the marks and walk the path
# --------------------------------------------------------------------------
_VK_LBUTTON = 0x01

#: The sample collector default, quoted by `hold` when it suggests a new one.
DATASET_DELAY = 0.25


def _lbutton_down() -> bool:
    """Is the left mouse button physically down right now?

    Polled rather than hooked. A press the user makes lands on the emulator and
    is never delivered to this process, so there is no event to listen for --
    and ``GetAsyncKeyState`` is already how
    :mod:`ttheart_sender.control.hotkey` reads the stop key, so nothing new is
    being relied on here.
    """
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_LBUTTON) & 0x8000)


def marks_on_board(before_crop: np.ndarray, held, tsums: Sequence[Tsum],
                   radius: float, *, pressed: int, threshold: float, aura: float):
    """Every tsum the game lit up while `pressed` was held.

    `held` is one frame from during the hold, or several. Several is worth the
    milliseconds whenever anything on the board moves by itself: a mark is
    present in *every* frame of the hold, while a FEVER sparkle, a popping
    bubble or a still-settling tsum is somewhere else by the next one. Taking
    the per-pixel *minimum* of the differences keeps only what persisted.

    Measured on six boards with 45 moving sparkles painted over them: one frame
    reports 10.7 false marks on average, two reports 0.7, three reports none,
    and the real marks survive all three unchanged.

    The board-wide counterpart to :func:`marked_by_game`. That one re-checks
    the members of a chain this module already proposed, so its risky move is
    *dropping* a member and it keeps anything under the glow regardless. Here
    the game is the only source of the chain -- nothing was guessed first -- so
    the risky move is the opposite one, adding a tsum the game never marked,
    and a reaction under the glow has to clear the same threshold as any other.

    Returns ``(hits, aura_only, diff)``. `aura_only` are hits close enough to
    the press that the glow alone could explain them (it washes over ~90px and
    that is where the warthog false positive in ``hold`` came from). They are
    still returned as hits, because dragging over a tsum the game will not
    accept costs a detour rather than a broken chain -- but they are counted
    separately so a bad reading shows up in the report instead of silently
    steering the path.
    """
    frames = [held] if isinstance(held, np.ndarray) else list(held)
    diff = np.minimum.reduce(
        [cv2.absdiff(f, before_crop) for f in frames]).max(axis=2)
    r = max(2, int(radius * 0.5))
    head = tsums[pressed]
    hits: list[int] = []
    aura_only: list[int] = []
    for i, t in enumerate(tsums):
        if i == pressed:
            continue
        m = np.zeros(diff.shape, np.uint8)
        cv2.circle(m, (int(t.x), int(t.y)), r, 1, -1)
        if float(diff[m.astype(bool)].mean()) <= threshold:
            continue
        hits.append(i)
        if math.hypot(t.x - head.x, t.y - head.y) <= aura:
            aura_only.append(i)
    return hits, aura_only, diff


def tour_from(start: int, members: Sequence[int], tsums: Sequence[Tsum]) -> list[int]:
    """Greedy nearest-neighbour order that is forced to begin at `start`.

    Neither existing router will do when a finger is already on a tsum:
    :func:`_nearest_neighbor_tour` chooses its own start, and
    :func:`orient_chain` may reverse the result to shorten the opening hop.
    Both are free choices only while nothing is being touched yet.
    """
    remaining = set(members) - {start}
    order = [start]
    while remaining:
        last = tsums[order[-1]]
        nxt = min(remaining,
                  key=lambda i: (tsums[i].x - last.x) ** 2 + (tsums[i].y - last.y) ** 2)
        order.append(nxt)
        remaining.remove(nxt)
    return order


def walk_path(points: Sequence[tuple[int, int]], *, step_px: float = 8.0,
              per_step: float = 0.004,
              still_down: Optional[Callable[[], bool]] = None,
              check_stop: Optional[Callable[[], None]] = None) -> int:
    """Move the cursor through `points` without ever pressing or releasing.

    :func:`drag_chain` owns a whole stroke -- press, walk, release. This is the
    middle third alone, for when the *user's* finger is what holds the button
    down. Windows carries the physical button state in the move messages a
    cursor warp generates, so the emulator sees touch movement with the button
    still held, exactly as it does when ``drag_chain`` pressed it itself.

    Same ~8px stepping and for the same reason: the chain is built from what
    the touch passes over, and a jump straight to the next tsum would teleport
    past everything in between.

    Returns the number of legs completed. It stops early when `still_down`
    goes false -- the user let go mid-path, and every further move would be a
    bare hover starting nothing.
    """
    import pyautogui

    pyautogui.PAUSE = 0.0
    walked = 0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / step_px))
        for s in range(1, steps + 1):
            if check_stop is not None:
                check_stop()
            if still_down is not None and not still_down():
                return walked
            pyautogui.moveTo(x0 + (x1 - x0) * s / steps, y0 + (y1 - y0) * s / steps)
            time.sleep(per_step)
        walked += 1
    return walked


def _settle(drv: Driver, *, max_wait: float, tol: float = 2.5,
            region: Optional[tuple] = None, out: Optional[dict] = None):
    """Wait for the board to stop moving, then return the frame it stopped on.

    Cleared tsums drop and the pile collapses; detecting mid-fall gives
    coordinates that are already stale by the time the drag runs. Frame diffing
    costs ~5ms, so this is faster than any fixed delay big enough to be safe --
    it returns the instant the board is still. Capped, because FEVER animates
    continuously and would otherwise wait forever.

    `region` restricts the comparison to the board. It matters more than it
    looks: measured over 26 logged rounds, **83% of a round is stroke and
    waiting and only 17% is thinking**, so this wait -- not detection -- is
    what sets how many chains a round gets to play. And :func:`cleared_by_drag`
    already records why a whole-frame diff is the wrong instrument for asking
    whether the BOARD moved: "the score counter ticks, the timer runs, the
    FEVER meter fills and idle tsums jiggle, so the mean across the crop clears
    the threshold whether or not anything popped." Every one of those lives
    outside the board rect, and each of them can hold this at its cap while the
    tsums have long since stopped falling.

    `out` collects what it did -- `waited`, and whether it `timed_out` -- so a
    round can report the cost instead of leaving it to be guessed at.
    """
    started = time.perf_counter()

    def view(f):
        if region is None:
            return f
        x, y, w, h = region
        return f[y:y + h, x:x + w]

    frame = drv.grab()
    prev = view(frame)
    deadline = started + max_wait
    while time.perf_counter() < deadline:
        drv.check_stop()
        frame = drv.grab()
        now = view(frame)
        if float(np.mean(cv2.absdiff(now, prev))) < tol:
            if out is not None:
                out["waited"] = time.perf_counter() - started
                out["timed_out"] = False
            return frame
        prev = now
    if out is not None:
        out["waited"] = time.perf_counter() - started
        out["timed_out"] = True
    return frame  # timed out: the newest frame is still the best guess


def _click_shuffle(drv: Driver, spec: Optional[str], times: int, delay: float,
                   hold: float, move_time: float) -> None:
    """Tap a fixed content-relative point a few times -- the shuffle button.

    `spec` is "x,y" in content coordinates, i.e. exactly what `main.py point`
    prints as `content=(x, y)`: the same space drag points are converted from,
    so it composes with everything else here.

    Goes straight to pyautogui rather than the configured mouse controller,
    whose press duration comes from config.yaml -- `hold`/`move_time` give the
    same speed control here that `drag_chain` has.
    """
    if not spec:
        return
    import pyautogui

    x, y = (int(v) for v in spec.split(","))
    sx, sy = drv.to_screen(x, y)
    pyautogui.PAUSE = 0.0
    for i in range(times):
        pyautogui.moveTo(sx, sy, duration=move_time)
        pyautogui.mouseDown()
        time.sleep(hold)
        pyautogui.mouseUp()
        if i + 1 < times:
            time.sleep(delay)


def skill_gold(frame: np.ndarray, spec: str, r_in: float, r_out: float) -> float:
    """Fraction of the skill button's ring that has gone gold.

    The button is awkward to template-match because the character portrait in
    the middle never changes -- only the plate behind it lights up when the
    skill is charged. So this ignores the portrait entirely and samples the
    ring around it. Measured across 14 uncharged captures the ring reads 0.000,
    so any real signal is unambiguous.
    """
    cx, cy = (int(v) for v in spec.split(","))
    h, w = frame.shape[:2]
    r_out = int(r_out)
    y0, y1 = max(0, cy - r_out), min(h, cy + r_out)
    x0, x1 = max(0, cx - r_out), min(w, cx + r_out)
    if y1 <= y0 or x1 <= x0:
        return 0.0

    sub = frame[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - cx, yy - cy)
    ring = (dist >= r_in) & (dist <= r_out)
    if not ring.any():
        return 0.0

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    gold = (hue >= 18) & (hue <= 38) & (sat >= 90) & (val >= 130)
    return float(gold[ring].mean())


def _load_bubbles(drv: Driver, names: str) -> list:
    """Resolve the comma-separated bubble template names, once.

    Resolution happens up front rather than per frame so a typo or a missing
    capture is reported once at startup instead of silently costing every
    bubble for the whole run.
    """
    templates = []
    for name in (n.strip() for n in str(names).split(",")):
        if not name:
            continue
        try:
            templates.append(drv.templates.get(name))
        except Exception:
            drv.say(f"    no template {name!r} -- that bubble will be ignored "
                    f"(capture it with `python main.py snip {name}`)")
    return templates


def _pop_bubbles(drv: Driver, frame, templates: Sequence, confidence: float,
                 hold: float, move_time: float, max_taps: int = 4) -> dict:
    """Tap any bubble on screen. Returns a template name -> how many it popped.

    Uses the project's own template matcher rather than a colour rule invented
    here -- capture the art once with `python main.py snip <name>` and this
    finds it, which also means a new special item costs a snip and a name in
    `--bubble` rather than a code change.

    Each kind is matched separately and the strongest hit across all of them
    wins the tap: bubbles drift and pop with an animation, so the frame is
    re-grabbed after every tap rather than trusting stale coordinates.
    """
    import pyautogui

    popped: dict[str, int] = {}
    for _ in range(max_taps):  # a few can be on screen at once
        best = None
        for template in templates:
            hit = drv.matcher.find(frame, template, confidence=confidence)
            if hit is not None and (best is None or hit.confidence > best[1].confidence):
                best = (template, hit)
        if best is None:
            break
        template, hit = best
        centre = hit.center
        sx, sy = drv.to_screen(centre.x, centre.y)

        pyautogui.PAUSE = 0.0
        pyautogui.moveTo(sx, sy, duration=move_time)
        pyautogui.mouseDown()
        time.sleep(hold)
        pyautogui.mouseUp()
        key = Path(template.name).stem
        popped[key] = popped.get(key, 0) + 1
        time.sleep(0.12)
        frame = drv.grab()
    return popped


def _open_dataset(opts, say):
    """Build the sample collector for this round, or None.

    Kept out of `play_loop` so a missing or unwritable directory is a message
    and a round played normally, never a round that does not happen.
    """
    directory = getattr(opts, "dataset", "")
    if not directory:
        return None
    from .dataset import DatasetWriter

    writer = DatasetWriter(Path(directory),
                           per_round=getattr(opts, "dataset_limit", 20),
                           every=getattr(opts, "dataset_every", 4),
                           quality=getattr(opts, "dataset_quality", 85),
                           delay=getattr(opts, "dataset_delay", 0.25),
                           frames=getattr(opts, "dataset_frames", 3),
                           gap=getattr(opts, "dataset_gap", 0.05),
                           floor_mult=getattr(opts, "dataset_floor_mult", 8.0),
                           max_motion=getattr(opts, "dataset_max_motion", 12.0),
                           max_mb=getattr(opts, "dataset_max_mb", 2048.0),
                           max_total=getattr(opts, "dataset_total", 0))
    if not writer.enabled:
        return None
    say(f"collecting up to {writer.per_round} detection sample(s) this round "
        f"({writer.frames} frame(s) at +{writer.delay:.2f}s)")
    return writer


def model_candidates(path: str) -> list:
    """Every place a model could be, nearest first.

    A bare `models/reject.onnx` in a flow is relative to the WORKING
    DIRECTORY, and a tray-launched .exe does not have the app root as its cwd.
    `Config.resolve` is what every other shipped path goes through -- templates
    and flows both -- so model files go through it too.
    """
    p = Path(path)
    out = [p]
    if p.is_absolute():
        return out
    try:
        from ..config import Config, bundle_dir
        for cand in (Config().resolve(p), (bundle_dir() / p) if bundle_dir() else None):
            # THE BUNDLE IS THE LAST RESORT AND THE ONE THAT WAS MISSING.
            # `default_app_root()` hands the whole data directory to the .exe's
            # own folder the moment a `config.yaml` appears there -- which is
            # the documented way to override flows and templates. It also means
            # `Config.resolve` stops looking at the copy baked INTO the .exe.
            # For config, flows and templates that is right: the point is to
            # override them. For a model it is not: nobody edits an .onnx, and
            # a build that carries one should be able to use it even when the
            # folder beside the .exe has been customised for something else.
            if cand is not None and cand not in out:
                out.append(cand)
    except Exception:                      # config is optional for the CLI
        pass
    return out


#: How many rounds this process has started under `--ab`. Module level because
#: alternation is a property of the RUN, not of a round: `play_loop` is called
#: once per round and cannot remember anything by itself, and the tray plays
#: rounds back to back in one process for hours.
#:
#: A restart resets it, which can leave the arms uneven. That is reported by
#: `scripts/ab_eval.py` rather than hidden -- an uneven split is a weaker
#: experiment, and a silent one would be a wrong experiment.
_AB_ROUNDS = 0


def _ab_arm(opts, say) -> str:
    """Set the option `--ab` names to this round's arm, and say which it is.

    **Why the alternation lives here and not in the flows.** The tray runs
    `launch` or `resume`, never `play`, and `run_flow` re-applies each flow's
    own `vars:` on the way down -- so a value has to be declared AND forwarded
    at every hop, and the three times this project has failed to do that cost
    it 11, 18 and 50 rounds. `play_loop` is the one place every round passes
    through exactly once, so an arm chosen here cannot be dropped by a hop.

    It is also self-recording. `play_settings` snapshots the whole namespace
    into every sample at write time, so the arm this sets appears in the corpus
    under the option's own name, with nothing to remember and no new field to
    keep in step.

    Returns the arm's name for the round record: "on", "off", or "" when no
    A/B is running.
    """
    global _AB_ROUNDS
    name = str(getattr(opts, "ab", "") or "")
    if not name:
        return ""
    if not hasattr(opts, name):
        # Loudly, and without stopping the round: a typo here would otherwise
        # alternate nothing at all while every round recorded `ab` set, which
        # is the exact shape of an experiment that looks like it ran.
        say(f"    A/B: there is no play option called {name!r} -- "
            f"nothing is being alternated")
        return ""
    on_value = getattr(opts, name)
    off_raw = getattr(opts, "ab_off", "")
    # Coerced to the live option's type so `ab_off: 0` reaches an int option as
    # 0 and a str option as "0", rather than arming a bool with a string that
    # happens to be truthy -- which is how an unresolved "${verify_clears}"
    # once turned a check on for every drag of a round nobody was measuring.
    if isinstance(on_value, bool):
        off_value = str(off_raw).strip().lower() in ("1", "true", "yes", "on")
    elif isinstance(on_value, int):
        off_value = int(float(off_raw or 0))
    elif isinstance(on_value, float):
        off_value = float(off_raw or 0.0)
    else:
        off_value = str(off_raw)
    if off_value == on_value:
        say(f"    A/B: {name} is {on_value!r} in both arms -- "
            f"nothing is being alternated")
        return ""
    arm = "on" if _AB_ROUNDS % 2 == 0 else "off"
    _AB_ROUNDS += 1
    if arm == "off":
        setattr(opts, name, off_value)
    say(f"    A/B round {_AB_ROUNDS}: {name}={getattr(opts, name)!r} "
        f"({arm.upper()} arm)")
    return arm


def _model_path(path: str) -> Path:
    """The model to load: the first candidate that has BOTH of its files.

    A crop model is two files -- the weights and the `.json` beside them that
    carries the class order and the normalisation. Picking the location on the
    weights alone found `./models/reject.onnx`, then looked for its sidecar at
    `./models/reject.json`, which was not there, and failed the round with an
    error naming a file nobody had asked for. Both, or keep looking.
    """
    tried = model_candidates(path)
    for c in tried:
        if c.exists() and c.with_suffix(".json").exists():
            return c
    for c in tried:                        # the weights alone: better error
        if c.exists():
            return c
    return tried[0]


def _model_missing(path: str, exc: Exception) -> str:
    """Why a model would not load, naming every place that was looked in."""
    lines = [f"{exc}"]
    for c in model_candidates(path):
        have = ("weights yes" if c.exists() else "weights NO")
        side = ("sidecar yes" if c.with_suffix(".json").exists()
                else "sidecar NO")
        lines.append(f"    tried {c}  ({have}, {side})")
    lines.append("    A crop model is TWO files: <name>.onnx and <name>.json. "
                 "Copy both, or rebuild -- `models/` now ships with the app.")
    return chr(10).join(lines)


def _load_palette(path: str, say) -> Optional[np.ndarray]:
    """Learned colour centres for this round, or None to fit per frame.

    Loud on every outcome. A palette is opt-in, so a round that silently ran
    without the one it was told to use would look exactly like a round where
    the palette did not help -- and that is the reading that would get a
    working palette thrown away.

    A missing or malformed file stops the round rather than falling back:
    asking for a palette and getting the old behaviour is not a degraded
    result, it is a different experiment, and it must not be mistaken for the
    one that was asked for.
    """
    if not path:
        return None
    from .learn import Palette

    p = Palette.load(Path(path))
    m = p.metrics or {}
    note = ""
    if m.get("agreement") is not None and m.get("baseline") is not None:
        note = (f", scored {100 * m['agreement']:.0f}% vs {100 * m['baseline']:.0f}% "
                f"per-frame on {'held-out' if m.get('held_out') else 'its own'} samples")
    say(f"    palette: k={p.k} learned from {p.meta.get('samples', '?')} sample(s) "
        f"in {p.meta.get('sessions', '?')} session(s){note}")
    return p.centres


def play_loop(drv: "Driver", opts, *, stop_when: Optional[Callable] = None) -> "PlayReport":
    """Grab the board, pick the best chain, drag it -- once, or on a loop.

    ``opts`` is the ``play`` command's option set (see :func:`play_defaults`);
    ``drv`` is where every pixel and every click goes, so the same loop serves
    the CLI and the ``play_tsum`` flow action.

    ``stop_when`` is called with each settled frame *before* anything is
    clicked on it. Returning a string ends the run and becomes the report's
    reason -- that is how a flow ends the round on "the scoreboard appeared"
    rather than on a stopwatch.
    """
    say = drv.say

    # FIRST, before the models load and before anything reads a setting: the
    # arm decides what `reject_model` (or whatever `--ab` names) is for this
    # round, and every line below has to see the arm's value, not the flow's.
    ab_option = str(getattr(opts, "ab", "") or "")
    report_arm = _ab_arm(opts, say)

    bubbles = _load_bubbles(drv, opts.bubble) if opts.bubble and not opts.dry_run else []

    deadline = time.perf_counter() + opts.duration if opts.duration > 0 else None
    report = PlayReport()
    report.ab, report.ab_arm = (ab_option, report_arm) if report_arm else ("", "")

    # A palette learned offline from collected samples, or None for the
    # per-frame fit this has always done. Loaded once, here, so a bad path
    # fails before a round starts rather than in the middle of one.
    #
    # What it changes: every `palette = None` below means "throw the colours
    # away and re-derive them from whatever this one frame shows", and with a
    # learned palette there is nothing to re-derive -- the colours came from
    # thousands of frames across many rounds and do not belong to any one of
    # them. So the resets restore the learned centres instead of clearing
    # them, and a cluster id keeps meaning the same character for the whole
    # round. See :mod:`ttheart_sender.game.learn`.
    learned = _load_palette(getattr(opts, "palette", ""), say)

    # Loaded here, with the palette, so a bad path fails before a round starts
    # rather than in the middle of one.
    rejector = None
    if getattr(opts, "reject_model", ""):
        try:
            rejector = load_reject_model(
                _model_path(opts.reject_model),
                float(getattr(opts, "reject_floor", RejectModel.FLOOR)))
            say(f"    board filter: dropping detections below "
                f"{rejector.floor:.2f}")
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            # TTHeartError, never SystemExit. SystemExit derives from
            # BaseException, so it goes straight past the runner's
            # `except TTHeartError` AND its `except Exception`, out of the
            # flow, out of the service thread, and kills the tray -- with no
            # message at all, because a windowed build has no stderr. That is
            # what "the app halts when the tsums drop" was: the model file was
            # not beside the .exe, and saying so killed the app instead.
            raise TTHeartError(
                f"could not load the board filter {opts.reject_model!r}. The "
                f"round is stopped rather than played without it -- a round "
                f"that silently drops the filter looks exactly like one that "
                f"ran with it and did nothing.\n"
                + _model_missing(opts.reject_model, exc)) from exc

    ranker = None
    if getattr(opts, "chain_model", ""):
        try:
            ranker = load_chain_model(
                _model_path(opts.chain_model),
                float(getattr(opts, "chain_bonus", 0.0)))
            say(f"    chain ranker: picking by expected accepted members"
                + (f", +{ranker.bonus:.2f} per member for length"
                   if ranker.bonus else ""))
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            raise TTHeartError(
                f"could not load the chain ranker {opts.chain_model!r}.\n"
                + _model_missing(opts.chain_model, exc)) from exc

    characters = None
    if getattr(opts, "character", ""):
        try:
            min_visible = float(getattr(opts, "character_min_visible",
                                        CHARACTER_MIN_VISIBLE))
            characters = load_character_model(_model_path(opts.character),
                                              opts.character_confidence,
                                              min_visible)
            say(f"    character model: {len(characters.classes)} classes, "
                f"naming above {opts.character_confidence:.2f} confidence "
                f"and {min_visible:.2f} visible "
                f"({', '.join(characters.classes[:6])}"
                f"{'...' if len(characters.classes) > 6 else ''})")
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            # Refused, not limped past: a round played with the model silently
            # off looks exactly like one played with it on and doing nothing,
            # and this project has lost two nights to that shape already.
            #
            # TTHeartError rather than SystemExit -- see the board filter's
            # loader below for why that distinction cost a build.
            raise TTHeartError(
                f"could not load the character model {opts.character!r}.\n"
                + _model_missing(opts.character, exc)) from exc

    # Clamped rather than rejected, the way `DatasetWriter` clamps its own:
    # a delay under the render floor is asking for a reading of a highlight
    # the game has not drawn yet, and obeying it silently is how a feature
    # gets measured in the one condition where it cannot work.
    if opts.verify_reach > 0:
        from .dataset import RENDER_FLOOR
        if opts.verify_delay < RENDER_FLOOR:
            say(f"    verify-delay {opts.verify_delay:.2f}s is below the "
                f"{RENDER_FLOOR:.2f}s render floor -- raising it, or the check "
                f"reads a mark that is not on screen yet")
            opts.verify_delay = RENDER_FLOOR
        say(f"    verifying chains that reach past {opts.verify_reach:.0f}px, "
            f"reading at {opts.verify_delay:.2f}s")
        if opts.verify_extend:
            say(f"    rebuilding checked chains from the marks, at "
                f"{opts.verify_floor_mult:.0f}x the board's noise floor")
    elif opts.verify_extend and not opts.verify_hold:
        # The rule has nothing to run on without a check to ride along with,
        # and a switch that silently does nothing is worse than one that says
        # so -- the twelfth round lost a night to exactly that shape.
        say("    --verify-extend needs --verify-reach or --verify-hold to have "
            "a reading to rebuild from; it will not fire")

    # Fitting the colour palette is the expensive half of detection, and the
    # tsums in play don't change mid-game, so it's fit once and reused. Same for
    # the radius and the base-tsum lookup.
    palette, radius, base = learned, opts.radius, None
    #: The equipped tsum's own icon colour, read once when `base` is, and
    #: carried into every sample so a corpus states which character played it.
    base_icon: Optional[dict] = None

    # The pile does not change size during a round, so the radius is a physical
    # constant and re-measuring it every time the fit is thrown away is a
    # liability rather than a refresh. Measured over 151 captured frames of one
    # round, the per-frame estimate ranges 8-38px on a board whose faces are
    # ~26px, and 55% of frames read under 18px.
    #
    # Why this matters beyond tidiness: a collapsed radius reads the board at
    # half scale, which roughly doubles the detection count, and a doubled
    # count sits comfortably inside `--min-tsums`..`--max-tsums`. The frame is
    # accepted, phantom chains are found and dragged, and because chains exist
    # the "nothing playable -> tap shuffle" recovery never runs. It is a
    # failure with no symptom except that nothing clears.
    #
    # Which is also why the count gate cannot be what decides a warm-up sample
    # is trustworthy: the very failure being screened for is one that inflates
    # the count. `_face_coverage` is the test that does work -- see there.
    # With collapsed frames screened out, the samples are no longer skewed one
    # way, so the summary is the MEDIAN. (Before the coverage gate existed the
    # lock had to take the max, because half the pool was collapsed and the
    # median inherited it; that version could still lock under 20px on ~2% of
    # rounds, and reached 38.6px on 5%.)
    radius_samples: deque = deque(maxlen=max(1, opts.radius_lock))
    locked: Optional[float] = None
    #: Said once per round, not once per frame -- see where it is set.
    warned_layout = False
    unlocked_frames = 0
    fever = FeverWatch(drv.matcher, drv.templates,
                       use_banner=getattr(opts, "fever_banner", True),
                       hold=getattr(opts, "fever_hold", FEVER_HOLD),
                       board=opts.board)
    if not fever.enabled:
        say("no max_fever or fever_bonus template -- FEVER will be played on "
            "the normal board rect")
    elif not fever.reads_state:
        say("no fever_bonus template -- FEVER will be inferred from the meter "
            "and a 10s timer rather than read off the screen")
    played = misses = stalls = 0
    samples = _open_dataset(opts, say)
    # Rolling window of "which chain did we just play". A board that keeps
    # offering the same handful of chains is a board nothing is clearing on --
    # see the loop-detection check below.
    recent: deque = deque(maxlen=opts.repeat_window)
    lengths: deque = deque(maxlen=max(opts.repeat_len, opts.repeat_window))
    per_step = opts.per_step
    skip_kinds: set[int] = set()
    #: Set when the last pass ended without touching the board, so the next one
    #: can skip waiting for movement that cannot have happened.
    no_settle = False
    #: Filled in from the first frame and reused: the board does not move.
    settle_rect: Optional[tuple] = None

    try:
        while True:
            drv.check_stop()
            if opts.dry_run or palette is None or no_settle:
                frame = drv.grab()
            else:
                probe_settle: dict = {}
                frame = _settle(drv, max_wait=opts.settle, out=probe_settle,
                                region=settle_rect if opts.settle_board else None)
                report.settle_s += probe_settle.get("waited", 0.0)
                report.settles += 1
                report.settle_timeouts += bool(probe_settle.get("timed_out"))
            no_settle = False

            # Asked before anything on this frame is touched: once the round is
            # over the board is gone, and a chain "found" on the results screen
            # would drag across whatever buttons are sitting there.
            if stop_when is not None:
                reason = stop_when(frame)
                if reason:
                    say(f"    {reason}")
                    report.reason = reason
                    break

            # Skill first: it is charged by the chains already played, and
            # firing it is worth more than any single chain.
            if opts.skill and not opts.dry_run:
                gold = skill_gold(frame, opts.skill, opts.skill_inner, opts.skill_outer)
                if gold >= opts.skill_gold:
                    say(f"    skill charged (ring {gold:.2f}) -- firing")
                    _click_shuffle(drv, opts.skill, 1, 0.0,
                                  opts.hold, opts.move_time)
                    frame = _settle(drv, max_wait=max(opts.settle, 1.2))

            # Bubbles are worth more than a chain and vanish on their own, so
            # they get checked first and the frame is re-grabbed if any popped.
            if bubbles:
                n_pop = _pop_bubbles(drv, frame, bubbles,
                                     opts.bubble_confidence, opts.hold, opts.move_time)
                if n_pop:
                    what = ", ".join(f"{n}x {name}" for name, n in n_pop.items())
                    say(f"    popped {what}")
                    frame = _settle(drv, max_wait=opts.settle)

            # The board rect is not fixed: FEVER gets the wider one. Cached
            # palette, radius and base all belong to the rect they were fitted
            # on, so a switch throws them away rather than reading the new crop
            # through the old crop's colours.
            fever.update(frame)
            flipped = fever.took_effect()
            if flipped:
                say(f"    {flipped.upper()} -- board rect now "
                    f"{_board_rect(frame.shape, opts.board, fever=fever.active)}")
                # Palette and base are colour-bound and cluster-id-bound, so
                # they have to go. The radius is not: FEVER repaints the board,
                # it does not resize the tsums. Re-estimating here means
                # re-estimating on the dimmest, most animated frames of the
                # round -- exactly where the estimator collapses -- and then
                # living with that number for the whole ten seconds.
                #
                # A learned palette is exempt from the first half of that: it
                # was fitted over frames from both sides of the flip, so it
                # already carries FEVER's neon as well as normal play's
                # colours. Refitting it here would replace a corpus of
                # thousands of frames with the single worst frame of the
                # round. `base` still goes -- the skill icon is re-read
                # against whatever centres are in force.
                palette, base = learned, None
                if locked is None:
                    radius = opts.radius
            bx, by, bw, bh = _board_rect(frame.shape, opts.board, fever=fever.active)
            settle_rect = (bx, by, bw, bh)
            report.frames += 1
            report.fever_frames += bool(fever.active)
            crop = frame[by:by + bh, bx:bx + bw]

            # A measured layout knows the tsum size outright, so there is
            # nothing to estimate and nothing to warm up. `--radius` still wins
            # over it, and an unmeasured capture size falls through to the
            # estimator and `--radius-lock` as before.
            if locked is None and opts.radius is None:
                known = _layout_radius(frame.shape)
                if known:
                    locked = known
                    say(f"    radius {locked:.1f}px, measured for this layout "
                        f"(not estimated -- every tsum is the same size)")
                elif not warned_layout:
                    # SAY SO, LOUDLY, ONCE. An unmeasured capture size is not a
                    # small degradation: the board rect falls back to fractions
                    # of the whole frame, which on LDPlayer swallows the score
                    # bar and the FEVER strip and detects them, and the radius
                    # falls back to an estimator that reads 8-38px on a board
                    # whose faces are 25. Both failures are silent -- the count
                    # stays inside `--min-tsums`..`--max-tsums`, so nothing
                    # else complains -- and the round simply plays badly.
                    #
                    # Resizing the emulator is the way this happens, and it
                    # happened: a resolution change moved the capture from
                    # 578x994 to 598x1031 and nothing said a word. See LAYOUTS.
                    warned_layout = True
                    say(f"    NO MEASURED LAYOUT for a {frame.shape[1]}x"
                        f"{frame.shape[0]} capture -- falling back to a "
                        f"fraction of the frame and an estimated radius, which "
                        f"plays worse. Add an entry to LAYOUTS in "
                        f"ttheart_sender/game/tsum.py, or pass --board/--radius")

            t0 = time.perf_counter()
            tsums, radius, palette = detect(crop, k=opts.k, radius=locked or radius,
                                            palette=palette,
                                            scale=opts.scale, include_dark=opts.include_dark,
                                            merge=opts.merge, bowl_reject=opts.bowl_reject,
                                            recolour=opts.recolour, kinds=opts.kinds,
                                            fit_effort=opts.fit_effort)

            # FEVER repaints the whole board in neon, so a palette fit during
            # normal play stops matching anything and the tsum count collapses.
            # Rather than trust a stale fit, throw it away and refit whenever
            # the count looks wrong -- that covers entering fever, leaving it,
            # and calibrating off a menu frame by accident. Cluster ids are only
            # meaningful within one fit, so the base tsum is re-read too.
            # Refit on anything implausible, not just a low count -- a cached
            # radius that has drifted small blows the count *up* rather than
            # down, and that reads as a healthy board unless the ceiling is
            # checked too.
            # `--min-tsums` is there to recognise that a frame is not a board at
            # all -- the Home screen scores 200+ "tsums" off portraits, the
            # results screen scores a handful. During FEVER that job is already
            # done, and done better, by the template `FeverWatch` matched: the
            # game does not show a FEVER meter over a menu. What FEVER does do
            # is fade and overlay the board, so a genuine in-play frame reads
            # ~20 tsums where normal play reads ~50, and the floor throws away
            # frames that were perfectly playable. Measured over the dim frames
            # of one round at a correct radius, the median count is 20 -- i.e.
            # sitting exactly on the default floor, half of them discarded.
            floor = (opts.fever_min_tsums
                     if fever.active and opts.fever_min_tsums else opts.min_tsums)
            plausible = floor <= len(tsums) <= opts.max_tsums
            if not plausible and palette is not None and learned is None:
                # Skipped outright on a learned palette, rather than run and
                # discarded. This branch exists because a per-frame fit goes
                # stale when the board is repainted, and it repairs that by
                # replacing the cached centres with fresh ones -- which on a
                # learned palette would swap a corpus of thousands of frames
                # for this one implausible frame, permanently, for the rest of
                # the round. That is the single worst frame to learn colours
                # from, and it is not a trade worth making: a learned palette
                # cannot go stale, so an implausible count under one is a
                # radius problem or a frame that is not a board, and both are
                # already handled by the count gate below.
                #
                # A refit re-fits the COLOURS. It must not also re-roll the
                # radius once that is locked, or the recovery path becomes the
                # way a collapsed estimate gets in.
                fresh, fresh_r, fresh_pal = detect(crop, k=opts.k, scale=opts.scale,
                                                   radius=locked,
                                                   include_dark=opts.include_dark,
                                                   merge=opts.merge,
                                                   bowl_reject=opts.bowl_reject,
                                                   recolour=opts.recolour, kinds=opts.kinds,
                                                   fit_effort=opts.fit_effort)
                if abs(len(fresh) - floor) < abs(len(tsums) - floor):
                    say(f"    recalibrated ({len(tsums)} -> {len(fresh)} tsums)")
                    tsums, radius, palette, base = fresh, fresh_r, fresh_pal, None

            # Sample the radius only from frames that read like a whole board.
            # A frame that fails the coverage test is not discarded as a frame
            # -- it is played normally -- it just does not get a vote on how
            # big a tsum is.
            if locked is None and opts.radius_lock:
                if _face_coverage(tsums, radius, crop.shape) >= opts.radius_cover:
                    radius_samples.append(radius)
                    if len(radius_samples) == radius_samples.maxlen:
                        locked = float(np.median(radius_samples))
                        say(f"    radius locked at {locked:.1f}px "
                            f"(median of {len(radius_samples)}: "
                            f"{', '.join(f'{r:.0f}' for r in radius_samples)})")
                else:
                    unlocked_frames += 1
                    # Not an error -- a sparse board or an unusual layout can
                    # simply never qualify, and then the round runs the way it
                    # did before any of this existed. Worth saying once, since
                    # the alternative is wondering why the lock line never
                    # appears in the log.
                    if unlocked_frames == opts.radius_lock * 10:
                        say(f"    no radius lock yet after {unlocked_frames} frames "
                            f"below {opts.radius_cover:.2f} coverage -- "
                            f"running unlocked (lower --radius-cover to change that)")

            # BEFORE identity and before any chain is built: a detection that
            # is board is not a tsum of any character, and leaving it in place
            # to be grouped and then chained is what spends a chain slot on a
            # stroke over empty bowl.
            if rejector is not None:
                tsums = rejector.keep(crop, tsums, radius)

            if characters is not None:
                characters.apply(crop, tsums, radius)
                # `read_base_kind` answers with a palette index, and a named
                # tsum no longer carries one -- so the equipped character has
                # to be found again from the icon's own colour, exactly as
                # `--recolour` needs. Without this the base priority points at
                # an arbitrary group and the skill quietly stops charging.
                if opts.use_base and base_icon:
                    base = _base_from_faces(crop, tsums, radius,
                                            base_icon.get("lab"))

            if base is None and opts.use_base:
                seen_base: dict = {}
                base, base_dist = read_base_kind(frame, palette, spec=opts.base,
                                                 out=seen_base)
                base_icon = {"kind": base,
                             "lab": seen_base.get("icon_lab"),
                             "distance": round(float(base_dist), 1)}
                say(f"base tsum: cluster #{base} (Lab distance {base_dist:.1f}, "
                    f"icon Lab {seen_base.get('icon_lab')})")
            if (opts.recolour > 0 or opts.kinds > 0) and opts.use_base and base_icon:
                # `read_base_kind` answers with an index into the PALETTE
                # centres, and `--recolour` throws those away: it renumbers
                # `kind` to its own group ids, which have nothing to do with
                # k-means cluster numbers. Left alone, turning recolour on
                # would silently point `base_kind` at an arbitrary group --
                # the bot would stop preferring the equipped character, which
                # is the thing that charges the skill, and nothing would say
                # so. Re-derived here per frame from the one identifier that
                # survives a renumbering: the icon's own Lab colour.
                base = _base_from_faces(crop, tsums, radius, base_icon.get("lab"))
            # Only quantise a second time when the rule that needs it is on:
            # detect() keeps the centres, not the label map, and refitting the
            # labels costs a GEMM over the crop.
            cluster_map = (_quantise(crop, opts.k, palette)[0]
                           if opts.mode == "blob" and palette is not None else None)
            chains = find_chains(tsums, radius, opts.link, block=opts.block,
                                 link_px=opts.link_px, base_kind=base, base_only=opts.base_only,
                                 mode=opts.mode, max_chain=opts.max_chain,
                                 first_leg_px=opts.first_leg_px, labels=cluster_map)
            think = (time.perf_counter() - t0) * 1000

            # A real board is crowded but bounded. Menus and the results screen
            # still yield blobs and can still produce a "chain", and dragging
            # that would swipe across live UI buttons -- so a frame outside the
            # plausible range is treated as no board at all rather than trusted.
            #
            # The ceiling matters as much as the floor: the Home screen scores
            # 200+ "tsums" off portraits and panel texture, sails past any
            # minimum, and produces a confident chain every single frame.
            if not (floor <= len(tsums) <= opts.max_tsums):
                chains = []

            # Chain length is set by detection recall, not by the search: a
            # tsum missed in the middle of a run of six splits it into a three
            # and a two. Holding out for a longer chain is therefore mostly a
            # bet that the next frame is better settled.
            longest = len(chains[0]) if chains else 0
            chains = [c for c in chains
                      if len(c) >= opts.min_chain and c.kind not in skip_kinds]

            # AFTER the length floor and the blacklist, so the ranker reorders
            # exactly the set that would otherwise be played longest-first,
            # and BEFORE the purity loop below, which takes chains[0] onwards.
            if ranker is not None and chains:
                chains = ranker.rank(chains, tsums, radius, fever.active)

            if not chains:
                misses += 1
                why = f"best was {longest}, want {opts.min_chain}" if longest else "no chain"
                say(f"  skip ({why}; {len(tsums)} tsums, {think:.0f}ms)")
                if deadline is None:
                    report.reason = why
                    break
                if time.perf_counter() >= deadline:
                    report.reason = f"{opts.duration:.0f}s elapsed"
                    say(report.reason)
                    break
                # A handful of misses in a row usually isn't an empty board --
                # restarting the command by hand routinely un-sticks exactly
                # this, and the only thing a restart does differently is start
                # with no cached palette, radius, base, or skip-list. So do
                # that here instead of giving up: throw every cached fit away
                # and refit clean, which is the same recovery `min_tsums`
                # already does, just triggered by "nothing playable" rather
                # than "barely anything detected". `skip_kinds` in particular
                # only ever clears on a successful drag -- a long miss streak
                # with no successful drag in between could otherwise leave it
                # blacklisting every kind on the board forever.
                if misses >= opts.max_misses:
                    say(f"    nothing playable {misses}x in a row -- "
                          f"tapping shuffle and recalibrating")
                    _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                                  opts.shuffle_delay, opts.hold, opts.move_time)
                    _settle(drv, max_wait=max(opts.settle, 1.5))
                    palette, base = learned, None
                    if locked is None:
                        radius = opts.radius
                    skip_kinds.clear()
                    misses = 0
                continue

            # Last gate before committing to a drag: make sure every stop in
            # this chain really is the same character. A single misclassified
            # member makes a 3-chain register as a 2-link, which the game
            # ignores entirely -- the drag runs and nothing clears.
            #
            # A chain failing this is a reason to try the NEXT chain, not to
            # throw the frame away: the board usually offers several, and
            # discarding all of them because the best one had an odd member
            # stalls the run outright.
            best = None
            floor_len = max(MIN_CHAIN, opts.min_chain)
            for cand in chains:
                if opts.purity <= 0:
                    best = cand
                    break
                clean = purity_filter(crop, tsums, cand.nodes, radius, opts.purity)
                if len(clean) >= floor_len:
                    if len(clean) < len(cand.nodes):
                        say(f"    trimmed {len(cand.nodes) - len(clean)} odd-coloured "
                              f"member(s) from a chain of {len(cand)}")
                    best = Chain(cand.kind, cand.colour, clean, cand.is_base)
                    break

            if best is None:
                # Every chain on the board failed. Fall into the same recovery
                # the other dead ends use rather than looping on it forever.
                misses += 1
                say(f"  skip (all {len(chains)} chain(s) had odd-coloured members; "
                      f"{len(tsums)} tsums)")
                if deadline is None:
                    report.reason = "every chain had odd-coloured members"
                    break
                if time.perf_counter() >= deadline:
                    report.reason = f"{opts.duration:.0f}s elapsed"
                    say(report.reason)
                    break
                if misses >= opts.max_misses:
                    say(f"    nothing playable {misses}x in a row -- "
                          f"tapping shuffle and recalibrating")
                    _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                                  opts.shuffle_delay, opts.hold, opts.move_time)
                    _settle(drv, max_wait=max(opts.settle, 1.5))
                    palette, base = learned, None
                    if locked is None:
                        radius = opts.radius
                    skip_kinds.clear()
                    misses = 0
                continue

            # Third and least obvious way to get stuck: chains keep being found
            # and drags keep looking like they landed, but the board never
            # actually advances -- the log shows the same few chains cycling
            # forever. Neither of the other guards catches it. It isn't a miss
            # streak (chains exist), and it isn't a stall streak either,
            # because idle tsums jiggle enough to push the before/after frame
            # diff past --change-tol even when nothing cleared.
            #
            # The giveaway is repetition: a board that keeps re-offering the
            # same chain is a board that chain isn't clearing. Quantising the
            # endpoints keeps detection jitter of a few pixels from reading as
            # a different chain each time.
            # Coarser companion to the signature check below: the same chain
            # *length* over and over is the tell when positions drift a little
            # each frame but the board never really advances -- the signature
            # differs every time so the exact-match check never fires, yet
            # nothing is being cleared.
            lengths.append(len(best))
            if (len(lengths) >= opts.repeat_len
                    and len(set(list(lengths)[-opts.repeat_len:])) == 1):
                say(f"    chain of {len(best)} x{opts.repeat_len} in a row -- "
                      f"board is not advancing; tapping shuffle")
                _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                              opts.shuffle_delay, opts.hold, opts.move_time)
                _settle(drv, max_wait=max(opts.settle, 1.5))
                palette, radius, base = learned, opts.radius, None
                skip_kinds.clear()
                lengths.clear()
                recent.clear()
                continue

            sig = (best.kind, len(best),
                   int(tsums[best.nodes[0]].x) // 12, int(tsums[best.nodes[0]].y) // 12,
                   int(tsums[best.nodes[-1]].x) // 12, int(tsums[best.nodes[-1]].y) // 12)
            recent.append(sig)
            if recent.count(sig) >= opts.max_repeats:
                say(f"    same chain {recent.count(sig)}x in {len(recent)} plays -- "
                      f"board is not advancing; tapping shuffle")
                _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                              opts.shuffle_delay, opts.hold, opts.move_time)
                _settle(drv, max_wait=max(opts.settle, 1.5))
                palette, radius, base = learned, opts.radius, None
                skip_kinds.clear()
                recent.clear()
                continue

            # crop -> content-area -> screen. pyautogui works in screen
            # coordinates, which is why the window gets parked at the top-left.
            screen = [
                drv.to_screen(int(tsums[i].x) + bx, int(tsums[i].y) + by)
                for i in best.nodes
            ]
            tag = "BASE" if best.is_base else f"#{best.kind}"
            say(f"  {tag} chain of {len(best)}  ({len(tsums)} tsums, {think:.0f}ms)"
                  + ("  " + " -> ".join(f"({x},{y})" for x, y in screen) if opts.dry_run else ""))

            if opts.dry_run:
                say("dry run -- not touching the mouse")
                report.reason = "dry run"
                break

            before = crop
            probe: dict = {}

            # HOW FAR THE CHAIN REACHES from the tsum being pressed, and
            # whether that is far enough to be worth checking before dragging.
            #
            # Asking the game costs the same on every drag, but being wrong
            # does not: measured over 303 collected drags, the share where the
            # game accepted every proposed member falls away with reach --
            #     under 90px  100% clean      150-220px   65%
            #     90-150px     81%            220-300px   33%
            #                                 over 300px  11%
            # A chain that stays near the press is almost always right and
            # paying to verify it is pure loss; one that reaches across the
            # board is usually wrong. So the check is bought only where it
            # pays. Off unless `--verify-reach` is set; `--verify-hold` still
            # means "every drag" and is unaffected.
            head_t = tsums[best.nodes[0]]
            reach_px = max(math.hypot(tsums[i].x - head_t.x, tsums[i].y - head_t.y)
                           for i in best.nodes)
            verifying = bool(opts.verify_hold) or (
                opts.verify_reach > 0 and reach_px > opts.verify_reach)
            # A reach-triggered check reads at its own delay. `--hold-delay`
            # defaults to 0.10, which is below the 0.15 floor the highlight
            # needs to have rendered at all -- that is why `--verify-hold` is
            # documented as measured-and-not-recommended, and re-using the
            # number here would reproduce the same finding.
            verify_delay = (opts.verify_delay
                            if (verifying and not opts.verify_hold) else opts.hold_delay)

            def _ask_the_game():
                """Read what the game marks while the first tsum is held.

                Two callers, one capture. `--verify-hold` uses the answer to
                trim the stroke; the sample collector wants the frame it was
                read from. Either alone is reason enough to press and look,
                so with only `--dataset` on, the marks are recorded and the
                chain is dragged exactly as proposed.
                """
                seen: dict = {}
                # A sampled drag reads the mark the collector's way -- long
                # enough for the game to have drawn it, and more than once, so
                # motion does not pass for a highlight. `--verify-hold` alone
                # keeps its own faster, blinder read: it pays on every drag,
                # and that cost is the reason it is tuned the way it is.
                collecting = samples is not None and sampling
                kept = marked_by_game(
                    drv, before, (bx, by, bw, bh), tsums, best.nodes,
                    delay=samples.delay if collecting else verify_delay,
                    threshold=opts.hold_threshold, aura=opts.hold_aura,
                    frames=samples.frames if collecting else 1,
                    gap=samples.gap if collecting else 0.0,
                    # `floor_mult` sets the bar the *label* is read at and
                    # nothing else -- the trim below keeps the fixed threshold
                    # either way, so arming the extend rule cannot silently
                    # re-score `--verify-reach`'s own A/B. The extend rule
                    # needs it because a mark read at the flat 8.0 admits 23
                    # to 35 tsums per press, and a chain built from that is
                    # built from the board's noise.
                    floor_mult=(samples.floor_mult if collecting
                                else (opts.verify_floor_mult
                                      if opts.verify_extend else 0.0)),
                    out=seen)
                if collecting:
                    samples.record(
                        before, seen.get("marked_frame"), reading=seen,
                        board=(bx, by, bw, bh),
                        radius=radius, tsums=tsums, head=best.nodes[0],
                        proposed=best.nodes, kept=kept, fever=fever.active,
                        base=base_icon, options=play_settings(opts),
                    )
                if not verifying:
                    # Collection must not change how the round is played.
                    return screen
                if not seen.get("marked"):
                    # The frame showed nothing at all -- not one tsum on the
                    # whole board cleared the bar, the pressed one included.
                    # That is not the game saying "refused", it is the game
                    # saying nothing, and trimming on it throws away a chain
                    # on the strength of a reading that failed. Measured over
                    # 726 collected drags it happens on 4.4% of them, and on 6
                    # of the 121 a `verify_reach 260` check would fire on --
                    # two of which are chains the trim cancelled outright.
                    probe["unreadable"] = True
                    return screen
                if opts.verify_extend:
                    # The check has already been paid for and the game has
                    # already named its partners. Trimming spends that answer
                    # on the members it takes away; this also spends it on the
                    # ones it hands over. Runs before the min_chain test on
                    # purpose: a chain the trim would abandon is exactly the
                    # one the marks are most likely to rebuild.
                    grown = chain_from_marks(
                        tsums, best.nodes, kept, seen.get("marked") or [],
                        radius, link_px=opts.link_px, block=opts.block,
                        max_chain=opts.max_chain)
                    if len(grown) > len(kept):
                        report.extended += 1
                        report.extended_by += len(grown) - len(kept)
                        kept = grown
                if opts.verify_no_trim and len(kept) <= len(best.nodes):
                    # SPEND THE CHECK ON THE REBUILD ONLY, NEVER ON THE TRIM.
                    #
                    # The trim exists because a refused member was believed to
                    # cost the whole drag. The fifteenth round measured that
                    # and it is false, and the sixteenth found the shape:
                    # refusals are a SUFFIX -- the chain dies at the first bad
                    # member, and 78% of refusing drags refuse everything
                    # after it. Either way the members the trim removes were
                    # already worth nothing, so dropping them buys no clears
                    # and dropping the whole chain (`abandoned`) buys less
                    # than none.
                    #
                    # What the check is still worth is the rebuild above,
                    # which the same round priced at 22 of 22 added members
                    # cleared. So: take the rebuild when it is longer, and
                    # otherwise drag exactly what was proposed. The chain can
                    # then never fall under min_chain, so nothing is
                    # abandoned either.
                    #
                    # Off by default. It is derived from 36 drags and a model
                    # this project has now been wrong about twice.
                    kept = list(best.nodes)
                # Never negative: `verify_extend` can hand back a chain
                # LONGER than the proposal, and a negative "dropped" would be
                # truthy, add itself to `report.trimmed`, and print as
                # "dropped -2 it would not accept".
                probe["dropped"] = max(0, len(best.nodes) - len(kept))
                probe["kept"] = len(kept)
                # The chain actually dragged, which is not `best.nodes` once
                # either verify rule has had its say. `cleared_by_drag` asks
                # "which of the DRAGGED tsums are gone", so it has to be told
                # what was dragged -- with `verify_extend` on, the members the
                # marks added are not in the proposal at all, and measuring
                # the proposal would score the rule's own clears as zero.
                probe["nodes"] = list(kept)
                if len(kept) < opts.min_chain:
                    # Abandon before moving. Releasing on one tsum clears
                    # nothing, which is cheaper than dragging a chain the game
                    # has already said it will not accept.
                    probe["abandoned"] = True
                    return [drv.to_screen(bx + tsums[best.nodes[0]].x,
                                          by + tsums[best.nodes[0]].y)]
                return [drv.to_screen(bx + tsums[i].x, by + tsums[i].y) for i in kept]

            if verifying:
                report.verified += 1
            sampling = samples is not None and samples.wants()
            if samples is not None:
                samples.seen_drag()
            drag_chain(screen, step_px=opts.step_px, per_step=per_step, hold=opts.hold,
                       after_press=_ask_the_game if (verifying or sampling) else None)

            if probe.get("unreadable"):
                report.unreadable += 1

            if probe.get("abandoned"):
                misses += 1
                report.abandoned += 1
                # Without this the next scan finds the identical chain, presses
                # it, and is rejected again -- the board has not changed, so
                # nothing about the answer would either. Same cure the stall
                # path uses: stop offering this kind until something clears.
                skip_kinds.add(best.kind)
                # And do not wait for a board to settle that never moved.
                no_settle = True
                say(f"    the game marked only {probe['kept']} of {len(best.nodes)} "
                      f"-- released without dragging, skipping #{best.kind}")
                # Same exits as every other dead end, or a one-shot run would
                # never return and a timed one could overrun --duration.
                if deadline is None:
                    report.reason = "the game rejected the only chain found"
                    break
                if time.perf_counter() >= deadline:
                    report.reason = f"{opts.duration:.0f}s elapsed"
                    say(report.reason)
                    break
                continue
            if probe.get("dropped"):
                report.trimmed += probe["dropped"]
                say(f"    game marked {probe['kept']}/{len(best.nodes)}; "
                      f"dropped {probe['dropped']} it would not accept")

            # Did anything actually clear? A drag the emulator only half-sampled
            # registers as a 2-link, which is below the game's minimum, so
            # nothing pops and the board is unchanged -- and the next scan finds
            # the identical chain and tries it again forever. That's the freeze.
            if opts.verify:
                after = drv.grab()[by:by + bh, bx:bx + bw]
                changed = float(np.mean(cv2.absdiff(after, before)))
                if changed < opts.change_tol:
                    stalls += 1
                    # `stalls` is a streak, cleared on the next success; this is
                    # the run total, which is what a comparison needs.
                    report.stalled += 1
                    # Two cures, applied together: walk the path more slowly so
                    # every tsum gets sampled, and stop re-offering the chain
                    # that just failed.
                    per_step = min(per_step * 2, 0.05)
                    skip_kinds.add(best.kind)
                    say(f"    no board change ({changed:.1f}) -- drag did not register; "
                          f"slowing to {per_step * 1000:.0f}ms/step, skipping #{best.kind}")
                    # Same story as the miss-streak case: bail out to a clean
                    # slate rather than give up outright. Coordinates are
                    # untouched by this (they come from `window`/`rect`, fixed
                    # for the run), so it's the per-step slowdown and the
                    # skip-list that reset -- if slow-and-skip hasn't found a
                    # working chain in max_stalls tries, that combination isn't
                    # it, so drop back to defaults and let the next
                    # recalibration try fresh.
                    if deadline is not None and time.perf_counter() < deadline \
                            and stalls >= opts.max_stalls:
                        say(f"    drag still not registering after {stalls} tries -- "
                              f"tapping shuffle, resetting speed, and recalibrating")
                        _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                                      opts.shuffle_delay, opts.hold, opts.move_time)
                        _settle(drv, max_wait=max(opts.settle, 1.5))
                        palette, radius, base = learned, opts.radius, None
                        per_step, stalls = opts.per_step, 0
                        skip_kinds.clear()
                    if deadline is None and stalls >= opts.max_stalls:
                        report.reason = "drag keeps failing to register"
                        say(f"{report.reason} -- stopping")
                        break
                    if deadline is not None and time.perf_counter() >= deadline:
                        report.reason = f"{opts.duration:.0f}s elapsed"
                        say(report.reason)
                        break
                    continue
                stalls = 0
                skip_kinds.clear()

                if opts.verify_clears:
                    # The stroke registered -- the board moved. Whether these
                    # particular tsums went is a separate question, and this
                    # is the only place with the frames to answer it.
                    dragged = probe.get("nodes") or best.nodes
                    gone, values, idle = cleared_by_drag(
                        before, after, tsums, dragged, tol=opts.clear_tol)
                    report.checked = True
                    report.cleared += len(gone)
                    if samples is not None:
                        # Onto the sample this drag started, so a round played
                        # to measure clears can be handed over as a dataset
                        # instead of as a log file on the machine that played
                        # it. `dragged` too: what cleared is only meaningful
                        # beside what was attempted, and the two differ once
                        # a verify rule has trimmed or rebuilt the chain.
                        samples.note_outcome(
                            dragged=[int(i) for i in dragged],
                            cleared=[int(i) for i in gone],
                            clear_values=[round(float(v), 1) for v in values],
                            clear_idle=round(float(idle), 2))
                    say(f"    popped {len(gone)}/{len(dragged)} "
                        f"({'/'.join(f'{v:.0f}' for v in values)} vs idle {idle:.0f})")
                    if not gone:
                        # Not a stall: the drag was delivered and the game
                        # declined it, which is a misread chain rather than a
                        # missed stroke -- so blacklist the kind and leave the
                        # drag speed alone.
                        report.rejected += 1
                        skip_kinds.add(best.kind)

            # `misses` is a streak of frames that produced nothing draggable,
            # and it clears here rather than anywhere earlier: every way of
            # ending up with nothing -- no chain, every chain impure, a drag
            # the game refused -- has to feed the same counter, or a board
            # that fails at a later gate than the one being counted keeps
            # resetting it and max_misses is never reached. That is exactly
            # what a purity-fail streak did: chains existed, so the counter
            # was cleared, so the shuffle that would have un-stuck the board
            # never fired.
            misses = 0
            played += 1
            # What was dragged, not what was proposed: with --verify-hold the
            # chain may have been trimmed after the press. Still not a count
            # of what cleared -- see `PlayReport.dragged`.
            report.dragged += probe.get("kept", len(best))

            if deadline is None:
                report.reason = "one chain played"
                break
            if time.perf_counter() >= deadline:
                report.reason = f"{opts.duration:.0f}s elapsed"
                say(report.reason)
                break
    except StopRequested:
        # The stop key is the user aborting everything, not this loop failing:
        # report what was played, then let it travel on to whoever is driving
        # (the CLI exits, the flow runner stops the whole flow).
        report.played, report.stopped = played, True
        say(f"stopped -- {report.describe()}")
        raise
    finally:
        # BEFORE the close, and that ordering is the whole point: `close()`
        # writes `round.json` from this report, and the assignment used to sit
        # after the `finally` block. Every round record ever written therefore
        # said `played: 0` while the log line beside it said 89 chains, and
        # anything derived from it -- mean chain length, chains per second,
        # the throughput half of every score correlation -- was silently
        # taken against a zero.
        report.played = played
        # An aborted round's samples are as good as a finished one's, so the
        # file is closed on every way out rather than only the tidy one.
        if samples is not None:
            samples.close(report)
            if samples.summary():
                say(samples.summary())

    say(report.describe())
    if characters is not None and characters.summary():
        say("    " + characters.summary())
    if ranker is not None and ranker.summary():
        say("    " + ranker.summary())
    if rejector is not None and rejector.summary():
        say("    " + rejector.summary())
    return report


def _play(args) -> int:
    """CLI wrapper: build the driver from an Application and run the loop."""
    from ..app import Application
    from ..control.hotkey import StopKeyWatcher

    app = Application.create()
    window = app.attach_window(prepare=not args.no_prepare)
    watcher = StopKeyWatcher(app.config.runner.stop_key)
    drv = Driver.from_app(app, app.content_rect(), watcher=watcher, say=print)

    if args.duration > 0 and not args.dry_run:
        print(f"running for {args.duration:.0f}s -- {watcher.describe()} to abort")
        time.sleep(args.countdown)

    try:
        report = play_loop(drv, args)
    except StopRequested as exc:
        print(exc)
        return 0
    finally:
        app.close()

    # One-shot mode (no --duration) reports failure when it never got a chain
    # away, so it stays usable as a probe from a shell script.
    return 0 if report.played or args.duration > 0 else 1


def _grab(args) -> int:
    """Save a burst of real boards off the live emulator, for labelling later.

    Tuning detection needs frames, and a round only lasts 60s -- long enough to
    take one screenshot by hand, not the ten it takes to say anything about a
    parameter. So this sits in the round and grabs on a timer.

    Frames whose detection count is outside [--min-tsums, --max-tsums] are
    dropped rather than saved: those are the countdown, the results screen and
    the Home screen, and a label file for one of those is worse than no label
    file, because `eval` would count the nonsense as ground truth.
    """
    from ..app import Application
    from ..control.hotkey import StopKeyWatcher, interruptible_sleep

    app = Application.create()
    app.attach_window(prepare=not args.no_prepare)
    rect = app.content_rect()
    watcher = StopKeyWatcher(app.config.runner.stop_key)

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing board: its .label.json is hand-made and the
    # two are matched by name.
    n = 1
    def _next_path() -> Path:
        nonlocal n
        while (p := out / f"{args.prefix}{n}.png").exists():
            n += 1
        return p

    print(f"grabbing {args.frames} boards every {args.interval:.1f}s into {out}")
    print(f"start a round now -- {watcher.describe()} to stop")

    saved, skipped = 0, 0
    palette, radius = None, args.radius
    try:
        # interruptible_sleep, not time.sleep: the stop key is read by polling
        # whether it is down *right now*, so a press during a plain 3s sleep is
        # simply never seen.
        interruptible_sleep(args.countdown, watcher)
        while saved < args.frames:
            watcher.check()
            frame = app.capture.grab(rect)
            bx, by, bw, bh = _board_rect(frame.shape, args.board)
            tsums, radius, palette = detect(frame[by:by + bh, bx:bx + bw], k=args.k,
                                            radius=radius, palette=palette,
                                            include_dark=args.include_dark)
            if not (args.min_tsums <= len(tsums) <= args.max_tsums):
                skipped += 1
                print(f"  skip: {len(tsums)} tsums -- not a board")
            else:
                path = _next_path()
                # imencode, not imwrite: imwrite can't take a non-ASCII path on
                # Windows, and the scratchpad may sit under a user folder.
                ok, buf = cv2.imencode(".png", frame)
                if ok:
                    buf.tofile(str(path))
                    saved += 1
                    print(f"  {saved}/{args.frames}  {path.name}  {len(tsums)} tsums")
            interruptible_sleep(args.interval, watcher)
    except StopRequested as exc:
        print(exc)
    finally:
        app.close()

    print(f"\nsaved {saved} board(s), skipped {skipped} non-board frame(s)")
    if saved:
        print(f"next: python -m ttheart_sender.game.tsum label {out}/{args.prefix}1.png")
    return 0 if saved else 1


def _idle(args) -> int:
    """Touch nothing and film the hint the game offers an idle player.

    The game suggests a move when you stop playing, and a suggestion is a chain
    it knows is legal -- which is character-identity ground truth, produced by
    the only party that actually knows it, for free. This films the hint rather
    than assuming its shape: if it steps through the chain one tsum at a time,
    the frames hold a complete labelled group; if it only ever marks one tsum,
    they say that instead.

    Too slow to play with -- a hint costs seconds and a round lasts sixty --
    but a labeller that needs no clicking is worth a great deal more than one
    that does.
    """
    from ..app import Application
    from ..control.hotkey import StopKeyWatcher, interruptible_sleep

    app = Application.create()
    app.attach_window(prepare=not args.no_prepare)
    rect = app.content_rect()
    watcher = StopKeyWatcher(app.config.runner.stop_key)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"start a round, then take your hands off -- {watcher.describe()} to abort")
    interruptible_sleep(args.countdown, watcher)

    drv = Driver.from_app(app, rect, watcher=watcher, say=print)
    # Generously: the first run of this probe was still watching the pile fall
    # eight seconds in, which drowned every per-tsum reading in settling.
    base = _settle(drv, max_wait=args.settle)
    bx, by, bw, bh = _board_rect(base.shape, args.board)
    crop = base[by:by + bh, bx:bx + bw]
    tsums, radius, _ = detect(crop, k=args.k, include_dark=args.include_dark)
    if not tsums:
        app.close()
        raise SystemExit("no tsums detected -- is a round actually running?")
    print(f"{len(tsums)} tsums, r~{radius:.1f}px -- filming {args.seconds:.0f}s "
          f"at {args.fps:.0f}fps, hands off")

    masks = []
    for t in tsums:
        m = np.zeros((bh, bw), np.uint8)
        cv2.circle(m, (int(t.x), int(t.y)), max(2, int(radius * 0.55)), 1, -1)
        masks.append(m.astype(bool))

    timeline, frames = [], []
    prev = crop
    deadline = time.perf_counter() + args.seconds
    while time.perf_counter() < deadline:
        watcher.check()
        frame = app.capture.grab(rect)[by:by + bh, bx:bx + bw]
        # Against the previous frame, not the baseline. A hint animates, and
        # frame-to-frame change finds anything that moves however far the board
        # has drifted from where filming started -- whereas a fixed baseline
        # reports every settled tsum forever and buries the signal.
        diff = cv2.absdiff(frame, prev).max(axis=2)
        lit = [(float(diff[m].mean()), i) for i, m in enumerate(masks)]
        lit = sorted((c for c in lit if c[0] > args.threshold), reverse=True)
        timeline.append((time.perf_counter(), float(diff.mean()), lit[:6]))
        frames.append(frame)
        prev = frame
        time.sleep(max(0.0, 1.0 / args.fps))
    app.close()

    t0 = timeline[0][0]
    print(f"\n{'t':>6} {'whole board':>11}  tsums changing past {args.threshold}")
    hot_any = set()
    quiet = 0
    for when, overall, lit in timeline:
        quiet += not lit
        if not lit:
            continue
        hot_any.update(i for _, i in lit)
        marks = "  ".join(f"#{i}@{int(tsums[i].x)},{int(tsums[i].y)}={c:.0f}" for c, i in lit)
        print(f"{when - t0:6.1f}s {overall:11.2f}  {marks}")

    print(f"\n{quiet}/{len(timeline)} frames were completely still")
    if not hot_any:
        print("nothing on the board moved at all. The hint did not fire in "
              f"{args.seconds:.0f}s -- try --seconds 45, and check the game is "
              "not simply waiting on something else.")
    else:
        print(f"\n{len(hot_any)} distinct tsum(s) lit at some point: "
              f"{sorted(hot_any)}")
        print("If they came up one after another, that sequence IS a labelled "
              "chain. If it is always the same one, the hint marks a start only.")

    # Keep the frames: whatever the numbers say, the pictures are the evidence.
    step = max(1, len(frames) // args.keep)
    for n, f in enumerate(frames[::step][:args.keep]):
        ok, buf = cv2.imencode(".png", f)
        if ok:
            buf.tofile(str(out / f"idle_{n:02d}.png"))
    print(f"wrote {min(args.keep, len(frames))} frame(s) to {out}/idle_*.png")
    return 0


def _hold(args) -> int:
    """Press a tsum, photograph the board while held, release. Then diff.

    The premise worth testing: the game knows which tsums are the same
    character and this module only guesses, so if pressing one makes the game
    mark the rest, that mark is ground truth free for the taking -- and it
    would replace the colour clustering that currently gets character identity
    wrong about 40% of the time.

    Writes before/held/diff frames and prints where the board changed. If only
    the pressed tsum lights up, the idea is dead and the diff says so plainly.
    """
    from ..app import Application
    from ..control.hotkey import StopKeyWatcher, interruptible_sleep

    app = Application.create()
    app.attach_window(prepare=not args.no_prepare)
    rect = app.content_rect()
    watcher = StopKeyWatcher(app.config.runner.stop_key)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.countdown > 0:
        print(f"start a round -- {watcher.describe()} to abort")
        interruptible_sleep(args.countdown, watcher)

    # No countdown by default. Waiting around is what made the game decide the
    # player was idle and put up a hint, which is the thing this probe was
    # briefly fooled by -- so it presses as soon as the board is still.
    drv = Driver.from_app(app, rect, watcher=watcher, say=print)
    before = _settle(drv, max_wait=args.settle)
    bx, by, bw, bh = _board_rect(before.shape, args.board)
    crop = before[by:by + bh, bx:bx + bw]
    tsums, radius, _ = detect(crop, k=args.k, include_dark=args.include_dark)
    if not tsums:
        app.close()
        raise SystemExit("no tsums detected -- is a round actually running?")

    # The gate `play_loop` has always had, and this probe did not. A frame
    # detection reads badly does not announce itself: one run here found 134
    # tsums at r~12px on a board whose tsums are ~28px, pressed a phantom
    # sitting on bare water, and -- because a press that lands on nothing
    # changes nothing -- reported that the game marks nothing and that the
    # whole approach was closed. `play` would have thrown that same frame away
    # untouched (134 is past its --max-tsums of 110). Refusing here is not a
    # detection fix; it is refusing to draw conclusions from a frame that was
    # never read properly.
    if not (args.min_tsums <= len(tsums) <= args.max_tsums):
        app.close()
        raise SystemExit(
            f"{len(tsums)} tsums at r~{radius:.1f}px is not a board detection "
            f"can read\n(expected {args.min_tsums}-{args.max_tsums}). Anything "
            f"pressed on this frame would be a\nphantom, so nothing it reported "
            f"would mean anything. Re-run -- and if it keeps\nhappening on this "
            f"board, `analyze` a saved frame and try another -k.")

    # Refuse to press a board that is still moving: every reading would be the
    # pile shifting rather than anything the press did, which is precisely how
    # the previous run produced a boardful of false highlights.
    again = _settle(drv, max_wait=args.settle)
    drift = float(cv2.absdiff(again, before).mean())
    if drift > 1.0:
        app.close()
        raise SystemExit(f"board still moving (drift {drift:.2f}) -- let the "
                         f"pile settle and run again, or raise --settle")
    before, crop = again, again[by:by + bh, bx:bx + bw]

    # Pick from the middle of the board only. A tsum at the edge is half out of
    # the play area and may sit under the FEVER bar, so pressing it tells you
    # nothing -- an earlier version weighted centrality so weakly that it just
    # took the largest tsum anywhere, and picked one on the bottom boundary.
    # Press the head of the best chain we can find, not merely a big tsum in
    # the middle. Pressing one with no partners proves nothing: the game has
    # nothing to mark, and the run cannot distinguish "it marks nothing" from
    # "there was nothing to mark" -- which is exactly what happened once.
    chains = find_chains(tsums, radius, block=args.block, link_px=args.link_px,
                         mode="touch", max_chain=args.max_chain)
    expect: list[int] = []
    if chains:
        expect = list(chains[0].nodes)
        pick = tsums[expect[0]]
    else:
        inner = [t for t in tsums
                 if 0.2 * bw < t.x < 0.8 * bw and 0.2 * bh < t.y < 0.8 * bh]
        if not inner:
            app.close()
            raise SystemExit("no tsum near the middle of the board to press")
        pick = max(inner, key=lambda t: t.r)
        print("no chain found -- pressing a central tsum instead, which can "
              "only show whether the press registers at all")

    sx, sy = int(rect.left + bx + pick.x), int(rect.top + by + pick.y)
    print(f"{len(tsums)} tsums, r~{radius:.1f}px, board still (drift {drift:.2f})")

    # The count gate above catches a frame read catastrophically badly. It
    # does not catch one read merely badly: 101 tsums at r~15px passed it, and
    # those detections were two or three per real tsum. The tell is a high
    # count *and* a radius the count cannot support -- tsums fill the board,
    # so n of them imply a size, and detections much smaller are fragments.
    #
    # Both halves are needed. On radius alone this fired on a perfectly good
    # 50-tsum board that happened to read 22.6px where another frame of the
    # same board read 27.3px; frame-to-frame radius wobbles by that much
    # without anything being wrong. A real board holds ~50-70 tsums, so it is
    # the count that says "split", and the radius only confirms it.
    #
    # A warning rather than a refusal: calibrated on five frames, which is
    # enough to flag one and not enough to discard one.
    implied = math.sqrt(0.7 * bw * bh / (len(tsums) * math.pi))
    if len(tsums) > 75 and radius < 0.75 * implied:
        print(f"  WARNING: {len(tsums)} tsums is already more than a board holds, "
              f"and that many\n  would measure ~{implied:.0f}px, not {radius:.1f}px. "
              f"Detection is probably splitting each\n  tsum into several, so the "
              f"one it presses may be a fragment and the partners\n  it predicts "
              f"may be fragments too.")
    if expect:
        print(f"we think {len(expect)} tsums share its character: "
              f"{', '.join(f'({tsums[i].x:.0f},{tsums[i].y:.0f})' for i in expect)}")
    print(f"pressing the tsum at board ({pick.x:.0f}, {pick.y:.0f}) "
          f"= screen ({sx}, {sy}) for {args.hold:.1f}s -- watch it")

    import pyautogui

    # Mirror drag_chain's opening exactly. It settles the cursor before
    # pressing, and -- the part that matters -- it *moves* afterwards. The
    # emulator turns mouse movement into touch movement, so a press with no
    # motion behind it appears never to flush a touch-down at all: the first
    # run of this probe measured zero change on the tsum it had just pressed.
    pyautogui.PAUSE = 0.0
    # Focus first, with a click we are happy to lose. An unfocused window eats
    # the first click to activate itself instead of passing it to the app, and
    # this probe presses within a second of starting -- unlike `play`, where
    # something has always clicked before the first drag, which is why the
    # problem never shows up there. A bare tap on a tsum clears nothing.
    pyautogui.moveTo(sx, sy)
    time.sleep(0.05)
    pyautogui.click()
    time.sleep(0.35)
    # Re-baseline after focusing, so nothing the focus click did counts as a
    # reaction to the hold.
    before = app.capture.grab(rect)
    crop = before[by:by + bh, bx:bx + bw]

    pyautogui.moveTo(sx, sy)
    time.sleep(0.05)
    # The clock starts at mouseDown, which is when the game learns about the
    # press -- not after the wiggle below. Starting it after cost ~0.1s of
    # unmeasured time and made every onset read as "+0.00s, already at peak",
    # which is indistinguishable from a contaminated baseline. It is the same
    # instant `--hold-delay` and `dataset.delay` are counted from, so the
    # number transfers to them directly.
    t0 = time.perf_counter()
    pyautogui.mouseDown()
    shots = []
    stamps = []
    try:
        # The wiggle is what makes the emulator flush the touch-down at all --
        # a press with no motion behind it never registers -- so it is part of
        # delivering the press, and the ~0.08s it takes is time the game has
        # already had.
        for dx in (1, 0, -1, 0):
            pyautogui.moveTo(sx + dx, sy)
            time.sleep(0.02)
        # Sample throughout rather than once at the end: a highlight that
        # animates in, or one that shows briefly and stops, would be missed by
        # a single grab timed badly.
        deadline = t0 + args.hold
        while time.perf_counter() < deadline:
            stamps.append(time.perf_counter() - t0)
            shots.append(app.capture.grab(rect))
            time.sleep(0.1)
    finally:
        pyautogui.mouseUp()
    app.close()

    held = shots[-1]
    # Strongest reaction at any moment during the hold, not just the last one.
    crops = [s[by:by + bh, bx:bx + bw] for s in shots]
    diff = np.maximum.reduce([cv2.absdiff(c, crop).max(axis=2) for c in crops])
    held_crop = crops[-1]
    full = max(float(cv2.absdiff(s, before).mean()) for s in shots)
    print(f"{len(shots)} frames during the hold; whole screen changed by "
          f"{full:.2f} at most")

    # How much did each tsum change? The pressed one is the control: anything
    # that moves with it is the game marking a match, and everything that stays
    # put is the game saying nothing.
    changes = []
    for t in tsums:
        m = np.zeros(diff.shape, np.uint8)
        cv2.circle(m, (int(t.x), int(t.y)), max(2, int(radius * 0.5)), 1, -1)
        changes.append(float(diff[m.astype(bool)].mean()))
    pressed = changes[tsums.index(pick)]
    others = sorted((c for t, c in zip(tsums, changes) if t is not pick), reverse=True)

    vis = held_crop.copy()
    for t, c in zip(tsums, changes):
        hot = c > args.threshold
        cv2.circle(vis, (int(t.x), int(t.y)), int(radius * 0.85),
                   (0, 255, 255) if t is pick else ((0, 255, 0) if hot else (90, 90, 90)),
                   2 if hot or t is pick else 1, cv2.LINE_AA)
        cv2.putText(vis, f"{c:.0f}", (int(t.x) - 12, int(t.y) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # The full frame too: a chain counter or badge drawn outside the board rect
    # would be invisible in the crop, and that is exactly the sort of thing the
    # game puts in its header.
    for name, im in (("hold_before", crop), ("hold_held", held_crop),
                     ("hold_diff", diff), ("hold_marked", vis),
                     ("hold_full", held), ("hold_full_diff", cv2.absdiff(held, before))):
        ok, buf = cv2.imencode(".png", im)
        if ok:
            buf.tofile(str(out / f"{name}.png"))

    print(f"\npressed tsum changed by {pressed:.1f}")
    print(f"every other tsum, most-changed first: "
          f"{', '.join(f'{c:.1f}' for c in others[:12])}")
    lit = sum(1 for c in others if c > args.threshold)
    print(f"{lit} other tsum(s) changed by more than {args.threshold}")

    # Before anything is concluded about marking: did the press land at all?
    #
    # Pressing a real tsum makes the game draw a chain counter on it, so a
    # pressed tsum that did not change is a press that went nowhere -- and a
    # run that pressed nothing cannot say whether the game marks. This check
    # used to be missing, and its absence was not harmless: a run that pressed
    # bare water between two tsums (detection had offered a phantom there,
    # among 134 detections at r~12px on a board whose tsums are ~28px)
    # reported "the pressed tsum reacted and its predicted partners did not"
    # and declared the whole approach closed. Every word of that was wrong --
    # the pressed tsum had read 0.0, and nine other tsums had reacted at
    # 112-124 against a board median of 0.0.
    if pressed <= args.threshold:
        print(f"\n-> INCONCLUSIVE: the press did not land. Nothing changed where "
              f"we pressed\n   ({pressed:.1f}), and a real press draws a chain "
              f"counter on the tsum it hits.")
        if lit:
            print(f"   {lit} other tsum(s) DID react (up to {others[0]:.1f}) -- see "
                  f"hold_diff.png. That is\n   something the game drew, but not an "
                  f"answer to this press.")
        if len(tsums) > 90 or radius < 15:
            print(f"   Likely cause: detection is over-splitting. {len(tsums)} "
                  f"detections at r~{radius:.1f}px\n   is roughly double what a real "
                  f"board holds, so the tsum it picked was a\n   phantom sitting on "
                  f"background. Fix detection on this board first --\n   `analyze` "
                  f"the saved frame and check the count and radius look sane.")
        print("   Re-run once a press actually registers.")
    else:
        # Read the game's answer off the board first, and only then ask how
        # well the clustering predicted it. The two questions were tangled
        # here, and the tangle gave a wrong verdict: with five predicted
        # partners of which one was real, the *median* partner reaction was
        # 2.1 against a board floor of 3.8, so a press that lit exactly one
        # tsum at 94.7 out of a still board was reported as "motion, not
        # marking". The clustering being wrong is the finding, not a reason to
        # disbelieve the mark.
        #
        # So: the floor is the board's own median, the bar is well clear of
        # it, and whatever clears the bar is what the game drew. That reading
        # does not care whether `expect` was any good.
        idx = tsums.index(pick)
        rest_all = [(i, c) for i, c in enumerate(changes) if i != idx]
        floor = float(np.median([c for _, c in rest_all]))
        bar = max(args.threshold, 5 * floor)
        marked = [i for i, c in rest_all if c > bar]
        share = len(marked) / max(1, len(rest_all))
        if expect:
            print(f"\npredicted partners: "
                  f"{', '.join(f'{changes[i]:.1f}' for i in expect if i != idx)}")
        print(f"board floor {floor:.1f}, so a mark has to clear {bar:.1f}")

        if share > 0.5:
            print(f"-> {len(marked)} of {len(rest_all)} tsums cleared it. That is most "
                  f"of the board, which is\n   motion or a screen-wide effect, not the "
                  f"game naming a character.\n   Re-run on a stiller board.")
        elif not marked:
            print("-> the pressed tsum reacted and nothing else on the board did. "
                  "The game\n   marks only what you are touching, so there is "
                  "nothing to read off and\n   this route is closed.")
        else:
            hit = [i for i in marked if i in expect]
            print(f"-> the game marked {len(marked)} tsum(s) "
                  f"({', '.join(f'{changes[i]:.0f}' for i in marked[:8])}). "
                  f"The mark is real\n   and it is character ground truth worth "
                  f"reading.")
            if expect:
                missed = [i for i in expect if i != idx and i not in marked]
                print(f"   Clustering predicted {len(expect) - 1} partner(s) and got "
                      f"{len(hit)} of the {len(marked)} right,\n   with {len(missed)} "
                      f"prediction(s) the game did not mark. That gap is exactly\n"
                      f"   what a collected sample records.")
            _mark_onset(shots, stamps, crop, (bx, by, bw, bh), tsums, marked,
                        radius, args.threshold)
    print(f"wrote hold_before/held/diff/marked.png to {out}")
    return 0


def _mark_onset(shots, stamps, before_crop, board, tsums, partners, radius,
                threshold) -> None:
    """When, after the press, did the mark actually appear?

    `hold` reports the strongest reaction at any moment of a three-second
    hold, which proves the mark exists but says nothing about when it arrived
    -- and "when" is the only number the sample collector needs. Its whole
    failure mode is photographing too early: 0.10s produced 5,729 samples with
    no label in them, and 0.25s produced a still board with no marks on it.

    Reported as the first frame at which the partners clear `threshold`, and
    the first at which they reach most of their eventual strength. A delay set
    between the two catches a mark that is drawn but still fading in; anything
    below the first is the mistake schema 1 made.
    """
    if not shots or not partners:
        return
    bx, by, bw, bh = board
    r = max(2, int(radius * 0.5))
    masks = []
    for i in partners:
        m = np.zeros((bh, bw), np.uint8)
        cv2.circle(m, (int(tsums[i].x), int(tsums[i].y)), r, 1, -1)
        masks.append(m.astype(bool))

    series = []
    for t, shot in zip(stamps, shots):
        d = cv2.absdiff(shot[by:by + bh, bx:bx + bw], before_crop).max(axis=2)
        series.append((t, float(np.median([d[m].mean() for m in masks]))))
    peak = max(v for _, v in series)
    if peak <= threshold:
        return

    # An onset in the very first frame is not a fast mark, it is a broken
    # measurement: the frame is grabbed within milliseconds of the touch and
    # nothing the game draws can be at full strength by then. What it really
    # says is that the baseline already differed from the hold -- a chain
    # still glowing from the previous press, a score popup mid-flight, or a
    # board that had not finished settling. Reporting "+0.00s, set the delay
    # to 0.15" off that would send the collector back to exactly the timing
    # that made schema 1 worthless.
    if series[0][1] >= 0.8 * peak:
        print(f"\n   mark onset: UNRESOLVED -- already at full strength "
              f"({series[0][1]:.0f} of a {peak:.0f} peak)\n   in the first frame, "
              f"at +{series[0][0]:.2f}s. Two readings and this cannot separate them:\n"
              f"   the mark renders faster than a press can be photographed, or the\n"
              f"   baseline already differed (a chain still glowing, a score popup in\n"
              f"   flight). Check hold_diff.png -- a chain counter or drifting score\n"
              f"   text means the second. If the frame is clean, the delay is not what\n"
              f"   is holding collection back.")
        return

    first = next((t for t, v in series if v > threshold), None)
    full = next((t for t, v in series if v >= 0.8 * peak), None)
    settled = f"80% of peak at +{full:.2f}s" if full is not None else "never settled"
    print(f"\n   mark onset: first over {threshold:g} at +{first:.2f}s, "
          f"{settled} (peak {peak:.0f})")
    if full is not None:
        # Rounded up to the next 50ms and given a little headroom: the reading
        # has to survive a frame arriving late, not merely the median case.
        suggest = math.ceil((full + 0.10) * 20) / 20
        print(f"   -> set dataset.delay to about {suggest:.2f} in config.yaml "
              f"(default {DATASET_DELAY:.2f})")
        if suggest > 0.6:
            print(f"      That is slow enough to be worth checking against "
                  f"round throughput\n      before collecting a long session.")


def _assist(args) -> int:
    """You hold a tsum; the app reads what the game marked and drags through it.

    The other half of ``hold``. That probe proved the game answers the question
    detection guesses at -- press a tsum and every tsum you could link to it
    lights up, identity and reachability together, and it is right exactly
    where the colour clustering is wrong. ``--verify-hold`` then tried to spend
    that answer inside ``play`` and lost decisively: reading it cost ~0.1s on
    every drag, which halved the chains played, and 196 dragged tsums lost to
    527. The lesson recorded there is that throughput dominates accuracy, so
    accuracy work has to be free at run time to be worth anything.

    Here it is free. The human chooses when to press, so there is no throughput
    to lose, and a human holds far longer than the 0.10s that run was racing to
    keep down -- which is the delay the docs warn sits below the 0.15s floor
    where marks have not finished rendering. This mode gets the cleaner read
    that ``--verify-hold`` could never afford.

    The loop is: watch for the press while keeping the newest frame as a
    baseline, wait for the marks to render, diff, order the marked tsums into a
    path starting at the one under the finger, and warp the cursor along it.
    Nothing here presses or releases the button -- that stays with the user,
    unless ``--auto-release`` is given.
    """
    from ..app import Application
    from ..control.hotkey import StopKeyWatcher

    import pyautogui

    app = Application.create()
    app.attach_window(prepare=not args.no_prepare)
    rect = app.content_rect()
    watcher = StopKeyWatcher(app.config.runner.stop_key)
    drv = Driver.from_app(app, rect, watcher=watcher, say=print)
    out = Path(args.dir)
    if args.debug:
        out.mkdir(parents=True, exist_ok=True)

    def release_wait() -> None:
        """Block until the button is up, so one hold only ever serves once."""
        while _lbutton_down():
            watcher.check()
            time.sleep(0.02)

    # The frame shape is fixed for the run, but the rect is not: FEVER gets a
    # wider one. So the shape is read once and the rect re-derived per press,
    # which is also the only moment its answer can change anything.
    shape = drv.grab().shape
    fever = FeverWatch(drv.matcher, drv.templates)
    bx, by, bw, bh = _board_rect(shape, args.board)
    if not fever.enabled:
        print("no max_fever template -- FEVER will be read on the normal board rect")

    print(f"assist ready -- hold a tsum and keep holding; {watcher.describe()}")
    if args.dry_run:
        print("dry run: the path is reported and drawn to disk, the cursor is not moved")

    served = failed = 0
    try:
        while True:
            watcher.check()

            # -- wait for the press, keeping the newest frame as the baseline.
            # The marks are read as a difference, so the "before" has to come
            # from before a press whose timing is the user's to choose. A grab
            # costs ~5ms, so the baseline is at most a frame or two stale --
            # far inside the ~150ms the glow takes to render, which is what
            # makes a rolling baseline clean rather than a race.
            release_wait()
            # Whole frames, not crops: the rect can change under FEVER, and the
            # baseline has to be re-croppable once that is known. Only the
            # drift reading is taken through the current rect.
            prev = drv.grab()
            drift = 0.0
            while not _lbutton_down():
                watcher.check()
                frame = drv.grab()
                # Drift over the *board*, not the whole frame. Measured over
                # the frame it never reads still during a round and never
                # could: the score counter, the timer and the FEVER meter all
                # animate continuously, so a full-frame reading is a HUD
                # activity meter that says nothing about the pile.
                drift = float(cv2.absdiff(frame[by:by + bh, bx:bx + bw],
                                          prev[by:by + bh, bx:bx + bw]).mean())
                prev = frame
            press = pyautogui.position()
            baseline = prev

            # The button is watched globally -- there is no event to subscribe
            # to for a press that lands on the emulator -- so a click anywhere
            # else on the desktop arrives here too. Ignore those silently
            # rather than counting them as a press that came to nothing.
            if not rect.contains(press):
                release_wait()
                continue

            # -- let the marks render, and give up if the hold does not last.
            deadline = time.perf_counter() + args.delay
            while time.perf_counter() < deadline and _lbutton_down():
                watcher.check()
                time.sleep(0.01)
            if not _lbutton_down():
                print("  released before the marks rendered -- keep holding until "
                      "the path has been drawn")
                failed += 1
                continue

            # Several frames, spaced, rather than one. Anything that moves by
            # itself -- FEVER's sparkles, a bubble popping, a tsum still
            # falling -- lands somewhere different in each, while a mark holds
            # still, and `marks_on_board` keeps only what persisted. The gap
            # has to be long enough for the animation to actually move.
            shots = []
            for i in range(max(1, args.mark_frames)):
                if i:
                    time.sleep(args.mark_gap)
                    if not _lbutton_down():
                        break
                watcher.check()
                shots.append(drv.grab())
            held = shots[-1]

            # Off by default. `hold` aborts on a moving board because a press
            # on a settling pile reads the settling, and that is still true --
            # but refusing on it up front proved wrong in play: the pile is
            # rarely perfectly still mid-round, and the user is the one who
            # picked the moment. The reading itself is checked instead, below,
            # where motion is both easier to recognise and the thing that
            # actually matters.
            if args.drift > 0 and drift > args.drift:
                print(f"  board still moving (drift {drift:.2f}) -- let the pile "
                      f"settle before pressing")
                failed += 1
                release_wait()
                continue

            # Which rect applies to *this* press. Read off the held frames --
            # the meter is in every one of them, and asking here means the
            # answer is current rather than however old the last press was.
            for s in shots:
                fever.update(s)
            was_fever = fever.active
            bx, by, bw, bh = _board_rect(shape, args.board, fever=was_fever)
            crop = baseline[by:by + bh, bx:bx + bw]

            t0 = time.perf_counter()
            tsums, radius, _ = detect(crop, k=args.k, include_dark=args.include_dark)
            if not tsums:
                print("  no tsums detected -- is a round actually running?")
                failed += 1
                release_wait()
                continue

            # Which one is under the finger? Board coordinates, via the content
            # area: the same conversion `to_screen` does, run backwards.
            px = press.x - rect.left - bx
            py = press.y - rect.top - by
            # By index, not by value: Tsum is a plain dataclass, so `.index()`
            # would match the first tsum with equal fields rather than this one.
            pressed = min(range(len(tsums)),
                          key=lambda i: (tsums[i].x - px) ** 2 + (tsums[i].y - py) ** 2)
            near = tsums[pressed]
            gap = math.hypot(near.x - px, near.y - py)
            # Generous, with an absolute floor. Being slightly wrong about
            # which tsum is under the finger costs little -- the path starts
            # from the cursor's real position either way, and the marks come
            # from the game rather than from this guess -- whereas refusing a
            # press the game did accept costs the whole chain. The floor is
            # there because the radius estimate itself dips on a frame caught
            # mid-animation, and 1.5 radii of a bad radius rejected presses
            # that were 18px from a centre.
            if gap > max(radius * 2.0, 30.0):
                print(f"  pressed board ({px:.0f}, {py:.0f}), which is not on any "
                      f"tsum I can see -- the nearest is {gap:.0f}px away")
                failed += 1
                release_wait()
                continue

            hits, aura_only, diff = marks_on_board(
                crop, [s[by:by + bh, bx:bx + bw] for s in shots], tsums, radius,
                pressed=pressed, threshold=args.threshold, aura=args.aura)

            # The reading, not the precondition. A board that moved between
            # the baseline and the hold lights up almost everything, because
            # every tsum that shifted differs from where it was -- so an
            # implausible share of the board reacting is motion (or FEVER)
            # rather than marks. This catches what the drift guard was aiming
            # at, at the point where it is unambiguous: a real press marks the
            # tsums of one character, never most of the pile.
            share = len(hits) / max(1, len(tsums) - 1)
            if share > args.max_marked:
                print(f"  {len(hits)} of {len(tsums) - 1} tsums lit up "
                      f"({share:.0%}) -- that is the board moving, not the game "
                      f"marking; let the pile settle")
                failed += 1
                release_wait()
                continue

            # No cap by default. `play` caps chain length to keep a guessed
            # tour tight, but these are not guesses -- the game marked exactly
            # what is linkable, and a longer chain scores better.
            members = hits
            if args.max_chain > 0 and len(members) + 1 > args.max_chain:
                members = sorted(members, key=lambda i: (tsums[i].x - near.x) ** 2
                                 + (tsums[i].y - near.y) ** 2)[:args.max_chain - 1]
            order = tour_from(pressed, [pressed] + members, tsums)
            think = (time.perf_counter() - t0) * 1000

            note = f", {len(aura_only)} of them inside the glow" if aura_only else ""
            print(f"  {'[FEVER] ' if was_fever else ''}{len(tsums)} tsums, "
                  f"r~{radius:.1f}px, drift {drift:.2f} -- "
                  f"game marked {len(hits)}{note}  ({think:.0f}ms)")

            if args.debug:
                vis = held[by:by + bh, bx:bx + bw].copy()
                for i, t in enumerate(tsums):
                    colour = ((0, 255, 255) if i == pressed
                              else (0, 165, 255) if i in aura_only
                              else (0, 255, 0) if i in hits else (90, 90, 90))
                    cv2.circle(vis, (int(t.x), int(t.y)), int(radius * 0.85), colour,
                               2 if i == pressed or i in hits else 1, cv2.LINE_AA)
                for a, b in zip(order, order[1:]):
                    cv2.line(vis, (int(tsums[a].x), int(tsums[a].y)),
                             (int(tsums[b].x), int(tsums[b].y)), (255, 0, 255), 2, cv2.LINE_AA)
                stamp = time.strftime("%H%M%S")
                for name, im in ((f"assist_{stamp}_marked", vis),
                                 (f"assist_{stamp}_diff", diff)):
                    ok, buf = cv2.imencode(".png", im)
                    if ok:
                        buf.tofile(str(out / f"{name}.png"))

            if len(order) < args.min_chain:
                print(f"  only {len(order)} tsum(s) in the chain, want {args.min_chain} "
                      f"-- not moving; release and try another")
                # Nothing marked at all usually means the game never saw the
                # press. An unfocused window eats the first click to activate
                # itself, which is the same trap that cost `hold` a run -- and
                # it looks identical to a tsum with no partners.
                if not hits:
                    print("     (nothing lit up: if you just alt-tabbed back, "
                          "the window ate that press -- try again)")
                failed += 1
                release_wait()
                continue

            # Start from where the finger actually is, not from the pressed
            # tsum's centre. They are within one tsum of each other, and the
            # cursor's real position is the truth about where the touch is --
            # warping to the centre first would add a leg that buys nothing.
            points = [(press.x, press.y)]
            points += [drv.to_screen(bx + tsums[i].x, by + tsums[i].y) for i in order[1:]]

            if args.dry_run:
                print("  path: " + " -> ".join(f"({x},{y})" for x, y in points))
                served += 1
                release_wait()
                continue

            t1 = time.perf_counter()
            walked = walk_path(points, step_px=args.step_px, per_step=args.per_step,
                               still_down=_lbutton_down, check_stop=watcher.check)
            drew = (time.perf_counter() - t1)
            if walked < len(points) - 1:
                # The commonest way a press comes to nothing, by a distance. A
                # 40-tsum chain is seconds of walking and there is no way to
                # tell from the game that it is still going, so the honest fix
                # is --auto-release (on by default), not asking for a steadier
                # hand.
                print(f"  you released after {walked} of {len(points) - 1} legs "
                      f"({drew:.1f}s in) -- the rest of the chain was not drawn")
                failed += 1
            else:
                served += 1
                if args.auto_release:
                    pyautogui.mouseUp()
                    print(f"  drew {len(order)} tsums in {drew:.1f}s and released")
                else:
                    print(f"  drew {len(order)} tsums in {drew:.1f}s -- release now")
            release_wait()
    except StopRequested as exc:
        print(exc)
    finally:
        app.close()

    print(f"assist served {served} chain(s); {failed} press(es) came to nothing")
    return 0


def _label(args) -> int:
    """Mark up a board: the paths you'd drag, and where detection got it wrong.

    Four things get recorded, and they answer two different questions.

    `path` mode is the original one, and it calibrates *distance*: every
    consecutive pair in a path you mark is one example of a link the game
    accepts, so `score` can read the real link threshold straight off them
    instead of someone picking a number and hoping.

    `missed`, `false` and `group` calibrate *identification*, which had no
    ground truth at all before: a tsum detection can be wrong by not existing,
    by not being found, or by being found and filed under the wrong colour
    cluster. `eval` needs all three marked to score a parameter change, and it
    only counts boards you flag as fully reviewed (`r`) -- a half-marked board
    reads as a board with no errors, which would quietly reward whatever
    setting misses the most.
    """
    img = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    bx, by, bw, bh = _board_rect(img.shape, args.board)
    crop = img[by:by + bh, bx:bx + bw]
    tsums, radius, _ = detect(crop, k=args.k, radius=args.radius,
                              include_dark=args.include_dark, merge=args.merge)
    print(f"{len(tsums)} tsums detected, r~{radius:.1f}px")
    print("modes: 1=path  2=missed  3=false  4=same-kind group")
    print("       click to mark | n=next path/group  u=undo  c=clear")
    print("       r=toggle 'board fully reviewed'  s=save  q=quit")

    paths: list[list[int]] = [[]]
    groups: list[list[int]] = [[]]
    missed: list[tuple[float, float]] = []
    bad: list[int] = []            # indices of detections that are not tsums
    mode = "path"
    reviewed = False

    def _nearest(x: int, y: int) -> Optional[int]:
        if not tsums:
            return None
        i = min(range(len(tsums)),
                key=lambda n: (tsums[n].x - x) ** 2 + (tsums[n].y - y) ** 2)
        d2 = (tsums[i].x - x) ** 2 + (tsums[i].y - y) ** 2
        return i if d2 <= (radius * 1.4) ** 2 else None

    def redraw() -> np.ndarray:
        vis = crop.copy()
        for i, t in enumerate(tsums):
            colour = (0, 0, 255) if i in bad else t.colour
            cv2.circle(vis, (int(t.x), int(t.y)), int(radius * 0.9), colour, 2, cv2.LINE_AA)
        # A detection you called false gets an X through it, so "marked wrong"
        # never reads as "marked and kept" at a glance.
        for i in bad:
            t = tsums[i]
            d = int(radius * 0.6)
            for dx in (-d, d):
                cv2.line(vis, (int(t.x) - d, int(t.y) - dx), (int(t.x) + d, int(t.y) + dx),
                         (0, 0, 255), 2, cv2.LINE_AA)
        for (mx, my) in missed:
            cv2.circle(vis, (int(mx), int(my)), int(radius * 0.9), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, (int(mx), int(my)), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for gi, group in enumerate(groups):
            live = gi == len(groups) - 1
            for i in group:
                t = tsums[i]
                cv2.circle(vis, (int(t.x), int(t.y)), int(radius * 1.05),
                           (255, 255, 255) if live else (170, 170, 170), 2, cv2.LINE_AA)
                cv2.putText(vis, chr(ord("A") + gi), (int(t.x) - 18, int(t.y) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for pi, path in enumerate(paths):
            done = pi < len(paths) - 1
            colour = (150, 150, 150) if done else (60, 255, 255)
            pts = [(int(tsums[i].x), int(tsums[i].y)) for i in path]
            if len(pts) > 1:
                cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 0, 0), 7, cv2.LINE_AA)
                cv2.polylines(vis, [np.array(pts, np.int32)], False, colour, 3, cv2.LINE_AA)
            for n, (px, py) in enumerate(pts, 1):
                cv2.circle(vis, (px, py), 12, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.putText(vis, str(n), (px - 6, py + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, colour, 1, cv2.LINE_AA)
        banner = (f"[{mode}]  paths {sum(1 for p in paths if len(p) >= 2)}  "
                  f"missed {len(missed)}  false {len(bad)}  "
                  f"groups {sum(1 for g in groups if len(g) >= 2)}  "
                  f"{'REVIEWED' if reviewed else 'not reviewed'}")
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(vis, banner, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0) if reviewed else (200, 200, 200), 1, cv2.LINE_AA)
        return vis

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if mode == "missed":
            # No snapping: the whole point is that nothing was detected here,
            # so there is nothing to snap to.
            missed.append((float(x), float(y)))
        else:
            near = _nearest(x, y)
            if near is None:
                return
            if mode == "path":
                paths[-1].append(near)
            elif mode == "false":
                # Toggle, so a misclick is undone by clicking the same tsum.
                bad.remove(near) if near in bad else bad.append(near)
            elif mode == "group" and near not in groups[-1]:
                groups[-1].append(near)
        cv2.imshow("label", redraw())

    cv2.namedWindow("label")
    cv2.setMouseCallback("label", on_mouse)
    cv2.imshow("label", redraw())

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            mode = _LABEL_MODES[key - ord("1")]
            print(f"mode: {mode}")
        elif key == ord("u"):
            if mode == "path" and paths[-1]:
                paths[-1].pop()
            elif mode == "missed" and missed:
                missed.pop()
            elif mode == "false" and bad:
                bad.pop()
            elif mode == "group" and groups[-1]:
                groups[-1].pop()
        elif key == ord("c"):
            if mode == "path":
                paths[-1] = []
            elif mode == "missed":
                missed.clear()
            elif mode == "false":
                bad.clear()
            elif mode == "group":
                groups[-1] = []
        elif key == ord("n"):
            if mode == "path" and paths[-1]:
                paths.append([])
            elif mode == "group" and groups[-1]:
                groups.append([])
        elif key == ord("r"):
            reviewed = not reviewed
            print(f"board fully reviewed: {reviewed}")
        elif key == ord("s"):
            kept = [p for p in paths if len(p) >= 2]
            kept_groups = [g for g in groups if len(g) >= 2]
            if not (kept or kept_groups or missed or bad or reviewed):
                print("nothing to save")
                continue

            def pt(i: int) -> list:
                return [round(tsums[i].x + bx, 1), round(tsums[i].y + by, 1)]

            out = Path(args.image).with_suffix(".label.json")
            out.write_text(json.dumps({
                "image": Path(args.image).name,
                "board": [bx, by, bw, bh],
                "radius": radius,
                # Full-image coordinates, so a label survives re-detection with
                # different settings -- `score` and `eval` re-detect and match
                # by position rather than trusting these indices.
                "paths": [{"nodes": [pt(i) for i in p]} for p in kept],
                # Every detection present when the board was reviewed. Without
                # it "false positives" would only cover the ones clicked, and a
                # re-run under different parameters would have no way to tell a
                # newly-appeared detection from an approved one.
                "detected": [pt(i) for i in range(len(tsums))],
                "false_positives": [pt(i) for i in sorted(bad)],
                "missed": [[round(mx + bx, 1), round(my + by, 1)] for mx, my in missed],
                "groups": [{"nodes": [pt(i) for i in g]} for g in kept_groups],
                "reviewed": reviewed,
            }, indent=2))
            print(f"saved {len(kept)} path(s), {len(missed)} missed, {len(bad)} false, "
                  f"{len(kept_groups)} group(s), reviewed={reviewed} -> {out}")
        cv2.imshow("label", redraw())

    cv2.destroyAllWindows()
    return 0


def _skillcheck(args) -> int:
    """Watch the skill button's gold reading live, to pick a threshold."""
    from ..app import Application

    app = Application.create()
    app.attach_window(prepare=False)
    rect = app.content_rect()
    print(f"sampling ring r{args.skill_inner:.0f}-{args.skill_outer:.0f} at {args.skill}")
    print("charge the skill and watch the number climb; uncharged reads ~0.00")
    for _ in range(args.frames):
        gold = skill_gold(app.capture.grab(rect), args.skill, args.skill_inner, args.skill_outer)
        bar = "#" * int(gold * 40)
        print(f"  gold {gold:.3f}  {bar}")
        time.sleep(1.0)
    app.close()
    return 0


def _score(args) -> int:
    """Read every *.label.json and report what link rule your drawings imply."""
    labels = sorted(Path(args.dir).glob("*.label.json"))
    if not labels:
        raise SystemExit(f"no *.label.json found in {args.dir} -- run `label` first")

    gaps: list[float] = []
    px: list[float] = []
    per_file = []
    for lf in labels:
        data = json.loads(lf.read_text())
        r = float(data["radius"])
        n_pairs = 0
        for path in data["paths"]:
            nodes = path["nodes"]
            for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
                d = math.hypot(x1 - x0, y1 - y0)
                px.append(d)
                gaps.append(d / (2 * r))
                n_pairs += 1
        per_file.append((lf.name, len(data["paths"]), n_pairs))

    print(f"{len(labels)} labelled board(s), {len(gaps)} linked pairs\n")
    for name, np_, npair in per_file:
        print(f"  {name}: {np_} path(s), {npair} pairs")

    arr = np.array(gaps)
    arr_px = np.array(px)

    # Pixels first, because --link-px is the setting the play loop actually
    # uses; diameters follow only because --link is still there as a fallback.
    print("\ngap between consecutive tsums you linked, in PIXELS (--link-px):")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  p{p:<3d} {np.percentile(arr_px, p):6.1f}px")
    print("\nwhat each candidate --link-px would capture:")
    for t in (70, 80, 90, 100, 110, 125, 150):
        print(f"  link-px {t:<4}: {100 * (arr_px <= t).mean():5.1f}% of your links")

    print("\nsame gaps in tsum diameters (--link, only used when --link-px 0):")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  p{p:<3d} {np.percentile(arr, p):.2f}")

    print("\nwhat each candidate --link would capture:")
    for t in (1.2, 1.35, 1.5, 1.7, 2.0, 2.5, 3.0):
        print(f"  link {t:<4}: {100 * (arr <= t).mean():5.1f}% of your links")

    px90 = float(np.percentile(arr_px, 90))
    print(f"\n-> suggested --link-px {round(px90 / 5) * 5:.0f} (p90 = {px90:.0f}px)")

    _score_adjacency(labels, Path(args.dir), args.tol)

    # Deliberately NOT reading the verdict off p99: with a few dozen labelled
    # pairs, p99 is essentially the single largest value, so one stray click
    # decides the model. The bulk of the distribution is the real signal, and
    # outliers get reported separately rather than silently setting the answer.
    bulk = float(np.percentile(arr, 90))
    outliers = arr[arr > 1.8]
    print(f"\nbulk of your links (p90): {bulk:.2f} diameters")
    if len(outliers):
        print(f"outliers past 1.8: {len(outliers)}/{len(arr)} "
              f"({', '.join(f'{v:.2f}' for v in sorted(outliers))})")

    if len(arr) < 30:
        print(f"\n-> {len(arr)} pairs is thin. Treat this as provisional; "
              f"label a few more boards.")
    if bulk <= 1.8:
        print(f"-> the bulk sits at touching distance: 'touch' mode, "
              f"--link {max(1.5, round(bulk + 0.2, 1))}")
        if len(outliers):
            print("   the outliers are either mis-clicks or genuinely longer links -- "
                  "more labels will tell which")
    else:
        print("-> links routinely run well past touching: 'reach' is the right model")
    return 0


#: Every `detect` keyword `eval --sweep` is allowed to vary. Spelled out rather
#: than introspected so a typo is an error instead of a silently ignored knob.
_SWEEPABLE = ("k", "radius", "include_dark", "dark_l", "merge", "heal_frac",
              "open_ratio", "recolour", "kinds", "bowl_reject", "floor_frac", "hole_frac",
              "scale")

#: Not `detect` arguments -- these shrink the board rect before the crop is
#: taken, so they tune :data:`LAYOUTS` rather than detection. Sweepable because
#: the rect is not a free parameter you can reason about from a screenshot: it
#: decides which board furniture the k-means fit even sees, so a trim changes
#: the palette and the radius estimate, not just which detections survive.
_TRIMS = ("trim_top", "trim_bottom", "trim_left", "trim_right")


def _match(truth: Sequence[Sequence[float]], found: Sequence[Sequence[float]],
           tol: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair up two sets of points, closest pair first, one-to-one.

    Greedy rather than optimal (Hungarian): at `tol` well under the spacing
    between tsums the two agree, and greedy is deterministic and obvious to
    read. Returns (pairs, unmatched_truth, unmatched_found).
    """
    cand = []
    t2 = tol * tol
    for i, (tx, ty) in enumerate(truth):
        for j, (fx, fy) in enumerate(found):
            d2 = (tx - fx) ** 2 + (ty - fy) ** 2
            if d2 <= t2:
                cand.append((d2, i, j))
    cand.sort()
    used_t: set[int] = set()
    used_f: set[int] = set()
    pairs = []
    for _, i, j in cand:
        if i in used_t or j in used_f:
            continue
        used_t.add(i)
        used_f.add(j)
        pairs.append((i, j))
    return (pairs,
            [i for i in range(len(truth)) if i not in used_t],
            [j for j in range(len(found)) if j not in used_f])


def _truth_points(data: dict) -> list[list[float]]:
    """The tsums a reviewed board actually has, in full-image pixels.

    Everything detection found at label time, minus what the reviewer struck
    out, plus what it missed. Both parts matter: without the strike-outs a
    parameter that keeps a phantom scores as well as one that drops it, and
    without the misses one that finds nothing at all scores perfectly.
    """
    bogus = {tuple(p) for p in data.get("false_positives", [])}
    keep = [p for p in data.get("detected", []) if tuple(p) not in bogus]
    return keep + [list(p) for p in data.get("missed", [])]


def _detect_labelled(data: dict, folder: Path,
                     params: dict) -> tuple[list[Tsum], list[list[float]], float, np.ndarray]:
    """Re-detect a labelled board and return its tsums in full-image pixels.

    Ground truth is stored in full-image coordinates precisely so that a
    different board rect stays comparable: trimming the rect just means fewer
    detections to match, with every truth point still where it was.
    """
    img_path = folder / data["image"]
    img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {img_path} (referenced by the label file)")

    params = dict(params)
    trims = {name: int(params.pop(name, 0) or 0) for name in _TRIMS}
    bx, by, bw, bh = data["board"]
    bx += trims["trim_left"]
    by += trims["trim_top"]
    bw -= trims["trim_left"] + trims["trim_right"]
    bh -= trims["trim_top"] + trims["trim_bottom"]

    tsums, radius, centres = detect(img[by:by + bh, bx:bx + bw], **params)
    # The palette comes back so a caller can rebuild the label map this fit
    # produced -- `blob_adjacency` needs it, and refitting would renumber the
    # clusters and score a different segmentation than the one being tested.
    return tsums, [[t.x + bx, t.y + by] for t in tsums], radius, centres


def _eval_once(labels: Sequence[tuple[Path, dict]], params: dict,
               tol_frac: float, verbose: bool) -> dict:
    """Score one parameter set against every reviewed board."""
    tp = fp = fn = 0
    splits = merges = grouped = 0
    per_board = []

    for path, data in labels:
        truth = _truth_points(data)
        tsums, found, _, _centres = _detect_labelled(data, path.parent, params)
        tol = tol_frac * float(data["radius"])
        pairs, missed, extra = _match(truth, found, tol)
        tp += len(pairs)
        fn += len(missed)
        fp += len(extra)

        # Kind agreement. Each labelled group is a set of tsums a human says
        # are the same character; detection is right about a group only if it
        # filed every member under one cluster, and only if no *other* group
        # landed in that same cluster.
        by_group: list[set[int]] = []
        for g in data.get("groups", []):
            nodes = g["nodes"]
            gp, _, _ = _match(nodes, found, tol)
            kinds = {tsums[j].kind for _, j in gp}
            if not kinds:
                continue
            grouped += 1
            splits += len(kinds) - 1
            by_group.append(kinds)
        for gi, a in enumerate(by_group):
            for b in by_group[gi + 1:]:
                if a & b:
                    merges += 1

        per_board.append((path.name, len(pairs), len(extra), len(missed)))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    if verbose:
        for name, a, b, c in per_board:
            print(f"  {name:24s} hit {a:3d}  false {b:3d}  missed {c:3d}")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "splits": splits, "merges": merges, "grouped": grouped}


def _parse_sweep(specs: Sequence[str]) -> dict[str, list]:
    """`--sweep k=8,12,16` -> {"k": [8, 12, 16]}, with values typed."""
    def value(text: str):
        low = text.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(text)
        except ValueError:
            return float(text)

    out: dict[str, list] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--sweep wants NAME=v1,v2,... -- got {spec!r}")
        name, _, values = spec.partition("=")
        name = name.strip()
        if name not in _SWEEPABLE and name not in _TRIMS:
            raise SystemExit(f"--sweep {name}: not a detect parameter. "
                             f"Try one of: {', '.join(_SWEEPABLE + _TRIMS)}")
        out[name] = [value(v) for v in values.split(",")]
    return out


def _eval(args) -> int:
    """Score detection against the boards you reviewed in `label`.

    `score` answers "how far apart are tsums I chain"; this answers the other
    half -- "does detection find the right tsums at all", which until now was
    judged by looking at an overlay and forming an impression.
    """
    folder = Path(args.dir)
    all_labels = sorted(folder.glob("*.label.json"))
    if not all_labels:
        raise SystemExit(f"no *.label.json found in {folder} -- run `label` first")

    labels, unreviewed = [], []
    for lf in all_labels:
        data = json.loads(lf.read_text())
        (labels if data.get("reviewed") and data.get("detected") else unreviewed).append((lf, data))

    if unreviewed:
        print(f"skipping {len(unreviewed)} board(s) without a full identification review: "
              f"{', '.join(p.name for p, _ in unreviewed)}")
        print("  (open each in `label`, mark misses/false positives, press r, then s)\n")
    if not labels:
        raise SystemExit("no reviewed boards to score -- nothing to report")

    base = {"k": args.k, "radius": args.radius, "include_dark": args.include_dark,
            "merge": args.merge, "scale": args.scale, "bowl_reject": args.bowl_reject}
    truth_total = sum(len(_truth_points(d)) for _, d in labels)
    print(f"{len(labels)} reviewed board(s), {truth_total} ground-truth tsums, "
          f"match tolerance {args.tol:.2f}r\n")

    if not args.sweep:
        r = _eval_once(labels, base, args.tol, verbose=True)
        print(f"\nprecision {r['precision']:.3f}  recall {r['recall']:.3f}  f1 {r['f1']:.3f}"
              f"   (hit {r['tp']}, false {r['fp']}, missed {r['fn']})")
        if r["grouped"]:
            print(f"kind: {r['grouped']} labelled group(s), {r['splits']} split(s) "
                  f"(one character read as several colours), "
                  f"{r['merges']} merge(s) (two characters read as one colour)")
        else:
            print("kind: no same-kind groups labelled -- mode 4 in `label` fills this in")
        return 0

    grid = _parse_sweep(args.sweep)
    combos = [dict(zip(grid, values)) for values in product(*grid.values())]
    print(f"sweeping {len(combos)} combination(s) of {', '.join(grid)}\n")

    rows = []
    for combo in combos:
        r = _eval_once(labels, {**base, **combo}, args.tol, verbose=False)
        rows.append((combo, r))
        print(f"  {_describe(combo):32s} f1 {r['f1']:.3f}")

    rows.sort(key=lambda row: -row[1]["f1"])
    width = max(len(_describe(c)) for c, _ in rows)
    print(f"\n{'params'.ljust(width)}  {'prec':>5} {'rec':>5} {'f1':>5}  "
          f"{'hit':>4} {'fls':>4} {'mis':>4}  {'splt':>4} {'mrg':>4}")
    for combo, r in rows:
        print(f"{_describe(combo).ljust(width)}  {r['precision']:5.3f} {r['recall']:5.3f} "
              f"{r['f1']:5.3f}  {r['tp']:4d} {r['fp']:4d} {r['fn']:4d}  "
              f"{r['splits']:4d} {r['merges']:4d}")

    best = rows[0][0]
    print(f"\nbest f1: {_describe(best)}")
    # A sweep across a handful of boards overfits easily, and the gap between
    # first and second place is the honest read on whether it found anything.
    if len(rows) > 1:
        gap = rows[0][1]["f1"] - rows[1][1]["f1"]
        if gap < 0.01:
            print(f"-> only {gap:.3f} f1 ahead of {_describe(rows[1][0])} -- "
                  f"that is noise, not a winner. Label more boards.")
    if len(labels) < 5:
        print(f"-> {len(labels)} board(s) is thin; a winner here may not survive more data.")
    return 0


def _describe(combo: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in combo.items())


def _score_adjacency(label_files: Sequence[Path], folder: Path, tol_frac: float) -> None:
    """Of the links you actually drew, how many would `touch` mode accept?

    The distance percentiles above are only half the rule. :func:`adjacency`
    also rejects a pair when a third tsum sits on the line between them, and
    it only ever links two tsums it filed under the *same* colour cluster. A
    threshold read off distance alone therefore promises more than the code
    delivers, so this replays each hand-drawn link through the real test and
    reports which of the three gates it died at.
    """
    boards = []   # (tsums, radius, labels, [(i, j), ...]) -- one per labelled board
    total = 0
    for lf in label_files:
        data = json.loads(lf.read_text())
        if not data.get("paths"):
            continue
        params = {"k": 12, "include_dark": True}
        tsums, found, radius, centres = _detect_labelled(data, folder, params)
        tol = tol_frac * float(data["radius"])
        links = []
        for path in data["paths"]:
            nodes = path["nodes"]
            pairs, _, _ = _match(nodes, found, tol)
            at = {i: j for i, j in pairs}
            links += [(at[a], at[a + 1]) for a in range(len(nodes) - 1)
                      if a in at and a + 1 in at]
        if links:
            # The same fit detection ran on, so the mask blob_adjacency tests is
            # the one these tsums came out of.
            img = cv2.imdecode(np.fromfile(str(folder / data["image"]), np.uint8),
                               cv2.IMREAD_COLOR)
            bx, by, bw, bh = data["board"]
            cluster_map, _ = _quantise(img[by:by + bh, bx:bx + bw], params["k"], centres)
            boards.append((tsums, radius, cluster_map, links))
            total += len(links)

    if not total:
        print("\n(no labelled path survived re-detection, so the link rule "
              "cannot be replayed -- re-label a board and try again)")
        return

    print(f"\nreplaying {total} of your links through adjacency() itself:")
    print(f"  {'link-px':>8}  {'accepted':>8}  {'wrong kind':>10}  {'too far':>8}  {'blocked':>8}")
    for cap in (90, 100, 105, 120, 150, 200, 0):
        ok = kind = far = blocked = 0
        for tsums, radius, _labels, links in boards:
            # Once per board per threshold, not once per link: the graph is the
            # same for every pair on it. `link` is set absurdly high so that
            # link_px is the only distance gate in play.
            adj = adjacency(tsums, radius, link=1e6, link_px=(cap or None))
            for i, j in links:
                if tsums[i].kind != tsums[j].kind:
                    kind += 1
                elif j in adj[i]:
                    ok += 1
                elif cap and math.hypot(tsums[i].x - tsums[j].x,
                                        tsums[i].y - tsums[j].y) > cap:
                    far += 1
                else:
                    blocked += 1
        label = f"{cap}" if cap else "inf"
        print(f"  {label:>8}  {100 * ok / total:7.1f}%  {kind:10d}  {far:8d}  {blocked:8d}")

    print("  'wrong kind' is a detection error, not a distance one: two tsums you")
    print("  say are the same character were read as different colours, so no")
    print("  --link-px can ever join them. 'blocked' is the third-tsum test.")

    # `--mode blob` on the same links and the same detections. Reported here
    # rather than in a note, because the 86.6% it was landed on was measured
    # against one snapshot of detection: anything that moves detection moves
    # this, and the point of the row is that it is re-runnable.
    ok = kind = missed = 0
    edges_blob = edges_touch = 0
    t0 = time.perf_counter()
    for tsums, radius, labels, links in boards:
        adj = blob_adjacency(labels, tsums, radius)
        edges_blob += sum(len(a) for a in adj) // 2
        edges_touch += sum(len(a) for a in adjacency(tsums, radius, link=1e6,
                                                     link_px=105)) // 2
        for i, j in links:
            if tsums[i].kind != tsums[j].kind:
                kind += 1
            elif j in adj[i]:
                ok += 1
            else:
                missed += 1
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n  --mode blob, same links and detections: {100 * ok / total:.1f}% accepted "
          f"({kind} wrong kind, {missed} not connected)")
    print(f"  graph size: {edges_blob} edges against {edges_touch} for touch at "
          f"--link-px 105; {ms / max(1, len(boards)):.0f}ms per board")
    print("  Acceptance is positive-only -- the labels hold no 'these two are NOT")
    print("  linkable' pairs -- so read it beside the edge count, which is the")
    print("  control: a rule that accepts more by inventing edges shows up there.")


def _dataset(args) -> int:
    """Is a collected dataset actually labelled, or is it 803MB of noise?

    A sample is worth keeping only if the change between `before` and `marked`
    is the game's highlight. Nothing about a folder of JPEGs says whether it
    is, and the first collection ran for thirteen hours before anybody could
    tell -- so this is the check that has to run before a collection is
    trusted, trained on, or sent anywhere.

    Three questions, in the order that decides the answer:

    1. **Was the board still?** `baseline` is the median change over the tsums
       the stroke never touched. It is the floor any mark has to clear. When it
       sits above the threshold, the reading is the pile settling.
    2. **Do the marks look like one character?** The game marks same-character
       tsums, so the ones it lit should share an appearance -- measured here as
       Lab distance to the pressed tsum's own colour, against the distance to
       the rest of the board. A ratio near 1 means the reading carries no
       information about identity, whatever threshold it was taken at.
    3. **Does it agree with k-means?** Only worth asking once 1 and 2 pass. The
       point of the label is to correct the clustering, so disagreement is the
       signal being collected -- but disagreement from a reading that failed
       question 2 is just noise disagreeing with a working detector.
    """
    root = Path(args.dir)
    # Three different problems used to print the same line. The commonest by
    # far is pointing this at the default before anything has ever collected
    # into it, and "no session folders" reads like a verdict on the data
    # rather than what it is -- nothing has been collected yet.
    if not root.exists():
        raise SystemExit(
            f"{root.resolve()} does not exist.\n"
            f"Nothing has collected into it yet. Set `dataset.enabled: true` in "
            f"config.yaml\n(or tick Data collection in the tray), play a round, "
            f"then run this again --\nthe folder appears on the first sampled "
            f"drag. To check a collection that lives\nsomewhere else, pass "
            f"--dir <path to it>.")
    if not root.is_dir():
        raise SystemExit(f"{root} is a file, not a dataset directory")
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        raise SystemExit(
            f"{root.resolve()} holds no session folders.\n"
            f"A session is a `<timestamp>_<pid>` directory, written on the first "
            f"sampled drag\nof a round -- so an empty dataset directory means "
            f"collection never ran, or every\nsample was refused. The round "
            f"summary says which.")

    rows: list[tuple[Path, dict]] = []
    schemas: dict[int, int] = {}
    for folder in sessions:
        jl = folder / "samples.jsonl"
        if not jl.exists():
            continue
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            v = int(row.get("schema", 0))
            schemas[v] = schemas.get(v, 0) + 1
            rows.append((folder, row))
    if not rows:
        raise SystemExit(f"no samples in {root}")
    if args.limit:
        rows = rows[:args.limit]

    total_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"{len(rows)} sample(s) in {len(sessions)} session(s), "
          f"{total_bytes / 1024 / 1024:.0f}MB on disk")
    print("  schema: " + ", ".join(f"v{k} x{v}" for k, v in sorted(schemas.items())))

    baselines: list[float] = []
    marked_share: list[float] = []
    agree: list[float] = []
    base_rate: list[float] = []
    lifts: list[float] = []
    strong: list[tuple] = []
    read = 0
    for folder, row in rows:
        tsums = row.get("tsums") or []
        head = int(row.get("head", 0))
        if len(tsums) < 10 or head >= len(tsums):
            continue
        opts = row.get("options") or {}
        threshold = float(opts.get("hold_threshold", 8.0))
        aura = float(opts.get("hold_aura", 90.0))

        # Schema 2 wrote the reading down; schema 1 did not, so it has to be
        # recomputed from the images. Either way the number means the same
        # thing, which is what makes the two collections comparable.
        #
        # The images get decoded regardless, up to `--appearance` samples: the
        # appearance test below is the one that decides the verdict, and a run
        # that skips it can only report that it does not know. Skipping it and
        # passing on the other two measures is how this command first blessed
        # a collection whose marks were not there.
        values = row.get("marks")
        before = None
        if not values or args.reread or len(lifts) < args.appearance:
            i = int(row["index"])
            before = cv2.imread(str(folder / f"{i:04d}_before.jpg"))
            marked = cv2.imread(str(folder / f"{i:04d}_marked.jpg"))
            if before is None or marked is None or before.shape != marked.shape:
                continue
            if not values or args.reread:
                values = _disk_means(cv2.absdiff(marked, before).max(axis=2), tsums)
        values = np.asarray(values, np.float32)
        if len(values) != len(tsums):
            continue
        read += 1

        chain = set(row.get("proposed") or [])
        idle = [v for i, v in enumerate(values) if i not in chain]
        baseline = row.get("baseline")
        if baseline is None:
            baseline = float(np.median(idle)) if idle else 0.0
        baselines.append(float(baseline))

        hx, hy = tsums[head]["x"], tsums[head]["y"]
        far = np.array([math.hypot(t["x"] - hx, t["y"] - hy) > aura for t in tsums])
        # Score against the bar the sample was actually judged on. Schema 1
        # had only the fixed threshold, and that is most of why every reading
        # looked like half the board: measured with `hold`, real marks sit
        # 8x-25x above the board's floor, and 8.0 sits inside the floor's own
        # noise on a live board. `--floor-mult` re-scores either collection at
        # a floor-relative bar so the two can be compared on equal terms.
        floor = float(np.median(values[far])) if far.any() else 0.0
        bar = float(row.get("bar") or 0.0)
        if args.floor_mult:
            bar = max(threshold, args.floor_mult * floor)
        lit = values > (bar or threshold)
        lit[head] = False
        if far.sum() < 5:
            continue
        marked_share.append(float(lit[far].mean()))

        # Needs pixels, so only when the images are being decoded anyway.
        if before is not None and (far & lit).any() and (far & ~lit).any():
            lab = _disk_lab(cv2.cvtColor(before, cv2.COLOR_BGR2Lab), tsums)
            dist = np.linalg.norm(lab - lab[head], axis=1)
            lifts.append(float(dist[far & ~lit].mean() /
                               max(1e-6, dist[far & lit].mean())))

        # The shape of the reaction, reported rather than interpreted. A count
        # of "strongly reacting" tsums was tried here as a way to separate a
        # real board-wide highlight from the pressed tsum glowing alone, at
        # four different thresholds; none of them separated the two, because
        # the pressed tsum's own glow is *weaker* than a score popup drifting
        # over the board. The appearance test below is the discriminator.
        strong.append((float(values[head]), float(np.median(values[far])),
                       float(values[far].max())))

        kind = np.array([t["kind"] for t in tsums])
        same = (kind == kind[head]) & far
        if (far & lit).any():
            agree.append(float(same[far & lit].mean()))
            base_rate.append(float(same[far].mean()))

    if not read:
        raise SystemExit("no sample could be read -- are the .jpg files present?")

    arr = np.asarray(baselines)
    print("\nboard motion at press time (median change over untouched tsums):")
    for q in (10, 50, 90):
        print(f"  p{q:<3d} {np.percentile(arr, q):6.1f}")
    over = float((arr > 8.0).mean())
    print(f"  past the 8.0 mark threshold on {over * 100:5.1f}% of samples"
          f"{'   <- the diff is reading motion' if over > 0.2 else ''}")

    print(f"\nshare of the board outside the glow read as marked: "
          f"{np.mean(marked_share) * 100:.1f}%")

    if strong:
        sa = np.asarray(strong)
        print(f"\nreaction to the press: pressed tsum {np.median(sa[:, 0]):.0f}, "
              f"board median {np.median(sa[:, 1]):.0f},\n  strongest other tsum "
              f"{np.median(sa[:, 2]):.0f}. The strongest is usually a score popup "
              f"drifting\n  across rather than a mark, which is why none of these "
              f"decides anything alone.")

    lift = float(np.mean(lifts)) if lifts else None
    if lift is not None:
        print(f"\nappearance test ({len(lifts)} samples): tsums read as marked sit "
              f"{lift:.2f}x closer in\n  colour to the pressed tsum than the rest of "
              f"the board. 1.00x means the\n  reading knows nothing about identity -- "
              f"the game's own marks are all one\n  character, so a real label scores "
              f"well above 1.")

        # Each sample's figure is a RATIO, and a ratio has no upper bound: a
        # sample whose marked tsums are nearly the pressed tsum's own colour
        # divides by something close to zero. Those samples are good ones, but
        # a mean over them describes the tail rather than the collection --
        # measured on the first 303-sample corpus, the mean read 3.94x while
        # 13 samples (4.6%) supplied 62% of it and the median sat at 1.30x.
        # Both are printed because they answer different questions: the mean
        # says how strong the best samples are, the median says what a typical
        # one looks like, and it is the typical one that decides whether
        # anything can be learned from the collection as a whole.
        med = float(np.median(lifts))
        share = float(np.sum([v for v in lifts if v > 10]) / np.sum(lifts))
        print(f"  typical sample (median): {med:.2f}x. "
              f"{sum(1 for v in lifts if v > 10)} sample(s) above 10x carry "
              f"{100 * share:.0f}%\n  of the mean -- where these two diverge, "
              f"trust the median.")
        if med < 1.6 <= lift:
            print(f"  NOTE: the mean clears the 1.6x bar below and the median "
                  f"does not. The colour\n  signal is strong in a minority of "
                  f"samples and weak in most of them. `learn`\n  scores the "
                  f"collection as a whole and is the check that prices that.")

    if agree:
        print(f"\nagreement with k-means: {np.mean(agree) * 100:.1f}% of marked tsums "
              f"share the pressed\n  tsum's cluster, against {np.mean(base_rate) * 100:.1f}% "
              f"for marking at random.")

    # Three outcomes, not two. "Motion" and "no highlight" produce nearly
    # identical numbers everywhere except the motion baseline, and they need
    # opposite fixes -- settle the board, or hold longer. Telling a user to
    # fix motion on a collection whose motion is already clean sends them
    # round the loop for nothing.
    print()
    if over > 0.2:
        print("-> UNUSABLE: the diff is reading the board move. What changed between")
        print("   the two frames is the pile settling, not the game answering.")
        print("   Collect again past the 0.15s render floor and with more than one")
        print("   frame (the schema 2 defaults do both), and re-run until board")
        print("   motion p50 sits well under 8.")
        return 1
    if lift is None:
        print("-> INCONCLUSIVE. The board was still, but the test that decides this")
        print("   needs the images and none could be decoded. Check the .jpg files")
        print("   are present next to samples.jsonl.")
        return 2
    if lift < 1.3:
        print("-> UNUSABLE: the board was still, and the marks still are not there.")
        print("   The tsums that reacted look no more like the pressed character than")
        print("   the rest of the board does, so whatever changed was not the game")
        print("   naming a character. Motion is no longer the problem, so the delay")
        print("   is the next thing to rule out -- find when the mark actually")
        print("   renders before collecting more:")
        print("     python -m ttheart_sender.game.tsum hold --hold 3.0")
        print("   then raise dataset.delay in config.yaml past what that reports.")
        print("   If `hold` shows no board-wide highlight at any delay, this build")
        print("   does not have the feature and the label has to come from elsewhere.")
        return 1
    # A lift just over the bar is not the same as a clean label, and saying so
    # matters: `hold` reads real marks at 8x-25x the board floor with the lit
    # tsums plainly one character, which lands far above this. Treating 1.35
    # from 19 samples as a green light would repeat the mistake this command
    # was written to stop.
    if lift < 1.6 or len(lifts) < 100:
        print(f"-> MARGINAL. The marks are there ({lift:.2f}x) but not cleanly, "
              f"on {len(lifts)} sample(s).")
        print("   For comparison, `hold` reads a real mark at 8x-25x the board's")
        print("   floor with the lit tsums obviously one character. Collect more")
        print("   before training on this, and re-run -- a lift that climbs with")
        print("   sample count is a real signal, one that sits still is not.")
        return 0
    print("-> the marks are in the frames. Worth training on.")
    return 0


def _learn(args) -> int:
    """Fit a persistent palette from a collection, and say whether to use it.

    This is the step that closes the loop. Everything before it writes: the
    collector saves boards and the game's own answer, `dataset` says whether
    those answers are real. Nothing read them back, so the only way experience
    reached the bot was a person reading `docs/` and editing a constant.

    What comes out is one file of colour centres. It is not a strategy and it
    does not decide anything -- it replaces the per-frame k-means fit, so a
    cluster id means the same character on every frame of every round instead
    of being re-invented each time the fit is thrown away.

    The verdict is a straight comparison on samples the fit never saw: how
    often the game's own "these are the same character" survives the palette,
    against how often it survived the per-frame clustering that actually ran
    live on those same frames. Nothing is enabled here either way -- the file
    is written, the numbers are printed, and turning it on is a line in
    `play.yaml`.
    """
    from . import learn as learn_mod

    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(
            f"{root.resolve()} does not exist.\n"
            f"Nothing has collected into it yet. Set `dataset.enabled: true` in "
            f"config.yaml\n(or tick Data collection in the tray), play a few "
            f"rounds, then run this again.")

    rows = [(f, r) for f, r in learn_mod.iter_rows(root) if learn_mod.usable(r)]
    sessions = {f for f, _ in rows}
    labelled = sum(1 for _, r in rows if r.get("marked"))
    if not rows:
        raise SystemExit(
            f"no usable samples under {root.resolve()}.\n"
            f"Either nothing has been collected, or every row was a screen with "
            f"fewer than\n{learn_mod.MIN_TSUMS} detections -- which is a menu, "
            f"not a board. Check with `tsum dataset` first.")

    print(f"{len(rows)} usable sample(s) in {len(sessions)} session(s), "
          f"{labelled} carrying marks")
    if len(sessions) < 2:
        # Said before the fit rather than after, because the fix is to play
        # another round and the run is otherwise about to print a number that
        # cannot mean what it looks like.
        print("  NOTE: one session only, so there is nothing to hold out. The "
              "score below is\n  measured on the frames it was fitted on and is "
              "not evidence it generalises.")

    t0 = time.perf_counter()
    palette = learn_mod.build(root, k=args.k, holdout=args.holdout,
                              px_per_frame=args.px_per_frame, seed=args.seed,
                              limit=args.limit,
                              progress=lambda n: print(f"  ...{n} frames read"))
    took = time.perf_counter() - t0

    m = palette.metrics
    if not m.get("pairs"):
        raise SystemExit(
            "not one sample carried marks, so there is nothing to score the "
            "palette against.\nThe palette itself is fine -- the label is what "
            "is missing. Run `tsum dataset`:\nit reports why samples were "
            "refused, and a collection with no marks is the fault\nit exists "
            "to catch.")

    palette.save(Path(args.out))
    print(f"\nfitted k={palette.k} over {palette.meta['fit_samples']} frame(s) "
          f"in {took:.1f}s -> {args.out}")

    faces = palette.faces()
    landed = ", ".join(f"#{i} x{palette.face_counts[i]}" for i in faces[:8])
    print(f"  ids the game's marks landed on: {landed or 'none'}")
    print(f"  {palette.k - len(faces)} of {palette.k} centres are board, "
          f"outline or menu -- expected, and kept")

    where = "held-out session(s)" if m["held_out"] else "the SAME frames it was fitted on"
    print(f"\nscored on {where}: {m['pairs']} pair(s) the game confirmed and "
          f"{m['negatives']}\nweak negative(s), over {m['samples']} sample(s). "
          f"Marks inside the {learn_mod.AURA:.0f}px glow are\nexcluded -- there a "
          f"reaction means proximity, not identity.")
    print(f"\n{'':<18}{'agree':>8}{'split':>8}{'balanced':>10}")
    print(f"  {'learned palette':<16}{100 * m['agreement']:7.1f}%{100 * m['split']:7.1f}%"
          f"{100 * m['balanced']:9.1f}%")
    print(f"  {'per-frame k-means':<16}{100 * m['baseline']:7.1f}%"
          f"{100 * m['baseline_split']:7.1f}%{100 * m['baseline_balanced']:9.1f}%"
          f"   <- what runs today")

    # `balanced`, not `agreement`. Merging two characters into one id RAISES
    # agreement and only costs split, so a verdict read off agreement alone
    # pays a palette to collapse -- measured on the first real corpus, that
    # misreading turned a 1.5-point loss at k=6 into an apparent 3.9-point win.
    lift = m["balanced"] - m["baseline_balanced"]
    print()
    if not m["held_out"]:
        print(f"-> UNPROVEN. {lift * 100:+.1f} points, but measured on its own fit "
              f"set.\n   Play rounds in another session and re-run: two sessions "
              f"is the minimum\n   for this number to mean anything.")
        return 0
    if m["split"] < 0.5:
        # A palette this merged is not worth reporting a lift for at all.
        print(f"-> COLLAPSED. Only {100 * m['split']:.0f}% of weak negatives got a "
              f"different id, so\n   this palette is calling most of the board one "
              f"character. Re-fit with a larger -k.")
        return 1
    if lift < 0.02:
        print(f"-> NO BETTER ({lift * 100:+.1f} points balanced). Per-frame "
              f"clustering is already doing\n   this job as well on your boards. Do "
              f"not enable it.")
        if m["agreement"] > m["baseline"]:
            # The trap this command exists to not fall into, named out loud
            # whenever the data is shaped to spring it.
            print(f"   NOTE: it does win on agreement alone "
                  f"({100 * m['agreement']:.1f}% vs {100 * m['baseline']:.1f}%), and "
                  f"that is not a\n   result -- it separates fewer tsums "
                  f"({100 * m['split']:.1f}% vs {100 * m['baseline_split']:.1f}%), "
                  f"which is what merging\n   characters looks like. Re-fitting to "
                  f"chase that number makes it worse.")
        print(f"   More sessions, or a different -k, are the two things worth trying.")
        return 0
    print(f"-> BETTER by {lift * 100:.1f} points balanced, on sessions it never saw.")
    print(f"   To try it live, add one line under `options:` in flows/play.yaml:")
    print(f"       palette: {args.out}")
    print(f"   and delete that line to revert. Watch the per-kind counts in the "
          f"play log:\n   they should stop reshuffling between rounds.")
    return 0


def _disk_means(diff: np.ndarray, tsums: Sequence[dict]) -> list[float]:
    """Mean change inside each tsum's disk -- the reading `marked_by_game` takes."""
    r = max(2, int(min(t["r"] for t in tsums) * 0.55))
    out = []
    for t in tsums:
        m = np.zeros(diff.shape[:2], np.uint8)
        cv2.circle(m, (int(t["x"]), int(t["y"])), r, 1, -1)
        out.append(float(diff[m.astype(bool)].mean()))
    return out


def _disk_lab(lab: np.ndarray, tsums: Sequence[dict]) -> np.ndarray:
    """Mean Lab colour inside each tsum's disk."""
    r = max(2, int(min(t["r"] for t in tsums) * 0.55))
    out = []
    for t in tsums:
        m = np.zeros(lab.shape[:2], np.uint8)
        cv2.circle(m, (int(t["x"]), int(t["y"])), r, 1, -1)
        out.append(lab[m.astype(bool)].mean(axis=0))
    return np.asarray(out, np.float32)


def _live(args) -> int:
    """Grab the real emulator repeatedly and time the whole loop.

    The point is the numbers, not the picture: capture + detect + path is the
    per-decision cost, and it tells you the fastest honest polling rate.
    """
    from ..app import Application

    app = Application.create()
    window = app.attach_window()
    rect = app.content_rect()
    print(f"attached: {window.info().describe()}  content={rect}")

    grab_ms, det_ms, path_ms = [], [], []
    frame = overlay = None
    # Frame 1 pays for the k-means fit and the radius estimate; every frame
    # after reuses both, which is what a real loop would do.
    palette, radius = None, args.radius
    for _ in range(args.frames):
        t = time.perf_counter()
        frame = app.capture.grab(rect)
        grab_ms.append((time.perf_counter() - t) * 1000)

        bx, by, bw, bh = _board_rect(frame.shape, args.board)
        crop = frame[by:by + bh, bx:bx + bw]

        t = time.perf_counter()
        tsums, radius, palette = detect(crop, k=args.k, radius=radius, palette=palette,
                                        scale=args.scale, include_dark=args.include_dark,
                                        bowl_reject=args.bowl_reject,
                                        debug_dir=args.debug_dir)
        det_ms.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        base = None
        if args.use_base:
            base, _ = read_base_kind(frame, palette, spec=args.base, debug_dir=args.debug_dir)
        cluster_map = (_quantise(crop, args.k, palette)[0]
                       if args.mode == "blob" and palette is not None else None)
        chains = find_chains(tsums, radius, args.link, block=args.block,
                             link_px=args.link_px, base_kind=base, base_only=args.base_only,
                             mode=args.mode, max_chain=args.max_chain,
                             first_leg_px=getattr(args, "first_leg_px", 0.0),
                             labels=cluster_map)
        path_ms.append((time.perf_counter() - t) * 1000)

        overlay = frame.copy()
        overlay[by:by + bh, bx:bx + bw] = draw(crop, tsums, chains, radius)

    if overlay is not None:
        cv2.imwrite(str(args.out), overlay)

    def stat(name, xs):
        print(f"  {name:8s} min {min(xs):6.1f}ms  median {sorted(xs)[len(xs) // 2]:6.1f}ms  max {max(xs):6.1f}ms")

    print(f"{args.frames} frames of {rect.width}x{rect.height}")
    stat("grab", grab_ms)
    stat("detect", det_ms)
    stat("path", path_ms)
    total = [a + b + c for a, b, c in zip(grab_ms, det_ms, path_ms)]
    med = sorted(total)[len(total) // 2]
    print(f"  {'TOTAL':8s} median {med:6.1f}ms  ->  {1000 / med:.1f} decisions/sec")
    print(f"wrote {args.out}")
    app.close()
    return 0


def add_play_args(play, *, merge_default: bool):
    play.add_argument("--board")
    play.add_argument("--base")
    play.add_argument("-k", type=int, default=12)
    play.add_argument("--radius", type=float)
    play.add_argument("--link", type=float, default=1.75,
                      help="max centre gap in tsum diameters; above ~1.4 it "
                           "links tsums that are not touching (--mode touch only)")
    play.add_argument("--link-px", type=float, default=100.0, help="link distance in PIXELS (stable across frames; from labelled links: p90=95). Set 0 to fall back to --link diameters")
    play.add_argument("--max-chain", type=int, default=8, help="cap chain length; picks the tightest cluster of N rather than truncating a board-wide path. 0 = no cap")
    play.add_argument("--verify-hold", action="store_true",
                      help="while the first tsum is held, read which tsums the "
                           "game marks as linkable and drop the chain members it "
                           "did not mark. Costs one capture per drag, because the "
                           "press is the start of the stroke either way. "
                           "MEASURED AND NOT RECOMMENDED: at --hold-delay 0.10 "
                           "the highlight has not rendered and the board is "
                           "still moving, so the reading is noise -- replayed "
                           "over 5,729 collected drags it dropped a mean 3 of 4 "
                           "members on 43%% of them. See docs/DATASET-FINDINGS.md. "
                           "--verify-reach is the version of this that pays: "
                           "same check, bought only on the chains whose reach "
                           "says they are likely to be refused")
    # The doubled percent above is deliberate: argparse %-formats every help
    # string, so a lone "%" followed by a space and a letter is read as a
    # conversion and raises TypeError the moment anyone asks for --help.
    # `tests/test_optional_rules.py` walks every subparser to keep that fixed.
    play.add_argument("--verify-reach", type=float, default=0.0,
                      help="ask the game to check a chain ONLY when it reaches "
                           "further than this many pixels from the tsum being "
                           "pressed; 0 (the default) never does. --verify-hold "
                           "pays that check on every drag, which costs more "
                           "than the waste it removes; the waste is not spread "
                           "evenly. Measured over 303 collected drags, the "
                           "share where the game accepted every proposed "
                           "member runs 100%% under 90px, 81%% at 90-150, 65%% "
                           "at 150-220, 33%% at 220-300 and 11%% beyond -- so "
                           "buying the check only past ~260px caught most of "
                           "the bad chains for a third of the cost. See "
                           "docs/DATASET-FINDINGS.md")
    play.add_argument("--verify-extend", action="store_true",
                      help="on a --verify-reach check, REBUILD the chain from "
                           "what the game marked instead of only trimming the "
                           "chain to it. The press has already been paid for "
                           "and the marks name partners the proposal never "
                           "contained: over 4,306 collected drags the game "
                           "marks a mean 6.1 tsums and 4.0 of them were never "
                           "proposed. Replayed at an identical reading cost "
                           "this clears +6%% over the trim in every one of the "
                           "three cost columns, and lengthens chains rather "
                           "than shortening them (6+ chains 6.7%% -> 9.4%%). "
                           "OFF by "
                           "default: it rests on the game accepting a member "
                           "it marked, which --verify-clears has still not "
                           "measured. See docs/DATASET-FINDINGS.md")
    play.add_argument("--verify-no-trim", action="store_true",
                      help="on a check, use the REBUILD but never the trim: if "
                           "the marks do not produce a longer chain, drag the "
                           "proposal exactly as it was rather than cutting it "
                           "to what the game marked, and never abandon. The "
                           "trim assumed a refused member costs the whole "
                           "drag; measured, a drag with one refused member of "
                           "five still clears the other four (89.8%% of marked "
                           "members clear, 22.9%% of refused), and refusals are "
                           "a suffix -- the chain dies at the first bad member, "
                           "so what the trim removes was already worth nothing. "
                           "Needs --verify-reach or --verify-hold. OFF by "
                           "default: derived from 36 drags")
    play.add_argument("--verify-floor-mult", type=float, default=8.0,
                      help="multiple of the board's own noise floor a mark has "
                           "to clear before --verify-extend will build a chain "
                           "through it. Only read when --verify-extend is on, "
                           "and it never moves the trim, which keeps its fixed "
                           "--hold-threshold. 8.0 is the bar 11,537 samples "
                           "settled: below it a tsum that reacts is no more "
                           "the pressed character than the board average is")
    play.add_argument("--verify-delay", type=float, default=0.25,
                      help="seconds to wait before reading the marks on a "
                           "--verify-reach check. Its own setting rather than "
                           "--hold-delay because that one defaults to 0.10, "
                           "below the 0.15 floor at which the highlight has "
                           "rendered at all -- the reason --verify-hold reads "
                           "noise. Clamped up to that floor")
    play.add_argument("--palette", default="",
                      help="use colour centres learned offline by `tsum learn` "
                           "instead of re-fitting k-means on every frame; empty "
                           "(the default) is the per-frame fit. This is the "
                           "other end of --dataset: a cluster id then means the "
                           "same character on every frame of every round, "
                           "rather than being re-invented whenever the fit is "
                           "thrown away. Fit it and score it against the game's "
                           "own labels first -- `tsum learn` refuses to "
                           "recommend one that is no better. A file that will "
                           "not load stops the round rather than quietly "
                           "playing without it")
    play.add_argument("--dataset", default="",
                      help="collect training samples for detection into this "
                           "directory; empty (the default) collects nothing. "
                           "Each sampled drag saves the board before the press "
                           "and again while the game highlights what it marked "
                           "-- that highlight is the label. Costs one extra "
                           "capture and --hold-delay on sampled drags only")
    play.add_argument("--dataset-limit", type=int, default=20,
                      help="most samples to keep from one round")
    play.add_argument("--dataset-every", type=int, default=4,
                      help="sample every Nth drag; boards inside one round are "
                           "alike, so a stride buys more variety than a burst")
    play.add_argument("--dataset-quality", type=int, default=85,
                      help="JPEG quality for the saved crops")
    play.add_argument("--dataset-delay", type=float, default=0.25,
                      help="seconds to wait after the press before photographing "
                           "the marks, on sampled drags only. Not --hold-delay: "
                           "that one is paid on every drag and is set below the "
                           "0.15s floor where the game has finished drawing the "
                           "highlight, which is why the first 5,729 samples "
                           "carried no label at all")
    play.add_argument("--dataset-frames", type=int, default=3,
                      help="frames to read the marks from. Only what changed in "
                           "ALL of them counts -- a mark is in every frame, a "
                           "settling tsum is not. 1 frame is what schema 1 used "
                           "and it read motion")
    play.add_argument("--dataset-gap", type=float, default=0.05,
                      help="seconds between those frames")
    play.add_argument("--dataset-floor-mult", type=float, default=8.0,
                      help="a mark must beat the board's own noise floor by "
                           "this multiple. Measured with `hold`, real marks sit "
                           "8x-25x above it, while the fixed --hold-threshold "
                           "sits inside that floor on a live board. Swept over "
                           "11,537 collected samples: everything under 8x is "
                           "indistinguishable from the board average, so 5 "
                           "spent about a third of each label on noise. 0 = use "
                           "the fixed threshold")
    play.add_argument("--dataset-max-motion", type=float, default=12.0,
                      help="refuse to keep a sample whose board was already "
                           "jiggling this hard at press time -- the reading is "
                           "motion, not marks. 0 = keep everything")
    play.add_argument("--dataset-max-mb", type=float, default=2048.0,
                      help="stop collecting once the dataset directory reaches "
                           "this size. There is no per-day limit; without a "
                           "budget a night of play wrote 803MB. 0 = no cap")
    play.add_argument("--dataset-total", type=int, default=0,
                      help="stop collecting once the dataset directory holds "
                           "this many samples in total. 0 = no cap")
    play.add_argument("--hold-delay", type=float, default=0.10,
                      help="seconds to wait after pressing before reading the marks. "
                           "Paid on every drag, so it is the main speed cost of "
                           "--verify-hold; too low and the marks have not rendered")
    play.add_argument("--hold-threshold", type=float, default=8.0,
                      help="mean pixel change that counts as 'the game marked it'")
    play.add_argument("--hold-aura", type=float, default=90.0,
                      help="the glow washes over this radius, so members within it "
                           "are kept regardless -- only distant ones get dropped")
    play.add_argument("--first-leg-px", type=float, default=0.0,
                      help="drop leading tsums until the chain's OPENING hop is "
                           "no longer than this. The first tsum sets the "
                           "character for the whole stroke, so an opening hop "
                           "that does not register wastes the entire drag. 0 = "
                           "off, which is the measured default: orienting the "
                           "drag to start at its tighter end already brings the "
                           "opening hop to ~64px, and capping below that only "
                           "shortens chains")
    play.add_argument("--mode", choices=["touch", "reach", "blob"], default="touch", help='"touch" (default): only tsums within --link-px of each other, with nothing blocking the line between them, count as linked. "reach": any two same-kind tsums link regardless of distance, which is closer to how the game is actually played -- distance stops mattering and --link-px/--link/--block are ignored. "blob": like "touch" but reads contact off the colour mask instead of inferring it from distance -- accepts more real links, ~45x slower, --link-px/--link/--block are ignored')
    play.add_argument("--block", type=float, default=0.75,
                      help="a tsum within this many radii of the line blocks the link")
    play.add_argument("--radius-lock", type=int, default=0, help='measure the tsum radius over the first N frames that look like a board, then hold it for the round. 0 = off, which re-measures on every refit and on the FEVER transition. The pile does not change size mid-round, so the radius is a constant; measured over 151 frames of one round the per-frame estimate ranges 8-38px on ~26px faces, and a collapsed estimate reads the board at half scale, doubles the count, and still passes --min-tsums. The lock takes the MAX of the samples, not the median: the estimator only ever collapses, never inflates')
    play.add_argument("--fever-hold", type=float, default=FEVER_HOLD,
                      help="seconds one sighting of the FEVER BONUS banner keeps "
                           "FEVER open. Has to outlast the longest stretch the "
                           "banner can go unread mid-run, which is a skill "
                           "animation (a skill pauses FEVER while it plays), not "
                           "the banner's own fade. Too short and one real FEVER "
                           "logs as several, each flip refitting the palette and "
                           "re-reading the base tsum; too long and the end of "
                           "FEVER is noticed a few seconds late, which is mild")
    play.add_argument("--no-fever-banner", dest="fever_banner", action="store_false",
                      help="do not read the FEVER BONUS banner; infer FEVER from the "
                           "max_fever meter and a 10s timer instead, as before the "
                           "banner was added. The banner is a state the game shows "
                           "for the whole of FEVER, so it can be asked every frame "
                           "rather than assumed from one trigger -- this switch is "
                           "the way back if that misreads")
    play.add_argument("--radius-cover", type=float, default=0.25, help='how much of the board rect the detected faces must add up to before a frame gets a vote on the radius, as a fraction of the rect area. Only consulted when --radius-lock is on. The detection count cannot screen a collapsed radius because a collapse INFLATES the count; area can, because a half-scale read covers a quarter of the area. Measured: good reads sit at 0.22-0.38, collapsed reads at 0.08-0.23, and 0.25 keeps 39 of 48 good frames while rejecting 48 of 51 collapsed ones. Lower it if the log says the lock never engages')
    play.add_argument("--fever-min-tsums", type=int, default=0, help="the --min-tsums floor to use while FEVER is running. 0 = use --min-tsums. FEVER fades and overlays the board, so a genuine in-play frame reads ~20 detections where normal play reads ~50, and the normal floor discards half of them. Safe to lower because the floor's real job -- noticing that a frame is a menu rather than a board -- is already done by the FEVER template, which the game does not draw over a menu")
    play.add_argument("--bowl-reject", type=float, default=0.0, help="drop detections whose face colour sits closer than this (Lab) to the board's own colour -- a detection that landed on the bowl instead of on a tsum carries the bowl's colour. 0 = off (the default). Over the ten labelled boards: off f1 0.762, 40 -> 0.785, 60 -> 0.791, 80 -> 0.766, so 40-60 is a plateau. It buys precision with recall, and only a played round prices that trade")
    play.add_argument("--scale", type=float, default=1.0)
    play.add_argument("--fit-effort", type=int, default=1, choices=(1, 2, 3),
                      help="how hard the per-frame colour fit tries when there "
                           "is no cached palette to reuse. The fit is k-means "
                           "with a seed, so the same frame read twice can come "
                           "back different: measured over 60 collected boards, "
                           "two reads agree on 73%% of tsums at level 1, 83%% at "
                           "2 and 91%% at 3, and the count swings by 9 tsums "
                           "between reads at level 1 against 3 at level 3. "
                           "Costs 51/138/349ms per fit, and a round fits about "
                           "five times -- on the fever transition, a shuffle "
                           "and a recalibration, which are the worst frames to "
                           "read badly. Default 1: it is what every measurement "
                           "in docs/DATASET-FINDINGS.md was taken under")
    play.add_argument("--no-dark", dest="include_dark", action="store_false")
    play.add_argument("--no-base", dest="use_base", action="store_false")
    play.add_argument("--base-only", action="store_true")
    play.add_argument("--no-prepare", action="store_true",
                      help="do not move/focus the emulator first")
    play.add_argument("--dry-run", action="store_true",
                      help="print the drag path and stop, without moving the mouse")
    play.add_argument("--duration", type=float, default=0.0,
                      help="keep playing for this many seconds (0 = one chain and exit)")
    play.add_argument("--settle", type=float, default=0.9,
                      help="max seconds to wait for the board to stop moving")
    play.add_argument("--min-chain", type=int, default=3,
                      help="skip chains shorter than this and re-look (3 = play anything legal)")
    play.add_argument("--max-tsums", type=int, default=110,
                      help="frames with MORE detections than this are not a board "
                           "(the Home screen reads 200+)")
    play.add_argument("--repeat-len", type=int, default=5,
                      help="same chain length this many times in a row = stuck, tap shuffle")
    play.add_argument("--min-tsums", type=int, default=20,
                      help="frames with fewer detections are treated as 'not a board'")
    play.add_argument("--max-misses", type=int, default=6,
                      help="consecutive no-chain frames before giving up")
    play.add_argument("--no-verify", dest="verify", action="store_false",
                      help="skip the did-it-clear check after each drag")
    play.add_argument("--verify-clears", action="store_true",
                      help="after each drag, check the dragged tsums' own pixels "
                           "rather than the whole crop, and report how many "
                           "actually left the board. Costs no extra capture -- "
                           "--verify already grabs the frame it reads")
    play.add_argument("--clear-tol", type=float, default=20.0,
                      help="mean change inside a tsum's disk that counts as "
                           "'it is gone'. The log prints the board's idle noise "
                           "beside every reading, so a round tells you whether "
                           "this needs moving")
    play.add_argument("--change-tol", type=float, default=2.0,
                      help="mean pixel change below this means the drag did not register")
    play.add_argument("--max-repeats", type=int, default=3,
                      help="same chain seen this many times in the window = stuck, tap shuffle")
    play.add_argument("--repeat-window", type=int, default=10,
                      help="how many recent plays the repeat check looks back over")
    play.add_argument("--max-stalls", type=int, default=4,
                      help="consecutive non-registering drags before giving up")
    play.add_argument("--shuffle", default="459,859",
                      help='content-relative "x,y" of the shuffle button -- from '
                           "`python main.py point`, its `content=(x, y)` output. "
                           "Tapped on recalibration; empty string disables it.")
    play.add_argument("--shuffle-clicks", type=int, default=3)
    play.add_argument("--shuffle-delay", type=float, default=0.3,
                      help="seconds between shuffle taps")
    play.add_argument("--character", default="",
                      help="path to a character classifier (models/character.onnx "
                           "plus its .json). Names each tsum instead of trusting "
                           "the per-frame colour cluster, WHERE IT IS SURE -- an "
                           "unsure crop keeps its `kind`, so an unlabelled "
                           "character plays exactly as it does now. `kind` agrees "
                           "with the game about a confirmed partner 37%% of the "
                           "time; the model reads ~95%% on held-out sessions. "
                           "Costs ~11ms a board. Empty = off")
    play.add_argument("--character-confidence", type=float, default=0.85,
                      help="softmax floor for naming a tsum. Below it the crop "
                           "keeps its colour cluster rather than being guessed "
                           "at -- a wrong name is worse than no name, because "
                           "`adjacency` believes it and a chain dies at its "
                           "first wrong member")
    play.add_argument("--reject-model", default="",
                      help="ONNX that says whether a detection is a tsum at "
                           "all, dropping the ones that are board. Trained on "
                           "705 crops a person labelled `board`, verified "
                           "against the game's own marks: it confirmed 0.0%% "
                           "of them. Empty by default -- see flows/play.yaml")
    play.add_argument("--reject-floor", type=float, default=RejectModel.FLOOR,
                      help="drop a detection whose probability of being a tsum "
                           "is below this")
    play.add_argument("--chain-model", default="",
                      help="ONNX that scores how likely the GAME is to accept "
                           "each member of a proposed chain. With it, chains "
                           "are ranked by expected accepted members instead "
                           "of by raw length. Empty by default -- it has never "
                           "been played; see flows/play.yaml")
    play.add_argument("--chain-bonus", type=float, default=0.0,
                      help="added to a chain's score per member, so a "
                           "preference for length can be mixed back in")
    play.add_argument("--ab", default="",
                      help="run an A/B: the name of another play option to "
                           "ALTERNATE between rounds, one round on and the "
                           "next off, e.g. --ab reject_model. The arm is "
                           "chosen inside the play loop, so no flow hop can "
                           "drop it, and it records itself in every sample "
                           "because the option's own value is what changes. "
                           "Empty by default -- read the result with "
                           "scripts/ab_eval.py")
    play.add_argument("--ab-off", default="",
                      help="the value the --ab option takes on its OFF "
                           "rounds, coerced to that option's type. The ON "
                           "value is whatever the option is already set to")
    play.add_argument("--character-min-visible", type=float,
                      default=CHARACTER_MIN_VISIBLE,
                      help="least of a tsum that must be showing before the "
                           "model is asked at all. The labelled crops are ALL "
                           "above 0.55 and the median board detection shows "
                           "0.41, so below the floor the model is answering "
                           "about pictures it has no example of -- and it does "
                           "so at full confidence, which is why the softmax "
                           "floor above cannot catch it. 0 asks about everything")
    play.add_argument("--recolour", type=float, default=0.0,
                      help="re-decide identity by sampling each FACE once and "
                           "merging groups closer than this in Lab, instead of "
                           "trusting the pixel-level k-means. 0 = off. Pixel "
                           "k-means splits one character across two clusters "
                           "when it is lit differently across the board, and "
                           "`adjacency` will not link across a kind "
                           "difference, so those partners are unreachable. "
                           "MEASURED BOTH WAYS AND THEY DISAGREE: scored "
                           "symmetrically against the game's marks it is flat "
                           "(balanced 55.1%% at k-means, 55.6%% at its best, "
                           "over 685 samples), but scored on the chains it "
                           "builds it gains, and that second score prices an "
                           "unknown tail member as positive so it rewards "
                           "merging without bound. Only a played round "
                           "decides. Try 35 -- it keeps mean chain length near "
                           "6, inside the range the clear model was fitted on. "
                           "Costs ~18ms a frame",
                      )
    play.add_argument("--kinds", type=int, default=0,
                      help="sort the board into exactly this many identities "
                           "by face colour, instead of trusting the "
                           "pixel-level k-means. 0 = off. THE GAME BOUNDS "
                           "THIS: a board holds at most 5 characters, 4 with "
                           "an item, so identity is a closed problem -- sort "
                           "the board into 5 piles -- rather than the open one "
                           "of naming 40-odd characters most of which nobody "
                           "has labelled. Unlike --recolour this cannot run "
                           "away, because a count is bounded where a distance "
                           "threshold is not. Scored over 1,117 saved boards "
                           "against the game's own marks, in tsums cleared per "
                           "drag: 3.26 today, 3.31 at 3, 3.41 at 4, 3.34 at 5, "
                           "3.23 at 6 -- an interior optimum, +4.6%% at its "
                           "best against a paired 2 s.e. of 0.047. Real, and "
                           "small. Try 4. Costs ~2ms a frame",
                      )
    play.add_argument("--settle-board", action="store_true",
                      help="wait only for the BOARD to stop moving, not the "
                           "whole screen. The score counter, the timer and the "
                           "FEVER meter animate continuously and all sit "
                           "outside the board, so a whole-frame wait can hold "
                           "at its cap long after the tsums have stopped "
                           "falling -- and that wait is the biggest number in "
                           "a round: 83%% of the gap between chains is stroke "
                           "and waiting, against 17%% thinking. Score tracks "
                           "CHAINS PLAYED (r=+0.91 over five rounds), not "
                           "chain length, so this is aimed at the thing that "
                           "scores. OFF until a round says otherwise",
                      )
    play.add_argument("--purity", type=float, default=35.0,
                      help="drop chain members whose colour is this far (Lab) "
                           "from the chain median; 0 disables")
    play.add_argument("--skill", default="76,854",
                      help="content-relative \"x,y\" of the skill button; fired when its "
                           "ring turns gold. Empty string disables")
    play.add_argument("--skill-gold", type=float, default=0.12,
                      help="gold fraction of the ring that counts as charged "
                           "(uncharged measures 0.00)")
    play.add_argument("--skill-inner", type=float, default=30.0)
    play.add_argument("--skill-outer", type=float, default=52.0)
    play.add_argument("--bubble", default="bubble,time_bubble,coin_bubble,score_bubble",
                      help="comma-separated template names to tap when found "
                           "(capture each with `python main.py snip <name>`); "
                           "missing ones are skipped, empty string disables")
    play.add_argument("--bubble-confidence", type=float, default=0.80)
    play.add_argument("--countdown", type=float, default=3.0)
    play.add_argument("--step-px", type=float, default=8.0,
                      help="cursor step along each leg; larger drops touch events")
    play.add_argument("--per-step", type=float, default=0.004,
                      help="seconds paused at each step -- the main drag-speed knob")
    play.add_argument("--hold", type=float, default=0.05,
                      help="seconds the button stays down at drag endpoints and shuffle taps")
    play.add_argument("--move-time", type=float, default=0.05,
                      help="cursor travel time to the shuffle button")
    play.set_defaults(merge=merge_default)


def play_settings(opts: argparse.Namespace) -> dict:
    """Every play setting in force, for a sample to be written with.

    Deliberately *not* a chosen list of the interesting ones. This used to be
    a hand-written dict of a dozen keys, and it went stale three rounds
    running: `verify_reach` was missing when the eleventh round needed it,
    `fit_effort` and `verify_extend` were each bolted on by the round that
    happened to add them, and `verify_clears` was never there at all -- so a
    corpus could not say whether the switch being A/B'd had been on while it
    was collected. Curating the list is the bug; the fix is not to curate it.

    So the whole namespace goes in, and a flag added tomorrow is recorded by
    this function without anyone remembering to come back here. Every value
    the parser can produce is a JSON scalar (`tests/test_dataset_options.py`
    holds that true), and the effective value is taken at write time, so a
    setting the loop clamped mid-round -- `verify_delay` up to the render
    floor -- is recorded as what was actually used rather than what was asked
    for.

    What it costs: about 1.5KB of JSONL per sample, against ~90KB of JPEG.
    """
    return {k: v for k, v in sorted(vars(opts).items())
            if isinstance(v, (bool, int, float, str)) or v is None}


def play_defaults(*, merge: bool = False) -> argparse.Namespace:
    """The `play` command's options, at their defaults.

    The parser is the single definition of what a play run can be tuned with,
    so callers that are not the CLI -- the `play_tsum` flow action -- start
    from this and override the handful of keys they care about, rather than
    keeping a second copy of forty defaults that can drift out of step.
    """
    ap = argparse.ArgumentParser(add_help=False)
    add_play_args(ap, merge_default=merge)
    return ap.parse_args([])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("image", type=Path)
    a.add_argument("-o", "--out", type=Path, default=Path("tsum_path_out.png"))
    a.add_argument("--board", help='"x,y,w,h" in pixels, or "full". Default: play area of a 540x960 grab.')
    a.add_argument("-k", type=int, default=12, help="colour clusters (distinct tsums + board + ink)")
    a.add_argument("--radius", type=float, help="tsum radius in px; auto-estimated if omitted")
    a.add_argument("--link", type=float, default=1.35, help="link distance in tsum diameters (--mode touch only)")
    a.add_argument("--link-px", type=float, default=100.0, help="link distance in PIXELS (stable across frames; from labelled links: p90=95). Set 0 to fall back to --link diameters")
    a.add_argument("--max-chain", type=int, default=8, help="cap chain length; picks the tightest cluster of N rather than truncating a board-wide path. 0 = no cap")
    a.add_argument("--mode", choices=["touch", "reach", "blob"], default="touch", help='"touch" (default): only tsums within --link-px of each other, with nothing blocking the line between them, count as linked. "reach": any two same-kind tsums link regardless of distance, which is closer to how the game is actually played -- distance stops mattering and --link-px/--link/--block are ignored. "blob": like "touch" but reads contact off the colour mask instead of inferring it from distance -- accepts more real links, ~45x slower, --link-px/--link/--block are ignored')
    a.add_argument("--block", type=float, default=0.75, help="occlusion radius for blocking a link")
    a.add_argument("--bowl-reject", type=float, default=0.0, help="drop detections whose face colour sits closer than this (Lab) to the board's own colour -- a detection that landed on the bowl instead of on a tsum carries the bowl's colour. 0 = off (the default). Over the ten labelled boards: off f1 0.762, 40 -> 0.785, 60 -> 0.791, 80 -> 0.766, so 40-60 is a plateau. It buys precision with recall, and only a played round prices that trade")
    a.add_argument("--scale", type=float, default=1.0, help="run detection on a downscaled copy (0.5 = ~4x faster)")
    a.add_argument("--no-dark", dest="include_dark", action="store_false",
                   help="skip black-faced tsums (Mickey) instead of detecting them")
    a.add_argument("--base", help='skill-icon sample as "cx,cy,r" in px; default is the bottom-left button')
    a.add_argument("--no-base", dest="use_base", action="store_false", help="rank purely by chain length")
    a.add_argument("--base-only", action="store_true", help="ignore chains that are not the equipped tsum")
    a.add_argument("--merge", action="store_true",
                   help="fold two-tone faces into one tsum before detecting")
    a.add_argument("--debug-dir", type=Path)

    s = sub.add_parser("synth")
    s.add_argument("-o", "--out", type=Path, default=Path("synth_board.png"))
    s.add_argument("--count", type=int, default=46)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--tilt", type=float, default=1.0,
                   help="0 = all upright, 1 = fully random rotation")

    g = sub.add_parser("grab", help="save a burst of real boards to label later")
    g.add_argument("-n", "--frames", type=int, default=10, help="how many boards to keep")
    g.add_argument("--dir", default="scratchpad", help="where to write them")
    g.add_argument("--prefix", default="board", help="filename stem; a free number is appended")
    g.add_argument("--interval", type=float, default=3.0,
                   help="seconds between grabs -- long enough that the board has changed")
    g.add_argument("--countdown", type=float, default=5.0,
                   help="seconds before the first grab, to get a round started")
    g.add_argument("--board")
    g.add_argument("-k", type=int, default=12)
    g.add_argument("--radius", type=float)
    g.add_argument("--no-dark", dest="include_dark", action="store_false")
    g.add_argument("--no-prepare", action="store_true")
    g.add_argument("--min-tsums", type=int, default=20)
    g.add_argument("--max-tsums", type=int, default=110)

    idl = sub.add_parser("idle", help="film the hint the game shows an idle player")
    idl.add_argument("--seconds", type=float, default=45.0, help="how long to film")
    idl.add_argument("--fps", type=float, default=5.0)
    idl.add_argument("--countdown", type=float, default=3.0)
    idl.add_argument("--settle", type=float, default=10.0,
                     help="seconds to wait for the pile to stop moving first")
    idl.add_argument("--keep", type=int, default=12, help="frames to save")
    idl.add_argument("--threshold", type=float, default=8.0)
    idl.add_argument("--dir", default="scratchpad")
    idl.add_argument("--board")
    idl.add_argument("-k", type=int, default=12)
    idl.add_argument("--no-dark", dest="include_dark", action="store_false")
    idl.add_argument("--no-prepare", action="store_true")

    hp = sub.add_parser("hold", help="press one tsum and photograph what the game marks")
    hp.add_argument("--hold", type=float, default=3.0,
                    help="seconds to keep it pressed -- long enough to watch")
    hp.add_argument("--countdown", type=float, default=0.0,
                    help="0 = press as soon as the board is still. Waiting is "
                         "what makes the game put up its idle hint")
    hp.add_argument("--settle", type=float, default=3.0)
    hp.add_argument("--dir", default="scratchpad")
    hp.add_argument("--threshold", type=float, default=8.0,
                    help="mean pixel change that counts as 'the game marked this one'")
    hp.add_argument("--link-px", type=float, default=105.0)
    hp.add_argument("--block", type=float, default=1.25)
    hp.add_argument("--max-chain", type=int, default=8)
    hp.add_argument("--board")
    hp.add_argument("-k", type=int, default=12)
    hp.add_argument("--min-tsums", type=int, default=20,
                    help="refuse to press a frame with fewer detections than "
                         "this -- the same gate `play` applies")
    hp.add_argument("--max-tsums", type=int, default=110,
                    help="and refuse one with more. Over-split frames offer "
                         "phantoms to press, and a press that lands on nothing "
                         "looks exactly like a game that marks nothing")
    hp.add_argument("--no-dark", dest="include_dark", action="store_false")
    hp.add_argument("--no-prepare", action="store_true")

    asi = sub.add_parser("assist", help="you hold a tsum, the app reads the game's "
                                        "marks and drags the chain for you")
    asi.add_argument("--delay", type=float, default=0.25,
                     help="seconds to wait after your press before reading the marks. "
                          "0.15 is the floor below which they have not rendered; "
                          "unlike --verify-hold there is no throughput to lose here, "
                          "so this is deliberately generous")
    asi.add_argument("--threshold", type=float, default=8.0,
                     help="mean pixel change that counts as 'the game marked this one'")
    asi.add_argument("--mark-frames", type=int, default=3,
                     help="how many frames to read the marks from. Only what "
                          "changed in ALL of them counts, which is what keeps "
                          "FEVER's sparkles and a settling pile out of the "
                          "reading. Measured: 1 frame gives ~11 false marks "
                          "against moving sparkles, 2 gives 0.7, 3 gives none")
    asi.add_argument("--mark-gap", type=float, default=0.05,
                     help="seconds between those frames -- long enough that "
                          "anything animating has moved on")
    asi.add_argument("--aura", type=float, default=90.0,
                     help="the glow washes over this radius, so hits within it are "
                          "reported separately -- they may be splash rather than marks")
    asi.add_argument("--min-chain", type=int, default=3,
                     help="below this the cursor is not moved at all: the game clears "
                          "nothing under three, so a short path is worse than none")
    asi.add_argument("--max-chain", type=int, default=0,
                     help="0 = drag everything the game marked. A cap keeps a guessed "
                          "tour tight in `play`, but nothing here is guessed")
    asi.add_argument("--step-px", type=float, default=8.0)
    asi.add_argument("--per-step", type=float, default=0.004)
    asi.add_argument("--drift", type=float, default=0.0,
                     help="refuse to read a board drifting by more than this, "
                          "measured over the board only. 0 = never refuse up "
                          "front; --max-marked catches the same thing after the "
                          "fact, and more reliably")
    asi.add_argument("--max-marked", type=float, default=0.5,
                     help="refuse a reading where more than this share of the "
                          "board lit up -- that is the pile moving, not the game "
                          "marking one character")
    asi.add_argument("--no-auto-release", dest="auto_release", action="store_false",
                     help="wait for you to let go instead of releasing for you "
                          "once the path is walked. Releasing early is the "
                          "commonest way a press is wasted, which is why the "
                          "release is taken over by default")
    asi.add_argument("--dry-run", action="store_true",
                     help="report the path it would draw without moving the cursor")
    asi.add_argument("--debug", action="store_true",
                     help="write the marked-up board and the raw diff per press")
    asi.add_argument("--dir", default="scratchpad")
    asi.add_argument("--board")
    asi.add_argument("-k", type=int, default=12)
    asi.add_argument("--no-dark", dest="include_dark", action="store_false")
    asi.add_argument("--no-prepare", action="store_true")

    lab = sub.add_parser("label", help="mark up a board: chains you'd drag, and detection's mistakes")
    lab.add_argument("image", type=Path)
    lab.add_argument("--board")
    lab.add_argument("-k", type=int, default=12)
    lab.add_argument("--radius", type=float)
    lab.add_argument("--no-dark", dest="include_dark", action="store_false")
    lab.add_argument("--merge", action="store_true")

    ev = sub.add_parser("eval", help="score detection against your reviewed boards")
    ev.add_argument("--dir", default="scratchpad", help="folder holding *.label.json")
    ev.add_argument("-k", type=int, default=12)
    ev.add_argument("--radius", type=float)
    ev.add_argument("--scale", type=float, default=1.0)
    ev.add_argument("--no-dark", dest="include_dark", action="store_false")
    ev.add_argument("--merge", action="store_true")
    ev.add_argument("--bowl-reject", type=float, default=0.0,
                    help="reject detections this close (Lab) to the board colour; "
                         "0 = off. Sweepable, e.g. --sweep bowl_reject=0,40,60,80")
    ev.add_argument("--tol", type=float, default=0.6,
                    help="a detection counts as the same tsum within this many radii "
                         "of a ground-truth point")
    ev.add_argument("--sweep", action="append", metavar="NAME=v1,v2",
                    help="try every value of a detect parameter and rank by f1; "
                         "repeatable, e.g. --sweep k=8,12,16 --sweep floor_frac=0.35,0.42")

    sk = sub.add_parser("skillcheck", help="print the skill button's gold reading right now")
    sk.add_argument("--skill", default="76,854")
    sk.add_argument("--skill-inner", type=float, default=30.0)
    sk.add_argument("--skill-outer", type=float, default=52.0)
    sk.add_argument("-n", "--frames", type=int, default=10)

    sc = sub.add_parser("score", help="read your labels and report the link rule they imply")
    sc.add_argument("--dir", default="scratchpad", help="folder holding *.label.json")
    sc.add_argument("--tol", type=float, default=0.6,
                    help="how close a re-detected tsum must be to a labelled one "
                         "to count as the same, in radii")

    ds = sub.add_parser("dataset", help="check whether a collected dataset is "
                                       "actually labelled before trusting it")
    ds.add_argument("--dir", default="dataset",
                    help="the dataset directory holding the session folders")
    ds.add_argument("--limit", type=int, default=0,
                    help="stop after this many samples. 0 = all of them; a "
                         "full pass over a night's collection decodes "
                         "thousands of JPEGs")
    ds.add_argument("--floor-mult", type=float, default=0.0,
                    help="re-score with a bar this many times the board's own "
                         "noise floor instead of the bar the sample recorded. "
                         "Real marks sit 8x-25x above that floor, so this is "
                         "how a schema 1 collection (fixed threshold only) is "
                         "compared with a schema 2 one on equal terms")
    ds.add_argument("--appearance", type=int, default=200,
                    help="decode this many samples for the appearance test -- "
                         "the one that decides the verdict. Lower it to skip "
                         "faster over a huge collection; 0 turns it off, and "
                         "the verdict then reports that it does not know")
    ds.add_argument("--reread", action="store_true",
                    help="recompute the reading from the images instead of "
                         "trusting what the row recorded. Required for schema "
                         "1 folders, which recorded nothing, and the way to "
                         "re-score a schema 2 collection at a different "
                         "threshold")

    ln = sub.add_parser("learn", help="fit a persistent colour palette from a "
                                      "collection, and score it against the labels")
    ln.add_argument("--dir", default="dataset",
                    help="the dataset directory holding the session folders")
    ln.add_argument("-o", "--out", type=Path, default=Path("models/palette.json"),
                    help="where to write the fitted palette")
    ln.add_argument("-k", type=int, default=12,
                    help="colour clusters, same meaning as `play -k`. It must "
                         "match what play runs with, because the palette IS "
                         "play's k-means result")
    ln.add_argument("--holdout", type=float, default=0.25,
                    help="share of SESSIONS (not samples) kept back to score "
                         "on. Samples inside one session are near-duplicates, "
                         "so a per-sample split would score the fit against "
                         "frames it effectively saw. 0 = score on everything, "
                         "which measures nothing")
    ln.add_argument("--px-per-frame", type=int, default=4000,
                    help="pixels sampled from each frame. The fit needs the "
                         "colour distribution, not every pixel; raising this "
                         "costs memory over a big corpus and changes little")
    ln.add_argument("--limit", type=int, default=0,
                    help="stop after this many samples. 0 = all of them")
    ln.add_argument("--seed", type=int, default=0)

    live = sub.add_parser("live", help="grab the emulator N times and report real throughput")
    live.add_argument("-n", "--frames", type=int, default=20)
    live.add_argument("-o", "--out", type=Path, default=Path("live_out.png"))
    live.add_argument("--board")
    live.add_argument("-k", type=int, default=8)
    live.add_argument("--radius", type=float)
    live.add_argument("--link", type=float, default=1.35)
    live.add_argument("--link-px", type=float, default=100.0, help="link distance in PIXELS (stable across frames; from labelled links: p90=95). Set 0 to fall back to --link diameters")
    live.add_argument("--max-chain", type=int, default=8, help="cap chain length; picks the tightest cluster of N rather than truncating a board-wide path. 0 = no cap")
    live.add_argument("--mode", choices=["touch", "reach", "blob"], default="touch", help='"touch" (default): only tsums within --link-px of each other, with nothing blocking the line between them, count as linked. "reach": any two same-kind tsums link regardless of distance, which is closer to how the game is actually played -- distance stops mattering and --link-px/--link/--block are ignored. "blob": like "touch" but reads contact off the colour mask instead of inferring it from distance -- accepts more real links, ~45x slower, --link-px/--link/--block are ignored')
    live.add_argument("--block", type=float, default=0.75)
    live.add_argument("--bowl-reject", type=float, default=0.0, help="drop detections whose face colour sits closer than this (Lab) to the board's own colour -- a detection that landed on the bowl instead of on a tsum carries the bowl's colour. 0 = off (the default). Over the ten labelled boards: off f1 0.762, 40 -> 0.785, 60 -> 0.791, 80 -> 0.766, so 40-60 is a plateau. It buys precision with recall, and only a played round prices that trade")
    live.add_argument("--scale", type=float, default=1.0)
    live.add_argument("--no-dark", dest="include_dark", action="store_false")
    live.add_argument("--base")
    live.add_argument("--no-base", dest="use_base", action="store_false")
    live.add_argument("--base-only", action="store_true")
    live.add_argument("--debug-dir", type=Path)

    add_play_args(
        sub.add_parser("play", help="grab the board, pick a chain, and drag it for real"),
        merge_default=False)
    # Same loop, but folds two-tone faces (Pluto's muzzle, Piglet's snout) into
    # one tsum before detecting. Measured across 10 real frames it finds Dumbo
    # and Piglet better but fuses adjacent Mickeys, costing more base chains
    # than it gains -- kept separate so it can be A/B'd rather than assumed.
    add_play_args(
        sub.add_parser("play2", help="like play, with two-tone face merging (experimental)"),
        merge_default=True)

    args = ap.parse_args()

    if args.cmd == "grab":
        return _grab(args)

    if args.cmd == "hold":
        return _hold(args)

    if args.cmd == "assist":
        return _assist(args)

    if args.cmd == "idle":
        return _idle(args)

    if args.cmd == "label":
        return _label(args)

    if args.cmd == "eval":
        return _eval(args)

    if args.cmd == "skillcheck":
        return _skillcheck(args)

    if args.cmd == "score":
        return _score(args)

    if args.cmd == "dataset":
        return _dataset(args)

    if args.cmd == "learn":
        return _learn(args)

    if args.cmd in ("play", "play2"):
        return _play(args)

    if args.cmd == "synth":
        cv2.imwrite(str(args.out), synth(count=args.count, seed=args.seed, tilt=args.tilt))
        print(f"wrote {args.out}")
        return 0

    if args.cmd == "live":
        return _live(args)

    img = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    bx, by, bw, bh = _board_rect(img.shape, args.board)
    crop = img[by:by + bh, bx:bx + bw]

    t0 = time.perf_counter()
    tsums, radius, centres = detect(crop, k=args.k, radius=args.radius, scale=args.scale,
                                    include_dark=args.include_dark, merge=args.merge,
                                    bowl_reject=args.bowl_reject, debug_dir=args.debug_dir)
    t_detect = time.perf_counter() - t0

    # Read the skill icon off the FULL frame, not the board crop -- the button
    # sits below the play area.
    base, base_dist = (None, float("inf"))
    if args.use_base:
        base, base_dist = read_base_kind(img, centres, spec=args.base, debug_dir=args.debug_dir)

    t0 = time.perf_counter()
    cluster_map = _quantise(crop, args.k, centres)[0] if args.mode == "blob" else None
    chains = find_chains(tsums, radius, args.link, block=args.block,
                         link_px=args.link_px, base_kind=base, base_only=args.base_only,
                         mode=args.mode, max_chain=args.max_chain,
                         first_leg_px=getattr(args, "first_leg_px", 0.0),
                         labels=cluster_map)
    t_path = time.perf_counter() - t0

    overlay = draw(crop, tsums, chains, radius)
    canvas = img.copy()
    canvas[by:by + bh, bx:bx + bw] = overlay
    cv2.imwrite(str(args.out), canvas)

    print(f"detected {len(tsums)} tsums, r~{radius:.1f}px  "
          f"({t_detect * 1000:.0f}ms detect, {t_path * 1000:.0f}ms path)")
    counts: dict[int, int] = {}
    for t in tsums:
        counts[t.kind] = counts.get(t.kind, 0) + 1
    print("  per kind:", ", ".join(
        f"#{k}:{v}{' <- BASE' if k == base else ''}" for k, v in sorted(counts.items())))
    if args.use_base:
        # A big distance means the icon matched nothing on the board; say so
        # rather than silently prioritising a wrong colour.
        note = "" if base_dist < 30 else "  (WEAK -- check debug base_icon.png / --base)"
        print(f"  base tsum: cluster #{base}, Lab distance {base_dist:.1f}{note}")
    for c in chains:
        print(f"  chain kind #{c.kind}: {len(c)} tsums{'  [BASE]' if c.is_base else ''}")

    waypoints = [
        {"x": round(tsums[i].x + bx, 1), "y": round(tsums[i].y + by, 1)}
        for i in (chains[0].nodes if chains else [])
    ]
    side = args.out.with_suffix(".json")
    side.write_text(json.dumps({
        "detected": len(tsums), "radius": radius,
        "base_kind": base, "base_distance": None if base is None else round(base_dist, 2),
        "chains": [{"kind": c.kind, "length": len(c), "base": c.is_base} for c in chains],
        "drag": waypoints,
    }, indent=2))
    print(f"wrote {args.out} and {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
