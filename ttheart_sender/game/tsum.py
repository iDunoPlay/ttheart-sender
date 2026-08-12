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
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from ..exceptions import StopRequested

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
        "board": (8, 265, 522, 535),
        "base": "83,858,26",
    },
    (956, 542): {                       # saved screenshot, no emulator chrome
        "board": (8, 258, 524, 505),
        "base": "76,830,26",
    },
}


def _layout(shape) -> dict:
    return LAYOUTS.get((shape[0], shape[1]), {})

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


def _quantise(bgr: np.ndarray, k: int, palette: Optional[np.ndarray] = None,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means the image in Lab space. Returns (label image, cluster centres).

    Pass `palette` to reuse centres from an earlier frame. Worth doing in a live
    loop: the fit is most of the cost here and the colours on the board don't
    change mid-run, so refitting every frame buys nothing.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    flat = lab.reshape(-1, 3).astype(np.float32)

    if palette is None:
        # Fitting on every pixel is pointless -- 40k samples pins the centres
        # just as well.
        step = max(1, flat.shape[0] // 40_000)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(seed)
        _, _, palette = cv2.kmeans(flat[::step], k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

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
    floor_frac: float = 0.42,
    hole_frac: float = 0.8,
    palette: Optional[np.ndarray] = None,
    scale: float = 1.0,
    debug_dir: Optional[Path] = None,
) -> tuple[list[Tsum], float, np.ndarray]:
    """Locate every tsum. Coordinates and radius come back in `bgr` pixels.

    `scale` < 1 runs the whole pipeline on a downscaled copy: cost is roughly
    quadratic in resolution, and tsums are big enough that half-size still
    separates them. Measured on a 523x542 board: 49ms at 1.0 -> 13ms at 0.5,
    same chain, a couple of deeply buried tsums lost.
    """
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if radius is not None:
            radius *= scale

    labels, centres = _quantise(bgr, k, palette)
    skip = _background_clusters(labels, centres)
    # Dark pixels are two different things at once: every tsum's outline, ears
    # and shadow, AND the face of a black tsum like classic Mickey. Treating
    # them as one cluster fuses the whole board; discarding them loses Mickey
    # entirely. So they get split off and processed under their own rules below.
    dark = {i for i, c in enumerate(centres) if c[0] < dark_l} - skip
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

    found: list[Tsum] = []
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
            found.append(Tsum(cx, cy, min(score, radius), kind, colour))

    # A blob can peak twice on one tsum (a highlight splits the mask, or a
    # loose peak floor finds two maxima on one face). Keep only the deepest
    # peak in a neighbourhood. 1.6r is set by geometry, not taste: two tsums
    # that genuinely touch sit ~2.44r apart, so anything closer than 1.6r is
    # two readings of one tsum. Measured over 14 real boards, 1.6 removes every
    # impossible overlap (223 -> 0) while keeping 730 detections.
    found.sort(key=lambda t: -t.r)
    kept: list[Tsum] = []
    for t in found:
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

    if recolour > 0:
        kept = _recolour(bgr, kept, radius, recolour)

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
) -> tuple[Optional[int], float]:
    """Identify which colour cluster the bottom-left skill icon shows.

    That icon is the tsum you equipped, so it's the one whose chains actually
    charge the skill gauge -- which makes it the chain worth finding, not
    whichever type happens to have the longest run on the board.

    Returns (cluster index, Lab distance). A large distance means no cluster
    really matched and the caller should not trust it.
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


def _board_rect(shape, spec: Optional[str]):
    h, w = shape[:2]
    if spec is None:
        measured = _layout(shape).get("board")
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
    #: Total tsums in chains that verifiably cleared. The throughput number.
    cleared: int = 0
    #: Drags that ran but changed nothing -- the cost of an over-permissive
    #: link rule, and the half a positive-only label set cannot measure.
    stalled: int = 0
    #: --verify-hold only: chain members the game declined to mark, dropped
    #: before the stroke walked to them.
    trimmed: int = 0
    #: --verify-hold only: presses released without dragging, because what the
    #: game marked was already below min_chain.
    abandoned: int = 0
    #: Why the loop ended -- shown by the CLI and returned to the flow.
    reason: str = ""
    #: True when the stop key ended it rather than a normal exit condition.
    stopped: bool = False

    def describe(self) -> str:
        chains = f"played {self.played} chains, cleared {self.cleared} tsums"
        if self.played:
            chains += f" (mean {self.cleared / self.played:.1f}/chain)"
        out = f"{chains}, {self.stalled} drag(s) did not register"
        if self.trimmed or self.abandoned:
            out += (f"; the game trimmed {self.trimmed} member(s) and rejected "
                    f"{self.abandoned} chain(s) outright")
        return out


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


def marked_by_game(drv, before_crop: np.ndarray, board: tuple, tsums: Sequence[Tsum],
                   nodes: Sequence[int], *, delay: float, threshold: float,
                   aura: float) -> list[int]:
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
    """
    bx, by, bw, bh = board
    time.sleep(delay)
    crop = drv.grab()[by:by + bh, bx:bx + bw]
    diff = cv2.absdiff(crop, before_crop).max(axis=2)

    head = tsums[nodes[0]]
    r = max(2, int(min(t.r for t in tsums) * 0.55))
    keep = [nodes[0]]
    for n in nodes[1:]:
        t = tsums[n]
        if math.hypot(t.x - head.x, t.y - head.y) <= aura:
            keep.append(n)
            continue
        m = np.zeros(diff.shape, np.uint8)
        cv2.circle(m, (int(t.x), int(t.y)), r, 1, -1)
        if float(diff[m.astype(bool)].mean()) > threshold:
            keep.append(n)
    return keep


def _settle(drv: Driver, *, max_wait: float, tol: float = 2.5):
    """Wait for the board to stop moving, then return the frame it stopped on.

    Cleared tsums drop and the pile collapses; detecting mid-fall gives
    coordinates that are already stale by the time the drag runs. Frame diffing
    costs ~5ms, so this is faster than any fixed delay big enough to be safe --
    it returns the instant the board is still. Capped, because FEVER animates
    continuously and would otherwise wait forever.
    """
    frame = prev = drv.grab()
    deadline = time.perf_counter() + max_wait
    while time.perf_counter() < deadline:
        drv.check_stop()
        frame = drv.grab()
        if float(np.mean(cv2.absdiff(frame, prev))) < tol:
            return frame
        prev = frame
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

    bubbles = _load_bubbles(drv, opts.bubble) if opts.bubble and not opts.dry_run else []

    deadline = time.perf_counter() + opts.duration if opts.duration > 0 else None
    report = PlayReport()

    # Fitting the colour palette is the expensive half of detection, and the
    # tsums in play don't change mid-game, so it's fit once and reused. Same for
    # the radius and the base-tsum lookup.
    palette, radius, base = None, opts.radius, None
    played = misses = stalls = 0
    # Rolling window of "which chain did we just play". A board that keeps
    # offering the same handful of chains is a board nothing is clearing on --
    # see the loop-detection check below.
    from collections import deque

    recent: deque = deque(maxlen=opts.repeat_window)
    lengths: deque = deque(maxlen=max(opts.repeat_len, opts.repeat_window))
    per_step = opts.per_step
    skip_kinds: set[int] = set()
    #: Set when the last pass ended without touching the board, so the next one
    #: can skip waiting for movement that cannot have happened.
    no_settle = False

    try:
        while True:
            drv.check_stop()
            frame = (drv.grab() if opts.dry_run or palette is None or no_settle
                     else _settle(drv, max_wait=opts.settle))
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

            bx, by, bw, bh = _board_rect(frame.shape, opts.board)
            crop = frame[by:by + bh, bx:bx + bw]

            t0 = time.perf_counter()
            tsums, radius, palette = detect(crop, k=opts.k, radius=radius, palette=palette,
                                            scale=opts.scale, include_dark=opts.include_dark,
                                            merge=opts.merge)

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
            plausible = opts.min_tsums <= len(tsums) <= opts.max_tsums
            if not plausible and palette is not None:
                fresh, fresh_r, fresh_pal = detect(crop, k=opts.k, scale=opts.scale,
                                                   include_dark=opts.include_dark,
                                                   merge=opts.merge)
                if abs(len(fresh) - opts.min_tsums) < abs(len(tsums) - opts.min_tsums):
                    say(f"    recalibrated ({len(tsums)} -> {len(fresh)} tsums)")
                    tsums, radius, palette, base = fresh, fresh_r, fresh_pal, None

            if base is None and opts.use_base:
                base, base_dist = read_base_kind(frame, palette, spec=opts.base)
                say(f"base tsum: cluster #{base} (Lab distance {base_dist:.1f})")
            chains = find_chains(tsums, radius, opts.link, block=opts.block,
                                 link_px=opts.link_px, base_kind=base, base_only=opts.base_only,
                                 mode=opts.mode, max_chain=opts.max_chain,
                                 first_leg_px=opts.first_leg_px)
            think = (time.perf_counter() - t0) * 1000

            # A real board is crowded but bounded. Menus and the results screen
            # still yield blobs and can still produce a "chain", and dragging
            # that would swipe across live UI buttons -- so a frame outside the
            # plausible range is treated as no board at all rather than trusted.
            #
            # The ceiling matters as much as the floor: the Home screen scores
            # 200+ "tsums" off portraits and panel texture, sails past any
            # minimum, and produces a confident chain every single frame.
            if not (opts.min_tsums <= len(tsums) <= opts.max_tsums):
                chains = []

            # Chain length is set by detection recall, not by the search: a
            # tsum missed in the middle of a run of six splits it into a three
            # and a two. Holding out for a longer chain is therefore mostly a
            # bet that the next frame is better settled.
            longest = len(chains[0]) if chains else 0
            chains = [c for c in chains
                      if len(c) >= opts.min_chain and c.kind not in skip_kinds]

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
                    palette, radius, base = None, opts.radius, None
                    skip_kinds.clear()
                    misses = 0
                continue
            misses = 0

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
                    palette, radius, base = None, opts.radius, None
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
                palette, radius, base = None, opts.radius, None
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
                palette, radius, base = None, opts.radius, None
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

            def _ask_the_game():
                """Trim the chain to what the game marks, while it is held."""
                kept = marked_by_game(drv, before, (bx, by, bw, bh), tsums,
                                      best.nodes, delay=opts.hold_delay,
                                      threshold=opts.hold_threshold,
                                      aura=opts.hold_aura)
                probe["dropped"] = len(best.nodes) - len(kept)
                probe["kept"] = len(kept)
                if len(kept) < opts.min_chain:
                    # Abandon before moving. Releasing on one tsum clears
                    # nothing, which is cheaper than dragging a chain the game
                    # has already said it will not accept.
                    probe["abandoned"] = True
                    return [drv.to_screen(bx + tsums[best.nodes[0]].x,
                                          by + tsums[best.nodes[0]].y)]
                return [drv.to_screen(bx + tsums[i].x, by + tsums[i].y) for i in kept]

            drag_chain(screen, step_px=opts.step_px, per_step=per_step, hold=opts.hold,
                       after_press=_ask_the_game if opts.verify_hold else None)

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
                        palette, radius, base = None, opts.radius, None
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

            played += 1
            # What was dragged, not what was proposed: with --verify-hold the
            # chain may have been trimmed after the press.
            report.cleared += probe.get("kept", len(best))

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

    report.played = played
    say(report.describe())
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
    pyautogui.mouseDown()
    shots = []
    try:
        for dx in (1, 0, -1, 0):
            pyautogui.moveTo(sx + dx, sy)
            time.sleep(0.02)
        # Sample throughout rather than once at the end: a highlight that
        # animates in, or one that shows briefly and stops, would be missed by
        # a single grab timed badly.
        deadline = time.perf_counter() + args.hold
        while time.perf_counter() < deadline:
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

    # The one comparison that settles it. If the game marks matches, the tsums
    # we predicted share the character should react and the rest should not.
    if expect and len(expect) > 1:
        idx = tsums.index(pick)
        partners = [changes[i] for i in expect if i != idx]
        rest = [c for i, c in enumerate(changes) if i != idx and i not in expect]
        print(f"\npredicted partners ({len(partners)}): "
              f"{', '.join(f'{c:.1f}' for c in partners)}")
        print(f"everything else: median {np.median(rest):.1f}, "
              f"max {max(rest):.1f}" if rest else "everything else: none")
        hot_partners = sum(1 for c in partners if c > args.threshold)
        if hot_partners and rest and np.median(partners) > 3 * max(1.0, np.median(rest)):
            print(f"-> {hot_partners}/{len(partners)} predicted partners reacted and "
                  f"the rest of the board did not. The game IS marking matches, and "
                  f"that mark is character ground truth worth reading.")
        elif hot_partners:
            print(f"-> {hot_partners} partners reacted, but so did the rest of the "
                  f"board -- that is motion, not marking. Re-run on a stiller board.")
        else:
            print("-> the pressed tsum reacted and its predicted partners did not. "
                  "The game marks only what you are touching, so there is nothing "
                  "to read off and this route is closed.")
    elif lit == 0:
        print("-> only the pressed tsum reacted, but it had no partners to mark, "
              "so this run cannot tell you which. Re-run for a board with a chain.")
    print(f"wrote hold_before/held/diff/marked.png to {out}")
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
              "open_ratio", "recolour", "floor_frac", "hole_frac", "scale")

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


def _detect_labelled(data: dict, folder: Path, params: dict) -> tuple[list[Tsum], list[list[float]], float]:
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

    tsums, radius, _ = detect(img[by:by + bh, bx:bx + bw], **params)
    return tsums, [[t.x + bx, t.y + by] for t in tsums], radius


def _eval_once(labels: Sequence[tuple[Path, dict]], params: dict,
               tol_frac: float, verbose: bool) -> dict:
    """Score one parameter set against every reviewed board."""
    tp = fp = fn = 0
    splits = merges = grouped = 0
    per_board = []

    for path, data in labels:
        truth = _truth_points(data)
        tsums, found, _ = _detect_labelled(data, path.parent, params)
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
            "merge": args.merge, "scale": args.scale}
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
    boards = []   # (tsums, radius, [(i, j), ...]) -- one entry per labelled board
    total = 0
    for lf in label_files:
        data = json.loads(lf.read_text())
        if not data.get("paths"):
            continue
        tsums, found, radius = _detect_labelled(data, folder, {"k": 12, "include_dark": True})
        tol = tol_frac * float(data["radius"])
        links = []
        for path in data["paths"]:
            nodes = path["nodes"]
            pairs, _, _ = _match(nodes, found, tol)
            at = {i: j for i, j in pairs}
            links += [(at[a], at[a + 1]) for a in range(len(nodes) - 1)
                      if a in at and a + 1 in at]
        if links:
            boards.append((tsums, radius, links))
            total += len(links)

    if not total:
        print("\n(no labelled path survived re-detection, so the link rule "
              "cannot be replayed -- re-label a board and try again)")
        return

    print(f"\nreplaying {total} of your links through adjacency() itself:")
    print(f"  {'link-px':>8}  {'accepted':>8}  {'wrong kind':>10}  {'too far':>8}  {'blocked':>8}")
    for cap in (90, 100, 105, 120, 150, 200, 0):
        ok = kind = far = blocked = 0
        for tsums, radius, links in boards:
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
                                        debug_dir=args.debug_dir)
        det_ms.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        base = None
        if args.use_base:
            base, _ = read_base_kind(frame, palette, spec=args.base, debug_dir=args.debug_dir)
        chains = find_chains(tsums, radius, args.link, block=args.block,
                         link_px=args.link_px, base_kind=base, base_only=args.base_only,
                                 mode=args.mode, max_chain=args.max_chain,
                                 first_leg_px=getattr(args, "first_leg_px", 0.0))
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
                           "press is the start of the stroke either way")
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
    play.add_argument("--mode", choices=["touch", "reach"], default="touch", help='"touch" (default): only tsums within --link-px of each other, with nothing blocking the line between them, count as linked. "reach": any two same-kind tsums link regardless of distance, which is closer to how the game is actually played -- distance stops mattering and --link-px/--link/--block are ignored')
    play.add_argument("--block", type=float, default=0.75,
                      help="a tsum within this many radii of the line blocks the link")
    play.add_argument("--scale", type=float, default=1.0)
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
    a.add_argument("--mode", choices=["touch", "reach"], default="touch", help='"touch" (default): only tsums within --link-px of each other, with nothing blocking the line between them, count as linked. "reach": any two same-kind tsums link regardless of distance, which is closer to how the game is actually played -- distance stops mattering and --link-px/--link/--block are ignored')
    a.add_argument("--block", type=float, default=0.75, help="occlusion radius for blocking a link")
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
    hp.add_argument("--no-dark", dest="include_dark", action="store_false")
    hp.add_argument("--no-prepare", action="store_true")

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

    live = sub.add_parser("live", help="grab the emulator N times and report real throughput")
    live.add_argument("-n", "--frames", type=int, default=20)
    live.add_argument("-o", "--out", type=Path, default=Path("live_out.png"))
    live.add_argument("--board")
    live.add_argument("-k", type=int, default=8)
    live.add_argument("--radius", type=float)
    live.add_argument("--link", type=float, default=1.35)
    live.add_argument("--link-px", type=float, default=100.0, help="link distance in PIXELS (stable across frames; from labelled links: p90=95). Set 0 to fall back to --link diameters")
    live.add_argument("--max-chain", type=int, default=8, help="cap chain length; picks the tightest cluster of N rather than truncating a board-wide path. 0 = no cap")
    live.add_argument("--mode", choices=["touch", "reach"], default="touch", help='"touch" (default): only tsums within --link-px of each other, with nothing blocking the line between them, count as linked. "reach": any two same-kind tsums link regardless of distance, which is closer to how the game is actually played -- distance stops mattering and --link-px/--link/--block are ignored')
    live.add_argument("--block", type=float, default=0.75)
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
                                    debug_dir=args.debug_dir)
    t_detect = time.perf_counter() - t0

    # Read the skill icon off the FULL frame, not the board crop -- the button
    # sits below the play area.
    base, base_dist = (None, float("inf"))
    if args.use_base:
        base, base_dist = read_base_kind(img, centres, spec=args.base, debug_dir=args.debug_dir)

    t0 = time.perf_counter()
    chains = find_chains(tsums, radius, args.link, block=args.block,
                         link_px=args.link_px, base_kind=base, base_only=args.base_only,
                                 mode=args.mode, max_chain=args.max_chain,
                                 first_leg_px=getattr(args, "first_leg_px", 0.0))
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
