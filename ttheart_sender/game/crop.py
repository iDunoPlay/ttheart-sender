"""The one place a tsum crop is cut, and the profile that says how.

**Why this module exists at all.** A crop is cut in two places -- once offline
when `scripts/crops.py` builds the training set, and once per frame at play time
when a model is asked about a board. When those two disagree the model is served
a picture it never trained on, and nothing raises: it just gets quietly worse.
This project has already paid for that lesson once, on a colour feature that was
computed two different ways and cost 312 rounds of play before anyone noticed.
So there is one implementation, and both callers import it.

**The profile is part of the model, not part of the code.** A model trained on
crops cut one way must be served crops cut the same way, and "remember how you
trained it" is not a mechanism. Every exported model records its profile name in
its `.json`, the runtime reads it back, and a model that asks for a profile this
build does not have is refused rather than served the default.

    PLAIN       what every model up to 1.11.x was trained on
    PADDED_V1   the frame is extended outward so a tsum at the board's rim
                yields a crop instead of nothing

Why PADDED_V1 might matter: a crop is a square of one radius about a tsum's
centre, and `PLAIN` REFUSES it when the frame edge would clip it -- a sliver
stretched to a square is a picture of the board edge rather than of a character.
That refusal costs **40.7% of tsums above the visibility floor, and 52% of the
clearest ones**, because a tsum at the rim has nothing lying on top of it and
sits a median of 1 pixel from the edge. Padding replaces the missing pixels by
replicating the border, which is a guess -- but a consistent one, applied in
training and at play time both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CropProfile:
    """Everything that decides what picture a model sees.

    Frozen, and named, so that a model can record which one it was trained
    under and the runtime can refuse to serve it anything else.
    """

    name: str
    #: Half-width of the crop in detected radii. 1.0, not 1.5, and the
    #: difference is the whole quality of the training set: two touching tsums
    #: sit ~2.44r apart, so a 1.5r window reaches well into the neighbour.
    window: float = 1.0
    #: The size a crop is stored at on disk. Models upsample from here rather
    #: than from the frame, so a model is never shown a sharper picture than
    #: the training set held.
    store: int = 64
    #: Extend the frame instead of refusing a crop the edge would clip.
    pad: bool = False
    #: How the extension is filled. Replicate rather than a constant: a flat
    #: black border is a strong, learnable "this one is at the edge" cue, and
    #: the model would use it instead of looking at the tsum.
    border: int = cv2.BORDER_REPLICATE


PLAIN = CropProfile("plain")
PADDED_V1 = CropProfile("padded_v1", pad=True)

#: Every profile a model may name. Looking one up by name is how the runtime
#: refuses a model it cannot serve correctly.
PROFILES = {p.name: p for p in (PLAIN, PADDED_V1)}

#: What a model's `.json` gets when it does not say. Every model trained before
#: profiles existed was trained on `PLAIN`, so this is a fact about the past
#: rather than a preference.
DEFAULT = PLAIN.name


def profile(name: Optional[str]) -> CropProfile:
    """A profile by name. An unknown name raises rather than defaulting.

    Silently falling back to `PLAIN` for a model trained with padding is the
    precise failure this module exists to prevent, so it is not possible here.
    """
    key = name or DEFAULT
    if key not in PROFILES:
        raise ValueError(
            "unknown crop profile %r -- this model was trained with a crop "
            "rule this build does not have. Known: %s"
            % (key, ", ".join(sorted(PROFILES))))
    return PROFILES[key]


def cut(bgr: np.ndarray, x: float, y: float, radius: float,
        prof: CropProfile = PLAIN) -> Optional[np.ndarray]:
    """One square crop centred on a tsum at `prof.store` px, or None.

    Returns None when the crop cannot be made honestly: off the frame entirely,
    or clipped by the edge under a profile that does not pad. Clamping to the
    bounds and resizing anyway turns a thin strip into a square, which is why a
    whole cluster once came back looking "too zoomed in" -- it was a sliver of a
    tsum stretched to fill 64x64. A partial crop is not a smaller picture of a
    character, it is a different picture.
    """
    half = max(4, int(round(radius * prof.window)))
    h, w = bgr.shape[:2]
    x0, y0 = int(x) - half, int(y) - half
    x1, y1 = int(x) + half + 1, int(y) + half + 1
    clipped = x0 < 0 or y0 < 0 or x1 > w or y1 > h
    if clipped and not prof.pad:
        return None

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None                      # centre is off the frame entirely
    patch = bgr[sy0:sy1, sx0:sx1]
    if clipped:
        # The PATCH is padded, not the frame. Padding a 525x456 board once per
        # tsum would cost more than the model does, and the result is identical.
        patch = cv2.copyMakeBorder(patch, sy0 - y0, y1 - sy1,
                                   sx0 - x0, x1 - sx1, prof.border)
    if patch.size == 0 or min(patch.shape[:2]) < 4:
        return None
    return cv2.resize(patch, (prof.store, prof.store),
                      interpolation=cv2.INTER_AREA)


def for_model(bgr: np.ndarray, x: float, y: float, radius: float, size: int,
              prof: CropProfile = PLAIN) -> Optional[np.ndarray]:
    """The picture a model is served: :func:`cut`, then up to `size`.

    Two resizes, not one, and deliberately: the training crops were written to
    disk at `store` and upsampled from there, so going straight to `size` would
    hand the model a sharper image than it has ever seen.
    """
    small = cut(bgr, x, y, radius, prof)
    if small is None:
        return None
    if size == prof.store:
        return small
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
