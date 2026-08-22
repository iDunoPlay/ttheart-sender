"""Reading FEVER off the screen instead of inferring it from a timer.

`max_fever` is the meter an instant before FEVER starts -- a trigger. Turning
one match into a ten-second assumption is the only thing it can support, and
that assumption is wrong whenever the trigger is missed or FEVER does not last
exactly ten seconds.

`fever_bonus` is the banner the game shows for the whole of FEVER, so it can be
asked every frame. Getting it to work took two dead ends, both pinned below,
because the obvious approaches fail in ways that look like success: one scores
~0.61 on every frame and the other ~0.966 on every frame, and neither is
distinguishable from "working" without checking a negative.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.tsum import (FEVER_BANNER_CONFIDENCE, FEVER_HOLD,
                                      FEVER_SECONDS, FeverWatch,
                                      fever_banner_score)
from ttheart_sender.screen.matcher import TemplateMatcher
from ttheart_sender.screen.templates import TemplateLibrary

ROOT = Path(__file__).resolve().parents[1]

#: Where the banner renders in a 994x578 capture, measured off a real frame.
BANNER_AT = (279, 173)

#: The top of the board rect for that capture size -- what `update` passes.
BOARD_TOP = 314


@pytest.fixture
def library():
    return TemplateLibrary(ROOT / "templates")


@pytest.fixture
def ink(library):
    return tsum._glyph_ink(library.get("fever_bonus").image)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class NoMatch:
    """`max_fever` never matches, so only the banner can decide."""

    def find(self, frame, template, confidence=None):
        return None


class Always:
    """`max_fever` matches on every frame -- the trigger's side of the story."""

    def find(self, frame, template, confidence=None):
        return object()


def _frame(banner, ink_template):
    """A 994x578 frame with, or without, the banner painted where it renders."""
    frame = np.full((994, 578, 3), 40, np.uint8)
    if banner:
        raw = cv2.imdecode(np.fromfile(str(ROOT / "templates" / "fever_bonus.png"),
                                       np.uint8), cv2.IMREAD_UNCHANGED)
        y, x = BANNER_AT
        h, w = ink_template.shape[:2]
        frame[y:y + h, x:x + w] = raw[:, :, :3]
    return frame


# --------------------------------------------------------------------------
# reading the banner
# --------------------------------------------------------------------------
def test_the_banner_is_read_where_the_game_draws_it(ink):
    assert fever_banner_score(_frame(True, ink), ink, BOARD_TOP) >= FEVER_BANNER_CONFIDENCE


def test_a_board_without_the_banner_scores_below_the_threshold(ink):
    assert fever_banner_score(_frame(False, ink), ink, BOARD_TOP) < FEVER_BANNER_CONFIDENCE


def test_the_read_survives_a_different_board_showing_through(ink):
    """The whole point: the letters are constant, the background is not.

    The banner is text over the live board, so most of the template's box is
    whatever tsums happen to be behind it. Painting noise there must not move
    the answer -- if it does, the reader is matching the background.
    """
    rng = np.random.default_rng(4)
    scores = []
    for _ in range(5):
        frame = _frame(True, ink)
        y, x = BANNER_AT
        h, w = ink.shape[:2]
        noise = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        patch = frame[y:y + h, x:x + w]
        gaps = tsum._glyph_ink(patch) == 0
        patch[gaps] = noise[gaps]
        scores.append(fever_banner_score(frame, ink, BOARD_TOP))

    assert min(scores) >= FEVER_BANNER_CONFIDENCE, \
        f"background noise moved the read: {scores}"


def test_the_two_approaches_that_do_not_work(library):
    """Kept as a test so nobody 'simplifies' the reader back into one of them.

    Both fail by scoring the SAME on frames with and without the banner, which
    is the failure mode that looks like a working matcher until you check a
    negative.
    """
    matcher = TemplateMatcher(confidence=0.85)
    plain = library.get("fever_bonus")
    glyphs = tsum._glyph_ink(plain.image)
    with_banner, without = _frame(True, glyphs), _frame(False, glyphs)

    # 1. Straight correlation over the whole template box.
    near = abs(matcher.best_score(with_banner, plain)
               - matcher.best_score(without, plain))

    # 2. Masking the template to the glyphs, which makes TemplateMatcher
    #    switch to TM_CCORR_NORMED -- no mean subtraction, ~1.0 on anything.
    masked = type(plain)(name="m", path=plain.path, image=plain.image, mask=glyphs)
    near_masked = abs(matcher.best_score(with_banner, masked)
                      - matcher.best_score(without, masked))

    shape = abs(fever_banner_score(with_banner, glyphs, BOARD_TOP)
                - fever_banner_score(without, glyphs, BOARD_TOP))

    assert shape > near, "the shape reader must separate better than raw correlation"
    assert shape > near_masked, "...and better than a mask, which switches method"


# --------------------------------------------------------------------------
# turning sightings into a state
# --------------------------------------------------------------------------
def test_no_assumption_is_made_about_how_long_fever_lasts(library, ink):
    """A run has no fixed length to lean on, so one sighting buys one hold.

    An earlier version spent a full FEVER_SECONDS on the first sighting, on the
    theory that ten seconds is a game rule. It is not: a skill firing pauses
    FEVER until its animation finishes, and some skills start FEVER outright.
    Leaning on the length also made a single spurious read cost ten seconds of
    wrong state instead of one hold.
    """
    clock = Clock()
    watch = FeverWatch(NoMatch(), library, clock=clock)
    assert watch.reads_state

    assert watch.update(_frame(True, ink)) is True
    clock.advance(FEVER_HOLD - 0.1)
    assert watch.active
    clock.advance(0.2)
    assert not watch.active, "one sighting must not buy more than one hold"

    assert FEVER_HOLD < FEVER_SECONDS, "the hold is a dropout bridge, not a duration"


def test_fever_ends_promptly_once_the_banner_goes(library, ink):
    """The other half: a long opening window must not make the END late.

    Sightings after the first only top up by FEVER_HOLD, so FEVER lapses within
    that of the banner disappearing -- not ten seconds after the last sighting.
    """
    clock = Clock()
    watch = FeverWatch(NoMatch(), library, clock=clock)

    for _ in range(30):
        watch.update(_frame(True, ink))
        clock.advance(0.4)

    clock.advance(FEVER_HOLD + 0.1)
    assert not watch.active, "FEVER should end within the hold of the banner going"


def test_the_live_log_that_broke_the_flat_window(library, ink):
    """Replays the round that showed the defect, at the timings it logged.

    One real eleven-second FEVER read as three, because two dropouts each
    outlasted a flat 1.0s window: FEVER :37, NORMAL :40, FEVER :40, NORMAL :43,
    FEVER :43, NORMAL :48. Each flip refits the palette and re-reads the base
    tsum, and the base came back a different cluster every time.
    """
    times = [0.0, 0.6, 1.2, 2.0, 2.6, 3.0, 3.4, 4.4, 5.0, 5.6,
             6.0, 6.5, 7.6, 8.2, 9.0, 10.0, 10.6]
    # Two stretches the banner could not be read: a skill animation covering
    # the screen, and the same again later in the run.
    dark = {3.0, 3.4, 4.4, 6.0, 6.5, 7.6}
    seen = {at: at not in dark for at in times}

    def replay(hold):
        clock = Clock()
        watch = FeverWatch(NoMatch(), library, clock=clock, hold=hold)
        flips, last = [], 0.0
        for at, banner in sorted(seen.items()):
            clock.advance(at - last)
            last = at
            watch.update(_frame(banner, ink))
            moved = watch.took_effect()
            if moved:
                flips.append((at, moved))
        return flips, watch.active

    flips, running = replay(FEVER_HOLD)
    assert flips == [(0.0, "fever")], f"FEVER should read as one stretch, got {flips}"
    assert running, "and still be running at the end of the log"

    # The window this replaced, on the same frames, to keep the test honest:
    # if the log stopped separating the two there would be nothing here to pin.
    flat, _ = replay(1.0)
    assert sum(1 for _, moved in flat if moved == "fever") == 3,         f"the 1.0s window should still read this as three FEVERs, got {flat}"


def test_the_hold_re_arms_every_frame_the_banner_is_up(library, ink):
    """FEVER can run longer than one hold; each sighting extends it."""
    clock = Clock()
    watch = FeverWatch(NoMatch(), library, clock=clock)

    for _ in range(20):
        assert watch.update(_frame(True, ink)) is True
        clock.advance(FEVER_HOLD - 0.5)


def test_a_missed_frame_mid_fever_does_not_end_it(library, ink):
    """The banner fades in and out, and those frames score too low to match.

    Measured on real captures: the fades read 0.20-0.28 against a 0.35
    threshold. The hold exists to cover exactly them, so a dropout shorter than
    it must not flip the state.
    """
    clock = Clock()
    watch = FeverWatch(NoMatch(), library, clock=clock)

    watch.update(_frame(True, ink))
    clock.advance(0.6)
    assert watch.update(_frame(False, ink)) is True, \
        "a single faded frame is not the end"


# --------------------------------------------------------------------------
# what is left for the trigger
# --------------------------------------------------------------------------
def test_the_trigger_no_longer_assumes_ten_seconds_when_the_banner_leads(library, ink):
    """`max_fever` fires just BEFORE the banner is drawn, so it bridges a frame
    or two -- not the whole of FEVER, which the banner now reports directly.

    Measured over 151 captured frames the two never overlap: the trigger
    matches 3 frames, all immediately before FEVER, the banner the 54 of FEVER.
    """
    clock = Clock()
    watch = FeverWatch(Always(), library, clock=clock)

    watch.update(_frame(False, ink))
    assert watch.active
    clock.advance(FEVER_HOLD + 0.1)
    assert not watch.active, "the trigger should no longer buy ten seconds"


def test_without_the_banner_the_trigger_keeps_its_ten_seconds(library, ink):
    """The fallback has to behave exactly as it did before the banner existed."""
    clock = Clock()
    watch = FeverWatch(Always(), library, clock=clock, use_banner=False)
    assert not watch.reads_state

    watch.update(_frame(False, ink))
    clock.advance(FEVER_SECONDS - 0.1)
    assert watch.active
    clock.advance(0.2)
    assert not watch.active


def test_the_banner_can_be_switched_off_from_a_flow():
    defaults = vars(tsum.play_defaults())
    assert defaults["fever_banner"] is True, "the banner leads by default"

    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    tsum.add_play_args(ap, merge_default=False)
    assert ap.parse_args(["--no-fever-banner"]).fever_banner is False
